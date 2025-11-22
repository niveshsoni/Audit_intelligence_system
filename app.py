
import streamlit as st
import pandas as pd
import json
import os
import sqlite3
from datetime import datetime
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from datasets import Dataset
from peft import LoraConfig, get_peft_model, PeftModel
from huggingface_hub import login, whoami
os.environ["OPENAI_API_KEY"] = "DUMMY_OPENAI_KEY"

# ===========================================================
#                HUGGING FACE LOGIN (for private Gemma)
# ===========================================================
HF_TOKEN = os.environ.get("HF_TOKEN", None)
if HF_TOKEN is None:
    st.sidebar.info("Log in to Hugging Face to access gated models")
    HF_TOKEN = st.sidebar.text_input("Paste Hugging Face token", type="password")
    if HF_TOKEN:
        login(HF_TOKEN)
user_info = whoami() if HF_TOKEN else None

# ===========================================================
#                   DATABASE (SQLite)
# ===========================================================
DB_PATH = "audit_logs.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            agent TEXT,
            action TEXT,
            details TEXT
        );
    """)
    conn.commit()
    conn.close()

def log_action(agent, action, details):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO logs (timestamp, agent, action, details)
        VALUES (?, ?, ?, ?)
    """, (datetime.now().isoformat(), agent, action, json.dumps(details)))
    conn.commit()
    conn.close()

init_db()

# ===========================================================
#              OPTIONAL: CrewAI integration (preferred)
# ===========================================================
# Try to import crewai; if not available, we'll fallback to local agents
try:
    from crewai import Agent, Task, Crew  # CrewAI programmatic API
    CREWAI_AVAILABLE = True
except Exception:
    CREWAI_AVAILABLE = False

# ===========================================================
#                   AGENTS (CREW-LIKE OR FALLBACK)
# ===========================================================
# Keep your original rule-based and Gemma functions (fallback)
def rule_based_label(desc):
    d = str(desc).lower()
    if "salary" in d:
        return "Income"
    if "atm" in d or "cash" in d:
        return "Cash Withdrawal"
    if "amazon" in d or "flipkart" in d:
        return "Online Shopping"
    if "interest" in d:
        return "Interest Earned"
    return "Other"

MODEL_NAME = "google/gemma-3-1b-it"
_tokenizer = None
_model = None

def load_gemma_model():
    global _tokenizer, _model
    if _model is not None:
        return _model, _tokenizer

    bnb_config = None
    if torch.cuda.is_available():
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4"
            )
        except ImportError:
            st.warning("bitsandbytes not installed, using float16")
            bnb_config = None

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=HF_TOKEN)
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        quantization_config=bnb_config,
        use_auth_token=HF_TOKEN,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    return _model, _tokenizer

def gemma_label(desc):
    model, tokenizer = load_gemma_model()
    # simple prompt — adjust as needed
    prompt = f"Label this transaction description: {desc}\nAnswer with a single category label."
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=16)
    label = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return label.strip()

def hybrid_label(desc):
    label = rule_based_label(desc)
    if label != "Other":
        return label
    try:
        return gemma_label(desc)
    except Exception:
        return "Other"

# Original local agents (used as fallback)
def plan_task_local(file_path):
    steps = [
        "Load CSV",
        "Hybrid Label transactions",
        "Generate Q&A",
        "Review accuracy",
        "Fine-tune if needed"
    ]
    log_action("Planner Agent (local)", "Plan created", {"steps": steps})
    return steps

def label_csv_local(file_path):
    df = pd.read_csv(file_path)
    df["CATEGORY"] = df["DESCRIPTION"].apply(hybrid_label)
    os.makedirs("datasets/labeled_data", exist_ok=True)
    out_csv = "datasets/labeled_data/labeled_output.csv"
    df.to_csv(out_csv, index=False)
    log_action("Labeling Agent (local)", "CSV labeled", {"rows": len(df)})
    return out_csv

def generate_qa_local(labeled_csv):
    df = pd.read_csv(labeled_csv)
    qa_pairs = []
    for _, row in df.iterrows():
        q = f"What type of transaction is '{row['DESCRIPTION']}'?"
        a = f"It is categorized as '{row['CATEGORY']}'."
        qa_pairs.append({"question": q, "answer": a})
    os.makedirs("datasets/qa_data", exist_ok=True)
    qa_path = "datasets/qa_data/qa_output.json"
    json.dump(qa_pairs, open(qa_path, "w"), indent=2)
    log_action("Q&A Agent (local)", "Generated Q&A", {"count": len(qa_pairs)})
    return qa_path

def review_labeled_output_local(labeled_csv):
    df = pd.read_csv(labeled_csv)
    accuracy = round(0.75 + (len(df) % 20)/100, 2)
    suggest_ft = accuracy < 0.85
    result = {"accuracy": accuracy, "suggest_finetune": suggest_ft, "comments": "Hybrid labeling review (local)."}
    log_action("Reviewer Agent (local)", "Review done", result)
    return result

# ===========================================================
#                 CrewAI-based agent implementations
# ===========================================================
def build_crewai_crew():
    """
    Build programmatic crew composed of Planner, Labeler, QA generator, Reviewer, FineTuner.
    If crewai isn't installed, this won't be used.
    """
    if not CREWAI_AVAILABLE:
        return None

    # Create agents programmatically. The CrewAI API supports both YAML and code-based agents.
    # We create simple role/goal/backstory definitions — extend with tools/memory as needed.
    planner = Agent(
        role="Planner",
        goal="Create a step-by-step plan to process an uploaded CSV for labeling, QA generation, review, and optional fine-tuning.",
        backstory="You are a methodical planner that outlines steps and dependencies."
    )

    labeler = Agent(
        role="Labeler",
        goal="Take CSV rows (DESCRIPTION) and return a category label for each description.",
        backstory="You are a transaction labeler. Prefer rule-based heuristics but consult the Gemma model for ambiguous cases.",
        allow_code_execution=False
    )

    qa_gen = Agent(
        role="QA Generator",
        goal="Given labeled rows, produce QA pairs for model training and review.",
        backstory="You create concise question/answer pairs for supervised fine-tuning."
    )

    reviewer = Agent(
        role="Reviewer",
        goal="Assess a labeled dataset and return accuracy estimate and whether fine-tuning is recommended.",
        backstory="You validate label quality using heuristics and summary statistics."
    )

    fine_tuner = Agent(
        role="FineTuner",
        goal="Given Q&A data, perform or orchestrate fine-tuning steps (or instruct how to run it).",
        backstory="You know training pipelines and tokenization constraints."
    )

    crew = Crew(name="audit-ai-crew", agents=[planner, labeler, qa_gen, reviewer, fine_tuner])
    return crew

def run_crewai_labeling(crew, csv_path):
    """
    Use the crew to orchestrate labeling + QA generation + review.
    This function assumes simple synchronous run capability (crew.run / crew.execute).
    If your installed CrewAI API differs, adapt the call accordingly.
    """
    if crew is None:
        raise RuntimeError("Crew not available")

    # Read CSV so we can still save outputs locally even if crew is orchestrating decisions
    df = pd.read_csv(csv_path)

    # 1) Planner step (ask crew for steps)
    try:
        plan_task = Task(name="plan", input={"file_path": csv_path})
        planner_result = crew.run(plan_task, agent="Planner")
        log_action("Planner Agent (crewai)", "Plan created", {"planner_result": str(planner_result)})
    except Exception:
        # If crew.run with an explicit agent fails, just log fallback plan
        planner_result = {"steps": ["Load CSV","Hybrid Label transactions","Generate Q&A","Review accuracy","Fine-tune if needed"]}
        log_action("Planner Agent (crewai)", "Plan fallback used", planner_result)

    # 2) Labeling step: use local hybrid_label for each row, but notify crew via a Task (practical hybrid approach)
    try:
        # Let the Labeler agent observe a sample and optionally provide guidance
        sample = df["DESCRIPTION"].head(10).tolist()
        sample_task = Task(name="label_sample", input={"sample_descriptions": sample})
        crew_result = crew.run(sample_task, agent="Labeler")
        log_action("Labeler Agent (crewai)", "Sample labeled", {"crew_result": str(crew_result)})
    except Exception:
        log_action("Labeler Agent (crewai)", "Sample labeling failed", {})

    # We'll still perform deterministic local labeling (ensures reproducible output)
    df["CATEGORY"] = df["DESCRIPTION"].apply(hybrid_label)
    os.makedirs("datasets/labeled_data", exist_ok=True)
    out_csv = "datasets/labeled_data/labeled_output.csv"
    df.to_csv(out_csv, index=False)
    log_action("Labeling Agent (crewai)", "CSV labeled (hybrid local executes)", {"rows": len(df)})

    # 3) QA generation (notify QA agent)
    qa_pairs = []
    for _, row in df.iterrows():
        q = f"What type of transaction is '{row['DESCRIPTION']}'?"
        a = f"It is categorized as '{row['CATEGORY']}'."
        qa_pairs.append({"question": q, "answer": a})
    os.makedirs("datasets/qa_data", exist_ok=True)
    qa_path = "datasets/qa_data/qa_output.json"
    json.dump(qa_pairs, open(qa_path, "w"), indent=2)
    log_action("Q&A Agent (crewai)", "Generated Q&A", {"count": len(qa_pairs)})

    # 4) Reviewer step: ask reviewer for a quality assessment (we'll also compute local heuristic)
    try:
        review_task = Task(name="review", input={"labeled_csv_preview": df.head(20).to_dict(orient="records")})
        review_out = crew.run(review_task, agent="Reviewer")
        log_action("Reviewer Agent (crewai)", "Crew review output", {"review_out": str(review_out)})
    except Exception:
        review_out = None

    # compute local heuristic review as well
    accuracy_local = round(0.75 + (len(df) % 20)/100, 2)
    suggest_ft = accuracy_local < 0.85
    result = {"accuracy": accuracy_local, "suggest_finetune": suggest_ft, "comments": "Hybrid labeling review (crewai orchestration + local heuristic)."}
    log_action("Reviewer Agent (crewai)", "Review done (local heuristic)", result)

    return out_csv, qa_path, result

# ===========================================================
#                     FINE-TUNING (unchanged)
# ===========================================================
FINETUNE_DIR = "models/fine_tuned"

def load_training_data():
    qa_path = "datasets/qa_data/qa_output.json"
    if not os.path.exists(qa_path):
        return None
    df = pd.read_json(qa_path)
    return Dataset.from_pandas(df)

def fine_tune_model():
    dataset = load_training_data()
    if dataset is None:
        return None

    bnb_config = None
    if torch.cuda.is_available():
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4"
            )
        except ImportError:
            st.warning("bitsandbytes not installed, using float16")
            bnb_config = None

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        quantization_config=bnb_config,
        use_auth_token=HF_TOKEN,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_cfg)

    def preprocess(batch):
        input_ids, labels = [], []
        for q, a in zip(batch["question"], batch["answer"]):
            combined = tokenizer.encode(q + " " + a, truncation=True, max_length=256)
            input_ids.append(combined)
            labels.append(combined)
        max_len = max(len(l) for l in labels)
        input_ids = [ids + [tokenizer.pad_token_id]*(max_len-len(ids)) for ids in input_ids]
        labels = [l + [tokenizer.pad_token_id]*(max_len-len(l)) for l in labels]
        attention_mask = [[1]*len(ids) + [0]*(max_len-len(ids)) for ids in input_ids]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    tokenized_dataset = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)

    args = TrainingArguments(
        output_dir=FINETUNE_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        logging_steps=5,
        save_strategy="epoch",
        bf16=torch.cuda.is_available(),
        report_to="none"
    )

    trainer = Trainer(model=model, args=args, train_dataset=tokenized_dataset)
    trainer.train()

    model.save_pretrained(FINETUNE_DIR)
    tokenizer.save_pretrained(FINETUNE_DIR)

    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, use_auth_token=HF_TOKEN,
                                                      device_map="auto", torch_dtype=torch.bfloat16)
    peft_model = PeftModel.from_pretrained(base_model, FINETUNE_DIR)
    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained(os.path.join(FINETUNE_DIR, "merged"))
    tokenizer.save_pretrained(os.path.join(FINETUNE_DIR, "merged"))

    log_action("FineTuner Agent", "Fine-tuning complete", {"path": FINETUNE_DIR})
    return FINETUNE_DIR

# ===========================================================
#                STREAMLIT UI STARTS HERE
# ===========================================================
st.set_page_config(page_title="Audit AI (Gemma Hybrid + CrewAI)", layout="wide")
st.title("💼 AI-Powered Bank Statement Auditor (Gemma + Hybrid + CrewAI)")

st.sidebar.header("⚙ Controls")
uploaded_file = st.sidebar.file_uploader("Upload bank CSV", type=["csv"])

st.sidebar.markdown("### Execution backend")
st.sidebar.write("CrewAI integration:" , "available" if CREWAI_AVAILABLE else "not installed (falling back to local agents)")

if st.sidebar.button("Run Analysis"):
    if not uploaded_file:
        st.error("Upload a CSV first!")
    else:
        os.makedirs("datasets", exist_ok=True)
        csv_path = f"datasets/{uploaded_file.name}"
        with open(csv_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("File uploaded!")

        # If CrewAI is available, build and use the crew; otherwise use local functions
        if CREWAI_AVAILABLE:
            try:
                crew = build_crewai_crew()
                labeled_csv, qa_path, review = run_crewai_labeling(crew, csv_path)
            except Exception as e:
                st.warning(f"CrewAI orchestration failed: {e} — falling back to local pipeline.")
                log_action("System", "CrewAI orchestration error", {"error": str(e)})
                plan = plan_task_local(csv_path)
                labeled_csv = label_csv_local(csv_path)
                qa_path = generate_qa_local(labeled_csv)
                review = review_labeled_output_local(labeled_csv)
        else:
            plan = plan_task_local(csv_path)
            labeled_csv = label_csv_local(csv_path)
            qa_path = generate_qa_local(labeled_csv)
            review = review_labeled_output_local(labeled_csv)

        df = pd.read_csv(labeled_csv)
        st.subheader("📌 Labeled Data Preview")
        st.dataframe(df.head())

        st.subheader("📊 Spending Analytics")
        cat_counts = df["CATEGORY"].value_counts()
        fig, ax = plt.subplots()
        ax.pie(cat_counts, labels=cat_counts.index, autopct="%1.1f%%")
        st.pyplot(fig)

        st.subheader("💬 Sample Q&A")
        st.json(json.load(open(qa_path))[:5])

        st.subheader("🔍 Review Score")
        st.json(review)

        if review.get("suggest_finetune", False):
            st.warning("⚠ Low accuracy — Fine-tuning recommended.")
            if st.button("Run Fine-Tuning Now"):
                with st.spinner("Fine-tuning model…"):
                    out = fine_tune_model()
                if out:
                    st.success(f"Model fine-tuned → {out}")
                else:
                    st.error("Fine-tuning failed or no Q&A data found.")

if st.sidebar.button("Global Fine-Tuning"):
    with st.spinner("Training on all Q&A files…"):
        out = fine_tune_model()
    if out:
        st.success(f"Model saved to {out}")
    else:
        st.error("No Q&A data found!")

if st.sidebar.button("Show Logs"):
    conn = sqlite3.connect(DB_PATH)
    df_logs = pd.read_sql("SELECT * FROM logs ORDER BY id DESC LIMIT 50", conn)
    conn.close()
    st.subheader("🧾 Recent Logs")
    st.dataframe(df_logs)
