

# 💼 AI-Powered Bank Statement Auditor (Gemma Hybrid)

A Streamlit-based application that automates the auditing of bank statements using **rule-based labeling** combined with **Gemma 2B IT** for hybrid AI-assisted labeling. The app also generates Q&A datasets for fine-tuning and provides a simple accuracy-based review system.

**For better understanding of the code, everything is also attached in a Python notebook — feel free to refer to that as well.**
**For the review system, we are currently using accuracy.**

---

## Features

* **Hybrid Labeling**
  Combines rule-based classification with predictions from **Gemma 2B IT** for better category accuracy.

* **Review with Accuracy**
  Shows a basic accuracy score to review labeling quality.

* **Fine-Tuning Available**
  If accuracy is low, the app recommends fine-tuning — and you can trigger the fine-tuning step directly from the UI.

* **Q&A Generation**
  Creates Q&A pairs from labeled data that can be used for further training or audits.

* **Interactive Dashboard**
  Upload CSVs, analyze spending patterns, visualize outputs, and inspect results.

* **Audit Logging**
  Automatically logs actions, labeling events, and fine-tuning processes into a local SQLite DB.

---

## Model Used

This project uses:

### **Gemma 2B IT (Instruction-Tuned Model)**

* Lightweight, fast, and ideal for on-device or small-GPU setups
* Used here for:

  * AI-assisted labeling
  * Q&A generation
  * Fine-tuning (LoRA-based)

You can replace it with a larger Gemma model later if needed.

---

## Getting Started

### 1. Clone the repository

```bash
[git clone <repository-url>](https://github.com/niveshsoni/Audit_intelligence_system.git)
cd <repository-folder>
```

### 2. Install dependencies

```bash
python -m venv venv
source venv/bin/activate  # macOS / Linux
venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

### 3. Set Hugging Face token

```bash
export HF_TOKEN=<your_token>     # macOS / Linux
set HF_TOKEN=<your_token>        # Windows
```

### 4. Run the app

```bash
streamlit run app.py
```

---

## Usage

1. **Upload bank statement CSV**
   Required columns:

   ```
   DATE,DESCRIPTION,DEBIT,CREDIT,BALANCE
   ```

2. **Run Analysis**
   The app will:

   * Generate an audit plan
   * Apply hybrid labeling with Gemma 2B IT
   * Create Q&A data
   * Visualize category spending
   * Show **accuracy-based review**
   * Suggest fine-tuning if accuracy is low

3. **Fine-Tune Model (Optional)**
   Train on generated Q&A to improve future predictions.

4. **Global Fine-Tuning**
   Uses all accumulated Q&A datasets.

5. **View Logs**
   All operations are saved in an SQLite database.
6. 5. **Output storage**
   Persistent File Storage for All Outputs.
---

## File Structure

```
app.py
audit_logs.db
datasets/
  labeled_data/
    labeled_output.csv
  qa_data/
    qa_output.json
models/
  fine_tuned/
    merged/
```

---

## How Hybrid Labeling Works

1. **Rule-Based Step**
   Simple keyword-based matching.

2. **Gemma 2B IT Prediction**
   Used when rule-based logic is unsure or returns "Other".

3. **Final Decision**
   Chooses the best label between rule-based and model output.


Sure — here’s **only the directory section**, exactly as you asked:

---

## Where Outputs Are Saved (Important)

When you run the full system, the app automatically creates a structured directory where all outputs are stored.
This helps you keep track of everything — labeled files, generated Q&A, fine-tuned models, logs, etc.

### 📁 Output Directory Structure

```
project-root/
│
├── app.py
├── audit_logs.db                  # SQLite DB storing logs
│
├── datasets/
│   ├── labeled_data/              # Final outputs after running analysis
│   │   ├── labeled_output.csv     # Categorized bank statement
│   │   └── <more labeled files>   # (Timestamped if processed multiple times)
│   │
│   ├── qa_data/                   # Q&A datasets generated from labeled data
│   │   ├── qa_output.json
│   │   └── <timestamped_qa.json>
│
├── models/
│   ├── fine_tuned/
│   │   ├── merged/                # Final merged LoRA + base model
│   │   │   ├── config.json
│   │   │   ├── pytorch_model.bin
│   │   │   └── tokenizer.json
│   │   └── adapter/               # LoRA adapter weights (before merge)
│
└── logs/
    └── streamlit.log (optional)


```

---

## Fine-Tuning Details

* Uses **LoRA** adapters
* Base model: **Gemma 2B IT**
* Trains on generated Q&A pairs
* Outputs a merged fine-tuned model for future runs

---

## Notes

* Accuracy review is currently a **placeholder logic**, not ground-truth comparison.
* Notebook version of the workflow is included for easier understanding.
* BitsAndBytes 4-bit optimization is optional.

---


