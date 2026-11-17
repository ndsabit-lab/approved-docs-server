"""
Fine-tune DistilBERT (multilingual) to classify documents into the four archive types.

    pip install -r requirements-ml.txt
    python train_classifier.py my_documents.csv            # CSV with columns: text,label
    MODEL_ID=./models/doc-classifier uvicorn app:app --port 8000

You can build the CSV from the app itself: every saved document is a labeled example.
Open http://localhost:8000/export/training.csv (add ?key=... if you set an API key).
Labels must be: procurement, financial, general_internal, general_external.
"""
import sys

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          Trainer, TrainingArguments)

BASE_MODEL = "distilbert-base-multilingual-cased"
LABELS = ["procurement", "financial", "general_internal", "general_external"]
OUT_DIR = "models/doc-classifier"


def main(csv_path: str) -> None:
    df = pd.read_csv(csv_path).dropna(subset=["text", "label"])
    unknown = set(df["label"]) - set(LABELS)
    if unknown:
        sys.exit(f"Unknown labels in CSV: {sorted(unknown)}. Expected: {LABELS}")
    print(df["label"].value_counts().to_string(), "\n")

    label2id = {l: i for i, l in enumerate(LABELS)}
    df["labels"] = df["label"].map(label2id)
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["labels"], random_state=42)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def to_dataset(frame):
        ds = Dataset.from_dict({"text": frame["text"].tolist(), "labels": frame["labels"].tolist()})
        return ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=512), batched=True,
                      remove_columns=["text"])

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=label2id)
    args = TrainingArguments(output_dir="models/tmp", num_train_epochs=4, per_device_train_batch_size=8,
                             learning_rate=3e-5, save_strategy="no", report_to="none")
    trainer = Trainer(model=model, args=args, train_dataset=to_dataset(train_df), tokenizer=tokenizer,
                      data_collator=DataCollatorWithPadding(tokenizer))
    trainer.train()

    val_ds = to_dataset(val_df)
    pred = np.argmax(trainer.predict(val_ds).predictions, axis=-1)
    true = val_df["labels"].to_numpy()
    print("\nMacro-F1:", round(f1_score(true, pred, average="macro"), 3))
    print(classification_report(true, pred, labels=range(len(LABELS)), target_names=LABELS, zero_division=0))
    print("Confusion matrix (rows = true, columns = predicted):\n", confusion_matrix(true, pred))

    trainer.save_model(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"\nSaved to {OUT_DIR}. Use it with:  MODEL_ID=./{OUT_DIR} uvicorn app:app --port 8000")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python train_classifier.py my_documents.csv")
    main(sys.argv[1])
