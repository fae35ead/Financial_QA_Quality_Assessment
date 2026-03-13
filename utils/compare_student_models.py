'''该文件用于比较两个学生模型（student_20k 和 student_100k）在黄金样本上的表现。它会加载测试数据，使用两个模型进行预测，计算各种评估指标，并将结果保存到指定目录。'''

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ROOT_LABELS = {
    0: "Direct",
    1: "Intermediate",
    2: "Evasive",
}
LABEL_ORDER = [0, 1, 2]
MAX_LEN = 384
BATCH_SIZE = 32

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_DATA_FILE = PROJECT_ROOT / "data" / "processed" / "labeled_golden_samples.csv"
MODEL_PATHS = {
    "student_20k": PROJECT_ROOT / "models" / "student_distilbert",
    "student_100k": PROJECT_ROOT / "models" / "student_100k_distilbert",
}
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "evaluation" / "student_compare"


class QATestDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer, max_len: int):
        self.questions = dataframe["Qsubj"].fillna("").astype(str).tolist()
        self.replies = dataframe["Reply"].fillna("").astype(str).tolist()
        self.labels = dataframe["label"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, item):
        encoding = self.tokenizer(
            text=self.questions[item],
            text_pair=self.replies[item],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(self.labels[item], dtype=torch.long),
        }


def load_eval_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_columns = {"Qsubj", "Reply", "label"}
    missing = required_columns - set(df.columns)
    if missing:
        raise KeyError(f"测试集缺少必要列: {sorted(missing)}")

    df = df[df["label"].isin(LABEL_ORDER)].copy()
    df["Qsubj"] = df["Qsubj"].fillna("").astype(str)
    df["Reply"] = df["Reply"].fillna("").astype(str)
    df = df[(df["Qsubj"].str.strip() != "") & (df["Reply"].str.strip() != "")].reset_index(drop=True)
    return df


def predict(model_dir: Path, eval_df: pd.DataFrame, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    dataset = QATestDataset(eval_df, tokenizer, MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].cpu().numpy()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_labels.extend(labels.tolist())
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def build_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    report = classification_report(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
        target_names=[ROOT_LABELS[i] for i in LABEL_ORDER],
        output_dict=True,
        zero_division=0,
    )
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
        zero_division=0,
    )

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }

    per_class_rows = []
    for idx, label_id in enumerate(LABEL_ORDER):
        per_class_rows.append(
            {
                "label_id": label_id,
                "label_name": ROOT_LABELS[label_id],
                "precision": precision[idx],
                "recall": recall[idx],
                "f1": f1[idx],
                "support": support[idx],
            }
        )

    cm = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)
    return metrics, per_class_rows, report, cm


def save_confusion_matrix(cm: np.ndarray, out_path: Path, title: str):
    import matplotlib.pyplot as plt

    labels = [ROOT_LABELS[i] for i in LABEL_ORDER]
    cm_norm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    for ax, matrix, matrix_title, fmt in [
        (axes[0], cm, "Count", "d"),
        (axes[1], cm_norm, "Row-normalized", ".2f"),
    ]:
        im = ax.imshow(matrix, cmap="Blues")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{title} - {matrix_title}")

        threshold = matrix.max() / 2 if matrix.size else 0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = format(matrix[i, j], fmt)
                color = "white" if matrix[i, j] > threshold else "black"
                ax.text(j, i, value, ha="center", va="center", color=color, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"评估设备: {device}")

    eval_df = load_eval_data(TEST_DATA_FILE)
    print(f"有效黄金样本数: {len(eval_df)}")
    print("标签分布:")
    print(eval_df["label"].value_counts().sort_index())

    summary_rows = []
    all_prediction_frames = []

    for model_name, model_dir in MODEL_PATHS.items():
        if not model_dir.exists():
            raise FileNotFoundError(f"模型目录不存在: {model_dir}")

        print(f"\n开始评估: {model_name} -> {model_dir}")
        y_true, y_pred, y_prob = predict(model_dir, eval_df, device)
        metrics, per_class_rows, report, cm = build_metrics(y_true, y_pred)

        summary_rows.append({"model": model_name, **metrics})
        pd.DataFrame(per_class_rows).to_csv(OUTPUT_DIR / f"{model_name}_per_class_metrics.csv", index=False, encoding="utf-8-sig")
        save_confusion_matrix(cm, OUTPUT_DIR / f"{model_name}_confusion_matrix.png", model_name)

        with open(OUTPUT_DIR / f"{model_name}_classification_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        prediction_df = eval_df.copy()
        prediction_df[f"{model_name}_pred"] = y_pred
        prediction_df[f"{model_name}_pred_name"] = [ROOT_LABELS[int(x)] for x in y_pred]
        prediction_df[f"{model_name}_confidence"] = y_prob.max(axis=1)
        for idx, label_id in enumerate(LABEL_ORDER):
            prediction_df[f"{model_name}_prob_{label_id}"] = y_prob[:, idx]
        all_prediction_frames.append(prediction_df)

        print(pd.DataFrame([{"model": model_name, **metrics}]).to_string(index=False))

    summary_df = pd.DataFrame(summary_rows).sort_values(by="macro_f1", ascending=False)
    summary_df.to_csv(OUTPUT_DIR / "model_summary.csv", index=False, encoding="utf-8-sig")

    merged = eval_df.copy()
    for frame in all_prediction_frames:
        extra_cols = [c for c in frame.columns if c not in merged.columns]
        merged = pd.concat([merged, frame[extra_cols]], axis=1)

    merged["models_disagree"] = merged["student_20k_pred"] != merged["student_100k_pred"]
    merged["student_20k_correct"] = merged["student_20k_pred"] == merged["label"]
    merged["student_100k_correct"] = merged["student_100k_pred"] == merged["label"]
    merged.to_csv(OUTPUT_DIR / "golden_predictions_compare.csv", index=False, encoding="utf-8-sig")

    bad_cases = merged[(~merged["student_20k_correct"]) | (~merged["student_100k_correct"])].copy()
    bad_cases.to_csv(OUTPUT_DIR / "bad_cases_compare.csv", index=False, encoding="utf-8-sig")

    disagreement_cases = merged[merged["models_disagree"]].copy()
    disagreement_cases.to_csv(OUTPUT_DIR / "disagreement_cases.csv", index=False, encoding="utf-8-sig")

    print("\n评估完成，输出目录:")
    print(OUTPUT_DIR)
    print("\n模型汇总:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
