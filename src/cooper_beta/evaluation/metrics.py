from __future__ import annotations

try:
    import pandas as pd
except Exception:
    pd = None


def ensure_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.copy()
    for column in [
        "filename",
        "chain",
        "result",
        "score_adjust",
        "valid_layers",
        "all_adjusted_layers",
        "all_layers",
        "reason",
    ]:
        if column not in dataframe.columns:
            dataframe[column] = None
    return dataframe


def drop_error_chain_level(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    result_series = dataframe["result"].astype(str).str.upper()
    keep_mask = result_series != "ERROR"
    dropped = int((~keep_mask).sum())
    return dataframe.loc[keep_mask].copy(), dropped


def compute_confusion_metrics(tp: int, fp: int, tn: int, fn: int) -> dict[str, float | int]:
    def safe_div(numerator: float, denominator: float) -> float:
        return (numerator / denominator) if denominator else 0.0

    recall = safe_div(tp, tp + fn)
    precision = safe_div(tp, tp + fp)
    specificity = safe_div(tn, tn + fp)
    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    balanced_accuracy = 0.5 * (recall + specificity)
    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "specificity": specificity,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
    }


def print_metrics(title: str, metrics: dict[str, float | int]) -> None:
    if title:
        print(title)
    print("Confusion matrix:")
    print(f"  TP={metrics['TP']}  FP={metrics['FP']}")
    print(f"  FN={metrics['FN']}  TN={metrics['TN']}\n")
    print("Metrics:")
    print(f"  Recall      : {metrics['recall']:.4f}")
    print(f"  Precision   : {metrics['precision']:.4f}")
    print(f"  F1          : {metrics['f1']:.4f}")
    print(f"  Specificity : {metrics['specificity']:.4f}")
    print(f"  Accuracy    : {metrics['accuracy']:.4f}")
    print(f"  Bal. Acc.   : {metrics['balanced_accuracy']:.4f}\n")


def compute_chain_metrics(
    positive_dataframe: pd.DataFrame,
    negative_dataframe: pd.DataFrame,
) -> tuple[dict[str, float | int], dict[str, int]]:
    positive_used, dropped_positive = drop_error_chain_level(positive_dataframe)
    negative_used, dropped_negative = drop_error_chain_level(negative_dataframe)

    def predicted_barrel(series: pd.Series) -> pd.Series:
        return series.astype(str).str.upper().eq("BARREL")

    true_positive_mask = predicted_barrel(positive_used["result"])
    false_positive_mask = predicted_barrel(negative_used["result"])

    tp = int(true_positive_mask.sum())
    fn = int((~true_positive_mask).sum())
    fp = int(false_positive_mask.sum())
    tn = int((~false_positive_mask).sum())

    metrics = compute_confusion_metrics(tp, fp, tn, fn)
    extra = {
        "dropped_true_error": int(dropped_positive),
        "dropped_false_error": int(dropped_negative),
        "n_true_used": int(len(positive_used)),
        "n_false_used": int(len(negative_used)),
    }
    return metrics, extra


def compute_file_metrics(
    aggregated_dataframe: pd.DataFrame,
) -> tuple[dict[str, float | int], dict[str, int]]:
    positives = aggregated_dataframe[aggregated_dataframe["split"] == "true"].copy()
    negatives = aggregated_dataframe[aggregated_dataframe["split"] == "false"].copy()

    tp = int(positives["pred_barrel_any"].astype(int).sum()) if len(positives) else 0
    fn = int(len(positives) - tp) if len(positives) else 0
    fp = int(negatives["pred_barrel_any"].astype(int).sum()) if len(negatives) else 0
    tn = int(len(negatives) - fp) if len(negatives) else 0

    metrics = compute_confusion_metrics(tp, fp, tn, fn)
    extra = {
        "n_true_files": int(len(positives)),
        "n_false_files": int(len(negatives)),
        "true_any_skip": int(positives["any_skip"].astype(int).sum()) if len(positives) else 0,
        "false_any_skip": int(negatives["any_skip"].astype(int).sum()) if len(negatives) else 0,
    }
    return metrics, extra
