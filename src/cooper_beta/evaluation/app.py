from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

try:
    import pandas as pd
except Exception:
    pd = None

from .runner import evaluate


def ablation_suite() -> list[tuple[str, dict[str, object]]]:
    base = {
        "USE_ADJUSTED_SCORE": False,
        "NN_RULE_ENABLED": False,
        "NN_FAIL_AS_JUNK": False,
        "ANGLE_RULE_ENABLED": False,
        "ANGLE_ORDER_RULE_ENABLED": False,
    }
    return [
        ("A0_baseline_ellipse", base),
        ("A1_nn_only", {**base, "NN_RULE_ENABLED": True, "NN_FAIL_AS_JUNK": True}),
        ("A2_adjusted_only", {**base, "USE_ADJUSTED_SCORE": True}),
        (
            "A3_angle_only",
            {**base, "ANGLE_RULE_ENABLED": True, "ANGLE_ORDER_RULE_ENABLED": True},
        ),
        (
            "A4_adjusted_plus_nn",
            {**base, "USE_ADJUSTED_SCORE": True, "NN_RULE_ENABLED": True, "NN_FAIL_AS_JUNK": True},
        ),
        (
            "A5_adjusted_plus_angle",
            {
                **base,
                "USE_ADJUSTED_SCORE": True,
                "ANGLE_RULE_ENABLED": True,
                "ANGLE_ORDER_RULE_ENABLED": True,
            },
        ),
        (
            "A6_full",
            {
                **base,
                "USE_ADJUSTED_SCORE": True,
                "NN_RULE_ENABLED": True,
                "NN_FAIL_AS_JUNK": True,
                "ANGLE_RULE_ENABLED": True,
                "ANGLE_ORDER_RULE_ENABLED": True,
            },
        ),
    ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Cooper-Beta on positive and negative folders and save CSV outputs "
            "for ROC/PR analysis."
        )
    )
    parser.add_argument(
        "--true",
        default="data-true",
        help="Directory containing positive examples (beta-barrel-like structures).",
    )
    parser.add_argument(
        "--false",
        default="data-false",
        help="Directory containing negative examples (non-barrel structures).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of analysis workers (default: pipeline default).",
    )
    parser.add_argument(
        "--prepare",
        type=int,
        default=None,
        help="Number of preparation workers (default: follows --workers).",
    )
    parser.add_argument(
        "--save-dir",
        default="eval_outputs",
        help="Directory where evaluation CSV files are written.",
    )
    parser.add_argument(
        "--metric-level",
        choices=["chain", "file", "both"],
        default="both",
        help="Which metric level to print: chain, file, or both.",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run the 7-experiment ablation suite and write a summary CSV.",
    )
    args = parser.parse_args(argv)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.save_dir)

    if not args.ablation:
        row = evaluate(
            true_dir=Path(args.true),
            false_dir=Path(args.false),
            workers=args.workers,
            prepare_workers=args.prepare,
            save_dir=save_dir,
            metric_level=args.metric_level,
            tag=timestamp,
            detector_overrides=None,
            print_metric_tables=True,
        )
        print("\nSaved for ROC/PR:")
        print(f"  chain-level: {row['chain_csv']}")
        print(f"  file-level : {row['file_csv']}\n")
        return

    output_dir = save_dir / f"ablation_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    print("\n=== Ablation study (7 exps: main effects & interactions) ===")
    print(f"Output dir: {output_dir}\n")

    for experiment_name, overrides in ablation_suite():
        print(f"[{experiment_name}] overrides={overrides}")
        row = evaluate(
            true_dir=Path(args.true),
            false_dir=Path(args.false),
            workers=args.workers,
            prepare_workers=args.prepare,
            save_dir=output_dir,
            metric_level=args.metric_level,
            tag=f"{timestamp}_{experiment_name}",
            detector_overrides=overrides,
            print_metric_tables=False,
        )
        row["exp"] = experiment_name
        rows.append(row)

        summary_parts: list[str] = []
        if row.get("chain_f1") is not None:
            summary_parts.append(
                "chain: "
                f"F1={row['chain_f1']:.4f}  "
                f"BalAcc={row['chain_balanced_accuracy']:.4f}  "
                f"(TP={row['chain_TP']} FP={row['chain_FP']} "
                f"TN={row['chain_TN']} FN={row['chain_FN']})"
            )
        if row.get("file_f1") is not None:
            summary_parts.append(
                "file: "
                f"F1={row['file_f1']:.4f}  "
                f"BalAcc={row['file_balanced_accuracy']:.4f}  "
                f"(TP={row['file_TP']} FP={row['file_FP']} "
                f"TN={row['file_TN']} FN={row['file_FN']})"
            )
        print("  " + " | ".join(summary_parts) + "\n")

    if pd is None:
        return

    dataframe = pd.DataFrame(rows)
    preferred_columns = [
        "exp",
        "chain_recall",
        "chain_precision",
        "chain_f1",
        "chain_specificity",
        "chain_accuracy",
        "chain_balanced_accuracy",
        "file_recall",
        "file_precision",
        "file_f1",
        "file_specificity",
        "file_accuracy",
        "file_balanced_accuracy",
        "chain_TP",
        "chain_FP",
        "chain_TN",
        "chain_FN",
        "file_TP",
        "file_FP",
        "file_TN",
        "file_FN",
        "chain_csv",
        "file_csv",
    ]
    ordered_columns = [column for column in preferred_columns if column in dataframe.columns]
    ordered_columns += [column for column in dataframe.columns if column not in ordered_columns]
    dataframe = dataframe[ordered_columns]

    output_path = output_dir / f"ablation_summary_{timestamp}.csv"
    dataframe.to_csv(output_path, index=False)

    print("=== Ablation summary ===")
    display_columns = [
        column
        for column in [
            "exp",
            "chain_f1",
            "chain_balanced_accuracy",
            "file_f1",
            "file_balanced_accuracy",
        ]
        if column in dataframe.columns
    ]
    if display_columns:
        print(dataframe[display_columns].to_string(index=False))
    print(f"\nSaved: {output_path}\n")


if __name__ == "__main__":  # pragma: no cover
    main()
