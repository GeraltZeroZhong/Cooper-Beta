from __future__ import annotations

import argparse
import sys
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
        "EXCEPTION_LAYER_ENABLED": False,
    }

    def variant(
        *,
        adjusted: bool = False,
        nn: bool = False,
        angle: bool = False,
        exception_layer: bool = False,
    ) -> dict[str, object]:
        return {
            **base,
            "USE_ADJUSTED_SCORE": adjusted,
            "NN_RULE_ENABLED": nn,
            "NN_FAIL_AS_JUNK": nn,
            "ANGLE_RULE_ENABLED": angle,
            "ANGLE_ORDER_RULE_ENABLED": angle,
            "EXCEPTION_LAYER_ENABLED": exception_layer,
        }

    return [
        ("A0_raw_ellipse_baseline", variant()),
        ("A1_raw_plus_nn", variant(nn=True)),
        ("A2_raw_plus_angle", variant(angle=True)),
        ("A3_raw_plus_nn_angle", variant(nn=True, angle=True)),
        ("A4_adjusted_only", variant(adjusted=True)),
        ("A5_adjusted_plus_nn", variant(adjusted=True, nn=True)),
        ("A6_adjusted_plus_angle", variant(adjusted=True, angle=True)),
        ("A7_core_full", variant(adjusted=True, nn=True, angle=True)),
        ("A8_adjusted_plus_exception", variant(adjusted=True, exception_layer=True)),
        (
            "A9_adjusted_nn_exception",
            variant(adjusted=True, nn=True, exception_layer=True),
        ),
        (
            "A10_adjusted_angle_exception",
            variant(adjusted=True, angle=True, exception_layer=True),
        ),
        (
            "A11_production_full",
            variant(adjusted=True, nn=True, angle=True, exception_layer=True),
        ),
    ]


def _run(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Cooper-Beta on positive and negative folders and save CSV outputs "
            "for ROC/PR analysis."
        )
    )
    parser.add_argument(
        "--positives",
        "--true",
        dest="true",
        required=True,
        help="Directory containing positive examples (beta-barrel-like structures).",
    )
    parser.add_argument(
        "--negatives",
        "--false",
        dest="false",
        required=True,
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
        default="evaluation-results",
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
        help="Run the 12-experiment ablation suite and write a summary CSV.",
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
    print("\n=== Ablation study (12-experiment suite: core components + exception layer) ===")
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
                f"R={row['chain_recall']:.4f}  "
                f"P={row['chain_precision']:.4f}  "
                f"F1={row['chain_f1']:.4f}  "
                f"BalAcc={row['chain_balanced_accuracy']:.4f}  "
                f"MCC={row['chain_mcc']:.4f}  "
                f"(TP={row['chain_TP']} FP={row['chain_FP']} "
                f"TN={row['chain_TN']} FN={row['chain_FN']})"
            )
        if row.get("file_f1") is not None:
            summary_parts.append(
                "file: "
                f"R={row['file_recall']:.4f}  "
                f"P={row['file_precision']:.4f}  "
                f"F1={row['file_f1']:.4f}  "
                f"BalAcc={row['file_balanced_accuracy']:.4f}  "
                f"MCC={row['file_mcc']:.4f}  "
                f"(TP={row['file_TP']} FP={row['file_FP']} "
                f"TN={row['file_TN']} FN={row['file_FN']})"
            )
        print("  " + " | ".join(summary_parts) + "\n")

    if pd is None:
        return

    dataframe = pd.DataFrame(rows)
    preferred_columns = [
        "exp",
        "USE_ADJUSTED_SCORE",
        "NN_RULE_ENABLED",
        "ANGLE_RULE_ENABLED",
        "EXCEPTION_LAYER_ENABLED",
        "chain_recall",
        "chain_precision",
        "chain_specificity",
        "chain_f1",
        "chain_balanced_accuracy",
        "chain_mcc",
        "chain_TP",
        "chain_FP",
        "chain_TN",
        "chain_FN",
        "file_recall",
        "file_precision",
        "file_specificity",
        "file_f1",
        "file_balanced_accuracy",
        "file_mcc",
        "file_TP",
        "file_FP",
        "file_TN",
        "file_FN",
        "chain_accuracy",
        "file_accuracy",
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
            "chain_recall",
            "chain_precision",
            "chain_specificity",
            "chain_f1",
            "chain_balanced_accuracy",
            "chain_mcc",
            "chain_TP",
            "chain_FP",
            "chain_TN",
            "chain_FN",
            "file_recall",
            "file_precision",
            "file_specificity",
            "file_f1",
            "file_balanced_accuracy",
            "file_mcc",
            "file_TP",
            "file_FP",
            "file_TN",
            "file_FN",
        ]
        if column in dataframe.columns
    ]
    if display_columns:
        print(dataframe[display_columns].to_string(index=False))
    print(f"\nSaved: {output_path}\n")


def main(argv: list[str] | None = None) -> None:
    try:
        _run(argv)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":  # pragma: no cover
    main()
