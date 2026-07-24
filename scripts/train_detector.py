#!/usr/bin/env python3
"""Train BlotGrid's optional TorchVision band detector."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from wblot.ml_detector import (  # noqa: E402
    DEFAULT_ARCHITECTURE,
    MLDetectorError,
    SUPPORTED_ARCHITECTURES,
)
from wblot.ml_training import train_detector  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune a one-class TorchVision detector on reviewed BlotGrid "
            "COCO boxes."
        )
    )
    parser.add_argument("annotations", type=Path, help="COCO annotation JSON")
    parser.add_argument("images", type=Path, help="directory containing COCO images")
    parser.add_argument("output", type=Path, help="best checkpoint output (.pt)")
    parser.add_argument(
        "--architecture",
        choices=SUPPORTED_ARCHITECTURES,
        default=DEFAULT_ARCHITECTURE,
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--split-json",
        type=Path,
        help=(
            "explicit disjoint {'train': [...], 'val': [...]} image split; "
            "recommended for experiment-aware leakage control"
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or a CUDA device such as cuda:0",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="do not download or initialize from TorchVision pretrained weights",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help=(
            "resume from this run's restricted-load best or .last checkpoint; "
            "use .last to continue from the most recently completed epoch"
        ),
    )
    parser.add_argument(
        "--allow-pseudo-labels",
        action="store_true",
        help=(
            "allow explicitly marked pseudo-labels in training only; "
            "pseudo-label images are always excluded from validation"
        ),
    )
    parser.add_argument(
        "--allow-unverified-images",
        action="store_true",
        help=(
            "allow trusted COCO images without checksum metadata; any provided "
            "checksum is still verified"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = train_detector(
            args.annotations,
            args.images,
            args.output,
            architecture=args.architecture,
            pretrained=not args.no_pretrained,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
            split_json=args.split_json,
            device=args.device,
            num_workers=args.num_workers,
            score_threshold=args.score_threshold,
            iou_threshold=args.iou_threshold,
            resume=args.resume,
            allow_pseudo_labels=args.allow_pseudo_labels,
            allow_unverified_images=args.allow_unverified_images,
        )
    except MLDetectorError as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
