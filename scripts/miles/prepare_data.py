#!/usr/bin/env python3
"""Convert seta-env harbor dataset to miles training format.

Uses load_harbor_dataset() to read task directories and produces
a parquet file consumable by miles' RolloutDataSource.

Output format per row:
    prompt:   str  — task instruction text
    metadata: dict — {"instance_id": "stack_overflow__888", "agent_name": "tito_train_agent"}

Usage:
    python scripts/miles/prepare_data.py \
        --input dataset/seta-env-v2 \
        --output scripts/miles/seta_train.parquet

    # JSONL output instead:
    python scripts/miles/prepare_data.py \
        --input dataset/seta-env-v2 \
        --output scripts/miles/seta_train.jsonl
"""

import argparse
import json
from pathlib import Path

from seta_env.utils.dataset_loader import load_harbor_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Convert seta-env harbor dataset to miles format",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to harbor dataset folder (e.g. dataset/seta-env-v2)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path (.parquet or .jsonl)",
    )
    parser.add_argument(
        "--agent-name",
        type=str,
        default="tito_train_agent",
        help="Agent name for metadata (default: tito_train_agent)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of tasks (for testing)",
    )
    args = parser.parse_args()

    # Load harbor tasks
    ds = load_harbor_dataset(args.input)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    # Transform to miles format
    records = []
    for row in ds:
        records.append({
            "prompt": row["instruction"],
            "metadata": json.dumps({
                "instance_id": row["task_name"],
                "agent_name": args.agent_name,
            }),
        })

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == ".parquet":
        from datasets import Dataset
        out_ds = Dataset.from_list(records)
        out_ds.to_parquet(str(output_path))
        print(f"Wrote {len(records)} samples to {output_path} (parquet)")
    elif output_path.suffix == ".jsonl":
        with open(output_path, "w") as f:
            for r in records:
                # metadata needs to be a dict for miles, not a JSON string
                r["metadata"] = json.loads(r["metadata"])
                f.write(json.dumps(r) + "\n")
        print(f"Wrote {len(records)} samples to {output_path} (jsonl)")
    else:
        raise ValueError(f"Unsupported output format: {output_path.suffix}")


if __name__ == "__main__":
    main()
