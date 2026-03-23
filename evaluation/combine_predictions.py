import argparse

from pathlib import Path

from datasets import concatenate_datasets, load_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default="output/mmlu_predictions/by_model",
        help="Directory containing per-model Parquet files to merge.",
    )
    parser.add_argument("--pattern", default="*.parquet")
    parser.add_argument(
        "--output-file",
        default="output/mmlu_predictions/test.parquet",
        help="Path to the merged Parquet output file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    files = sorted(str(file_path) for file_path in input_dir.glob(args.pattern))

    if not files:
        msg = f"No Parquet files found matching: {input_dir / args.pattern}"
        raise FileNotFoundError(msg)

    datasets = [load_dataset("parquet", data_files=file_path, split="train") for file_path in files]
    merged_dataset = concatenate_datasets(datasets)

    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    merged_dataset.to_parquet(str(output_file))


if __name__ == "__main__":
    main()
