#!/usr/bin/env python3
"""Write a runnable white-robot training config with a selected data root."""

import argparse
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a runtime training YAML with data.preprocessed_data_root overridden."
    )
    parser.add_argument("--config", required=True, type=Path, help="Source training YAML")
    parser.add_argument("--preprocessed-root", required=True, type=Path, help="Preprocessed dataset root")
    parser.add_argument("--output", required=True, type=Path, help="Output runtime YAML")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_config = args.config.resolve()
    preprocessed_root = args.preprocessed_root.resolve()
    output_config = args.output.resolve()

    if not source_config.is_file():
        raise FileNotFoundError(f"Source config does not exist: {source_config}")
    if not preprocessed_root.is_dir():
        raise NotADirectoryError(f"Preprocessed dataset root does not exist: {preprocessed_root}")

    with source_config.open() as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Training config must be a YAML mapping: {source_config}")

    data_config = config.setdefault("data", {})
    if not isinstance(data_config, dict):
        raise ValueError("Training config field 'data' must be a mapping")
    data_config["preprocessed_data_root"] = str(preprocessed_root)

    output_config.parent.mkdir(parents=True, exist_ok=True)
    with output_config.open("w") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)

    print(output_config)


if __name__ == "__main__":
    main()
