import os
import typing

import pandas as pd

from datasets import load_dataset
from dotenv import load_dotenv
from translation_utils import DIALECTS


load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    msg = "Please set the HF_TOKEN environment variable to your Hugging Face API token to push datasets to the Hub."
    raise ValueError(msg)

# MMLU -----------------------------------------------------------------------------------------------------------------

mmlu_dataset = load_dataset("output/mmlu_translation", data_files={"test": "test.parquet", "validation": "dev.parquet"})

print("MMLU " + "-" * 50 + "\n")

has_error = False

for split in ["test", "validation"]:
    print(f"Checking split '{split}' for expected dialects...\n")

    df = mmlu_dataset[split].to_pandas()
    df = typing.cast("pd.DataFrame", df)

    dialects = set(df["lang"].unique())
    missing_dialects = set(DIALECTS) - dialects

    if missing_dialects:
        has_error = True
        print(f"[ERROR] Missing dialects in split '{split}': {', '.join(missing_dialects)}\n")

    # each dialect should have the same number of rows as the original dataset (no missing translations)
    original_count = len(df[df["lang"] == "English"])
    for dialect in DIALECTS:
        dialect_count = len(df[df["lang"] == dialect])
        print(f"Dialect '{dialect}' has {dialect_count}/{original_count} rows in split '{split}'")

        if dialect_count != original_count:
            has_error = True

    print()

    # check for any null or empty values in the 'question' and 'choices' columns
    for column in ["question", "choices"]:
        null_count = df[column].isnull().sum()
        empty_count = (df[column].str.len() == 0).sum()

        if null_count > 0 or empty_count > 0:
            has_error = True
            print(f"[WARN] Column '{column}' in split '{split}' has {null_count} nulls and {empty_count} empty values")
            print()

if not has_error:
    print("\nNo errors found in MMLU translations!\n")
    info = mmlu_dataset.push_to_hub("saiteki-kai/mmlu-redux-dialects", private=True, token=HF_TOKEN)
    print(f"\nDataset pushed to Hugging Face Hub with info: {info}")
else:
    print("\nErrors were found in MMLU translations. Please fix them before pushing to the Hub.\n\n")

# XSTest ---------------------------------------------------------------------------------------------------------------

xstest_dataset = load_dataset("output/xstest_translation", data_files={"test": "test.parquet"})

print("XSTest " + "-" * 49 + "\n")

has_error = False

print("Checking split 'test' for expected dialects...")

df = xstest_dataset["test"].to_pandas()
df = typing.cast("pd.DataFrame", df)

dialects = set(df["lang"].unique())
missing_dialects = set(DIALECTS) - dialects

if missing_dialects:
    has_error = True
    print(f"[ERROR] Missing dialects in split 'test': {', '.join(missing_dialects)}\n")

# each dialect should have the same number of rows as the original dataset (no missing translations)
original_count = len(df[df["lang"] == "English"])
for dialect in DIALECTS:
    dialect_count = len(df[df["lang"] == dialect])
    print(f"Dialect '{dialect}' has {dialect_count}/{original_count} rows in split 'test'")

    if dialect_count != original_count:
        has_error = True
        print(f"[ERROR] Dialect '{dialect}' has {dialect_count}/{original_count} rows in split 'test'\n")

# check for any null or empty values in the 'prompt' column
null_count = df["prompt"].isnull().sum()
empty_count = (df["prompt"].str.len() == 0).sum()

if null_count > 0 or empty_count > 0:
    has_error = True
    print(f"[WARN] Column 'prompt' in split 'test' has {null_count} nulls and {empty_count} empty values")
    print()

if not has_error:
    print("\nNo errors found in XSTest translations!\n")
    info = xstest_dataset.push_to_hub("saiteki-kai/xstest-dialects", private=True, token=HF_TOKEN)
    print(f"\nDataset pushed to Hugging Face Hub with info: {info}")
else:
    print("\nErrors were found in XSTest translations. Please fix them before pushing to the Hub.")
