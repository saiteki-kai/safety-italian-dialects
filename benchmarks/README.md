# Translation of Benchmarks Into Italian Dialects

This package translates benchmark datasets from English into the dialects defined in
`translation_utils.py`, with checkpoint-based resume support for long-running jobs.

## Scripts

- `translate_mmlu.py`: translates MMLU Redux test split and writes `output/mmlu_translation/test.parquet`.
- `translate_mmlu_dev.py`: translates MMLU dev split and writes `output/mmlu_translation/dev.parquet`.
- `translate_xstest.py`: translates XSTest split and writes `output/xstest_translation/test.parquet`.

Each script stores one JSON checkpoint per dialect under `output/**/checkpoints` and can resume from the
last completed row.

## Typical Usage

Run from the repository root:

```bash
python benchmarks/translate_mmlu.py
python benchmarks/translate_mmlu_dev.py
python benchmarks/translate_xstest.py
```

Translate one dialect only:

```bash
python benchmarks/translate_mmlu.py --dialect Sicilian
python benchmarks/translate_mmlu_dev.py --dialect Venetian
python benchmarks/translate_xstest.py --dialect Friulian
```

## Checkpoint Notes

- Checkpoints are validated against expected metadata before reuse.
- If metadata changes, translation restarts with a fresh state.
- Final combined parquet outputs are generated only from complete dialect checkpoints.
