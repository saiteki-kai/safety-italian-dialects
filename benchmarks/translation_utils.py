import json
import time

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import googletrans as gt

from datasets import concatenate_datasets
from tqdm import tqdm


GOOGLE_TRANSLATE_CODES = {
    "Sicilian": "scn",
    "Friulian": "fur",
    "Lombard": "lmo",
    "Ligurian": "lij",
    "Venetian": "vec",
    "Italian": "it",
}
DIALECTS = list(GOOGLE_TRANSLATE_CODES.keys())
ENGLISH_LABEL = "English"


@dataclass(frozen=True)
class TranslationRetryConfig:
    max_retries: int = 3
    retry_sleep_sec: float = 0.5
    backoff_base_sec: float = 3.0
    backoff_max_sec: float = 30.0
    request_pacing_sec: float = 0.5
    from_language: str = "en"
    failure_prefix: str = "[FAILED]"


@dataclass(frozen=True)
class DatasetTranslationRunConfig:
    progress_label: str
    sequence_fields: tuple[str, ...]
    save_every_n_rows: int
    cooldown_every_n_rows: int = 0
    cooldown_sec: float = 0.0


def build_checkpoint_path(checkpoint_dir: Path, name: str) -> Path:
    slug = name.lower().replace(" ", "_")
    return checkpoint_dir / f"{slug}.json"


def save_json_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")

    with tmp.open("w", encoding="utf-8") as file:
        json.dump(state, file, ensure_ascii=False)

    tmp.replace(path)


def existing_dialect_checkpoints(checkpoint_dir: Path, dialects: list[str]) -> list[tuple[str, Path]]:
    """Return existing checkpoint paths paired with dialect names in input order."""
    checkpoints = [(dialect, build_checkpoint_path(checkpoint_dir, dialect)) for dialect in dialects]
    return [(dialect, path) for dialect, path in checkpoints if path.exists()]


def load_checkpoint_state(
    path: Path,
    *,
    metadata: dict[str, Any],
    sequence_fields: tuple[str, ...],
) -> dict[str, Any]:
    state: dict[str, Any] = {"sample_idx": 0, **dict(metadata)}
    for field in sequence_fields:
        state[field] = []

    if not path.exists():
        return state

    try:
        with path.open("r", encoding="utf-8") as file:
            loaded = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON checkpoint at {path.as_posix()}") from error

    # If metadata changed, start clean to avoid mixing incompatible checkpoints.
    if any(loaded.get(key) != value for key, value in metadata.items()):
        return state

    for field in sequence_fields:
        value = loaded.get(field, [])
        if not isinstance(value, list):
            msg = f"Checkpoint field '{field}' must be a list. File: {path.as_posix()}"
            raise TypeError(msg)
        state[field] = value

    sample_idx = loaded.get("sample_idx", 0)
    try:
        state["sample_idx"] = int(sample_idx)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Checkpoint field 'sample_idx' must be an int. File: {path.as_posix()}") from error

    for field in sequence_fields:
        rows = len(state[field])
        if rows != state["sample_idx"]:
            raise ValueError(
                "Checkpoint is inconsistent: "
                f"sample_idx={state['sample_idx']} but field '{field}' has {rows} rows. "
                f"File: {path.as_posix()}"
            )

    return state


def require_complete_checkpoint(
    state: dict[str, Any],
    sequence_fields: tuple[str, ...],
    expected_rows: int,
    label: str,
    path: Path,
) -> int:
    field_row_counts = {field: len(state.get(field, [])) for field in sequence_fields}
    incomplete = {field: rows for field, rows in field_row_counts.items() if rows != expected_rows}
    if incomplete:
        details = ", ".join(f"{field}={rows}" for field, rows in sorted(incomplete.items()))
        raise ValueError(
            f"Checkpoint for {label} is incomplete or inconsistent ({details}); "
            f"expected {expected_rows} rows each. File: {path.as_posix()}"
        )
    return expected_rows


def validate_sequence_fields(
    state: dict[str, Any],
    checkpoint_path: Path,
    sequence_fields: tuple[str, ...],
) -> None:
    if not sequence_fields:
        raise ValueError("sequence_fields cannot be empty")

    for field in sequence_fields:
        if field not in state or not isinstance(state[field], list):
            raise ValueError(f"State is missing list field '{field}'. File: {checkpoint_path.as_posix()}")


def resolve_resume_row_idx(dataset: Any, state: dict[str, Any]) -> tuple[int, int]:
    start_idx = int(state["sample_idx"])
    total_rows = len(dataset)
    if start_idx < 0:
        raise ValueError(f"Invalid checkpoint sample_idx={start_idx}")
    if start_idx > total_rows:
        # Clamp instead of failing hard: dataset may have shrunk between runs.
        start_idx = total_rows
        state["sample_idx"] = total_rows
    return start_idx, total_rows


def append_transformed_fields(
    transformed_row: dict[str, Any],
    state: dict[str, Any],
    sequence_fields: tuple[str, ...],
    checkpoint_path: Path,
) -> None:
    missing_fields = [field for field in sequence_fields if field not in transformed_row]
    if missing_fields:
        missing = ", ".join(missing_fields)
        raise ValueError(f"Missing field(s) {missing} in transformed row for {checkpoint_path.as_posix()}")

    for field in sequence_fields:
        state[field].append(transformed_row[field])


def persist_and_cooldown_if_needed(
    processed_rows: int,
    state: dict[str, Any],
    checkpoint_path: Path,
    run_config: DatasetTranslationRunConfig,
) -> None:
    if run_config.save_every_n_rows > 0 and processed_rows % run_config.save_every_n_rows == 0:
        save_json_state(checkpoint_path, state)

    if run_config.cooldown_every_n_rows > 0 and processed_rows % run_config.cooldown_every_n_rows == 0:
        time.sleep(run_config.cooldown_sec)


def translate_dataset_with_checkpoint(
    dataset: Any,
    state: dict[str, Any],
    checkpoint_path: Path,
    transform_row: Callable[[Any], dict[str, Any]],
    run_config: DatasetTranslationRunConfig,
) -> None:
    sequence_fields = run_config.sequence_fields
    validate_sequence_fields(state, checkpoint_path, sequence_fields)
    start_idx, total_rows = resolve_resume_row_idx(dataset, state)

    print(f"{run_config.progress_label} (resume from row {start_idx}/{total_rows})")

    for row_idx in tqdm(range(start_idx, total_rows)):
        transformed_row = transform_row(dataset[row_idx])
        append_transformed_fields(transformed_row, state, sequence_fields, checkpoint_path)

        processed_rows = row_idx + 1
        state["sample_idx"] = processed_rows
        persist_and_cooldown_if_needed(processed_rows, state, checkpoint_path, run_config)

    save_json_state(checkpoint_path, state)


def save_combined_dataset(output_path: Path, original_dataset: Any, translated_datasets: list[Any]) -> None:
    if not translated_datasets:
        raise ValueError("No translated checkpoints found. Run translation first.")

    combined_translated = concatenate_datasets(list(translated_datasets))
    final_dataset = concatenate_datasets([original_dataset, combined_translated])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_dataset.to_parquet(output_path.as_posix())
    print(f"Combined dataset saved to {output_path.as_posix()}")


def is_rate_limit_error(error: Exception) -> bool:
    status_code = getattr(error, "status_code", None)
    response_status_code = getattr(getattr(error, "response", None), "status_code", None)
    error_text = str(error).lower()

    return status_code == 429 or response_status_code == 429 or "429" in error_text or "too many requests" in error_text  # noqa: PLR2004


def retry_delay_for_error(error: Exception, attempt: int, config: TranslationRetryConfig) -> float:
    if is_rate_limit_error(error):
        backoff_sec = min(config.backoff_base_sec * (2 ** (attempt - 1)), config.backoff_max_sec)
        print(f"429 received; backing off for {backoff_sec:.1f}s before retrying")
        return backoff_sec

    print(f"Error during translation: {error}. Retrying in {config.retry_sleep_sec:.1f}s...")
    return config.retry_sleep_sec


def translate_with_retry(text: str, target: str, config: TranslationRetryConfig) -> str:
    last_err: Exception | None = None
    saw_empty_translation = False

    for attempt in range(1, config.max_retries + 1):
        try:
            if config.request_pacing_sec > 0:
                time.sleep(config.request_pacing_sec)

            translated = gt.translate(text, to_language=target, from_language=config.from_language)
            if translated and translated.strip():
                return translated

            saw_empty_translation = True
            print(
                f"[DEBUG] Empty translation for target='{target}' on attempt {attempt}/{config.max_retries}; retrying."
            )
            if attempt < config.max_retries:
                time.sleep(config.retry_sleep_sec)
        except Exception as error:
            last_err = error
            if attempt < config.max_retries:
                time.sleep(retry_delay_for_error(error, attempt, config))

    if saw_empty_translation:
        print(f"[DEBUG] Translation stayed empty for target='{target}' after {config.max_retries} attempts.")
        return f"{config.failure_prefix}{text}"

    if last_err is not None:
        raise RuntimeError("Translation failed after retries") from last_err

    raise RuntimeError("Translation failed after retries")
