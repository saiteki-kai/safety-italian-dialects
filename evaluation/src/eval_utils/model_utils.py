QWEN35_IDENTIFIER = "Qwen/Qwen3.5"

DEFAULT_MODELS = [
    "Fastweb/FastwebMIIA-7B",
    "Qwen/Qwen3.5-9B",  # (open bug on github) temporary fix: gdn_prefill_backend="triton"
    "sapienzanlp/Minerva-7B-instruct-v1.0",
    "swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA",
    "utter-project/EuroLLM-9B-Instruct-2512",
    "Qwen/Qwen3-8B",
    "swiss-ai/Apertus-8B-Instruct-2509",
]

SYSTEM_PROMPTS = {
    "swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA": (
        "Sei un an assistente AI per la lingua Italiana di nome LLaMAntino-3 ANITA "
        "(Advanced Natural-based interaction for the ITAlian language)."
        " Rispondi nella lingua usata per la domanda in modo chiaro, semplice ed esaustivo."
    ),
    "utter-project/EuroLLM-9B-Instruct-2512": (
        "You are EuroLLM --- an AI assistant specialized in European languages "
        "that provides safe, educational and helpful answers."
    ),
}


def model_to_filename(model_id: str) -> str:
    return model_id.replace("/", "__")


def get_system_message(model_id: str) -> str | None:
    return SYSTEM_PROMPTS.get(model_id)


def get_additional_config(model_id: str) -> dict[str, str | None]:
    # Temporary workaround for Qwen 3.5 prefill behavior.
    return {"gdn_prefill_backend": "triton" if QWEN35_IDENTIFIER in model_id else None}


__all__ = [
    "DEFAULT_MODELS",
    "SYSTEM_PROMPTS",
    "get_additional_config",
    "get_system_message",
    "model_to_filename",
]
