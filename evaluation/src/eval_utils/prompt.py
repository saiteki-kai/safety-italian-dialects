from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import yaml


PromptConfig = dict[str, Any]
FewShotExample = dict[str, Any]
ChatMessage = dict[str, str]


class BasePromptBuilder(ABC):
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.fewshots: list[FewShotExample] = []
        self.is_chat: bool = False

    @abstractmethod
    def build(self, **kwargs: Any) -> str | list[dict[str, str]]:
        raise NotImplementedError

    @abstractmethod
    def build_few_shot(
        self,
        shots: list[FewShotExample],
        **kwargs: Any,
    ) -> str | list[dict[str, str]]:
        raise NotImplementedError


class CompletionPromptBuilder(BasePromptBuilder):
    def __init__(self, config: PromptConfig):
        super().__init__(config)

        instruction = self.config.get("instruction")
        template = self.config.get("template")

        if not isinstance(instruction, str) or not instruction:
            raise ValueError("Missing or invalid 'instruction' in prompt config")

        if not isinstance(template, str) or not template:
            raise ValueError("Missing or invalid 'template' in prompt config")

        self.instruction = instruction
        self.template = template

    def build(self, **kwargs: Any) -> str:
        content = self.template.format(**kwargs)
        return f"{self.instruction}\n\n{content}"

    def build_few_shot(self, shots: list[FewShotExample], **kwargs: Any) -> str:
        shot_texts: list[str] = []
        for shot in shots:
            shot_data = dict(shot)
            if "answer" not in shot_data:
                raise ValueError("Missing 'answer' in few-shot example")

            answer = str(shot_data.pop("answer"))
            shot_content = self.template.format(**shot_data)
            shot_texts.append(f"{self.instruction}\n\n{shot_content}{answer}")

        current_content = self.template.format(**kwargs)
        return "\n\n".join([self.instruction, *shot_texts, current_content])


class ChatPromptBuilder(BasePromptBuilder):
    def __init__(self, config: PromptConfig, system_message: str | None = None):
        super().__init__(config)
        self.is_chat = True
        self.system_message = system_message

        system_content = self.config.get("system_instruction")
        user_template = self.config.get("user_template")
        assistant_prefill = self.config.get("assistant_prefill")
        assistant_template = self.config.get("assistant_template")

        if system_content is not None and not isinstance(system_content, str):
            raise TypeError("Invalid 'system_instruction' in prompt config: expected string")

        if assistant_prefill is not None and not isinstance(assistant_prefill, str):
            raise TypeError("Invalid 'assistant_prefill' in prompt config: expected string")

        if not isinstance(user_template, str) or not user_template:
            raise ValueError("Missing or invalid 'user_template' in prompt config")

        if not isinstance(assistant_template, str) or not assistant_template:
            raise ValueError("Missing or invalid 'assistant_template' in prompt config")

        self.system_content = system_content
        self.user_template = user_template
        self.assistant_prefill = assistant_prefill
        self.assistant_template = assistant_template

    def build(self, **kwargs: Any) -> list[ChatMessage]:
        user_content = self.user_template.format(**kwargs)

        messages = []
        messages.extend(self._build_system_messages(self.system_message))
        messages.extend(self._build_conversation_turn(user_content, self.assistant_prefill))

        return messages

    def _build_system_messages(self, sys_msg: str | None) -> list[ChatMessage]:
        messages: list[ChatMessage] = []
        if sys_msg:
            messages.append({"role": "system", "content": sys_msg})

        if self.system_content:
            messages.append({"role": "system", "content": self.system_content})

        return messages

    def _build_conversation_turn(
        self,
        user_content: str,
        assistant_content: str | None = None,
    ) -> list[ChatMessage]:
        if assistant_content is not None:
            return [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ]

        return [{"role": "user", "content": user_content}]

    def build_few_shot(self, shots: list[FewShotExample], **kwargs: Any) -> list[ChatMessage]:
        messages: list[ChatMessage] = []
        messages.extend(self._build_system_messages(self.system_message))

        for shot in shots:
            shot_data = dict(shot)
            if "answer" not in shot_data:
                raise ValueError("Missing 'answer' in few-shot example")

            answer = str(shot_data.pop("answer"))
            shot_user_content = self.user_template.format(**shot_data)
            shot_assistant_content = self.assistant_template.format(answer=answer, **shot_data)

            messages.extend(self._build_conversation_turn(shot_user_content, shot_assistant_content))

        user_content = self.user_template.format(**kwargs)
        messages.extend(self._build_conversation_turn(user_content, self.assistant_prefill))

        return messages


class PromptBuilderFactory:
    @staticmethod
    def from_yaml(path: Path, language: str, **kwargs: Any) -> BasePromptBuilder:
        with path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if not isinstance(config, dict):
            raise TypeError(f"Invalid prompt config format in {path}: expected language mapping at root")

        localized_config = config.get(language)

        if not localized_config:
            raise ValueError(f"Language '{language}' not found in prompt config: {path}")

        if not isinstance(localized_config, dict):
            raise TypeError(f"Invalid language config for '{language}' in {path}: expected object")

        if localized_config.get("user_template"):
            return ChatPromptBuilder(localized_config, **kwargs)

        return CompletionPromptBuilder(localized_config)


__all__ = [
    "BasePromptBuilder",
    "ChatPromptBuilder",
    "CompletionPromptBuilder",
    "PromptBuilderFactory",
]
