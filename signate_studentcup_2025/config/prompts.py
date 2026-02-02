"""プロンプト管理モジュール.

このモジュールは、YAMLファイルで管理されたプロンプトテンプレートを読み込み、
文字列フォーマットのためのユーティリティ機能を提供します。
"""

from pathlib import Path
from typing import Any

import yaml
from loguru import logger


class PromptConfig:
    """プロンプト設定クラス."""

    # プロンプトディレクトリのパス
    PROMPTS_DIR = Path(__file__).parent / "prompts"

    @classmethod
    def list_available_prompts(cls) -> list[str]:
        """利用可能なプロンプト名のリストを取得.

        Returns:
            利用可能なプロンプト名のリスト（例: ["base", "v1_cot", "v2_fewshot", ...]）
        """
        if not cls.PROMPTS_DIR.exists():
            logger.warning(f"Prompts directory not found: {cls.PROMPTS_DIR}")
            return []

        prompts = []
        for prompt_file in cls.PROMPTS_DIR.glob("*.yaml"):
            prompts.append(prompt_file.stem)
        return sorted(prompts)

    @classmethod
    def load_prompt(cls, name: str) -> dict[str, Any]:
        """指定された名前のプロンプト設定を読み込む.

        Args:
            name: プロンプト名（例: "base", "v1_cot", "v2_fewshot"）

        Returns:
            プロンプト設定を含む辞書

        Raises:
            FileNotFoundError: 指定されたプロンプトファイルが存在しない場合
            yaml.YAMLError: YAMLファイルの解析に失敗した場合
        """
        prompt_path = cls.PROMPTS_DIR / f"{name}.yaml"

        if not prompt_path.exists():
            available = cls.list_available_prompts()
            raise FileNotFoundError(
                f"Prompt not found: {name}.yaml. "
                f"Available prompts: {available}"
            )

        logger.debug(f"Loading prompt: {prompt_path}")
        with open(prompt_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # バリデーション
        required_keys = ["name", "version", "system_message", "user_template"]
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValueError(
                f"Invalid prompt config {name}.yaml. "
                f"Missing keys: {missing_keys}"
            )

        logger.info(f"Loaded prompt: {config['name']} v{config['version']} - {config.get('description', 'N/A')}")
        return config

    @classmethod
    def format_prompt(
        cls,
        prompt_config: dict[str, Any],
        **variables: dict[str, str],
    ) -> tuple[str, str]:
        """プロンプトテンプレートを変数でフォーマット.

        Args:
            prompt_config: load_prompt()で読み込んだプロンプト設定
            **variables: テンプレート変数（例: works_list="...", query_text="..."）

        Returns:
            (system_message, formatted_user_message) のタプル

        Raises:
            KeyError: 必須のテンプレート変数が提供されていない場合
        """
        system_message = prompt_config["system_message"]
        user_template = prompt_config["user_template"]

        try:
            formatted_user = user_template.format(**variables)
        except KeyError as e:
            defined_vars = prompt_config.get("variables", [])
            var_names = [v.get("name") for v in defined_vars]
            raise KeyError(
                f"Missing template variable: {e}. "
                f"Expected variables: {var_names}"
            ) from e

        return system_message, formatted_user

    @classmethod
    def get_api_params(cls, prompt_config: dict[str, Any]) -> dict[str, Any]:
        """APIパラメータを取得.

        Args:
            prompt_config: load_prompt()で読み込んだプロンプト設定

        Returns:
            APIパラメータ（response_format, temperature, max_tokensなど）
        """
        return prompt_config.get("api_params", {})


# 便利エイリアス関数
def load_prompt(name: str) -> dict[str, Any]:
    """プロンプト設定を読み込む便利関数.

    Args:
        name: プロンプト名

    Returns:
        プロンプト設定辞書
    """
    return PromptConfig.load_prompt(name)


def format_prompt(
    prompt_config: dict[str, Any],
    **variables: dict[str, str],
) -> tuple[str, str]:
    """プロンプトをフォーマットする便利関数.

    Args:
        prompt_config: プロンプト設定辞書
        **variables: テンプレート変数

    Returns:
        (system_message, formatted_user_message) のタプル
    """
    return PromptConfig.format_prompt(prompt_config, **variables)


def list_prompts() -> list[str]:
    """利用可能なプロンプトリストを取得する便利関数.

    Returns:
        プロンプト名のリスト
    """
    return PromptConfig.list_available_prompts()
