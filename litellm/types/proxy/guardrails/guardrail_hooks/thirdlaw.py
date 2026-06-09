from typing import Optional

from pydantic import Field

from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel


class ThirdlawGuardrailConfigModel(GuardrailConfigModel):
    api_base: Optional[str] = Field(
        default=None,
        description="Thirdlaw Guardrail API Base URL. Env: THIRDLAW_API_BASE.",
        json_schema_extra={
            "examples": [
                "http://localhost:9090",
                "https://guardrails.thirdlaw.com",
            ]
        },
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for Thirdlaw. Env: THIRDLAW_API_KEY.",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Thirdlaw"
