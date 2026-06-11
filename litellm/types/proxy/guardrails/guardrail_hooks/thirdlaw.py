from typing import Optional

from pydantic import Field

from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel

class ThirdlawGuardrailConfigModel(GuardrailConfigModel):
    api_base: Optional[str] = Field(
        default=None,
        description="ThirdLaw Guardrail API Base URL. Env: THIRDLAW_API_BASE.",
        json_schema_extra={
            "examples": [
                "http://localhost:9090",
                "https://guardrails.thirdlaw.com",
            ]
        },
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for ThirdLaw. Env: THIRDLAW_API_KEY.",
    )

    additional_headers: Optional[str] = Field(
        default=None,
        description="Additional headers to forward to the ThirdLaw API. Comma-separated list of header names.",
    )
    guardrail_timeout: Optional[int] = Field(
        default=None,
        description="Timeout for the ThirdLaw API request. In seconds.",
    )
    ingest_only: Optional[bool] = Field(
        default=False,
        description="Whether to only ingest the request and response data, without running the guardrail.",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "ThirdLaw"
