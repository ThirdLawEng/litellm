from typing import Optional, Literal

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
    unreachable_fallback: Literal["fail_closed", "fail_open"] = Field(
        default="fail_closed",
        description="What to do when ThirdLaw is unreachable. 'fail_open' = allow, 'fail_closed' = block.",
    )

    guardrail_timeout: Optional[int] = Field(
        default=60,
        description="Timeout for the ThirdLaw API request. In seconds.",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "ThirdLaw"
