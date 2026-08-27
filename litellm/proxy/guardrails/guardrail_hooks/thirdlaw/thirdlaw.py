import json
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, Type

import httpx
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import log_guardrail_information
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy.guardrails.guardrail_hooks.generic_guardrail_api.generic_guardrail_api import (
    GenericGuardrailAPI,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel

GUARDRAIL_NAME = "thirdlaw"

RAW_REQUEST_BODY_PARAM = "thirdlaw_raw_request_body"
RAW_RESPONSE_BODY_PARAM = "thirdlaw_raw_response_body"

# LiteLLM bookkeeping that rides along in ``request_data`` but is not part of the
# provider request body. ``proxy_server_request`` in particular holds a shallow
# copy of the same dict and so is self-referential; serializing it would recurse.
_NON_BODY_KEYS = frozenset(
    {
        "proxy_server_request",
        "metadata",
        "litellm_metadata",
        "litellm_logging_obj",
        "response",
        "responses",
        "secret_fields",
        "user_api_key_dict",
    }
)

# Per-request bridge: the parent class builds the outbound payload from
# ``get_guardrail_dynamic_request_body_params``, which only receives the inbound
# body, not the full ``request_data`` the raw bodies live in. A ContextVar keeps
# concurrent requests isolated where instance state would race.
_RAW_BODIES: ContextVar[dict[str, Any]] = ContextVar("thirdlaw_raw_bodies", default={})


def _jsonable(value: Any) -> Any:
    """Round-trip through JSON so the payload cannot carry live objects."""
    return json.loads(json.dumps(value, default=str))


class ThirdlawGuardrailMissingConfig(ValueError):
    pass


class ThirdlawGuardrail(GenericGuardrailAPI):
    def __init__(
        self,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        additional_headers: Optional[str] = None,
        guardrail_timeout: Optional[int] = 60,
        streaming_buffer_until_moderated: bool = True,
        streaming_end_of_stream_only: bool = True,
        streaming_sampling_rate: int = 5,
        **kwargs,
    ):
        resolved_base = api_base or get_secret_str("THIRDLAW_API_BASE")
        if not resolved_base:
            raise ThirdlawGuardrailMissingConfig(
                "ThirdLaw api_base is required. Set api_base in the guardrail "
                "config or the THIRDLAW_API_BASE environment variable."
            )
        resolved_key = api_key or get_secret_str("THIRDLAW_API_KEY")
        thirdlaw_headers = []
        if additional_headers:
            thirdlaw_headers.extend(
                [h.strip() for h in additional_headers.split(",") if h.strip()]
            )
        existing = kwargs.get("extra_headers") or []
        kwargs["extra_headers"] = thirdlaw_headers + [
            h for h in existing if h not in thirdlaw_headers
        ]
        self.guardrail_timeout = httpx.Timeout(timeout=guardrail_timeout, connect=5.0)
        self.streaming_buffer_until_moderated = streaming_buffer_until_moderated
        self.streaming_end_of_stream_only = streaming_end_of_stream_only
        self.streaming_sampling_rate = streaming_sampling_rate
        if "supported_event_hooks" not in kwargs:
            kwargs["supported_event_hooks"] = [
                GuardrailEventHooks.pre_call,
                GuardrailEventHooks.post_call,
                GuardrailEventHooks.during_call,
            ]
        super().__init__(
            api_base=resolved_base,
            api_key=resolved_key,
            streaming_buffer_until_moderated=streaming_buffer_until_moderated,
            streaming_end_of_stream_only=streaming_end_of_stream_only,
            streaming_sampling_rate=streaming_sampling_rate,
            **kwargs,
        )
        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback,
            params={"timeout": self.guardrail_timeout},
        )

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> GenericGuardrailAPIInputs:
        inputs["structured_messages"] = (
            inputs.get("structured_messages", request_data.get("messages", [])) or []
        )
        token = _RAW_BODIES.set(self._capture(request_data))
        try:
            return await super().apply_guardrail(
                inputs=inputs,
                request_data=request_data,
                input_type=input_type,
                logging_obj=logging_obj,
            )
        finally:
            _RAW_BODIES.reset(token)

    def get_guardrail_dynamic_request_body_params(self, request_data: dict) -> dict:
        return {
            **super().get_guardrail_dynamic_request_body_params(request_data),
            **_RAW_BODIES.get(),
        }

    def _capture(self, request_data: Optional[dict]) -> dict[str, Any]:
        """Best-effort extraction of the raw bodies.

        Never raises: a guardrail that throws here would fail live traffic, and a
        missing raw body only costs fidelity because intervene-service still
        synthesizes one.
        """
        if not isinstance(request_data, dict):
            return {}
        return {
            **self._capture_one(RAW_REQUEST_BODY_PARAM, self._request_body, request_data),
            **self._capture_one(RAW_RESPONSE_BODY_PARAM, self._response_body, request_data),
        }

    @staticmethod
    def _capture_one(
        param: str,
        extract: Callable[[dict], Optional[dict]],
        request_data: dict,
    ) -> dict[str, Any]:
        try:
            body = extract(request_data)
        except Exception:
            verbose_proxy_logger.warning(
                "ThirdLaw guardrail: could not capture %s", param, exc_info=True
            )
            return {}
        return {param: _jsonable(body)} if body else {}

    @staticmethod
    def _request_body(request_data: dict) -> Optional[dict]:
        """The outbound chat-completions body.

        ``proxy_server_request.body`` is LiteLLM's own snapshot, but it is a
        shallow copy of ``request_data`` taken after the snapshot key already
        existed, so it points back at itself. Strip the bookkeeping keys from
        either source before use.
        """
        proxy_request = request_data.get("proxy_server_request")
        snapshot = proxy_request.get("body") if isinstance(proxy_request, dict) else None
        source = snapshot if isinstance(snapshot, dict) else request_data
        body = {k: v for k, v in source.items() if k not in _NON_BODY_KEYS}
        return body if body.get("messages") else None

    @staticmethod
    def _response_body(request_data: dict) -> Optional[dict]:
        """The provider response, present only on the post-call hook."""
        response = request_data.get("response")
        if response is None:
            return None
        dump = getattr(response, "model_dump", None)
        body = dump(mode="json") if callable(dump) else response
        return body if isinstance(body, dict) and body.get("choices") else None

    @staticmethod
    def get_config_model() -> Optional[Type["GuardrailConfigModel"]]:
        from litellm.types.proxy.guardrails.guardrail_hooks.thirdlaw import (
            ThirdlawGuardrailConfigModel,
        )

        return ThirdlawGuardrailConfigModel
