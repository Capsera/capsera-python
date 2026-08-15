"""AI gateway integrations.

Capsera treats gateways (LiteLLM, Portkey, Bifrost, OpenRouter, Helicone,
Cloudflare AI Gateway, ...) as a first-class dimension alongside providers:

- **LiteLLM (Python SDK)** — ``install_litellm()`` registers a CustomLogger
  callback with litellm so every ``litellm.completion/acompletion/embedding``
  call is captured with LiteLLM's own cost figure, and wraps the top-level
  entry points with Capsera's pre-call budget check. Installed automatically
  by ``capsera.init()`` when litellm is importable.
- **LiteLLM Proxy (server)** — ``capsera.gateways.litellm_proxy`` ships a
  drop-in handler for the proxy's ``callbacks`` config, adding org-level
  capture and hard pre-call budget blocking at the gateway itself.
- **Portkey (Python SDK)** — ``install_portkey()`` patches the portkey_ai
  chat/embeddings surfaces. Installed automatically by ``capsera.init()``.
- **OpenAI-compatible gateways** (Bifrost, OpenRouter, Helicone, Cloudflare
  AI Gateway, a self-hosted LiteLLM proxy) need no integration code at all:
  they are addressed by pointing the OpenAI/Anthropic SDK ``base_url`` at the
  gateway, which the existing interceptor already captures — the gateway is
  detected from the host (``_catalog.GATEWAY_HOSTS``) or declared explicitly
  via ``capsera.init(gateway="bifrost")``.
"""
from ._litellm import install_litellm
from ._portkey import install_portkey

__all__ = ["install_litellm", "install_portkey"]
