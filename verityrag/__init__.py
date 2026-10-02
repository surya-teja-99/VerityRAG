"""VerityRAG: an agentic retrieval-augmented generation pipeline with a verifier agent."""

__version__ = "0.1.0"

import os as _os


def _sanitize_no_proxy() -> None:
    # Some proxied VMs ship no_proxy entries like "[::1]" (bracketed IPv6).
    # The httpx version used by huggingface_hub fails to parse those
    # ("InvalidURL: Invalid port"), which breaks model downloads. The
    # bracketed entries always duplicate an unbracketed one, so dropping them
    # is safe and keeps the proxy working for external hosts.
    for _key in ("no_proxy", "NO_PROXY"):
        _val = _os.environ.get(_key)
        if _val and "[" in _val:
            _os.environ[_key] = ",".join(p for p in _val.split(",") if "[" not in p)


_sanitize_no_proxy()
