import secrets

from fastapi import Header, HTTPException

from app.config import settings

LOCAL_TENANT = "local"


def get_tenant_id(authorization: str | None = Header(default=None)) -> str:
    """tenant_id is derived server-side from the Bearer key, never read from
    the request body — a body-supplied tenant_id would let any caller write
    into any tenant's vault. (The spec's example bodies show tenant_id as a
    field; its own auth section forbids trusting it. Auth section wins.)

    Keyless local mode: if no server.api_key is configured, requests are
    accepted unauthenticated and pinned to the 'local' tenant. The deployment
    is expected to bind 127.0.0.1 in that mode (the default host).
    """
    if not settings.server.api_key:
        return LOCAL_TENANT

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Bearer token required")
    supplied = authorization.removeprefix("Bearer ")
    if not secrets.compare_digest(supplied, settings.server.api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    # Single-key deployments are single-tenant; a key->tenant store arrives
    # with the multi-tenant control plane (Pro layer).
    return "default"
