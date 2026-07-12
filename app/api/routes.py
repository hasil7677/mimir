import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_tenant_id
from app.core import pipeline, recall as recall_pipeline, vault
from app.db import duckdb_client, redis_client, vector_store
from app.models.schemas import CaptureRequest, RecallRequest, SessionEndRequest

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/capture")
def capture(body: CaptureRequest, tenant_id: str = Depends(get_tenant_id)):
    """L0 ingest — see app.core.pipeline.capture for the degrade contract."""
    message_ids = pipeline.capture(
        tenant_id, body.user_id, body.session_id,
        [m.model_dump() for m in body.messages],
    )
    return {"message_ids": message_ids}


@router.post("/session/end")
def session_end(body: SessionEndRequest, tenant_id: str = Depends(get_tenant_id)):
    """Flushes this session into the vault + fact store (full L1 -> L1.5 ->
    L3 chain in app.core.pipeline.flush_session) AND recovers any other
    session for this user that captured turns but was never flushed — e.g.
    an adapter process that got recycled mid-conversation and lost track of
    its session_id. See pipeline.flush_all_pending.

    Response stays flat/backward-compatible for THIS session's result;
    any other sessions swept up in the recovery ride along under
    `recovered_sessions` rather than changing the top-level shape.
    """
    summary = pipeline.flush_all_pending(tenant_id, body.user_id, body.session_id)
    if summary["sessions_flushed"] == 0:
        raise HTTPException(status_code=404, detail="No captured messages for this session")

    results = dict(summary["results"])
    this_session = results.pop(body.session_id, None)
    if this_session is None:
        # this session itself had nothing to flush, but orphans did — report
        # the first recovered one as the primary result rather than 404ing
        # on data that plainly exists.
        session_id, this_session = next(iter(results.items()))
        results.pop(session_id, None)

    return {**this_session, "recovered_sessions": results}


@router.post("/recall")
def recall(body: RecallRequest, tenant_id: str = Depends(get_tenant_id)):
    """Memory context for prompt injection: hot turns + hybrid-searched,
    4-signal-scored facts + wikilink-expanded vault notes, one clean string."""
    return recall_pipeline.recall(tenant_id, body.user_id, body.query, body.session_id)


@router.delete("/user/{user_id}")
def erase_user(user_id: str, tenant_id: str = Depends(get_tenant_id)):
    """GDPR erasure across every store: DuckDB rows, Qdrant points, Redis
    keys, the vault directory. The audit log keeps only ids/actions (never
    content) and records the erasure itself — the receipt survives, the
    data doesn't. Redis being unreachable is reported, not fatal."""
    duck = duckdb_client.erase_user(user_id, tenant_id)
    vector_store.erase_user(tenant_id, user_id)
    notes_deleted = vault.erase_user(tenant_id, user_id)

    redis_result: int | str
    try:
        redis_result = redis_client.erase_user(tenant_id, user_id)
    except Exception:
        redis_result = "unreachable — hot keys expire via TTL within 24h"

    receipt = {
        "l0_deleted": duck["l0_deleted"], "l1_deleted": duck["l1_deleted"],
        "vault_notes_deleted": notes_deleted, "redis_keys_deleted": redis_result,
    }
    duckdb_client.log_audit(
        str(uuid.uuid4()), user_id, tenant_id, "erasure", [],
        datetime.now(timezone.utc), receipt if isinstance(redis_result, int) else {**receipt, "redis_keys_deleted": -1},
    )
    return receipt


@router.get("/export/{user_id}")
def export_user(user_id: str, tenant_id: str = Depends(get_tenant_id)):
    """Data portability: everything held on this user, in one JSON payload."""
    return {
        "l0_conversations": duckdb_client.get_all_l0(user_id, tenant_id),
        "l1_memories": duckdb_client.get_all_l1(user_id, tenant_id),
        "vault": vault.export_user(tenant_id, user_id),
    }


@router.get("/vault/notes")
def vault_notes(user_id: str, tenant_id: str = Depends(get_tenant_id)):
    return {"notes": vault.list_notes(tenant_id, user_id)}


@router.get("/vault/note/{target}")
def vault_note(target: str, user_id: str, tenant_id: str = Depends(get_tenant_id)):
    """Reads a note by wikilink target, plus its 1-hop linked notes — the
    same enrichment the recall pipeline will use."""
    note = vault.read_note(tenant_id, user_id, target)
    if note is None:
        raise HTTPException(status_code=404, detail="Note not found")
    frontmatter, note_body = note
    linked = vault.expand_links(tenant_id, user_id, [note_body], hops=1)
    return {"frontmatter": frontmatter, "body": note_body, "linked": linked}
