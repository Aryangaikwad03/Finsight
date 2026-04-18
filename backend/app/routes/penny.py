"""
app/routes/penny.py — Penny AI Assistant v2.0 endpoints
Supports: SSE streaming chat, chat history, feedback, vector refresh.
"""
import json
import logging
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from typing import List, Optional

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User

router = APIRouter(prefix="/penny", tags=["penny-ai"])
logger = logging.getLogger("finsight.penny")


# ── Schemas ───────────────────────────────────────────────────────────────────

class Message(BaseModel):
    role:    str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    question: str

class AutoCategorizeRequest(BaseModel):
    txn_ids: List[int]

class FeedbackRequest(BaseModel):
    message_id: int
    helpful: bool
    comment: Optional[str] = None


# ── Chat endpoint (SSE Streaming) ─────────────────────────────────────────────

@router.post("/chat")
async def chat(
    body: ChatRequest,
    db:   AsyncSession = Depends(get_db),
    user: User         = Depends(get_current_user),
):
    """
    Streaming chat with Penny using SSE.
    Frontend should use EventSource or fetch with ReadableStream.
    """
    from app.services.penny_service import chat_with_penny
    from app.core.db_config import save_chat_message

    user_id = str(user.id)
    question = body.question

    # Save user message to history (non-blocking intent capture first)
    try:
        save_chat_message(user_id, "user", question)
    except Exception:
        pass

    async def sse_generator():
        full_response = []
        detected_intent = "general"
        gen = None
        try:
            result = chat_with_penny(
                user_id=user_id,
                user_name=user.full_name,
                messages=[m.dict() for m in body.messages],
                question=question,
                stream=True,
            )
            gen, detected_intent = result
        except Exception as e:
            logger.error("Chat init error: %s", e, exc_info=True)
            err_msg = "I'm a bit overwhelmed right now — please try again in a minute! 🙏"
            if "rate_limit" in str(e).lower() or "429" in str(e):
                err_msg = "I've hit my daily response limit. I'll be fully available again shortly — please try in a few minutes!"
            yield f"data: {json.dumps({'content': err_msg})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
            return

        try:
            for chunk in gen:
                full_response.append(chunk)
                yield f"data: {json.dumps({'content': chunk})}\n\n"
        except Exception as e:
            logger.error("Chat stream error: %s", e, exc_info=True)
            err_msg = "Something went wrong mid-response. Please try again!"
            if "rate_limit" in str(e).lower() or "429" in str(e):
                err_msg = "I've hit my daily response limit. I'll be back in a few minutes!"
            yield f"data: {json.dumps({'content': err_msg})}\n\n"
        finally:
            full_text = "".join(full_response)
            if full_text:
                try:
                    msg_id = save_chat_message(user_id, "assistant", full_text, detected_intent)
                    yield f"data: {json.dumps({'done': True, 'message_id': msg_id, 'intent': detected_intent})}\n\n"
                except Exception:
                    yield f"data: {json.dumps({'done': True})}\n\n"
            else:
                yield f"data: {json.dumps({'done': True})}\n\n"


    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


# ── Chat History ─────────────────────────────────────────────────────────────

@router.get("/history")
async def get_history(
    limit: int = 50,
    user: User = Depends(get_current_user),
):
    """Fetch persistent chat history for the logged-in user."""
    from app.core.db_config import get_chat_history
    history = get_chat_history(str(user.id), limit=limit)
    return {"history": history}


@router.delete("/history")
async def clear_history(
    user: User = Depends(get_current_user),
):
    """Clear all chat history for the logged-in user."""
    from app.core.db_config import clear_chat_history
    clear_chat_history(str(user.id))
    return {"status": "success", "message": "Chat history cleared."}


# ── Feedback ─────────────────────────────────────────────────────────────────

@router.post("/feedback")
async def submit_feedback(
    body: FeedbackRequest,
    user: User = Depends(get_current_user),
):
    """Submit thumbs up/down feedback on a Penny response."""
    from app.core.db_config import save_feedback
    save_feedback(str(user.id), body.message_id, body.helpful, body.comment)
    return {"status": "success"}


# ── Vector Refresh ────────────────────────────────────────────────────────────

@router.post("/refresh-vectors")
async def refresh_vectors(
    user: User = Depends(get_current_user),
):
    """Trigger Pinecone re-indexing for the user's latest financial data."""
    from app.services.vector_store import upsert_user_vectors
    try:
        upsert_user_vectors(str(user.id), user.full_name)
        return {"status": "success", "message": "Vectors refreshed."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Bank Statement Upload ─────────────────────────────────────────────────────

@router.post("/upload-statement")
async def upload_statement(
    file: UploadFile = File(...),
    password: Optional[str] = Form(None),
    db:   AsyncSession = Depends(get_db),
    user: User         = Depends(get_current_user),
):
    """Upload a bank statement PDF/CSV. Parses and saves transactions to DB."""
    from app.services.penny_service import (
        parse_bank_statement, statement_to_fi_format, store_user_context_vectors
    )
    from app.core.db_config import save_session, save_consent, save_fi_data
    import uuid
    from datetime import datetime, timezone

    user_id = str(user.id)
    file_bytes = await file.read()
    if len(file_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Max 10MB.")

    try:
        parsed = parse_bank_statement(file_bytes, file.filename, user.full_name, password)

        stmt_consent_id = f"stmt-{user_id[:8]}-{uuid.uuid4().hex[:8]}"
        stmt_session_id = f"sess-{uuid.uuid4().hex[:12]}"

        save_consent(
            consent_id=stmt_consent_id,
            user_id=user_id,
            vua=f"statement-upload@{file.filename}",
            status="ACTIVE",
            data_range={"from": "2020-01-01T00:00:00Z", "to": datetime.now(timezone.utc).isoformat()}
        )
        save_session(stmt_session_id, user_id, stmt_consent_id, "COMPLETED")

        fi_data = statement_to_fi_format(parsed, user_id, stmt_session_id, stmt_consent_id)
        save_fi_data(stmt_session_id, user_id, stmt_consent_id, fi_data)

        # Refresh Pinecone vectors after upload (non-critical)
        try:
            store_user_context_vectors(user_id, user.full_name)
        except Exception:
            pass

        acc_info = parsed.get("account_info", {})
        txn_count = len(parsed.get("transactions", []))

        return {
            "success": True,
            "message": f"✅ Statement parsed successfully! Found {txn_count} transactions.",
            "account": acc_info.get("account_number", "Unknown"),
            "bank": acc_info.get("bank_name", "Unknown"),
            "txn_count": txn_count,
            "consent_id": stmt_consent_id,
        }

    except ValueError as e:
        if str(e) == "encrypted_pdf":
            raise HTTPException(status_code=401, detail="encrypted_pdf")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("Statement upload failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process statement: {str(e)}")


# ── Auto-categorize ───────────────────────────────────────────────────────────

@router.post("/auto-categorize")
async def auto_categorize(
    body: AutoCategorizeRequest,
    db:   AsyncSession = Depends(get_db),
    user: User         = Depends(get_current_user),
):
    """Use Penny to suggest better categories for selected transactions."""
    from app.services.penny_service import auto_categorize_transactions
    from app.core.db_config import get_connection
    from psycopg2 import extras as _extras

    user_id = str(user.id)
    conn = get_connection(); cur = conn.cursor(cursor_factory=_extras.RealDictCursor)
    try:
        cur.execute("""
            SELECT txn_id, txn_type, payment_mode, narration, amount, category, subcategory
            FROM transactions WHERE user_id = %s AND txn_id = ANY(%s)
        """, (user_id, body.txn_ids))
        txns = [dict(r) for r in cur.fetchall()]
        suggestions = auto_categorize_transactions(txns)
        updated = 0
        cur2 = conn.cursor()
        for s in suggestions:
            if float(s.get("confidence", 0)) >= 0.7:
                cur2.execute(
                    "UPDATE transactions SET category=%s, subcategory=%s WHERE txn_id=%s AND user_id=%s",
                    (s.get("category"), s.get("subcategory"), s.get("txn_id"), user_id)
                )
                if cur2.rowcount > 0: updated += 1
        conn.commit(); cur2.close()
        return {"suggestions": suggestions, "updated": updated,
                "message": f"Updated {updated} of {len(txns)} transactions"}
    finally:
        cur.close(); conn.close()


# ── Proactive Insights ────────────────────────────────────────────────────────

@router.get("/insights")
async def get_insights(
    db:   AsyncSession = Depends(get_db),
    user: User         = Depends(get_current_user),
):
    """Get Penny's proactive financial insights for the user."""
    from app.services.penny_service import _llm
    from app.services.intent_router import resolve_intent, format_db_facts

    user_id = str(user.id)
    # Pull rich context for insights
    spending_facts = format_db_facts(resolve_intent("spending_summary", user_id, ""))
    savings_facts  = format_db_facts(resolve_intent("savings_analysis", user_id, ""))
    budget_facts   = format_db_facts(resolve_intent("budget_status", user_id, ""))

    prompt = f"""You are Penny, a financial AI for FinSight.
Based on the following data, provide 4 specific, personalized insights:
{spending_facts}
{savings_facts}
{budget_facts}

Format as JSON array:
[{{"type": "warning|tip|positive|saving", "title": "short title", "message": "2-3 sentence insight"}}]
Return ONLY the JSON array, no extra text."""

    try:
        import re
        raw = _llm([{"role": "user", "content": prompt}], temperature=0.5, max_tokens=700)
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()
        insights = json.loads(raw)
        return {"insights": insights}
    except Exception as e:
        logger.error("Insights error: %s", e)
        return {"insights": [
            {"type": "tip", "title": "Welcome to Penny!",
             "message": "I'm analysing your transactions. Ask me anything about your finances!"}
        ]}
