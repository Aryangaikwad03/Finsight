"""
app/routes/penny.py — Penny AI Assistant endpoints
"""
import logging
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from typing import List, Optional
import json

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User

router = APIRouter(prefix="/penny", tags=["penny-ai"])
logger = logging.getLogger("finsight.penny")


# ── Schemas ───────────────────────────────────────────────────────────────────

class Message(BaseModel):
    role:    str   # user | assistant
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]   # conversation history
    question: str             # current user question


class AutoCategorizeRequest(BaseModel):
    txn_ids: List[int]        # transaction IDs to re-categorize


# ── Chat endpoint ─────────────────────────────────────────────────────────────

@router.post("/chat")
async def chat(
    body: ChatRequest,
    db:   AsyncSession = Depends(get_db),
    user: User         = Depends(get_current_user),
):
    """Send a message to Penny. Returns AI response with financial context."""
    from app.services.penny_service import chat_with_penny

    try:
        response = chat_with_penny(
            user_id   = str(user.id),
            user_name = user.full_name,
            messages  = [m.dict() for m in body.messages],
            question  = body.question,
        )
        return {"response": response, "role": "assistant"}
    except Exception as e:
        logger.error("Chat error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Penny unavailable: {str(e)}")


# ── Bank statement upload ─────────────────────────────────────────────────────

@router.post("/upload-statement")
async def upload_statement(
    file: UploadFile = File(...),
    password: Optional[str] = Form(None),
    db:   AsyncSession = Depends(get_db),
    user: User         = Depends(get_current_user),
):
    """
    Upload a bank statement (PDF/CSV/TXT).
    Penny parses it with LLM and saves transactions to the DB.
    """
    from app.services.penny_service import (
        parse_bank_statement, statement_to_fi_format, store_user_context_vectors
    )
    from app.core.db_config import save_session, save_consent, save_fi_data
    import uuid
    from datetime import datetime, timezone

    user_id = str(user.id)

    # Read file
    file_bytes = await file.read()
    if len(file_bytes) > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(status_code=400, detail="File too large. Max 10MB.")

    try:
        # 1. Parse statement with LLM
        parsed = parse_bank_statement(file_bytes, file.filename, user.full_name, password)

        # 2. Create a synthetic session/consent for the uploaded statement
        stmt_consent_id = f"stmt-{user_id[:8]}-{uuid.uuid4().hex[:8]}"
        stmt_session_id = f"sess-{uuid.uuid4().hex[:12]}"

        save_consent(
            consent_id = stmt_consent_id,
            user_id    = user_id,
            vua        = f"statement-upload@{file.filename}",
            status     = "ACTIVE",
            data_range = {"from": "2020-01-01T00:00:00Z", "to": datetime.now(timezone.utc).isoformat()}
        )
        save_session(stmt_session_id, user_id, stmt_consent_id, "COMPLETED")

        # 3. Convert to fi_parser format and save
        fi_data = statement_to_fi_format(parsed, user_id, stmt_session_id, stmt_consent_id)
        save_fi_data(stmt_session_id, user_id, stmt_consent_id, fi_data)

        # 4. Update vector store (async, non-blocking)
        try:
            store_user_context_vectors(user_id, user.full_name)
        except Exception:
            pass  # Non-critical

        acc_info = parsed.get("account_info", {})
        txn_count = len(parsed.get("transactions", []))

        return {
            "success":     True,
            "message":     f"✅ Statement parsed successfully! Found {txn_count} transactions.",
            "account":     acc_info.get("account_number", "Unknown"),
            "bank":        acc_info.get("bank_name", "Unknown"),
            "txn_count":   txn_count,
            "consent_id":  stmt_consent_id,
        }

    except ValueError as e:
        if str(e) == "encrypted_pdf":
            raise HTTPException(status_code=401, detail="encrypted_pdf")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("Statement upload failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process statement: {str(e)}")


# ── Auto-categorize endpoint ──────────────────────────────────────────────────

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
        # Fetch the transactions
        cur.execute("""
            SELECT txn_id, txn_type, payment_mode, narration, amount, category, subcategory
            FROM transactions
            WHERE user_id = %s AND txn_id = ANY(%s)
        """, (user_id, body.txn_ids))
        txns = [dict(r) for r in cur.fetchall()]

        # Get LLM suggestions
        suggestions = auto_categorize_transactions(txns)

        # Apply suggestions with confidence > 0.7
        updated = 0
        cur2 = conn.cursor()
        for s in suggestions:
            if float(s.get("confidence", 0)) >= 0.7:
                cur2.execute("""
                    UPDATE transactions SET category=%s, subcategory=%s
                    WHERE txn_id=%s AND user_id=%s
                """, (s.get("category"), s.get("subcategory"), s.get("txn_id"), user_id))
                if cur2.rowcount > 0:
                    updated += 1
        conn.commit()
        cur2.close()

        return {"suggestions": suggestions, "updated": updated,
                "message": f"Updated {updated} of {len(txns)} transactions"}
    finally:
        cur.close(); conn.close()


# ── Financial insights endpoint ───────────────────────────────────────────────

@router.get("/insights")
async def get_insights(
    db:   AsyncSession = Depends(get_db),
    user: User         = Depends(get_current_user),
):
    """Get Penny's proactive financial insights for the user."""
    from app.services.penny_service import build_user_context, _llm

    ctx = build_user_context(str(user.id), user.full_name)

    prompt = f"""{ctx}

Based on this user's financial data, provide 4-5 specific, personalized monthly insights:
1. One breakdown comparing their recent monthly spending vs their savings potential
2. One goal tracking projection (e.g. 'If you want to save ₹50k for a vacation, it will take X months based on your current ₹Y net flow.')
3. One budgeting suggestion based on 50/30/20 rule applied to their monthly income
4. One observation about their top spending category this month
5. One actionable money-saving tip for next month

Format as JSON array:
[{{"type": "warning|tip|positive|saving", "title": "short title", "message": "2-3 sentence insight"}}]
Return ONLY the JSON array."""

    try:
        raw = _llm([{"role": "user", "content": prompt}], temperature=0.6, max_tokens=1024)
        import re, json
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()
        insights = json.loads(raw)
        return {"insights": insights}
    except Exception as e:
        logger.error("Insights error: %s", e)
        return {"insights": [
            {"type": "tip", "title": "Welcome to Penny!", "message": "I'm analysing your transactions. Ask me anything about your finances!"}
        ]}
