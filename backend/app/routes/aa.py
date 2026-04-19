"""
app/routes/aa.py  —  Setu AA integration endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, status, Request, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from typing import Optional
import logging

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.services.aa_service import (
    create_consent_for_user,
    get_consent_status_from_setu,
    fetch_fi_data_for_consent,
    revoke_consent_for_user,
)

router = APIRouter(prefix="/aa", tags=["account-aggregator"])
logger = logging.getLogger("finsight.aa")


# ── Schemas ───────────────────────────────────────────────────────────────────

class CreateConsentRequest(BaseModel):
    mobile:    str
    aa_handle: str = "onemoney"
    preset:    str = "banking"

class CreateConsentResponse(BaseModel):
    consent_id:  str
    webview_url: str
    status:      str

class ConsentStatusResponse(BaseModel):
    consent_id:    str
    status:        str
    vua:           Optional[str]
    fi_data_ready: bool = False

class FetchDataResponse(BaseModel):
    session_id: str
    status:     str
    message:    str


# ── Consent endpoints ─────────────────────────────────────────────────────────

@router.post("/create-consent", response_model=CreateConsentResponse)
async def create_consent(
    body: CreateConsentRequest,
    db:   AsyncSession = Depends(get_db),
    user: User         = Depends(get_current_user),
):
    mobile = body.mobile.replace("+91", "").replace(" ", "").strip()
    vua    = f"{mobile}@{body.aa_handle}"
    result = await create_consent_for_user(db, user, vua, body.preset)
    return CreateConsentResponse(**result)


@router.get("/consent-status/{consent_id}", response_model=ConsentStatusResponse)
async def get_consent_status(
    consent_id: str,
    db:   AsyncSession = Depends(get_db),
    user: User         = Depends(get_current_user),
):
    result = await get_consent_status_from_setu(db, user, consent_id)
    return ConsentStatusResponse(**result)


@router.post("/fetch-data/{consent_id}", response_model=FetchDataResponse)
async def fetch_data(
    consent_id:       str,
    background_tasks: BackgroundTasks,
    db:   AsyncSession = Depends(get_db),
    user: User         = Depends(get_current_user),
):
    if user.aa_consent_id != consent_id:
        raise HTTPException(status_code=403, detail="Consent does not belong to this user")
    background_tasks.add_task(fetch_fi_data_for_consent, db, user, consent_id)
    return FetchDataResponse(session_id="pending", status="PROCESSING",
                             message="Data fetch started.")


@router.get("/my-consent")
async def get_my_consent(
    db:   AsyncSession = Depends(get_db),
    user: User         = Depends(get_current_user),
):
    return {
        "consent_id":     user.aa_consent_id,
        "consent_status": user.aa_consent_status,
        "vua":            user.vua,
    }


@router.delete("/revoke-consent/{consent_id}")
async def revoke_consent(
    consent_id: str,
    db:   AsyncSession = Depends(get_db),
    user: User         = Depends(get_current_user),
):
    if user.aa_consent_id != consent_id:
        raise HTTPException(status_code=403, detail="Consent does not belong to this user")
    return await revoke_consent_for_user(db, user, consent_id)


@router.post("/webhook")
async def setu_webhook(request: Request, db: AsyncSession = Depends(get_db)):
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    notification_type = payload.get("type", "")
    logger.info("Setu webhook: type=%s", notification_type)
    if notification_type == "CONSENT_STATUS_UPDATE":
        consent_id = payload.get("consentId")
        new_status = payload.get("data", {}).get("status")
        if consent_id and new_status:
            from sqlalchemy import update
            await db.execute(
                update(User)
                .where(User.aa_consent_id == consent_id)
                .values(aa_consent_status=new_status)
            )
            await db.commit()
    return {"status": "received"}


# ── FI Data endpoint ──────────────────────────────────────────────────────────

@router.get("/fi-data")
async def get_fi_data(
    month: int = None,
    year:  int = None,
    db:    AsyncSession = Depends(get_db),
    user:  User         = Depends(get_current_user),
):
    """
    Returns accounts + transactions + summary for the logged-in user.
    Always user-scoped — all sessions merged, deduplicated at DB level.
    """
    user_id = str(user.id)
    logger.info("GET /fi-data user=%s month=%s year=%s", user_id, month, year)

    from app.core.db_config import (
        get_user_accounts, get_user_transactions,
        get_user_summary, get_category_breakdown
    )

    try:
        accounts     = get_user_accounts(user_id)
        transactions = get_user_transactions(user_id)
        summary      = get_user_summary(user_id, month=month, year=year)
        breakdown    = get_category_breakdown(user_id, month=month, year=year)
    except Exception as e:
        logger.error("fi-data query failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    # Fallback: retag rows by consent_id if nothing found by user_id
    if not accounts and user.aa_consent_id:
        logger.warning("Empty — running consent_id retag fallback")
        try:
            accounts, transactions, summary = _fetch_by_consent_id(
                user.aa_consent_id, user_id
            )
            breakdown = get_category_breakdown(user_id)
        except Exception as e:
            logger.error("fallback failed: %s", e, exc_info=True)

    logger.info("fi-data: accounts=%d txns=%d", len(accounts), len(transactions))
    return {
        "user_id":      user_id,
        "summary":      summary,
        "accounts":     accounts,
        "transactions": transactions,
        "breakdown":    breakdown,
    }


def _fetch_by_consent_id(consent_id: str, user_id: str):
    """
    Fallback: if data was stored before user_id was wired in,
    fetch by consent_id and re-tag with correct user_id.
    """
    from app.core.db_config import get_connection
    from psycopg2 import extras

    conn = get_connection()
    cur  = conn.cursor(cursor_factory=extras.RealDictCursor)
    try:
        # Re-tag existing rows with correct user_id
        cur.execute(
            "UPDATE fi_data SET user_id=%s WHERE consent_id=%s AND (user_id='' OR user_id='legacy')",
            (user_id, consent_id)
        )
        cur.execute(
            "UPDATE transactions SET user_id=%s WHERE fi_data_id IN "
            "(SELECT fi_data_id FROM fi_data WHERE consent_id=%s) AND (user_id='' OR user_id='legacy')",
            (user_id, consent_id)
        )
        cur.execute(
            "UPDATE summaries SET user_id=%s WHERE fi_data_id IN "
            "(SELECT fi_data_id FROM fi_data WHERE consent_id=%s) AND (user_id='' OR user_id='legacy')",
            (user_id, consent_id)
        )
        cur.execute(
            "UPDATE profiles SET user_id=%s WHERE fi_data_id IN "
            "(SELECT fi_data_id FROM fi_data WHERE consent_id=%s) AND (user_id='' OR user_id='legacy')",
            (user_id, consent_id)
        )
        conn.commit()
        logger.info("Re-tagged rows for consent_id=%s with user_id=%s", consent_id, user_id)

        # Now fetch by consent_id directly
        cur.execute("""
            SELECT f.fi_data_id, f.masked_acc_number, f.account_type,
                   f.fip_id, f.fi_status,
                   s.current_balance, s.balance_datetime,
                   s.branch, s.ifsc_code,
                   p.holder_name, p.holder_pan
            FROM fi_data f
            LEFT JOIN summaries s ON f.fi_data_id = s.fi_data_id
            LEFT JOIN profiles  p ON f.fi_data_id = p.fi_data_id
            WHERE f.consent_id = %s
            ORDER BY f.created_at DESC
        """, (consent_id,))
        accounts = [dict(r) for r in cur.fetchall()]

        cur.execute("""
            SELECT t.*, f.masked_acc_number, f.account_type
            FROM transactions t
            JOIN fi_data f ON t.fi_data_id = f.fi_data_id
            WHERE f.consent_id = %s
            ORDER BY t.txn_date DESC
            LIMIT 200
        """, (consent_id,))
        transactions = [dict(r) for r in cur.fetchall()]

        cur.execute("""
            SELECT
                COALESCE(SUM(s.current_balance), 0) AS total_balance,
                COUNT(DISTINCT f.fi_data_id)         AS account_count,
                COALESCE(SUM(CASE WHEN t.txn_type='CREDIT' THEN t.amount ELSE 0 END),0) AS total_income,
                COALESCE(SUM(CASE WHEN t.txn_type='DEBIT'  THEN t.amount ELSE 0 END),0) AS total_expenses
            FROM fi_data f
            LEFT JOIN summaries    s ON f.fi_data_id = s.fi_data_id
            LEFT JOIN transactions t ON f.fi_data_id = t.fi_data_id
            WHERE f.consent_id = %s
        """, (consent_id,))
        row     = cur.fetchone()
        summary = dict(row) if row else {}

        logger.info("Fallback found: accounts=%d txns=%d", len(accounts), len(transactions))
        return accounts, transactions, summary

    finally:
        cur.close()
        conn.close()


@router.get("/debug-fi")
async def debug_fi_data(
    db:   AsyncSession = Depends(get_db),
    user: User         = Depends(get_current_user),
):
    """
    Debug endpoint — shows exactly what's in the DB for this user.
    Also auto-fixes user_id mismatch by retagging rows via consent_id.
    Hit GET /api/aa/debug-fi to diagnose and fix in one call.
    """
    from app.core.db_config import get_connection
    from psycopg2 import extras as _extras

    user_id    = str(user.id)
    consent_id = user.aa_consent_id or ""

    conn = get_connection()
    cur  = conn.cursor(cursor_factory=_extras.RealDictCursor)

    try:
        # 1. What user_ids exist in fi_data?
        cur.execute("SELECT DISTINCT user_id, COUNT(*) as rows FROM fi_data GROUP BY user_id")
        fi_data_users = [dict(r) for r in cur.fetchall()]

        # 2. What user_ids exist in transactions?
        cur.execute("SELECT DISTINCT user_id, COUNT(*) as rows FROM transactions GROUP BY user_id")
        txn_users = [dict(r) for r in cur.fetchall()]

        # 3. What sessions exist for this consent?
        cur.execute("SELECT session_id, user_id, status, created_at FROM sessions WHERE consent_id=%s", (consent_id,))
        sessions = [dict(r) for r in cur.fetchall()]

        # 4. What fi_data exists for this consent?
        cur.execute("SELECT fi_data_id, user_id, masked_acc_number, account_type FROM fi_data WHERE consent_id=%s", (consent_id,))
        fi_rows = [dict(r) for r in cur.fetchall()]

        # 5. Transaction count for this consent
        cur.execute("""
            SELECT COUNT(*) as txn_count FROM transactions t
            JOIN fi_data f ON t.fi_data_id = f.fi_data_id
            WHERE f.consent_id = %s
        """, (consent_id,))
        txn_count = cur.fetchone()["txn_count"]

        # 6. AUTO-FIX: retag all rows for this consent with correct user_id
        fixed = 0
        if consent_id:
            cur2 = conn.cursor()
            cur2.execute("UPDATE fi_data SET user_id=%s WHERE consent_id=%s RETURNING fi_data_id", (user_id, consent_id))
            fi_ids = [r[0] for r in cur2.fetchall()]
            fixed += len(fi_ids)
            if fi_ids:
                cur2.execute("UPDATE transactions SET user_id=%s WHERE fi_data_id = ANY(%s)", (user_id, fi_ids))
                cur2.execute("UPDATE summaries   SET user_id=%s WHERE fi_data_id = ANY(%s)", (user_id, fi_ids))
                cur2.execute("UPDATE profiles    SET user_id=%s WHERE fi_data_id = ANY(%s)", (user_id, fi_ids))
            cur2.execute("UPDATE sessions SET user_id=%s, status='COMPLETED' WHERE consent_id=%s", (user_id, consent_id))
            cur2.execute("UPDATE consents SET user_id=%s WHERE consent_id=%s",                     (user_id, consent_id))
            conn.commit()
            cur2.close()

        return {
            "logged_in_user_id": user_id,
            "consent_id":        consent_id,
            "fi_data_users_in_db":   fi_data_users,
            "txn_users_in_db":       txn_users,
            "sessions_for_consent":  sessions,
            "fi_rows_for_consent":   fi_rows,
            "txn_count_for_consent": txn_count,
            "auto_fix_rows_retagged": fixed,
            "message": f"✅ Retagged {fixed} fi_data rows + all child rows with user_id={user_id}. Now reload /transactions."
        }
    finally:
        cur.close()
        conn.close()


# ── Category management ───────────────────────────────────────────────────────

class UpdateCategoryRequest(BaseModel):
    txn_id:      int
    category:    str
    subcategory: str = ""
    apply_all_narration: str = None


@router.patch("/transaction/category")
async def update_transaction_category(
    body: UpdateCategoryRequest,
    db:   AsyncSession = Depends(get_db),
    user: User         = Depends(get_current_user),
):
    """Update the category/subcategory of a single transaction."""
    from app.core.db_config import get_connection
    user_id = str(user.id)
    conn = get_connection(); cur = conn.cursor()
    try:
        if body.apply_all_narration is not None:
            search_pattern = f"%{body.apply_all_narration.strip()}%"
            cur.execute("""
                UPDATE transactions
                SET category    = %s,
                    subcategory = %s
                WHERE narration ILIKE %s AND user_id = %s
            """, (body.category.strip(), body.subcategory.strip(), search_pattern, user_id))
            if cur.rowcount == 0:
                conn.rollback()
                raise HTTPException(status_code=404, detail="Transactions not found")
            conn.commit()
            return {"success": True, "updated": cur.rowcount,
                    "category": body.category, "subcategory": body.subcategory}
        else:
            cur.execute("""
                UPDATE transactions
                SET category    = %s,
                    subcategory = %s
                WHERE txn_id = %s AND user_id = %s
            """, (body.category.strip(), body.subcategory.strip(), body.txn_id, user_id))
            if cur.rowcount == 0:
                conn.rollback()
                raise HTTPException(status_code=404, detail="Transaction not found")
            conn.commit()
            return {"success": True, "txn_id": body.txn_id,
                    "category": body.category, "subcategory": body.subcategory}
    except HTTPException:
        raise
    except Exception as e:
        conn.rollback(); raise HTTPException(status_code=500, detail=str(e))
    finally:
        cur.close(); conn.close()


@router.get("/categories")
async def get_categories(user: User = Depends(get_current_user)):
    """Return all available categories (built-in + user-created)."""
    from app.core.db_config import get_connection
    from psycopg2 import extras as _extras
    user_id = str(user.id)
    conn = get_connection(); cur = conn.cursor(cursor_factory=_extras.RealDictCursor)
    try:
        # Built-in categories
        # Built-in categories (20-category framework)
        builtin = [
            {"category": "Food & Dining", "subcategories": ["Restaurants/Cafe"]},
            {"category": "Groceries", "subcategories": []},
            {"category": "Transportation", "subcategories": ["Fuel", "Taxi/Ride Hailing", "Public Transport"]},
            {"category": "Shopping & Retail", "subcategories": ["E-commerce", "Clothing & Fashion", "Electronics"]},
            {"category": "Bills & Utilities", "subcategories": ["Utilities/BillPay"]},
            {"category": "Housing & Rent", "subcategories": ["Rent/Maintenance"]},
            {"category": "Healthcare & Medical", "subcategories": ["Medical/Healthcare"]},
            {"category": "Entertainment & Leisure", "subcategories": ["Sub/Movies/Events"]},
            {"category": "Travel", "subcategories": ["Hotel/Flight"]},
            {"category": "Education", "subcategories": ["Education/Fees"]},
            {"category": "Investments & Savings", "subcategories": ["Investments", "Interest Income", "Redemption", "Renewal"]},
            {"category": "Insurance", "subcategories": ["Insurance Premium"]},
            {"category": "Salary & Income", "subcategories": ["Salary/Payroll", "Refund/Cashback", "Freelance/Bonus", "Credit"]},
            {"category": "Transfers", "subcategories": ["Transfers", "UPI", "NEFT", "IMPS", "Fund Transfer", "RTGS", "Card Payment", "ACH", "NACH"]},
            {"category": "Taxes & Government", "subcategories": ["Tax/Govt Fees", "Tax Deducted at Source"]},
            {"category": "ATM / Cash Withdrawal", "subcategories": ["Cash Transaction", "Cash", "ATM"]},
            {"category": "Fees & Charges", "subcategories": ["Bank Fees/Penalty"]},
            {"category": "Donations & Charity", "subcategories": ["Donations"]},
            {"category": "Business / Professional Expenses", "subcategories": ["Business/Professional"]},
            {"category": "Subscription Services", "subcategories": ["Subscriptions"]},
            {"category": "Uncategorized / Unknown", "subcategories": ["Debit", "Other"]},
        ]
        # Also fetch any custom categories this user has created
        cur.execute("""
            SELECT DISTINCT category, subcategory
            FROM transactions
            WHERE user_id = %s
              AND category NOT IN (
                'Digital Payments','Bank Transfer','Card','Cash',
                'Investment Returns','Tax','Account','Investment',
                'Income','Expense','Other'
              )
            ORDER BY category
        """, (user_id,))
        custom_rows = cur.fetchall()
        custom = {}
        for r in custom_rows:
            cat = r["category"] or "Other"
            sub = r["subcategory"] or ""
            if cat not in custom:
                custom[cat] = []
            if sub and sub not in custom[cat]:
                custom[cat].append(sub)

        custom_list = [{"category": k, "subcategories": v, "custom": True}
                       for k, v in custom.items()]
        return {"builtin": builtin, "custom": custom_list}
    finally:
        cur.close(); conn.close()
@router.get("/user-summary")
async def get_user_summary_route(
    month: int = None,
    year: int = None,
    user: User = Depends(get_current_user)
):
    from app.core.db_config import get_user_summary
    return get_user_summary(str(user.id), month, year)


@router.get("/category-breakdown")
async def get_category_breakdown_route(
    month: int = None,
    year: int = None,
    user: User = Depends(get_current_user)
):
    from app.core.db_config import get_category_breakdown
    return get_category_breakdown(str(user.id), month, year)


@router.get("/category-drilldown")
async def get_category_drilldown_route(
    category: str,
    month: int = None,
    year: int = None,
    user: User = Depends(get_current_user)
):
    from app.core.db_config import get_category_drilldown
    return get_category_drilldown(str(user.id), category, month, year)
