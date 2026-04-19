"""
app/services/aa_service.py

setu_aa_client.py / fi_presets.py / fi_parser.py  → KAL root
db_config.py                                        → KAL/backend/app/core/
"""
import json
import logging
import os
import sys
from typing import Any, Dict

from sqlalchemy.ext.asyncio import AsyncSession
from app.core.config import get_settings
from app.models.user import User

logger   = logging.getLogger("finsight.aa_service")
settings = get_settings()

# ── KAL root: 3 levels up from this file ──────────────────────────────────────
# aa_service.py → services/ → app/ → backend/ → KAL/
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_KAL_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))

if _KAL_ROOT not in sys.path:
    sys.path.insert(0, _KAL_ROOT)

# Verify Setu AA files exist at KAL root
for _f in ("setu_aa_client.py", "fi_presets.py", "fi_parser.py"):
    _fp = os.path.join(_KAL_ROOT, _f)
    if not os.path.exists(_fp):
        raise ImportError(f"❌ {_f} not found at {_KAL_ROOT}")
    logger.info("✅ %s found", _f)

# Import Setu originals from KAL root
from setu_aa_client import SetuAAClient, SetuAAConfig  # noqa
from fi_presets     import PRESETS                      # noqa
from fi_parser      import parse_session_response       # noqa

# Import db_config from its correct location: app/core/
from app.core.db_config import (                        # noqa
    init_database, save_consent, save_session,
    save_fi_data, get_user_summary, get_user_accounts, get_user_transactions
)

# Initialise DB tables once on startup
try:
    init_database()
    logger.info("✅ DB tables ready")
except Exception as _e:
    logger.warning("DB init skipped (tables may already exist): %s", _e)

# ── In-memory cache: consent_id → full create-consent API response ────────────
# GET /consents/:id does NOT return detail.dataRange — only the create response
# does. Cache it here so fetch can read the approved dataRange exactly as
# main_data.py Step 3 does.
_consent_resp_cache: dict[str, dict] = {}


def _get_client() -> SetuAAClient:
    return SetuAAClient(SetuAAConfig(
        client_id           = settings.SETU_CLIENT_ID,
        client_secret       = settings.SETU_CLIENT_SECRET,
        product_instance_id = settings.SETU_PRODUCT_INSTANCE_ID,
        environment         = "sandbox" if "sandbox" in settings.SETU_BASE_URL else "production",
        max_poll_attempts   = 40,
        poll_interval_seconds = 8,
    ))


# ── Create Consent ─────────────────────────────────────────────────────────────

async def create_consent_for_user(
    db: AsyncSession, user: User, vua: str, preset: str = "banking",
) -> Dict[str, Any]:
    import asyncio

    def _call():
        client      = _get_client()
        consent_req = PRESETS.get(preset, PRESETS["banking"])(vua)
        consent_req.redirect_url = f"{settings.FRONTEND_URL}/consent"
        return client.create_consent(consent_req)

    loop = asyncio.get_event_loop()
    resp = await loop.run_in_executor(None, _call)

    consent_id  = resp.get("id")
    webview_url = resp.get("url")

    # Cache full response — fetch needs detail.dataRange from this
    _consent_resp_cache[consent_id] = resp

    # Persist consent to DB with user_id
    detail     = resp.get("detail", {})
    data_range = detail.get("dataRange", {})
    try:
        save_consent(consent_id, str(user.id), vua,
                     resp.get("status", "PENDING"), data_range)
    except Exception as e:
        logger.warning("Could not persist consent to DB: %s", e)

    user.vua               = vua
    user.aa_consent_id     = consent_id
    user.aa_consent_status = "PENDING"
    await db.flush()
    await db.refresh(user)

    logger.info("Consent created for %s → id=%s", user.email, consent_id)
    return {"consent_id": consent_id, "webview_url": webview_url, "status": "PENDING"}


# ── Get Consent Status ─────────────────────────────────────────────────────────

async def get_consent_status_from_setu(
    db: AsyncSession, user: User, consent_id: str,
) -> Dict[str, Any]:
    import asyncio

    def _call():
        return _get_client().get_consent_status(consent_id)

    loop = asyncio.get_event_loop()
    resp = await loop.run_in_executor(None, _call)

    setu_status = resp.get("status", "PENDING")
    user.aa_consent_status = setu_status
    await db.flush()

    try:
        save_consent(consent_id, str(user.id), user.vua or "", setu_status)
    except Exception as e:
        logger.warning("Could not update consent status in DB: %s", e)

    return {
        "consent_id":    consent_id,
        "status":        setu_status,
        "vua":           user.vua,
        "fi_data_ready": setu_status == "ACTIVE",
    }


# ── Fetch FI Data ──────────────────────────────────────────────────────────────

async def fetch_fi_data_for_consent(
    db: AsyncSession, user: User, consent_id: str,
) -> Dict[str, Any]:
    import asyncio
    from datetime import datetime, timezone

    user_id = str(user.id)

    def _fetch():
        client = _get_client()

        # Read dataRange from cached create-consent response
        # (mirrors main_data.py Step 3 exactly)
        cached_resp    = _consent_resp_cache.get(consent_id, {})
        approved_range = cached_resp.get("detail", {}).get("dataRange", {})

        if not (approved_range.get("from") and approved_range.get("to")):
            logger.warning("Cache miss for %s — reading dataRange from DB", consent_id)
            from app.core.db_config import get_connection
            conn = get_connection(); cur = conn.cursor()
            cur.execute("SELECT data_range_from, data_range_to FROM consents WHERE consent_id = %s", (consent_id,))
            row = cur.fetchone()
            cur.close(); conn.close()
            
            if row and row[0] and row[1]:
                data_from = row[0].replace(tzinfo=timezone.utc)
                data_to = row[1].replace(tzinfo=timezone.utc)
            else:
                raise ValueError(
                    f"Cannot read approved dataRange for consent {consent_id}. "
                    "Please revoke and create a new consent."
                )
        else:
            data_from = datetime.fromisoformat(approved_range["from"].replace("Z", "+00:00"))
            data_to   = datetime.fromisoformat(approved_range["to"].replace("Z", "+00:00"))

        now = datetime.now(timezone.utc)
        if data_to > now:
            data_to = now

        logger.info("Using approved dataRange: %s → %s", data_from.date(), data_to.date())

        # Create data session
        session    = client.create_data_session(consent_id,
                                                data_from=data_from,
                                                data_to=data_to)
        session_id = session.get("id")
        logger.info("session_id=%s  status=%s", session_id, session.get("status"))

        try:
            save_session(session_id, user_id, consent_id, session.get("status", "PENDING"))
        except Exception as e:
            logger.warning("Could not save session to DB: %s", e)

        # Poll until PARTIAL or COMPLETED
        logger.info("Waiting for FI data (up to ~320s)...")
        fi_resp = client.wait_for_fi_data(session_id)
        logger.info("Session status = %s", fi_resp.get("status"))

        # Save raw JSON backup to KAL root
        output_path = os.path.join(_KAL_ROOT, f"fi_data_{consent_id[:8]}.json")
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(fi_resp, fh, indent=2, ensure_ascii=False)
        logger.info("📄 Raw JSON saved → %s", output_path)

        # Parse + save to PostgreSQL
        try:
            parsed = parse_session_response(fi_resp)
            save_fi_data(session_id, user_id, consent_id, parsed)
            logger.info("✅ FI data persisted to PostgreSQL for user=%s", user_id)
        except Exception as e:
            logger.error("DB persist failed (JSON backup still saved): %s", e)

        return fi_resp

    try:
        loop    = asyncio.get_event_loop()
        fi_resp = await loop.run_in_executor(None, _fetch)
    except Exception as e:
        msg = str(e)
        if "Consent use exceeded" in msg:
            raise ValueError("This consent has already been used. Revoke and create a new one.")
        if "did not reach terminal state" in msg:
            raise ValueError("Fetch timed out. Click 'Refresh Data' in a few minutes.")
        raise

    user.aa_consent_status = "ACTIVE"
    await db.flush()
    return fi_resp


# ── Revoke Consent ─────────────────────────────────────────────────────────────

async def revoke_consent_for_user(
    db: AsyncSession, user: User, consent_id: str,
) -> Dict[str, Any]:
    import asyncio

    def _call():
        return _get_client()._post(f"/consents/{consent_id}/revoke", {})

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _call)

    user.aa_consent_id     = None
    user.aa_consent_status = "REVOKED"
    user.vua               = None
    await db.flush()

    try:
        save_consent(consent_id, str(user.id), "", "REVOKED")
    except Exception as e:
        logger.warning("Could not update revoke in DB: %s", e)

    _consent_resp_cache.pop(consent_id, None)
    return {"status": "REVOKED", "message": "Consent revoked successfully"}


# ── Dashboard data — user-scoped ───────────────────────────────────────────────

def get_dashboard_data(user_id: str) -> Dict[str, Any]:
    try:
        return {
            "summary":      get_user_summary(user_id),
            "accounts":     get_user_accounts(user_id),
            "transactions": get_user_transactions(user_id, limit=50),
        }
    except Exception as e:
        logger.error("get_dashboard_data failed: %s", e)
        return {"summary": {}, "accounts": [], "transactions": []}