"""
vector_store.py — Pinecone vector storage + retrieval for Penny v2.0
Uses sentence-transformers/all-MiniLM-L6-v2 for real semantic embeddings (384-dim).
Stores precomputed financial summaries per user.
"""
import logging
from typing import List

logger = logging.getLogger("finsight.vector_store")

_embedding_model = None

def _get_embedding_model():
    """Lazy-load the sentence-transformers model (loads once, cached)."""
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("✅ SentenceTransformer model loaded.")
        except Exception as e:
            logger.error("Failed to load SentenceTransformer: %s", e)
            _embedding_model = None
    return _embedding_model


def _embed(text: str) -> List[float]:
    """Embed a text string using all-MiniLM-L6-v2 (384 dims)."""
    model = _get_embedding_model()
    if model is None:
        # Graceful fallback — deterministic hash-based vector
        import hashlib
        h = hashlib.sha256(text.encode()).digest()
        return [(b - 128) / 128.0 for b in (list(h) * 13)[:384]]
    try:
        vec = model.encode(text, normalize_embeddings=True)
        return vec.tolist()
    except Exception as e:
        logger.warning("Embed failed: %s", e)
        return [0.0] * 384


def _pinecone_index():
    """
    Get or create the Pinecone index (384-dim for MiniLM).
    Gracefully handles free-tier 'max serverless indexes' limit:
      - If target index exists → use it.
      - If it doesn't exist and we're at the limit → try to reuse another existing index.
      - If nothing works → return None (Pinecone is non-critical, chat still works).
    """
    TARGET_INDEX = "finsight-penny-v2"

    try:
        from pinecone import Pinecone
        from app.core.config import get_settings
        settings = get_settings()
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)

        existing_indexes = pc.list_indexes()
        existing_names = [i.name for i in existing_indexes]

        # Case 1: target index already exists — just use it
        if TARGET_INDEX in existing_names:
            return pc.Index(TARGET_INDEX)

        # Case 2: target doesn't exist — try to create it
        try:
            from pinecone import ServerlessSpec
            pc.create_index(
                name=TARGET_INDEX,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            logger.info("Created Pinecone index: %s", TARGET_INDEX)
            return pc.Index(TARGET_INDEX)
        except Exception as create_err:
            err_str = str(create_err)
            # 403 = hit max free-tier indexes — try to reuse any existing index
            if "403" in err_str or "FORBIDDEN" in err_str or "max serverless" in err_str.lower():
                logger.warning(
                    "Pinecone max indexes hit — reusing an existing index. "
                    "Delete unused indexes at https://app.pinecone.io to fix this permanently."
                )
                if existing_names:
                    fallback = existing_names[0]
                    logger.warning("Falling back to existing Pinecone index: %s", fallback)
                    return pc.Index(fallback)
                logger.error("No existing Pinecone indexes to fall back to. Skipping vector store.")
                return None
            raise create_err

    except Exception as e:
        logger.error("Pinecone index failed: %s", e)
        return None




def upsert_user_vectors(user_id: str, user_name: str):
    """
    Precompute all financial summary chunks and upsert to Pinecone.
    Called after every bank statement upload or AA data fetch.
    """
    try:
        from app.core.db_config import (
            get_six_month_trend, get_category_breakdown,
            get_top_merchants, get_user_budgets, get_user_accounts,
            get_recurring_expenses, get_largest_transactions,
            get_user_summary
        )
        from datetime import datetime

        index = _pinecone_index()
        if index is None:
            return

        now = datetime.now()
        month, year = now.month, now.year
        vectors = []

        # 1. Monthly financial summaries (6 months)
        trend = get_six_month_trend(user_id)
        for t in trend:
            m = t.get("month", "")
            inc = float(t.get("total_income") or 0)
            exp = float(t.get("total_expenses") or 0)
            net = inc - exp
            rate = (net / inc * 100) if inc > 0 else 0
            text = (
                f"In {m}, {user_name}'s total income was ₹{inc:,.0f} and "
                f"total expenses were ₹{exp:,.0f}. Net savings: ₹{net:,.0f} "
                f"({rate:.1f}% savings rate)."
            )
            vectors.append({
                "id": f"{user_id}-monthly-{m}",
                "values": _embed(text),
                "metadata": {"user_id": user_id, "type": "monthly_summary", "text": text[:500], "month": m}
            })

        # 2. Category summaries (current month)
        breakdown = get_category_breakdown(user_id, month=month, year=year)
        for row in breakdown.get("breakdown", []):
            cat = row.get("category", "Other")
            amt = float(row.get("spent") or 0)
            text = f"{user_name} spent ₹{amt:,.0f} on {cat} in {month}/{year}."
            vectors.append({
                "id": f"{user_id}-cat-{cat.replace(' ', '-')}-{month}-{year}",
                "values": _embed(text),
                "metadata": {"user_id": user_id, "type": "category_summary", "text": text, "category": cat}
            })

        # 3. Top merchant summary
        merchants = get_top_merchants(user_id, month=month, year=year, limit=10)
        if merchants:
            top_text = ", ".join([
                f"{m.get('merchant', '?')} ₹{float(m.get('spent', 0)):,.0f}"
                for m in merchants[:5]
            ])
            text = f"{user_name}'s top merchants this month: {top_text}."
            vectors.append({
                "id": f"{user_id}-merchants-{month}-{year}",
                "values": _embed(text),
                "metadata": {"user_id": user_id, "type": "merchant_summary", "text": text}
            })

        # 4. Budget / goal summary
        budgets = get_user_budgets(user_id)
        if budgets:
            budget_breakdown = get_category_breakdown(user_id, month=month, year=year)
            cat_spent = {r.get("category", ""): float(r.get("spent") or 0) for r in budget_breakdown.get("breakdown", [])}
            budget_lines = []
            for b in budgets:
                cat = b.get("category") or "Global"
                target = float(b.get("target_amount") or 0)
                actual = cat_spent.get(cat, 0)
                pct = (actual / target * 100) if target > 0 else 0
                budget_lines.append(f"{cat} budget ₹{target:,.0f}, used ₹{actual:,.0f} ({pct:.0f}%)")
            text = f"{user_name}'s budget status: " + "; ".join(budget_lines) + "."
            vectors.append({
                "id": f"{user_id}-budgets-{month}-{year}",
                "values": _embed(text),
                "metadata": {"user_id": user_id, "type": "budget_summary", "text": text}
            })

        # 5. Recurring payments summary
        recurring = get_recurring_expenses(user_id)
        if recurring:
            rec_lines = [
                f"{r.get('merchant', '?')} ~₹{float(r.get('avg_amount') or 0):,.0f}/month"
                for r in recurring
            ]
            text = f"{user_name} has recurring payments: " + ", ".join(rec_lines) + "."
            vectors.append({
                "id": f"{user_id}-recurring",
                "values": _embed(text),
                "metadata": {"user_id": user_id, "type": "recurring_summary", "text": text}
            })

        # 6. Unusual transactions
        anomalies = get_largest_transactions(user_id, limit=5)
        if anomalies:
            anom_lines = [
                f"₹{float(a.get('amount') or 0):,.0f} on {str(a.get('narration',''))[:25]}"
                for a in anomalies
            ]
            text = f"Notable large transactions for {user_name} this month: " + ", ".join(anom_lines) + "."
            vectors.append({
                "id": f"{user_id}-anomalies-{month}-{year}",
                "values": _embed(text),
                "metadata": {"user_id": user_id, "type": "unusual_summary", "text": text}
            })

        # 7. Overall financial health summary
        summary = get_user_summary(user_id, month=month, year=year)
        inc = float(summary.get("total_income") or 0)
        exp = float(summary.get("total_expenses") or 0)
        bal = float(summary.get("total_balance") or 0)
        net = inc - exp
        rate = (net / inc * 100) if inc > 0 else 0
        text = (
            f"{user_name}'s financial health: total balance ₹{bal:,.0f}, "
            f"this month income ₹{inc:,.0f}, expenses ₹{exp:,.0f}, "
            f"savings rate {rate:.1f}%."
        )
        vectors.append({
            "id": f"{user_id}-health-{month}-{year}",
            "values": _embed(text),
            "metadata": {"user_id": user_id, "type": "health_summary", "text": text}
        })

        # 8. Per-account category vectors (enables account-specific semantic queries)
        try:
            from app.core.db_config import get_account_wise_category_breakdown
            acc_breakdowns = get_account_wise_category_breakdown(user_id, month=month, year=year)
            for acc_data in acc_breakdowns:
                top_cats = acc_data.get("categories", [])[:5]
                if not top_cats:
                    continue
                cats_str = ", ".join([
                    f"{c['category']} ₹{c['spent']:,.0f}" for c in top_cats
                ])
                text = (
                    f"{user_name}'s {acc_data['account_type']} account "
                    f"({acc_data['masked_acc_number']}) top spending in {month}/{year}: {cats_str}."
                )
                vectors.append({
                    "id": f"{user_id}-acc-{acc_data['fi_data_id']}-cats-{month}-{year}",
                    "values": _embed(text),
                    "metadata": {
                        "user_id": user_id, "type": "account_category_summary",
                        "text": text,
                        "account_type": acc_data['account_type'],
                        "masked_acc": acc_data['masked_acc_number']
                    }
                })
        except Exception as e:
            logger.warning("Per-account vector upsert failed: %s", e)

        # 9. Spending pattern summary vector
        try:
            from app.services.pattern_engine import get_spending_patterns, format_patterns
            patterns = get_spending_patterns(user_id)
            pattern_text = format_patterns(patterns)
            if pattern_text:
                full_text = f"{user_name}'s spending behaviour patterns: {pattern_text}"
                vectors.append({
                    "id": f"{user_id}-patterns-{month}-{year}",
                    "values": _embed(full_text[:500]),
                    "metadata": {"user_id": user_id, "type": "pattern_summary", "text": full_text[:500]}
                })
        except Exception as e:
            logger.warning("Pattern vector upsert failed: %s", e)

        if vectors:
            index.upsert(vectors=vectors, namespace=user_id)
            logger.info("✅ Upserted %d vectors for user=%s", len(vectors), user_id)


    except Exception as e:
        logger.warning("Vector upsert failed (non-critical): %s", e)


def retrieve_relevant_chunks(user_id: str, question: str, top_k: int = 4) -> List[str]:
    """Fetch top-k semantically relevant chunks from Pinecone for the user."""
    try:
        index = _pinecone_index()
        if index is None:
            return []
        
        q_vec = _embed(question)
        results = index.query(
            vector=q_vec,
            top_k=top_k,
            namespace=user_id,
            include_metadata=True
        )
        chunks = []
        for match in results.get("matches", []):
            text = match.get("metadata", {}).get("text", "")
            score = match.get("score", 0)
            if score > 0.3 and text:  # Only include relevant matches
                chunks.append(text)
        return chunks
    except Exception as e:
        logger.warning("Pinecone retrieval failed: %s", e)
        return []
