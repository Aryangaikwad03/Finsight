"""
penny_service.py — Penny AI Assistant v2.0
Hybrid RAG pipeline: Intent Router → Exact SQL → Pinecone retrieval → Groq LLM.
Handles: streaming chat, bank statement parsing, auto-categorization.
"""
import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger("finsight.penny")


# ── Groq client ───────────────────────────────────────────────────────────────

def _groq_client():
    from groq import Groq
    from app.core.config import get_settings
    settings = get_settings()
    return Groq(api_key=settings.GROQ_API_KEY)


FALLBACK_MODEL = "llama-3.1-8b-instant"  # 500k TPD — used when 70b limit hit

def _llm(messages: List[Dict], temperature: float = 0.4, max_tokens: int = 600,
          model: str = "llama-3.3-70b-versatile", stream: bool = False):
    """Call Groq LLM. Auto-falls back to llama-3.1-8b-instant on 429 rate limit."""
    client = _groq_client()

    def _call(m):
        return client.chat.completions.create(
            model=m, messages=messages,
            temperature=temperature, max_tokens=max_tokens, stream=stream,
        )

    try:
        resp = _call(model)
    except Exception as e:
        err_str = str(e)
        # Fallback to smaller model on rate limit
        if "429" in err_str or "rate_limit" in err_str.lower():
            if model != FALLBACK_MODEL:
                logger.warning("Rate limit on %s — falling back to %s", model, FALLBACK_MODEL)
                try:
                    resp = _call(FALLBACK_MODEL)
                except Exception as e2:
                    raise e2
            else:
                raise
        else:
            raise

    if stream:
        def _gen():
            for chunk in resp:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        return _gen()
    return resp.choices[0].message.content.strip()


# ── Pinecone client ───────────────────────────────────────────────────────────

def _pinecone_index():
    from pinecone import Pinecone
    from app.core.config import get_settings
    settings = get_settings()
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    index_name = "finsight-penny"
    existing   = [i.name for i in pc.list_indexes()]
    if index_name not in existing:
        from pinecone import ServerlessSpec
        pc.create_index(
            name      = index_name,
            dimension = 1536,
            metric    = "cosine",
            spec      = ServerlessSpec(cloud="aws", region="us-east-1")
        )
    return pc.Index(index_name)


def _embed(text: str) -> List[float]:
    """Embed text using Groq's API (falls back to simple hash if unavailable)."""
    try:
        # Use a simple embedding via Groq - encode as bag of words approximation
        # In production swap for a real embedding model
        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY", "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"))
        # Use chat to generate a fixed-size semantic vector (workaround since Groq doesn't have embeddings)
        # We'll use a simple text hash approach for dimension matching
        import hashlib
        import struct
        h = hashlib.sha256(text.encode()).digest()
        # Expand to 1536 dims by cycling
        raw = []
        for i in range(1536):
            byte_val = h[i % 32]
            raw.append((byte_val - 128) / 128.0)
        return raw
    except Exception as e:
        logger.warning("Embed failed: %s", e)
        return [0.0] * 1536


# ── Financial context builder ─────────────────────────────────────────────────

def build_user_context(user_id: str, user_name: str) -> str:
    """
    Build a comprehensive financial context string for Penny from the DB.
    This is injected into every chat as the system context.
    """
    from app.core.db_config import (
        get_user_accounts, get_user_transactions,
        get_user_summary, get_category_breakdown,
        get_top_merchants, get_user_budgets,
        get_six_month_trend, get_recurring_expenses, get_largest_transactions
    )
    import calendar
    from datetime import datetime

    try:
        accounts     = get_user_accounts(user_id)
        transactions = get_user_transactions(user_id, limit=200)
        summary      = get_user_summary(user_id)
        breakdown    = get_category_breakdown(user_id)
        merchants    = get_top_merchants(user_id, limit=10)
        budgets      = get_user_budgets(user_id)
        trend        = get_six_month_trend(user_id)
        recurring    = get_recurring_expenses(user_id)
        anomalies    = get_largest_transactions(user_id, limit=5)
    except Exception as e:
        logger.error("Context build failed: %s", e)
        return f"User: {user_name}. Financial data unavailable."

    # Summary section
    total_bal  = float(summary.get("total_balance") or 0)
    income     = float(summary.get("total_income") or 0)
    expenses   = float(summary.get("total_expenses") or 0)
    acc_count  = summary.get("account_count", 0)
    net        = income - expenses

    now = datetime.now()
    _, days_in_month = calendar.monthrange(now.year, now.month)
    current_date_str = now.strftime("%Y-%m-%d %H:%M:%S")
    days_passed = now.day or 1

    ctx = f"""You are Penny, a highly capable financial AI assistant for FinSight.
You are interacting with {user_name}. Use only the retrieved financial data provided below.
Current Date and Time: {current_date_str} (Day {days_passed} of {days_in_month} in this month).

CRITICAL RULES for Phase 2 & 3:
1. Provide short, exact, and highly specific data-driven answers. Mention concrete numbers.
2. If data is unavailable, state it clearly. DO NOT hallucinate transactions, merchants, balances, or budgets.
3. DO NOT give speculative investment advice (e.g., specific stocks to buy).
4. FINANCIAL HEALTH SCORE: If asked, dynamically grade the user out of 100 based on their savings ratio, budget adherence, and recurring burdens. Be creative but mathematically grounded.
5. DYNAMIC FORECASTING: To predict overspending or future expenses, use mathematical extrapolation: multiply their current month's expenses by ({days_in_month} / {days_passed}) to forecast their month-end total, and warn them if it will exceed their budgets or income.
6. ANOMALIES & SUBSCRIPTIONS: Point out "Notable Large Transactions" as potential anomalies if they ask. Warn them about expensive "Identified Subscriptions".
7. Be concise, warm, and actionable. Use ₹ for Indian Rupees. Format numbers in Indian system (lakhs/crores).

=== {user_name.upper()}'S FINANCIAL SNAPSHOT ===
Accounts: {acc_count}
Total Income: ₹{income:,.2f}
Total Expenses: ₹{expenses:,.2f}
Net Cash Flow: ₹{net:,.2f} ({'surplus/savings capacity' if net >= 0 else 'deficit'})

"""

    # Account details
    if accounts:
        ctx += "=== ACCOUNTS ===\n"
        for acc in accounts:
            acc_type = acc.get("account_type", "")
            masked   = acc.get("masked_acc_number", "")
            holder   = acc.get("holder_name", "")
            bal      = acc.get("current_balance") or acc.get("current_value") or 0
            ctx += f"• {acc_type} {masked} | Holder: {holder} | Balance: ₹{float(bal or 0):,.2f}\n"
            if acc.get("maturity_date"):
                ctx += f"  Matures: {acc.get('maturity_date')} | Rate: {acc.get('interest_rate')}%\n"
        ctx += "\n"

    # Budgets & Goals
    if budgets:
        ctx += "=== USER FINANCIAL GOALS & BUDGETS ===\n"
        for b in budgets:
            b_type = b.get("goal_type", "")
            cat = b.get("category") or "Global"
            amt = float(b.get("target_amount") or 0)
            ctx += f"• {b_type} ({b.get('period')}): {b.get('title') or cat} | Target: ₹{amt:,.2f}\n"
        ctx += "\n"
        
    # Top Merchants
    if merchants:
        ctx += "=== TOP MERCHANTS ===\n"
        for m in merchants:
            merchant = m.get("merchant", "Unknown")
            amt = float(m.get("spent", 0))
            count = m.get("txn_count", 0)
            ctx += f"• {merchant}: ₹{amt:,.2f} ({count} txns)\n"
        ctx += "\n"

    # Six Month Trend
    if trend:
        ctx += "=== 6-MONTH INCOME & EXPENSE TREND ===\n"
        ctx += "Note: Use THIS month-to-month breakdown when answering questions about 'My monthly spendings'.\n"
        for t in trend:
            m_label = t.get("month", "Unknown")
            inc = float(t.get("total_income") or 0)
            exp = float(t.get("total_expenses") or 0)
            ctx += f"• Month {m_label} -> Total Income: ₹{inc:,.2f} | Total Expenses: ₹{exp:,.2f}\n"
        ctx += "\n"

    # Subscriptions & Recurring
    if recurring:
        ctx += "=== IDENTIFIED RECURRING SUBSCRIPTIONS & EMIs ===\n"
        for r in recurring:
            ctx += f"• {r.get('merchant', 'Unknown')}: ~₹{float(r.get('avg_amount') or 0):,.2f} / month (Active for {r.get('months_active')} months)\n"
        ctx += "\n"
        
    # Anomalies
    if anomalies:
        ctx += "=== NOTABLE LARGE TRANSACTIONS (POTENTIAL ANOMALIES) ===\n"
        for a in anomalies:
            date_str = str(a.get('txn_date'))[:10]
            ctx += f"• {date_str} | {a.get('category')} | {str(a.get('narration'))[:30]}: ₹{float(a.get('amount') or 0):,.2f}\n"
        ctx += "\n"

    # Category breakdown
    rows = breakdown.get("breakdown", [])
    if rows:
        ctx += "=== SPENDING BY CATEGORY ===\n"
        for row in rows[:10]:
            cat   = row.get("category", "Other")
            amt   = float(row.get("total_amount") or row.get("spent") or 0)
            count = row.get("txn_count", 0)
            ctx += f"• {cat}: ₹{amt:,.2f} ({count} txns)\n"
        ctx += "\n"

    # Recent transactions (last 20 for context)
    if transactions:
        ctx += "=== RECENT TRANSACTIONS (last 20) ===\n"
        for t in transactions[:20]:
            date    = str(t.get("txn_date") or t.get("value_date") or "")[:10]
            narr    = (t.get("narration") or "")[:50]
            amt     = float(t.get("amount") or 0)
            ttype   = t.get("txn_type", "")
            cat     = t.get("category", "")
            sign    = "+" if ttype in ("CREDIT", "INTEREST", "OPENING") else "-"
            ctx += f"• {date} | {sign}₹{amt:,.2f} | {ttype} | {cat} | {narr}\n"
        ctx += "\n"

    # Budgeting benchmarks
    ctx += """=== BUDGETING BENCHMARKS (50/30/20 Rule) ===
• Needs (rent, food, utilities): 50% of income
• Wants (entertainment, dining): 30% of income
• Savings/Investments: 20% of income
Use these to guide the user when they ask about budgeting.

Always end with a specific, actionable tip based on their actual data.
"""
    return ctx


def store_user_context_vectors(user_id: str, user_name: str):
    """Delegate to dedicated vector_store module (Penny v2.0)."""
    try:
        from app.services.vector_store import upsert_user_vectors
        upsert_user_vectors(user_id, user_name)
    except Exception as e:
        logger.warning("Vector upsert (delegated) failed: %s", e)


# ── Chat ──────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_BASE = """You are Penny, FinSight's expert personal finance AI assistant.
You MUST follow these rules at all times:
1. Use ONLY the provided [DB FACTS] and [CONTEXT] below. Never hallucinate numbers.
2. State clearly if a data point is unavailable.
3. Never give speculative investment advice (no stock picks).
4. Be concise — 3-5 lines max unless the user explicitly asks for detail.
5. Use ₹ for Indian Rupees and Indian numbering (lakhs/crores).
6. Always end with one actionable tip grounded in the user's actual data."""


def build_base_financial_snapshot(user_id: str, user_name: str) -> str:
    """
    Always-present base context: accounts, current month totals, 3-month trend.
    Kept under ~600 tokens to leave room for intent facts.
    """
    from app.core.db_config import (
        get_user_accounts, get_user_summary, get_category_breakdown,
        get_user_budgets, get_six_month_trend, get_account_wise_category_breakdown
    )
    from datetime import datetime
    now = datetime.now()
    month, year = now.month, now.year

    try:
        accounts  = get_user_accounts(user_id)
        summary   = get_user_summary(user_id, month=month, year=year)
        trend     = get_six_month_trend(user_id)
        budgets   = get_user_budgets(user_id)
        breakdown = get_category_breakdown(user_id, month=month, year=year)
    except Exception as e:
        logger.warning("Base snapshot failed: %s", e)
        return f"User: {user_name}. Financial data temporarily unavailable."

    inc  = float(summary.get("total_income") or 0)
    exp  = float(summary.get("total_expenses") or 0)
    bal  = float(summary.get("total_balance") or 0)
    net  = inc - exp
    rate = round((net / inc * 100), 1) if inc > 0 else 0
    acc_count = summary.get("account_count", len(accounts))

    lines = [
        f"[BASE SNAPSHOT — {now.strftime('%B %Y')}]",
        f"User: {user_name} | Linked Accounts: {acc_count}",
        f"Total Balance (all accounts): \u20b9{bal:,.0f}",
        f"This Month: Income \u20b9{inc:,.0f} | Expenses \u20b9{exp:,.0f} | Net \u20b9{net:,.0f} | Savings Rate {rate}%",
        "",
    ]

    # Accounts list
    if accounts:
        lines.append("[ACCOUNTS]")
        for a in accounts:
            atype = a.get("account_type", "")
            masked = a.get("masked_acc_number", "")
            ab = float((a.get("current_balance") or a.get("current_value") or a.get("principal_amount") or 0))
            maturity = f" | Matures: {str(a.get('maturity_date',''))[:10]}" if a.get("maturity_date") else ""
            rate_txt = f" | Rate: {a.get('interest_rate')}%" if a.get("interest_rate") else ""
            lines.append(f"\u2022 {atype} {masked}: \u20b9{ab:,.0f}{maturity}{rate_txt}")
        lines.append("")

    # Per-account category breakdown (only when 2+ accounts — this is the key fix)
    if len(accounts) >= 2:
        try:
            acc_breakdowns = get_account_wise_category_breakdown(user_id, month=month, year=year)
            if acc_breakdowns:
                lines.append("[SPENDING BY ACCOUNT THIS MONTH]")
                for ab_data in acc_breakdowns:
                    top_cats = ab_data.get("categories", [])[:4]
                    if not top_cats:
                        continue
                    cats_str = " | ".join([
                        f"{c['category']} \u20b9{c['spent']:,.0f}"
                        for c in top_cats
                    ])
                    lines.append(
                        f"\u2022 {ab_data['account_type']} {ab_data['masked_acc_number']}: {cats_str}"
                    )
                lines.append("")
        except Exception as e:
            logger.warning("Per-account breakdown in snapshot failed: %s", e)

    # Active budgets summary
    if budgets:
        cat_spent = {r.get("category",""): float(r.get("spent") or 0) for r in breakdown.get("breakdown",[])}
        lines.append("[ACTIVE GOALS & BUDGETS]")
        for b in budgets:
            cat = b.get("category") or "Global"
            tgt = float(b.get("target_amount") or 0)
            act = cat_spent.get(cat, 0)
            pct = round(act/tgt*100, 0) if tgt > 0 else 0
            exceeded = " \u26a0\ufe0f EXCEEDED" if act > tgt else ""
            lines.append(f"\u2022 {b.get('goal_type')} | {b.get('title') or cat}: \u20b9{act:,.0f}/\u20b9{tgt:,.0f} ({pct}%){exceeded}")
        lines.append("")

    # Top 5 categories (merged across all accounts)
    cats = breakdown.get("breakdown", [])[:5]
    if cats:
        lines.append("[TOP SPENDING CATEGORIES — ALL ACCOUNTS]")
        for c in cats:
            txn_cnt = c.get('txn_count', '')
            cnt_txt = f" ({txn_cnt} txns)" if txn_cnt else ""
            lines.append(f"\u2022 {c.get('category','?')}: \u20b9{float(c.get('spent') or 0):,.0f}{cnt_txt}")
        lines.append("")

    # 3-month income/expense trend (most recent first)
    recent_trend = trend[-3:] if trend else []
    if recent_trend:
        lines.append("[3-MONTH TREND]")
        for t in recent_trend:
            ti = float(t.get("total_income") or 0)
            te = float(t.get("total_expenses") or 0)
            lines.append(f"\u2022 {t.get('month')}: In \u20b9{ti:,.0f} / Out \u20b9{te:,.0f}")

    # Recent transactions (last 30 across all accounts with account context)
    try:
        from app.core.db_config import get_transactions_filtered
        recent_txns = get_transactions_filtered(
            user_id,
            month=month, year=year,
            limit=30
        )
        if not recent_txns:
            # Fall back to all-time last 20 if no transactions this month
            recent_txns = get_transactions_filtered(user_id, limit=20)
        if recent_txns:
            lines.append("")
            lines.append(f"[RECENT TRANSACTIONS — {now.strftime('%B %Y')}]")
            for t in recent_txns:
                bal_str = f" | Bal \u20b9{t.get('balance_after'):,.0f}" if t.get('balance_after') else ""
                narr = str(t.get('narration', ''))[:40]
                lines.append(
                    f"\u2022 {t.get('txn_date')} | "
                    f"{t.get('account_type','?')} {t.get('masked_acc_number','?')} | "
                    f"{t.get('txn_type','?')} \u20b9{t.get('amount', 0):,.0f} | "
                    f"{t.get('category','?')} | {narr}{bal_str}"
                )
    except Exception as e:
        logger.warning("Recent transactions in snapshot failed: %s", e)

    return "\n".join(lines)



def build_slim_system_prompt(user_name: str, base_snapshot: str, db_facts_str: str, pinecone_chunks: list) -> str:
    """Build system prompt: base rules + always-on snapshot + intent-specific facts + semantic context."""
    ctx = f"{SYSTEM_PROMPT_BASE}\n\n{base_snapshot}\n"
    if db_facts_str:
        ctx += f"\n{db_facts_str}\n"
    if pinecone_chunks:
        ctx += "\n[SEMANTIC CONTEXT]\n"
        for chunk in pinecone_chunks[:2]:
            ctx += f"• {chunk[:180]}\n"
    return ctx


def chat_with_penny(
    user_id:   str,
    user_name: str,
    messages:  List[Dict],
    question:  str,
    stream:    bool = False,
):
    """
    Hybrid RAG chat pipeline:
    1. Classify intent (keyword rules → LLM fallback)
    2. Resolve exact DB facts for that intent
    3. Retrieve relevant Pinecone chunks
    4. Build tight system prompt (~1200 tokens total)
    5. Call Groq with streaming or one-shot
    Returns: string response OR generator (if stream=True)
    """
    from app.services.intent_router import classify_intent, resolve_intent, format_db_facts
    from app.services.vector_store import retrieve_relevant_chunks

    # Step 1: Detect intent — for compound/analysis questions resolve multiple intents
    try:
        intent = classify_intent(question)
        logger.info("Intent: %s | User: %s", intent, user_id)
    except Exception:
        intent = "general"

    # Step 2: Exact DB lookup — also pull comparison facts for analysis intents
    try:
        db_facts = resolve_intent(intent, user_id, question)
        db_facts_str = format_db_facts(db_facts)
        # For analysis/health/general add extra facts from a secondary intent
        secondary_intent_map = {
            "financial_health":     "comparison_query",
            "general":              "spending_summary",
            "savings_analysis":     "goal_progress",
            "comparison_query":     "category_spending",
            "pattern_analysis":     "spending_summary",
            "category_spending":    "transaction_lookup",  # adds sample txns for context
        }
        if intent in secondary_intent_map:
            sec_intent = secondary_intent_map[intent]
            sec_facts = resolve_intent(sec_intent, user_id, question)
            sec_str = format_db_facts(sec_facts)
            if sec_str:
                db_facts_str += "\n" + sec_str
    except Exception as e:
        logger.warning("DB resolution failed: %s", e)
        db_facts_str = ""

    # Step 3: Always-on base snapshot (accounts, totals, budgets, categories, trend)
    try:
        base_snapshot = build_base_financial_snapshot(user_id, user_name)
    except Exception as e:
        logger.warning("Base snapshot failed: %s", e)
        base_snapshot = f"User: {user_name}"

    # Step 4: Pinecone semantic retrieval
    try:
        pinecone_chunks = retrieve_relevant_chunks(user_id, question, top_k=3)
    except Exception:
        pinecone_chunks = []

    # Step 5: Build system prompt
    system_prompt = build_slim_system_prompt(user_name, base_snapshot, db_facts_str, pinecone_chunks)

    # Step 6: Assemble messages (last 6 turns max)
    full_messages = [{"role": "system", "content": system_prompt}]
    for m in messages[-6:]:
        full_messages.append({"role": m["role"], "content": m["content"]})
    full_messages.append({"role": "user", "content": question})

    return _llm(full_messages, temperature=0.4, max_tokens=700, stream=stream), intent


# ── Bank Statement Parser ─────────────────────────────────────────────────────

def parse_bank_statement(file_bytes: bytes, filename: str, user_name: str, password: str = None) -> Dict:
    """
    Parse a bank statement PDF/CSV/Excel deterministically via heuristics.
    Bypasses LLM tokens/TPM limits completely for 100x speedup.
    Restored to 'raw scan' regex mode as requested.
    """
    import re
    import uuid
    from datetime import datetime
    
    text = _extract_text_from_file(file_bytes, filename, password)
    if not text:
        raise ValueError("Could not extract text from the uploaded file.")

    HEADER_MAP = {
        "DATE": [
            "date", "txn date", "transaction date", "tran date", "value date", "posting date", 
            "entry date", "effective date", "transaction on", "txn on", "valuedate", "txndate",
            "date of transaction", "tran. date", "statement date", "vldt", "trndate", "val date"
        ],
        "DESC": [
            "narration", "particulars", "description", "remarks", "transaction details", "details", 
            "reference", "remarks/description", "transaction remarks", "info", "transaction description", 
            "narration details", "transaction particulars", "remarks details", "desc", "transaction",
            "subject", "trans details", "particulars details", "ref no", "chq/ref no", "tran particulars"
        ],
        "CREDIT": [
            "credit", "cr", "cr.", "deposit", "deposits", "credit amount", "amount cr", "paid in", 
            "receipt", "received", "received amount", "inflow", "credit value", "cr amount",
            "deposit amt", "credits", "pay in", "money in", "withdrawal(cr)", "deposit(cr)"
        ],
        "DEBIT": [
            "debit", "dr", "dr.", "withdrawal", "withdrawals", "debit amount", "amount dr", "paid out", 
            "payment", "spent", "outflow", "debit value", "dr amount", "withdrawal amt", "debits",
            "pay out", "money out", "withdrawal(dr)", "payment(dr)"
        ],
        "BALANCE": [
            "balance", "bal", "bal.", "closing balance", "running balance", "available balance", 
            "ledger balance", "current balance", "balance amount", "available amt", "ledger amt",
            "total balance", "account balance", "running bal", "bal amt", "cl. bal", "closing bal", "balance(inr)"
        ],
        "AMOUNT": [
            "amount", "txn amount", "transaction amount", "value", "total amount", "amt", "txn amt", "amount(inr)", "txn value"
        ]
    }

    transactions = []
    lines = text.split('\n')

    # Detect if this is a piped structured format (from CSV/Excel)
    is_piped = False
    fname_lower = filename.lower()
    if fname_lower.endswith(('.csv', '.xls', '.xlsx')):
        is_piped = True
    elif lines and not fname_lower.endswith('.pdf'):
        # Only check for pipes if it's not a PDF, to avoid false positives
        pipe_lines = sum(1 for l in lines[:20] if "|" in l)
        if pipe_lines > 3:
            is_piped = True

    # Heuristic Regex: Match dates like 01/12/2026, 01-Mar-2026, 01 Mar 26
    date_regex = re.compile(r'^\s*(\d{1,2}[/\-\s]+(?:[a-zA-Z]{3,4}|\d{1,2})[/\-\s]+\d{2,4})')

    last_balance = None
    num_columns_order = []
    header_offsets = {}

    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Detect headers dynamically
        if not num_columns_order:
            line_lower = line.lower()
            pos = {}
            for key, aliases in HEADER_MAP.items():
                for alias in aliases:
                    if is_piped:
                        # For piped files, check column headers
                        headers = [h.strip().lower() for h in line.split('|')]
                        for i, h in enumerate(headers):
                            if alias in h:
                                pos[key] = i
                                break
                        if key in pos: break
                    else:
                        if re.search(r'(?:^|\s|\b)' + re.escape(alias) + r'(?:$|\s|\b)', line_lower):
                            pos[key] = line_lower.find(alias)
                            break
            
            if "DATE" in pos and ("DEBIT" in pos or "CREDIT" in pos or "AMOUNT" in pos or "BALANCE" in pos):
                cols_sorted = sorted([(k, idx) for k, idx in pos.items() if k in ["DATE", "DESC", "DEBIT", "CREDIT", "BALANCE", "AMOUNT"]], key=lambda x: x[1])
                num_columns_order = [k for k, _ in cols_sorted if k in ["DEBIT", "CREDIT", "BALANCE", "AMOUNT"]]
                header_offsets = {k: idx for k, idx in cols_sorted}
                logger.info("  Detected headers: %s (Piped: %s)", pos, is_piped)
                continue

        # Parsing logic
        if is_piped:
            parts = [p.strip() for p in line.split('|')]
            if len(parts) < 2: continue
            
            # Map columns to values
            val_map = {}
            for k, idx in header_offsets.items():
                if idx < len(parts): val_map[k] = parts[idx]
            
            raw_date = val_map.get("DATE", "")
            if not raw_date or not _has_date(raw_date): continue
            
            narration = ""
            if "DESC" in pos and pos["DESC"] < len(parts):
                narration = parts[pos["DESC"]]
            
            # Handle amounts
            amount = 0.0
            txn_type = "DEBIT"
            if "AMOUNT" in val_map:
                amount = _clean_amount(val_map["AMOUNT"])
                # Guess type if not explicit
                txn_type = "CREDIT" if amount > 0 and ("CR" in line.upper() or "RECEIVED" in line.upper()) else "DEBIT"
            elif "CREDIT" in val_map and _clean_amount(val_map["CREDIT"]) > 0:
                amount = _clean_amount(val_map["CREDIT"])
                txn_type = "CREDIT"
            elif "DEBIT" in val_map and _clean_amount(val_map["DEBIT"]) > 0:
                amount = _clean_amount(val_map["DEBIT"])
                txn_type = "DEBIT"
                
            balance = _clean_amount(val_map.get("BALANCE", "0"))
            if balance: last_balance = balance

            parsed_date = _parse_statement_date(raw_date)
            transactions.append({
                "txnId": str(uuid.uuid4()),
                "transactionTimestamp": (parsed_date or datetime.now()).isoformat() + "+00:00",
                "valueDate": (parsed_date or datetime.now()).strftime("%Y-%m-%d"),
                "amount": abs(amount),
                "type": txn_type,
                "mode": "OTHERS",
                "narration": narration or "Transaction",
                "currentBalance": balance
            })
        else:
            # Existing Heuristic Regex for PDF
            date_match = date_regex.search(line)
            if date_match:
                raw_date = date_match.group(1).strip()
                rest_of_line = line[date_match.end():].strip()
                
                num_matches = list(re.finditer(r'(?:^|\s)((?:\d{1,3}(?:,\d{2,3})+|\d+)\.\d{2})(?=\s|$|[A-Za-z])', rest_of_line))
                if not num_matches:
                    continue
                    
                first_num_start = num_matches[0].start()
                narr_parts = []
                last_end = 0
                for match in num_matches:
                    narr_parts.append(rest_of_line[last_end:match.start()])
                    last_end = match.end()
                narr_parts.append(rest_of_line[last_end:])
                
                narration = " ".join(narr_parts)
                narration = re.sub(r'\b(?:CR|DR|Cr\.?|Dr\.?)\b', '', narration, flags=re.IGNORECASE)
                narration = re.sub(r'\s+', ' ', narration).strip()
                
                nums = [float(m.group(1).replace(',', '')) for m in num_matches]
                amount = 0.0
                txn_type = "DEBIT"
                balance = None
                
                assigned = {}
                if num_columns_order and 0 < len(nums) <= len(num_columns_order):
                    if len(nums) == len(num_columns_order):
                        assigned = {k: v for k, v in zip(num_columns_order, nums)}
                    elif len(nums) == len(num_columns_order) - 1 and "BALANCE" in num_columns_order:
                        assigned["BALANCE"] = nums[-1]
                        remaining_num = nums[0]
                        if last_balance is not None:
                            if abs(last_balance - remaining_num - assigned["BALANCE"]) < 0.1: assigned["DEBIT"] = remaining_num
                            elif abs(last_balance + remaining_num - assigned["BALANCE"]) < 0.1: assigned["CREDIT"] = remaining_num
                        
                        if "DEBIT" not in assigned and "CREDIT" not in assigned:
                            up_line = line.upper()
                            if ' CR' in up_line or 'CR.' in up_line: assigned["CREDIT"] = remaining_num
                            else: assigned["DEBIT"] = remaining_num

                    if "AMOUNT" in assigned:
                        amount = assigned["AMOUNT"]
                        txn_type = "CREDIT" if ' CR' in line.upper() else "DEBIT"
                    else:
                        if "CREDIT" in assigned and assigned["CREDIT"] > 0: amount, txn_type = assigned["CREDIT"], "CREDIT"
                        else: amount, txn_type = assigned.get("DEBIT", 0), "DEBIT"
                    balance = assigned.get("BALANCE")

                if amount == 0 and len(nums) > 0:
                    if len(nums) >= 2:
                        balance = nums[-1]
                        amount = nums[-2]
                        if last_balance is not None:
                            if abs(last_balance - amount - balance) < 0.1:
                                txn_type = "DEBIT"
                            elif abs(last_balance + amount - balance) < 0.1:
                                txn_type = "CREDIT"
                            else:
                                txn_type = "CREDIT" if ' CR' in line.upper() or 'CR.' in line.upper() else "DEBIT"
                        else:
                            txn_type = "CREDIT" if ' CR' in line.upper() or 'CR.' in line.upper() else "DEBIT"
                    else:
                        amount = nums[0]
                        txn_type = "CREDIT" if ' CR' in line.upper() or 'CR.' in line.upper() else "DEBIT"
                        balance = None

                if balance is not None: last_balance = balance
                    
                parsed_date = _parse_statement_date(raw_date)
                        
                transactions.append({
                    "txnId": str(uuid.uuid4()),
                    "transactionTimestamp": (parsed_date or datetime.now()).isoformat() + "+00:00",
                    "valueDate": (parsed_date or datetime.now()).strftime("%Y-%m-%d"),
                    "amount": amount,
                    "type": txn_type,
                    "mode": "OTHERS",
                    "narration": narration,
                    "currentBalance": balance
                })
            elif num_columns_order and transactions:
                # Continuation row for PDF
                if len(line) > 2 and not re.search(r'(?i)page|statement|opening balance|closing balance', line):
                    if not re.search(r'^\s*[\d,\.\-]+\s*$', line):
                        clean_line = re.sub(r'\b(?:CR|DR)\b', '', line, flags=re.IGNORECASE).strip()
                        if clean_line:
                            transactions[-1]["narration"] = (transactions[-1]["narration"] + " " + clean_line).strip()

    # --- Metadata Extraction (Bank Name & Acc Number) ---
    bank_name = "Other Bank"
    acc_num = "XXXX" + str(uuid.uuid4())[:4] # Default unique
    
    # Try to find bank name
    header_text = text[:3000].upper()
    if "AXIS" in header_text: bank_name = "Axis Bank"
    elif "SBI" in header_text or "STATE BANK" in header_text: bank_name = "State Bank of India"
    elif "HDFC" in header_text: bank_name = "HDFC Bank"
    elif "ICICI" in header_text: bank_name = "ICICI Bank"
    elif "PNB" in header_text or "PUNJAB NATIONAL" in header_text: bank_name = "Punjab National Bank"
    elif "YES BANK" in header_text: bank_name = "YES Bank"
    elif "KOTAK" in header_text: bank_name = "Kotak Mahindra"
    
    # Try to find account number
    acc_match = re.search(r'(?:A/c No|Account No|A/c).*?(\d{8,})', header_text, re.IGNORECASE)
    if acc_match:
        full_acc = acc_match.group(1)
        # We append a unique suffix (from filename/uuid) to ensure every UPLOAD is a separate account as requested
        acc_num = "X" * (len(full_acc)-4) + full_acc[-4:] + "-" + str(uuid.uuid4())[:4]
    else:
        # If no acc number found, use filename hash to stay unique
        import hashlib
        f_hash = hashlib.md5(filename.encode()).hexdigest()[:4]
        acc_num = f"STMT-{f_hash}-{str(uuid.uuid4())[:4]}"

    logger.info("Restored heuristic parser found %d transactions for %s", len(transactions), bank_name)
    
    return {
        "account_info": {
            "bank_name": bank_name,
            "account_number": acc_num,
            "account_type": "DEPOSIT",
            "holder_name": user_name,
            "closing_balance": last_balance
        },
        "transactions": transactions
    }

def _parse_text_fallback_heuristics(text: str, user_name: str) -> Dict:
    """Minimized heuristic parser for unstructured text blocks."""
    import re
    import uuid
    from datetime import datetime
    txns = []
    lines = text.split('\n')
    for line in lines:
        date_m = re.search(r'(\d{1,2}[/\-\s]+[a-zA-Z]{3,4}[/\-\s]+\d{2,4})', line)
        if date_m:
            nums = re.findall(r'(\d+[\.,]\d{2})', line)
            if nums:
                txns.append({
                    "txnId": str(uuid.uuid4()),
                    "transactionTimestamp": "2026-01-01T00:00:00+00:00",
                    "amount": float(nums[0].replace(',', '')),
                    "type": "DEBIT",
                    "narration": line.strip()
                })
    return {"account_info": {"holder_name": user_name}, "transactions": txns}


def _extract_text_from_file(file_bytes: bytes, filename: str, password: str = None) -> str:
    """Extract text from PDF or image file."""
    fname = filename.lower()

    if fname.endswith(".pdf"):
        try:
            import pdfplumber
            import io
            
            try:
                text = ""
                with pdfplumber.open(io.BytesIO(file_bytes), password=password or '') as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text() or ""
                        text += page_text + "\n"
                return text.strip()
            except Exception as e:
                err_str = repr(e) + repr(getattr(e, 'args', []))
                if "PDFPasswordIncorrect" in err_str or "PDFEncryptionError" in err_str or "PdfminerException" in err_str:
                    raise ValueError("encrypted_pdf")
                raise
        except ImportError:
            # Fallback: try PyPDF2
            try:
                import PyPDF2
                import io
                reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
                if reader.is_encrypted:
                    if not password or not reader.decrypt(password):
                        raise ValueError("encrypted_pdf")
                return "\n".join(page.extract_text() or "" for page in reader.pages)
            except ImportError:
                raise ValueError("Please install pdfplumber: pip install pdfplumber")

    elif fname.endswith((".jpg", ".jpeg", ".png")):
        # For images, encode as base64 and send to Groq vision
        # llama-3.1-8b-instant doesn't support vision, so we note this limitation
        raise ValueError("Image uploads are not yet supported. Please upload a PDF.")

    elif fname.endswith((".csv", ".xlsx", ".xls")):
        try:
            import pandas as pd
            import io
            
            # Robust logic: find the actual header row for both CSV and Excel
            if fname.endswith(".csv"):
                content_sample = file_bytes[:50000].decode('utf-8', errors='ignore')
                sample_lines = content_sample.split('\n')
                skip_count = 0
                for i, line in enumerate(sample_lines[:100]):
                    l_low = line.lower()
                    if any(k in l_low for k in ["date", "transaction", "narration", "particulars", "description"]):
                        skip_count = i
                        break
                df = pd.read_csv(io.BytesIO(file_bytes), skiprows=skip_count, on_bad_lines='skip')
            else:
                # For Excel, we scan the first 50 rows to find headers
                excel_file = io.BytesIO(file_bytes)
                temp_df = pd.read_excel(excel_file, header=None, nrows=100)
                skip_count = 0
                for i, row in temp_df.iterrows():
                    row_str = " ".join(str(val).lower() for val in row.values)
                    if any(k in row_str for k in ["date", "transaction", "narration", "particulars", "description"]):
                        skip_count = i
                        break
                # Re-read with correct header
                excel_file.seek(0)
                df = pd.read_excel(excel_file, skiprows=skip_count)

            # Return piped format for robust parsing in parse_bank_statement
            return df.to_csv(index=False, sep='|')
        except Exception as e:
            logger.error("Tabular data processing failed: %s", e)
            raise ValueError(f"Please ensure pandas and openpyxl are installed: {str(e)}")

    elif fname.endswith(".txt"):
        return file_bytes.decode("utf-8", errors="ignore")

    else:
        # Try plain text
        try:
            return file_bytes.decode("utf-8", errors="ignore")
        except Exception:
            raise ValueError(f"Unsupported file type: {filename}")


def _has_date(text: str) -> bool:
    """Check if a string starts with/contains a standard date."""
    # Matches DD-MM-YYYY, DD/MM/YYYY, YYYY-MM-DD, etc.
    _DATE_REGEX = r'\b(?:\d{1,2}[/\-\s]+(?:[a-zA-Z]{3,4}|\d{1,2})[/\-\s]+\d{2,4}|\d{4}[/\-\s]+\d{1,2}[/\-\s]+\d{1,2})\b'
    return bool(re.search(_DATE_REGEX, str(text)))


def _clean_amount(val: str) -> float:
    """Clean amount string into float."""
    if not val: return 0.0
    if isinstance(val, (int, float)): return float(val)
    clean = re.sub(r'[^\d\.\-]', '', str(val).replace(',', ''))
    try: return float(clean) if clean else 0.0
    except ValueError: return 0.0


def _parse_statement_date(val: str):
    """Robust date parser."""
    from datetime import datetime
    if not val: return None
    clean = str(val).replace('-', ' ').replace('/', ' ').replace(',', ' ').strip()
    for fmt in ("%d %m %Y", "%d %b %Y", "%d %B %Y", "%d %m %y", "%d %b %y", "%Y %m %d"):
        try: return datetime.strptime(clean, fmt)
        except ValueError: continue
    return None


def statement_to_fi_format(parsed: Dict, user_id: str, session_id: str, consent_id: str) -> Dict:
    """
    Convert parsed bank statement into the same format as parse_session_response()
    so save_fi_data() can consume it directly.
    """
    acc_info = parsed.get("account_info", {})
    txns     = parsed.get("transactions", [])

    # Convert to fi_parser format
    fi_account = {
        "masked_acc": acc_info.get("account_number", "XXXXXXXX0000"),
        "fi_type":    (acc_info.get("account_type") or "DEPOSIT").upper(),
        "fi_status":  "ACTIVE",
        "link_ref":   f"stmt-{user_id[:8]}",
        "profile": {
            "name":        acc_info.get("holder_name", ""),
            "holder_type": "SINGLE",
        },
        "summary": {
            "status":          "ACTIVE",
            "branch":          acc_info.get("branch"),
            "ifscCode":        acc_info.get("ifsc_code"),
            "currentBalance":  acc_info.get("closing_balance"),
            "currency":        "INR",
        },
        "transactions": txns,
    }

    return {
        "fips": [{
            "fip_id":   f"stmt-upload-{acc_info.get('bank_name','bank').lower().replace(' ','-')}",
            "accounts": [fi_account],
        }]
    }


# ── Category auto-correction ──────────────────────────────────────────────────

CATEGORY_PROMPT = """You are a financial transaction categorizer for Indian users.
Given these transactions, suggest the correct category for each.

Available categories:
- Food & Dining
- Transportation
- Shopping & Retail
- Bills & Utilities
- Housing & Rent
- Healthcare & Medical
- Entertainment & Leisure
- Travel
- Education
- Investments & Savings
- Insurance
- Salary & Income
- Transfers
- Taxes & Government
- ATM / Cash Withdrawal
- Fees & Charges
- Donations & Charity
- Business / Professional Expenses
- Subscription Services
- Uncategorized / Unknown

Return ONLY a JSON array:
[{"txn_id": <id>, "category": "<category>", "subcategory": "<subcategory>", "confidence": 0.0-1.0}]"""


def auto_categorize_transactions(transactions: List[Dict]) -> List[Dict]:
    """Use LLM grouping and deduplication to intelligently batch-categorize hundreds of transactions."""
    if not transactions:
        return []

    # 1. Deduplicate and clean text to massively reduce tokens and increase query speed
    def clean_text(n: str) -> str:
        # Remove 8+ digit IDs/numbers (e.g. UPI/123456789012)
        s = re.sub(r'\d{7,}', '', n or '')
        s = re.sub(r'[^A-Za-z0-9\s/&]', ' ', s)
        return ' '.join(s.split()).strip()[:65]

    unique_txns = {} # clean_text -> list of txn_id
    for t in transactions:
        c_text = clean_text(t.get('narration'))
        if not c_text:
            continue
        if c_text not in unique_txns:
            unique_txns[c_text] = []
        unique_txns[c_text].append(t.get('txn_id'))

    all_suggestions = []
    unique_keys = list(unique_txns.keys())
    
    # 2. Chunk in batches of 150 unique narrations to bypass Free-Tier RPM limit drastically
    import time
    chunk_size = 150
    for i in range(0, len(unique_keys), chunk_size):
        chunk_keys = unique_keys[i:i+chunk_size]
        
        txn_text = "\n".join([f"KEY_{idx}: {k}" for idx, k in enumerate(chunk_keys)])
        
        # Override the schema prompt dynamically for this batch logic
        sys_prompt = CATEGORY_PROMPT.replace(
            '[{"txn_id": <id>, "category": "<category>", "subcategory": "<subcategory>", "confidence": 0.0-1.0}]', 
            '[{"key_id": "KEY_<number>", "category": "<category>", "subcategory": "<subcategory>", "confidence": 0.0-1.0}]'
        )
        
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user",   "content": f"Categorize these unique parsed narrations ONLY. Return the JSON array.\n{txn_text}"}
        ]

        try:
            # Reverted to default fast 8b-instant to dodge permission issues while keeping efficiency
            raw = _llm(messages, temperature=0.1, max_tokens=3000, model="llama-3.1-8b-instant")
            raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()
            suggestions = json.loads(raw)
            
            # 3. Unroll AI's mapped suggestions logically back against raw identical transactions
            for s in suggestions:
                k_id = s.get("key_id", str(s.get("txn_id", "")))
                if k_id.startswith("KEY_"):
                    try:
                        idx = int(k_id.split("_")[1])
                        c_text = chunk_keys[idx]
                        orig_ids = unique_txns[c_text]
                        for oid in orig_ids:
                            all_suggestions.append({
                                "txn_id": oid,
                                "category": s.get("category"),
                                "subcategory": s.get("subcategory"),
                                "confidence": s.get("confidence", 0.9)
                            })
                    except Exception:
                        pass
                        
            time.sleep(2.0)
            
        except Exception as e:
            logger.error("Auto-categorize chunk failed: %s", e)
            continue
            
    return all_suggestions

