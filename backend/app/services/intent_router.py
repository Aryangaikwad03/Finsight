"""
intent_router.py — Penny v2.0 Intent Detection + DB-backed Resolution
Classifies user queries into 13 financial intents and resolves each with precise SQL facts.
"""
import logging
import re
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger("finsight.intent_router")

# All supported intents
INTENTS = [
    "spending_summary",
    "category_spending",
    "merchant_spending",
    "budget_status",
    "goal_progress",
    "savings_analysis",
    "income_analysis",
    "account_balance",
    "recurring_payments",
    "unusual_transaction",
    "fd_rd_query",
    "comparison_query",
    "financial_health",
    "pattern_analysis",
    "transaction_lookup",
    "account_transactions",
    "general",
]

INTENT_EXAMPLES = {
    "spending_summary":      "how much did i spend, total spending this month, what did i spend",
    "category_spending":     "which category, food spending, transport expenses, category breakdown",
    "merchant_spending":     "top merchants, which shops, amazon spending, swiggy, zomato",
    "budget_status":         "am i over budget, budget remaining, spending limit, exceeding budget",
    "goal_progress":         "goal progress, savings goal, behind schedule, target, which goal",
    "savings_analysis":      "how much saved, savings rate, savings ratio, net savings",
    "income_analysis":       "salary, income, how much money came in, earnings",
    "account_balance":       "balance, account balance, net worth, how much money do i have",
    "recurring_payments":    "subscriptions, EMI, recurring, monthly bills, payments every month",
    "unusual_transaction":   "unusual, anomaly, largest expense, biggest transaction, spike",
    "fd_rd_query":           "FD, fixed deposit, RD, recurring deposit, maturity, interest rate",
    "comparison_query":      "compare, last month vs this month, trend, increased, decreased",
    "financial_health":      "financial health, health score, rate my finances, am i doing well",
    "pattern_analysis":      "spending pattern, habit, behaviour, weekend spender, lifestyle inflation, impulse buying, rising categories",
    "transaction_lookup":    "show me transactions, find payments, list my netflix charges, search swiggy, show food transactions, all transactions for, find specific payment",
    "account_transactions":  "show transactions in my HDFC account, what happened in my savings account, account history, transactions in account",
    "general":               "anything else not specific to above",
}


def classify_intent(question: str) -> str:
    """
    Classify user query into one of 13 financial intents.
    Uses rule-based keyword matching first (fast, no API cost),
    falls back to llama-3.1-8b-instant for ambiguous queries.
    """
    q = question.lower().strip()

    # Fast keyword-based rules
    if any(w in q for w in ["fd", "fixed deposit", "rd", "recurring deposit", "maturity", "interest rate"]):
        return "fd_rd_query"
    if any(w in q for w in ["subscription", "emi", "recurring", "monthly bill", "every month"]):
        return "recurring_payments"
    if any(w in q for w in ["unusual", "anomaly", "largest", "biggest transaction", "spike", "highest expense"]):
        return "unusual_transaction"
    if any(w in q for w in ["financial health", "health score", "rate my", "am i doing well", "overall score"]):
        return "financial_health"
    if any(w in q for w in ["pattern", "habit", "behaviour", "when do i spend", "lifestyle", "impulse",
                             "weekend spender", "rising categor", "peak day", "inflation score"]):
        return "pattern_analysis"
    # Account transaction lookup (before generic transaction lookup)
    if any(w in q for w in ["in my hdfc", "in my icici", "in my sbi", "in my axis", "in my kotak",
                             "account history", "transactions in account", "in my savings account",
                             "in my current account", "this account"]):
        return "account_transactions"
    # Transaction-level search / lookup
    if any(w in q for w in ["show me transactions", "list transactions", "find transaction",
                             "search transaction", "find payment", "show payment",
                             "list my", "all my", "show all", "find all",
                             "transactions for", "payments for", "charges"]):
        return "transaction_lookup"
    if any(w in q for w in ["compare", "last month vs", "vs last", "trend", "increased", "decreased", "month over month"]):
        return "comparison_query"
    if any(w in q for w in ["goal", "target", "behind schedule", "savings goal", "progress"]):
        return "goal_progress"
    if any(w in q for w in ["budget", "over budget", "limit", "exceeding"]):
        return "budget_status"
    if any(w in q for w in ["salary", "income", "earnings", "money came in", "credited"]):
        return "income_analysis"
    if any(w in q for w in ["balance", "net worth", "how much do i have", "account balance"]):
        return "account_balance"
    if any(w in q for w in ["saved", "savings rate", "savings ratio", "how much did i save"]):
        return "savings_analysis"
    if any(w in q for w in ["merchant", "shop", "store", "amazon", "swiggy", "zomato", "flipkart", "who did i pay"]):
        return "merchant_spending"
    if any(w in q for w in ["category", "food", "transport", "dining", "utilities", "shopping", "which category"]):
        return "category_spending"
    if any(w in q for w in ["spend", "spent", "expense", "how much", "total spending"]):
        return "spending_summary"

    # LLM fallback for ambiguous queries (cheap 8b model)
    try:
        from app.services.penny_service import _groq_client
        client = _groq_client()
        intent_list = "\n".join([f"- {i}: {ex}" for i, ex in INTENT_EXAMPLES.items()])
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{
                "role": "user",
                "content": (
                    f"Classify this question into exactly one intent.\n"
                    f"Question: '{question}'\n\n"
                    f"Intents:\n{intent_list}\n\n"
                    f"Reply with ONLY the intent name, nothing else."
                )
            }],
            temperature=0,
            max_tokens=20
        )
        intent = resp.choices[0].message.content.strip().lower()
        if intent in INTENTS:
            return intent
    except Exception as e:
        logger.warning("Intent LLM fallback failed: %s", e)

    return "general"


def resolve_intent(intent: str, user_id: str, question: str) -> Dict[str, Any]:
    """
    Run the appropriate SQL query for the detected intent.
    Returns a structured dict of facts to inject into the prompt.
    """
    now = datetime.now()
    month, year = now.month, now.year

    try:
        from app.core.db_config import (
            get_user_summary, get_category_breakdown, get_top_merchants,
            get_user_budgets, get_six_month_trend, get_recurring_expenses,
            get_largest_transactions, get_user_accounts
        )

        if intent == "spending_summary":
            s = get_user_summary(user_id, month=month, year=year)
            prev_month = month - 1 if month > 1 else 12
            prev_year = year if month > 1 else year - 1
            prev_s = get_user_summary(user_id, month=prev_month, year=prev_year)
            curr_exp = float(s.get("total_expenses") or 0)
            prev_exp = float(prev_s.get("total_expenses") or 0)
            pct_change = ((curr_exp - prev_exp) / prev_exp * 100) if prev_exp > 0 else 0
            top_cats = get_category_breakdown(user_id, month=month, year=year)
            top3 = [r.get("category") for r in top_cats.get("breakdown", [])[:3]]
            return {
                "intent": intent,
                "current_month_expenses": curr_exp,
                "prev_month_expenses": prev_exp,
                "pct_change_vs_last_month": round(pct_change, 1),
                "total_income": float(s.get("total_income") or 0),
                "top_categories": top3,
                "month": month, "year": year
            }

        elif intent == "category_spending":
            bd = get_category_breakdown(user_id, month=month, year=year)
            return {"intent": intent, "breakdown": bd.get("breakdown", [])[:10], "month": month, "year": year}

        elif intent == "merchant_spending":
            merchants = get_top_merchants(user_id, month=month, year=year, limit=10)
            return {"intent": intent, "merchants": merchants, "month": month, "year": year}

        elif intent == "budget_status":
            budgets = get_user_budgets(user_id)
            bd = get_category_breakdown(user_id, month=month, year=year)
            cat_spent = {r.get("category", ""): float(r.get("spent") or 0) for r in bd.get("breakdown", [])}
            budget_status = []
            for b in budgets:
                cat = b.get("category") or "Global"
                target = float(b.get("target_amount") or 0)
                actual = cat_spent.get(cat, 0)
                pct = (actual / target * 100) if target > 0 else 0
                budget_status.append({
                    "category": cat, "target": target, "spent": actual,
                    "remaining": max(0, target - actual), "pct_used": round(pct, 1),
                    "exceeded": actual > target
                })
            return {"intent": intent, "budgets": budget_status, "month": month, "year": year}

        elif intent == "goal_progress":
            budgets = get_user_budgets(user_id)
            s = get_user_summary(user_id, month=month, year=year)
            net_monthly = float(s.get("total_income") or 0) - float(s.get("total_expenses") or 0)
            bd = get_category_breakdown(user_id, month=month, year=year)
            cat_spent = {r.get("category", ""): float(r.get("spent") or 0) for r in bd.get("breakdown", [])}
            goals_out = []
            for b in budgets:
                cat = b.get("category") or "Global"
                target = float(b.get("target_amount") or 0)
                curr = cat_spent.get(cat, 0) if b.get("goal_type") == "SPENDING_LIMIT" else net_monthly
                pct = (curr / target * 100) if target > 0 else 0
                goals_out.append({
                    "title": b.get("title") or cat, "goal_type": b.get("goal_type"),
                    "target": target, "current": curr, "pct": round(pct, 1)
                })
            return {"intent": intent, "goals": goals_out, "net_monthly_savings": net_monthly}

        elif intent == "savings_analysis":
            s = get_user_summary(user_id, month=month, year=year)
            trend = get_six_month_trend(user_id)
            income = float(s.get("total_income") or 0)
            expenses = float(s.get("total_expenses") or 0)
            net = income - expenses
            rate = (net / income * 100) if income > 0 else 0
            return {
                "intent": intent, "income": income, "expenses": expenses,
                "net_savings": net, "savings_rate": round(rate, 1),
                "six_month_trend": trend, "month": month, "year": year
            }

        elif intent == "income_analysis":
            s = get_user_summary(user_id, month=month, year=year)
            trend = get_six_month_trend(user_id)
            return {
                "intent": intent,
                "total_income": float(s.get("total_income") or 0),
                "monthly_trend": trend, "month": month, "year": year
            }

        elif intent == "account_balance":
            accounts = get_user_accounts(user_id)
            s = get_user_summary(user_id)
            return {
                "intent": intent,
                "total_balance": float(s.get("total_balance") or 0),
                "accounts": [{"type": a.get("account_type"), "masked": a.get("masked_acc_number"),
                               "balance": float((a.get("current_balance") or a.get("current_value") or 0))}
                              for a in accounts]
            }

        elif intent == "recurring_payments":
            recurring = get_recurring_expenses(user_id)
            total = sum(float(r.get("avg_amount") or 0) for r in recurring)
            return {"intent": intent, "recurring": recurring, "estimated_monthly_total": round(total, 2)}

        elif intent == "unusual_transaction":
            anomalies = get_largest_transactions(user_id, limit=5)
            return {"intent": intent, "anomalies": anomalies, "month": month, "year": year}

        elif intent == "fd_rd_query":
            accounts = get_user_accounts(user_id)
            fds = [a for a in accounts if a.get("account_type") in ("TERM_DEPOSIT", "RECURRING_DEPOSIT")]
            return {"intent": intent, "fd_rd_accounts": fds}

        elif intent == "comparison_query":
            trend = get_six_month_trend(user_id)
            bd_curr = get_category_breakdown(user_id, month=month, year=year)
            prev_month = month - 1 if month > 1 else 12
            prev_year = year if month > 1 else year - 1
            bd_prev = get_category_breakdown(user_id, month=prev_month, year=prev_year)
            return {
                "intent": intent, "six_month_trend": trend,
                "current_month_breakdown": bd_curr.get("breakdown", [])[:8],
                "prev_month_breakdown": bd_prev.get("breakdown", [])[:8],
                "current_month": f"{month}/{year}", "prev_month": f"{prev_month}/{prev_year}"
            }

        elif intent == "financial_health":
            s = get_user_summary(user_id, month=month, year=year)
            income = float(s.get("total_income") or 0)
            expenses = float(s.get("total_expenses") or 0)
            net = income - expenses
            rate = (net / income * 100) if income > 0 else 0
            budgets = get_user_budgets(user_id)
            bd = get_category_breakdown(user_id, month=month, year=year)
            cat_spent = {r.get("category", ""): float(r.get("spent") or 0) for r in bd.get("breakdown", [])}
            exceeded = sum(1 for b in budgets if float(b.get("target_amount") or 0) > 0
                          and cat_spent.get(b.get("category") or "", 0) > float(b.get("target_amount") or 0))
            recurring = get_recurring_expenses(user_id)
            recurring_burden = sum(float(r.get("avg_amount") or 0) for r in recurring)
            return {
                "intent": intent, "savings_rate": round(rate, 1), "net_savings": net,
                "income": income, "expenses": expenses, "budgets_exceeded": exceeded,
                "total_budgets": len(budgets), "recurring_monthly_burden": round(recurring_burden, 2),
                "month": month, "year": year
            }

        elif intent == "pattern_analysis":
            from app.services.pattern_engine import get_spending_patterns
            patterns = get_spending_patterns(user_id)
            return {"intent": intent, "patterns": patterns}

        elif intent == "transaction_lookup":
            from app.core.db_config import (
                get_transactions_filtered, get_subcategory_breakdown,
                search_transactions_by_keyword
            )
            q_lower = question.lower()
            # Extract category hint from question
            category_map = {
                "food": "Food & Dining", "dining": "Food & Dining", "restaurant": "Food & Dining",
                "grocery": "Food & Dining", "groceries": "Food & Dining",
                "transport": "Transportation", "travel": "Transportation", "fuel": "Transportation",
                "uber": "Transportation", "ola": "Transportation",
                "shopping": "Shopping & Retail", "amazon": "Shopping & Retail", "flipkart": "Shopping & Retail",
                "utility": "Bills & Utilities", "utilities": "Bills & Utilities", "bill": "Bills & Utilities",
                "health": "Healthcare", "medical": "Healthcare", "pharmacy": "Healthcare",
                "education": "Education", "school": "Education", "course": "Education",
                "entertainment": "Entertainment", "netflix": "Entertainment", "spotify": "Entertainment",
                "investment": "Investments & Savings", "mutual fund": "Investments & Savings", "sip": "Investments & Savings",
                "insurance": "Insurance",
                "emi": "EMI & Loans", "loan": "EMI & Loans",
                "transfer": "Transfers", "upi": "Transfers",
            }
            detected_category = None
            for kw, cat in category_map.items():
                if kw in q_lower:
                    detected_category = cat
                    break

            # Extract merchant keyword (words after "find"/"show"/"list"/"search")
            import re
            keyword_match = re.search(
                r'(?:find|show|search|list|fetch)\s+(?:my\s+|me\s+|all\s+)?([\w\s]+?)(?:\s+transactions?|\s+payments?|\s+charges?|$)',
                q_lower
            )
            merchant_kw = None
            if keyword_match:
                candidate = keyword_match.group(1).strip()
                # Don't use generic words as merchant keyword
                if candidate and candidate not in ('transactions', 'payments', 'charges',
                                                    'my', 'all', 'the', 'a', 'an', ''):
                    merchant_kw = candidate

            # If we have a specific merchant keyword, do narration search first
            if merchant_kw and not detected_category:
                txns = search_transactions_by_keyword(user_id, merchant_kw, limit=20)
            else:
                txns = get_transactions_filtered(
                    user_id,
                    category=detected_category,
                    month=month, year=year,
                    keyword=merchant_kw,
                    limit=30
                )

            # Also get subcategory breakdown for context
            subcat = get_subcategory_breakdown(user_id, month=month, year=year, category=detected_category)

            return {
                "intent": intent,
                "transactions": txns,
                "subcategory_breakdown": subcat[:10],
                "detected_category": detected_category,
                "merchant_keyword": merchant_kw,
                "month": month, "year": year
            }

        elif intent == "account_transactions":
            from app.core.db_config import get_account_transactions, get_user_accounts
            # Extract account hint from question
            q_lower = question.lower()
            accounts = get_user_accounts(user_id)

            # Try to match a specific account from user's accounts
            matched_acc = None
            matched_acc_type = None
            for acc in accounts:
                masked = (acc.get('masked_acc_number') or '').lower()
                atype = (acc.get('account_type') or '').lower()
                # Check if any part of the masked number or account type is in question
                if masked[-4:] in q_lower:
                    matched_acc = acc.get('masked_acc_number')
                    break
                if atype in q_lower:
                    matched_acc_type = acc.get('account_type')
                    break
            # Detect bank name patterns in question
            bank_map = {
                'hdfc': 'HDFC', 'icici': 'ICICI', 'sbi': 'SBI', 'axis': 'AXIS',
                'kotak': 'KOTAK', 'yes bank': 'YES', 'indusind': 'INDUSIND'
            }
            bank_kw = None
            for bname, _ in bank_map.items():
                if bname in q_lower:
                    bank_kw = bname.upper()
                    break

            txns = get_account_transactions(
                user_id,
                masked_acc=matched_acc,
                account_type=matched_acc_type,
                month=month if 'this month' in q_lower or 'current month' in q_lower else None,
                year=year if 'this month' in q_lower or 'current month' in q_lower else None,
                limit=35
            )
            return {
                "intent": intent,
                "transactions": txns,
                "matched_account": matched_acc or matched_acc_type or bank_kw or "all accounts",
                "accounts": [{"type": a.get("account_type"), "masked": a.get("masked_acc_number")} for a in accounts],
                "month": month, "year": year
            }

    except Exception as e:
        logger.error("Intent resolution failed for %s: %s", intent, e)

    return {"intent": intent}


def format_db_facts(facts: Dict[str, Any]) -> str:
    """
    Convert structured DB facts dict into a compact token-efficient string
    for the LLM system prompt (target: < 400 tokens).
    """
    intent = facts.get("intent", "general")
    lines = [f"[DB FACTS — Intent: {intent.upper()}]"]

    if intent == "spending_summary":
        lines.append(f"This month ({facts.get('month')}/{facts.get('year')}): Expenses ₹{facts.get('current_month_expenses', 0):,.0f} | Income ₹{facts.get('total_income', 0):,.0f}")
        lines.append(f"vs Last Month: ₹{facts.get('prev_month_expenses', 0):,.0f} ({facts.get('pct_change_vs_last_month', 0):+.1f}%)")
        lines.append(f"Top categories: {', '.join(facts.get('top_categories', []))}")

    elif intent == "category_spending":
        for r in facts.get("breakdown", []):
            cat = r.get("category", "Other")
            amt = float(r.get("spent") or 0)
            lines.append(f"• {cat}: ₹{amt:,.0f}")

    elif intent == "merchant_spending":
        for m in facts.get("merchants", []):
            merchant = m.get("merchant", "?")
            amt = float(m.get("spent") or 0)
            cnt = m.get("txn_count", 0)
            lines.append(f"• {merchant}: ₹{amt:,.0f} ({cnt} txns)")

    elif intent == "budget_status":
        for b in facts.get("budgets", []):
            exceeded_tag = " ⚠️ EXCEEDED" if b.get("exceeded") else ""
            lines.append(f"• {b['category']}: ₹{b['spent']:,.0f}/₹{b['target']:,.0f} ({b['pct_used']}%){exceeded_tag}")

    elif intent == "goal_progress":
        lines.append(f"Net monthly savings capacity: ₹{facts.get('net_monthly_savings', 0):,.0f}")
        for g in facts.get("goals", []):
            lines.append(f"• {g['title']} ({g['goal_type']}): ₹{g['current']:,.0f}/₹{g['target']:,.0f} ({g['pct']}%)")

    elif intent in ("savings_analysis",):
        lines.append(f"Income: ₹{facts.get('income', 0):,.0f} | Expenses: ₹{facts.get('expenses', 0):,.0f}")
        lines.append(f"Net Savings: ₹{facts.get('net_savings', 0):,.0f} | Rate: {facts.get('savings_rate', 0)}%")
        for t in facts.get("six_month_trend", [])[-6:]:
            m = t.get("month", "")
            inc = float(t.get("total_income") or 0)
            exp = float(t.get("total_expenses") or 0)
            lines.append(f"• {m}: In ₹{inc:,.0f} / Out ₹{exp:,.0f}")

    elif intent == "income_analysis":
        lines.append(f"This month income: ₹{facts.get('total_income', 0):,.0f}")
        for t in facts.get("monthly_trend", [])[-4:]:
            lines.append(f"• {t.get('month')}: ₹{float(t.get('total_income') or 0):,.0f}")

    elif intent == "account_balance":
        lines.append(f"Total Balance: ₹{facts.get('total_balance', 0):,.0f}")
        for a in facts.get("accounts", []):
            lines.append(f"• {a['type']} {a['masked']}: ₹{float(a['balance'] or 0):,.0f}")

    elif intent == "recurring_payments":
        lines.append(f"Est. monthly recurring total: ₹{facts.get('estimated_monthly_total', 0):,.0f}")
        for r in facts.get("recurring", []):
            lines.append(f"• {r.get('merchant', '?')}: ~₹{float(r.get('avg_amount') or 0):,.0f}/month")

    elif intent == "unusual_transaction":
        for a in facts.get("anomalies", []):
            lines.append(f"• {str(a.get('txn_date', ''))[:10]} | {a.get('category')} | ₹{float(a.get('amount') or 0):,.0f} | {str(a.get('narration', ''))[:30]}")

    elif intent == "fd_rd_query":
        for a in facts.get("fd_rd_accounts", []):
            bal = float(a.get("current_value") or a.get("principal_amount") or 0)
            lines.append(f"• {a.get('account_type')} {a.get('masked_acc_number')}: ₹{bal:,.0f} | Rate: {a.get('interest_rate')}% | Matures: {str(a.get('maturity_date', ''))[:10]}")

    elif intent == "comparison_query":
        lines.append(f"Comparison: {facts.get('current_month')} vs {facts.get('prev_month')}")
        for t in facts.get("six_month_trend", []):
            inc = float(t.get("total_income") or 0)
            exp = float(t.get("total_expenses") or 0)
            lines.append(f"• {t.get('month')}: In ₹{inc:,.0f} / Out ₹{exp:,.0f}")

    elif intent == "financial_health":
        lines.append(f"Savings Rate: {facts.get('savings_rate', 0)}% | Net: \u20b9{facts.get('net_savings', 0):,.0f}")
        lines.append(f"Budgets Exceeded: {facts.get('budgets_exceeded', 0)}/{facts.get('total_budgets', 0)}")
        lines.append(f"Monthly Recurring Burden: \u20b9{facts.get('recurring_monthly_burden', 0):,.0f}")

    elif intent == "pattern_analysis":
        from app.services.pattern_engine import format_patterns
        patterns = facts.get("patterns", {})
        return format_patterns(patterns)

    elif intent == "transaction_lookup":
        cat = facts.get('detected_category') or 'All'
        kw = facts.get('merchant_keyword') or ''
        header = f"Transactions"
        if cat != 'All': header += f" in {cat}"
        if kw: header += f" matching '{kw}'"
        lines.append(header + f" ({facts.get('month')}/{facts.get('year')})")
        for t in facts.get('transactions', [])[:25]:
            acc = t.get('masked_acc_number', '?')
            narr = str(t.get('narration', ''))[:35]
            lines.append(
                f"• {t.get('txn_date')} | {t.get('account_type','?')} {acc} | "
                f"{t.get('txn_type','?')} \u20b9{t.get('amount', 0):,.0f} | {narr}"
            )
        # Add subcategory breakdown if available
        subcats = facts.get('subcategory_breakdown', [])
        if subcats:
            lines.append("Subcategory breakdown:")
            for s in subcats[:6]:
                lines.append(f"  {s.get('subcategory','?')}: \u20b9{s.get('spent', 0):,.0f} ({s.get('txn_count',0)} txns)")

    elif intent == "account_transactions":
        matched = facts.get('matched_account', 'your account')
        lines.append(f"Recent transactions — {matched}:")
        for t in facts.get('transactions', [])[:30]:
            bal_str = f" | Bal \u20b9{t.get('balance_after'):,.0f}" if t.get('balance_after') else ""
            narr = str(t.get('narration', ''))[:35]
            lines.append(
                f"• {t.get('txn_date')} | {t.get('account_type','?')} {t.get('masked_acc_number','?')} | "
                f"{t.get('txn_type','?')} \u20b9{t.get('amount', 0):,.0f} | "
                f"{t.get('category','?')} | {narr}{bal_str}"
            )

    return "\n".join(lines)
