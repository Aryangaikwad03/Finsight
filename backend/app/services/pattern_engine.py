"""
pattern_engine.py — Spending Pattern Detection for Penny v2.0
Detects behavioural patterns from transaction history:
  - Weekend vs Weekday spending tendencies
  - Peak spending day of week
  - Rising / falling categories (month-over-month)
  - Lifestyle inflation score (expenses growing faster than income)
  - Merchant loyalty (consistent monthly visits)
  - Impulse payment detection (same merchant, high variance)
"""
import logging
from typing import Dict, List, Any
from datetime import datetime

logger = logging.getLogger("finsight.pattern_engine")


def get_spending_patterns(user_id: str) -> Dict[str, Any]:
    """
    Compute all behavioural spending patterns for a user.
    Returns a structured dict used by Penny's pattern_analysis intent.
    """
    try:
        from app.core.db_config import get_connection
        from psycopg2 import extras
        conn = get_connection()
        cur = conn.cursor(cursor_factory=extras.RealDictCursor)

        expense_types = (
            'DEBIT', 'TDS', 'PAYMENT', 'INSTALLMENT', 'WITHDRAWAL',
            'OUTWARD', 'FEES', 'CHARGES', 'TAX', 'OTHERS', 'REDEMPTION'
        )
        income_types = ('CREDIT', 'INTEREST', 'OPENING', 'REFUND', 'DEPOSIT', 'INWARD', 'REVERSAL')

        result: Dict[str, Any] = {}

        # ── 1. Weekend vs Weekday spending ─────────────────────────────────────
        try:
            cur.execute("""
                SELECT
                    CASE WHEN EXTRACT(DOW FROM txn_date) IN (0, 6) THEN 'weekend' ELSE 'weekday' END AS day_type,
                    COALESCE(SUM(amount), 0) AS total_spent,
                    COUNT(*) AS txn_count
                FROM transactions
                WHERE user_id = %s AND txn_type IN %s
                  AND txn_date >= CURRENT_DATE - INTERVAL '90 days'
                GROUP BY day_type
            """, (user_id, expense_types))
            rows = {r['day_type']: dict(r) for r in cur.fetchall()}
            weekend = float(rows.get('weekend', {}).get('total_spent', 0))
            weekday = float(rows.get('weekday', {}).get('total_spent', 0))
            wkd_txns = int(rows.get('weekend', {}).get('txn_count', 0))
            wdd_txns = int(rows.get('weekday', {}).get('txn_count', 0))
            # Per-day averages (2 weekend days vs 5 weekday days over 90 days)
            weekend_per_day = weekend / max((90 / 7) * 2, 1)
            weekday_per_day = weekday / max((90 / 7) * 5, 1)
            result['weekend_spend_total'] = round(weekend, 2)
            result['weekday_spend_total'] = round(weekday, 2)
            result['weekend_per_day_avg'] = round(weekend_per_day, 2)
            result['weekday_per_day_avg'] = round(weekday_per_day, 2)
            result['is_weekend_spender'] = weekend_per_day > weekday_per_day * 1.2
            result['weekend_txn_count'] = wkd_txns
            result['weekday_txn_count'] = wdd_txns
        except Exception as e:
            logger.warning("Weekend pattern failed: %s", e)

        # ── 2. Peak spending day of the week ───────────────────────────────────
        try:
            cur.execute("""
                SELECT
                    TO_CHAR(txn_date, 'Day') AS day_name,
                    EXTRACT(DOW FROM txn_date) AS day_num,
                    COALESCE(SUM(amount), 0) AS total_spent
                FROM transactions
                WHERE user_id = %s AND txn_type IN %s
                  AND txn_date >= CURRENT_DATE - INTERVAL '90 days'
                GROUP BY day_name, day_num
                ORDER BY total_spent DESC
                LIMIT 1
            """, (user_id, expense_types))
            row = cur.fetchone()
            if row:
                result['peak_spending_day'] = row['day_name'].strip()
                result['peak_day_total'] = float(row['total_spent'])
        except Exception as e:
            logger.warning("Peak day pattern failed: %s", e)

        # ── 3. Rising / Falling categories (last 2 full months) ───────────────
        try:
            now = datetime.now()
            curr_m, curr_y = now.month, now.year
            prev_m = curr_m - 1 if curr_m > 1 else 12
            prev_y = curr_y if curr_m > 1 else curr_y - 1

            cur.execute("""
                SELECT
                    category,
                    SUM(CASE WHEN EXTRACT(MONTH FROM txn_date) = %s AND EXTRACT(YEAR FROM txn_date) = %s
                             THEN amount ELSE 0 END) AS curr_month,
                    SUM(CASE WHEN EXTRACT(MONTH FROM txn_date) = %s AND EXTRACT(YEAR FROM txn_date) = %s
                             THEN amount ELSE 0 END) AS prev_month
                FROM transactions
                WHERE user_id = %s AND txn_type IN %s
                  AND category NOT IN ('Transfers', 'Account', 'Salary & Income')
                GROUP BY category
                HAVING SUM(amount) > 0
            """, (curr_m, curr_y, prev_m, prev_y, user_id, expense_types))

            rising, falling = [], []
            for row in cur.fetchall():
                curr = float(row['curr_month'] or 0)
                prev = float(row['prev_month'] or 0)
                if prev > 100 and curr > 0:
                    pct = ((curr - prev) / prev) * 100
                    entry = {
                        'category': row['category'],
                        'prev_month': round(prev, 0),
                        'curr_month': round(curr, 0),
                        'pct_change': round(pct, 1)
                    }
                    if pct >= 20:
                        rising.append(entry)
                    elif pct <= -20:
                        falling.append(entry)

            result['rising_categories'] = sorted(rising, key=lambda x: -x['pct_change'])[:5]
            result['falling_categories'] = sorted(falling, key=lambda x: x['pct_change'])[:5]
            result['comparison_months'] = {
                'current': f"{curr_m}/{curr_y}",
                'previous': f"{prev_m}/{prev_y}"
            }
        except Exception as e:
            logger.warning("Rising/falling categories failed: %s", e)

        # ── 4. Lifestyle Inflation Score ───────────────────────────────────────
        try:
            cur.execute("""
                SELECT
                    TO_CHAR(txn_date, 'YYYY-MM') AS month,
                    SUM(CASE WHEN txn_type IN %s THEN amount ELSE 0 END) AS income,
                    SUM(CASE WHEN txn_type IN %s THEN amount ELSE 0 END) AS expenses
                FROM transactions
                WHERE user_id = %s
                  AND txn_date >= DATE_TRUNC('month', CURRENT_DATE) - INTERVAL '5 months'
                GROUP BY TO_CHAR(txn_date, 'YYYY-MM')
                ORDER BY month ASC
            """, (income_types, expense_types, user_id))
            months = [dict(r) for r in cur.fetchall()]
            if len(months) >= 3:
                # Check if expense growth rate > income growth rate over last 3 months
                exp_vals = [float(m['expenses']) for m in months[-3:]]
                inc_vals = [float(m['income']) for m in months[-3:]]
                exp_growth = (exp_vals[-1] - exp_vals[0]) / max(exp_vals[0], 1) * 100
                inc_growth = (inc_vals[-1] - inc_vals[0]) / max(inc_vals[0], 1) * 100
                inflation_gap = exp_growth - inc_growth
                # Score 0-100: higher means more lifestyle inflation
                score = min(100, max(0, int(50 + inflation_gap)))
                result['lifestyle_inflation_score'] = score
                result['expense_growth_3m_pct'] = round(exp_growth, 1)
                result['income_growth_3m_pct'] = round(inc_growth, 1)
                result['inflation_warning'] = inflation_gap > 15
        except Exception as e:
            logger.warning("Lifestyle inflation failed: %s", e)

        # ── 5. Merchant Loyalty (consistent monthly visits, not EMI) ──────────
        try:
            cur.execute("""
                SELECT
                    SUBSTRING(UPPER(TRIM(narration)) FROM 1 FOR 25) AS merchant,
                    COUNT(DISTINCT TO_CHAR(txn_date, 'YYYY-MM')) AS months_seen,
                    ROUND(AVG(amount), 2) AS avg_amount,
                    ROUND(STDDEV(amount), 2) AS std_amount
                FROM transactions
                WHERE user_id = %s AND txn_type IN %s
                  AND txn_date >= DATE_TRUNC('month', CURRENT_DATE) - INTERVAL '5 months'
                GROUP BY SUBSTRING(UPPER(TRIM(narration)) FROM 1 FOR 25)
                HAVING COUNT(DISTINCT TO_CHAR(txn_date, 'YYYY-MM')) >= 3
                   AND (STDDEV(amount) / NULLIF(AVG(amount), 0)) > 0.15
                ORDER BY months_seen DESC, avg_amount DESC
                LIMIT 8
            """, (user_id, expense_types))
            result['loyal_merchants'] = [
                {
                    'merchant': r['merchant'],
                    'months_seen': r['months_seen'],
                    'avg_amount': float(r['avg_amount'] or 0)
                }
                for r in cur.fetchall()
            ]
        except Exception as e:
            logger.warning("Merchant loyalty failed: %s", e)

        # ── 6. High-variance / Impulse payments ───────────────────────────────
        try:
            cur.execute("""
                SELECT
                    SUBSTRING(UPPER(TRIM(narration)) FROM 1 FOR 25) AS merchant,
                    COUNT(*) AS txn_count,
                    ROUND(AVG(amount), 2) AS avg_amount,
                    MAX(amount) AS max_amount,
                    MIN(amount) AS min_amount
                FROM transactions
                WHERE user_id = %s AND txn_type IN %s
                  AND txn_date >= CURRENT_DATE - INTERVAL '90 days'
                GROUP BY SUBSTRING(UPPER(TRIM(narration)) FROM 1 FOR 25)
                HAVING COUNT(*) >= 3
                   AND (MAX(amount) - MIN(amount)) / NULLIF(AVG(amount), 0) > 0.5
                ORDER BY avg_amount DESC
                LIMIT 5
            """, (user_id, expense_types))
            result['impulse_merchants'] = [
                {
                    'merchant': r['merchant'],
                    'txn_count': r['txn_count'],
                    'avg': float(r['avg_amount'] or 0),
                    'max': float(r['max_amount'] or 0),
                    'min': float(r['min_amount'] or 0),
                }
                for r in cur.fetchall()
            ]
        except Exception as e:
            logger.warning("Impulse detection failed: %s", e)

        cur.close()
        conn.close()
        return result

    except Exception as e:
        logger.error("Pattern engine failed entirely: %s", e)
        return {}


def format_patterns(patterns: Dict[str, Any]) -> str:
    """Format spending patterns into a compact string for Penny's system prompt."""
    if not patterns:
        return ""
    lines = ["[SPENDING PATTERNS]"]

    # Weekend behaviour
    if 'is_weekend_spender' in patterns:
        tendency = "weekend" if patterns['is_weekend_spender'] else "weekday"
        lines.append(
            f"• Spending tendency: {tendency.upper()} spender "
            f"(Weekend avg ₹{patterns.get('weekend_per_day_avg', 0):,.0f}/day vs "
            f"Weekday avg ₹{patterns.get('weekday_per_day_avg', 0):,.0f}/day)"
        )

    if patterns.get('peak_spending_day'):
        lines.append(f"• Peak spending day: {patterns['peak_spending_day']} "
                     f"(₹{patterns.get('peak_day_total', 0):,.0f} over last 90 days)")

    # Rising categories
    rising = patterns.get('rising_categories', [])
    if rising:
        cats = ", ".join([f"{r['category']} +{r['pct_change']}%" for r in rising[:3]])
        lines.append(f"• Rising spend categories: {cats}")

    # Falling categories
    falling = patterns.get('falling_categories', [])
    if falling:
        cats = ", ".join([f"{r['category']} {r['pct_change']}%" for r in falling[:3]])
        lines.append(f"• Falling spend categories: {cats}")

    # Lifestyle inflation
    if 'lifestyle_inflation_score' in patterns:
        score = patterns['lifestyle_inflation_score']
        warn = " ⚠️ ALERT" if patterns.get('inflation_warning') else ""
        lines.append(
            f"• Lifestyle inflation score: {score}/100{warn} "
            f"(Expenses {patterns.get('expense_growth_3m_pct', 0):+.1f}% vs "
            f"Income {patterns.get('income_growth_3m_pct', 0):+.1f}% over 3 months)"
        )

    # Loyal merchants
    loyal = patterns.get('loyal_merchants', [])
    if loyal:
        mlist = ", ".join([f"{m['merchant']} ({m['months_seen']}mo)" for m in loyal[:4]])
        lines.append(f"• Regular merchants (3+ months): {mlist}")

    return "\n".join(lines)
