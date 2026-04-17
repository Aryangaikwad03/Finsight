"""
db_config.py — Full schema supporting all AA account types.

Account types handled:
  DEPOSIT          → savings/current — has currentBalance, OD fields, pending
  TERM_DEPOSIT     → has maturityDate, interestRate, principal, tenure
  RECURRING_DEPOSIT→ same as TD + recurringAmount, recurringDepositDay

Transaction types:
  DEPOSIT:  CREDIT, DEBIT
  TD/RD:    INTEREST, TDS, OPENING, OTHERS, REDEMPTION, RENEWAL

Category classification:
  By mode:  UPI→Digital, NEFT/IMPS→Bank Transfer, CARD→Card, CASH→Cash,
            ATM→ATM Withdrawal, FT→Fund Transfer
  By type:  INTEREST→Interest Income, TDS→Tax Deducted, OPENING→Opening,
            CREDIT→Income, DEBIT→Expense
"""
import logging
import os
import psycopg2
from psycopg2 import extras
from typing import Dict, List, Optional
from datetime import datetime, timezone


def _parse_dt(val: str):
    """
    Parse any date/datetime string from the AA response into a datetime object.
    """
    if not val:
        return None
    val = str(val).strip()
    for fmt in (
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d",
    ):
        try:
            dt = datetime.strptime(val, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            pass
    return None

logger = logging.getLogger("setu_aa.db")


def _get_db_config() -> dict:
    try:
        from app.core.config import get_settings
        s = get_settings()
        return {"host": s.DB_HOST, "port": s.DB_PORT, "database": s.DB_NAME,
                "user": s.DB_USER, "password": s.DB_PASSWORD}
    except Exception:
        return {"host": os.getenv("DB_HOST","localhost"), "port": int(os.getenv("DB_PORT","5432")),
                "database": os.getenv("DB_NAME","finsight"), "user": os.getenv("DB_USER","postgre"),
                "password": os.getenv("DB_PASSWORD","postgres")}


def get_connection():
    try:
        return psycopg2.connect(**_get_db_config())
    except psycopg2.Error as e:
        logger.error("DB connection failed: %s", e); raise


# ── Category classifier ───────────────────────────────────────────────────────

def classify_transaction(mode: str, txn_type: str, narration: str = "") -> tuple:
    mode = (mode or "").upper()
    txn_type = (txn_type or "").upper()
    n = (narration or "").upper()

    # --- Income / Salary ---
    if txn_type == "CREDIT":
        if any(w in n for w in ["SALARY", "SAL ", "PAYROLL", "WAGE", "STIPEND", "BONUS", "NPS"]):
            return ("Salary & Income", "Salary/Payroll")
        if any(w in n for w in ["REFUND", "CASHBACK", "REVERSAL", "REBATE", "CASH BACK", "SETTLEMENT", "SCRATCH"]):
            return ("Salary & Income", "Refund/Cashback")

    # --- Food & Dining ---
    if any(w in n for w in ["RESTAURANT", "CAFE", "ZOMATO", "SWIGGY", "MCDONALD", "KFC", "PIZZA", "DOMINO"]):
        return ("Food & Dining", "Restaurants/Cafe")
    if any(w in n for w in ["GROCERY", "DMART", "RELIANCE FRESH", "JIOMART", "BLINKIT", "ZEPTO", "INSTAMART"]):
        return ("Food & Dining", "Groceries")

    # --- Transportation ---
    if any(w in n for w in ["PETROL", "FUEL", "HPCL", "BPCL", "IOCL", "SHELL"]):
        return ("Transportation", "Fuel")
    if any(w in n for w in ["UBER", "OLA", "RAPIDO", "TAXI", "CAB"]):
        return ("Transportation", "Taxi/Ride Hailing")
    if any(w in n for w in ["METRO", "IRCTC", "REDBUS", "BUS", "PARKING", "TOLL", "FASTAG"]):
        return ("Transportation", "Public Transport")

    # --- Shopping & Retail ---
    if any(w in n for w in ["AMAZON", "FLIPKART", "MYNTRA", "AJIO", "MEESHO", "NYKAA"]):
        return ("Shopping & Retail", "E-commerce")
    
    # --- Bills & Utilities ---
    if any(w in n for w in ["ELECTRICITY", "WATER BILL", "GAS BILL", "INTERNET", "BROADBAND", "AIRTEL", "JIO", "VODAFONE"]):
        return ("Bills & Utilities", "Utilities/BillPay")

    # --- Transfers ---
    if any(w in n for w in ["UPI", "NEFT", "RTGS", "IMPS", "TRANSFER", "FUND TRANSFER"]):
        return ("Transfers", "Transfers")

    # Overrides
    type_map = {
        "INTEREST":   ("Investments & Savings", "Interest Income"),
        "TDS":        ("Taxes & Government",    "Tax Deducted at Source"),
        "OPENING":    ("Account",               "Account Opening"),
    }
    if txn_type in type_map:
        return type_map[txn_type]

    if txn_type == "CREDIT":
        return ("Salary & Income", "Credit")
    if txn_type == "DEBIT":
        return ("Uncategorized / Unknown", "Debit")
    
    return ("Uncategorized / Unknown", "Other")


def init_database():
    conn = get_connection(); cur = conn.cursor()
    try:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS consents (
                consent_id VARCHAR(255) PRIMARY KEY, user_id VARCHAR(255) NOT NULL,
                vua VARCHAR(255) NOT NULL, status VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                approved_at TIMESTAMP, expires_at TIMESTAMP,
                data_range_from TIMESTAMP, data_range_to TIMESTAMP
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id VARCHAR(255) PRIMARY KEY, user_id VARCHAR(255) NOT NULL,
                consent_id VARCHAR(255) NOT NULL REFERENCES consents(consent_id),
                status VARCHAR(50), created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS fi_data (
                fi_data_id        SERIAL PRIMARY KEY,
                user_id           VARCHAR(255) NOT NULL,
                session_id        VARCHAR(255) NOT NULL REFERENCES sessions(session_id),
                consent_id        VARCHAR(255) NOT NULL REFERENCES consents(consent_id),
                fip_id            VARCHAR(255),
                link_ref_number   VARCHAR(255),
                masked_acc_number VARCHAR(255),
                account_type      VARCHAR(50),
                fi_status         VARCHAR(50),
                created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (user_id, masked_acc_number, account_type)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS profiles (
                profile_id     SERIAL PRIMARY KEY,
                user_id        VARCHAR(255) NOT NULL,
                fi_data_id     INTEGER NOT NULL REFERENCES fi_data(fi_data_id) UNIQUE,
                holder_name    VARCHAR(255), holder_mobile VARCHAR(20),
                holder_email   VARCHAR(255), holder_dob DATE,
                holder_pan     VARCHAR(12),  holder_address TEXT,
                holder_type    VARCHAR(50),  ckyc_compliant BOOLEAN,
                nominee        VARCHAR(50),
                created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS summaries (
                summary_id   SERIAL PRIMARY KEY,
                user_id      VARCHAR(255) NOT NULL,
                fi_data_id   INTEGER NOT NULL REFERENCES fi_data(fi_data_id) UNIQUE,
                account_status  VARCHAR(50),
                opening_date    DATE,
                currency        VARCHAR(10) DEFAULT 'INR',
                current_balance       NUMERIC(20,2),
                balance_datetime      TIMESTAMP,
                branch                VARCHAR(100),
                ifsc_code             VARCHAR(20),
                micr_code             VARCHAR(20),
                account_type          VARCHAR(50),
                od_limit              NUMERIC(20,2),
                drawing_limit         NUMERIC(20,2),
                facility              VARCHAR(20),
                pending_amount        NUMERIC(20,2),
                pending_txn_type      VARCHAR(20),
                current_value         NUMERIC(20,2),
                principal_amount      NUMERIC(20,2),
                maturity_amount       NUMERIC(20,2),
                maturity_date         TIMESTAMP,
                interest_rate         NUMERIC(8,4),
                interest_computation  VARCHAR(30),
                interest_payout       VARCHAR(30),
                compounding_frequency VARCHAR(30),
                tenure_days           INTEGER,
                tenure_months         INTEGER,
                tenure_years          INTEGER,
                recurring_amount      NUMERIC(20,2),
                recurring_deposit_day INTEGER,
                credit_limit          NUMERIC(20,2),
                current_dues          NUMERIC(20,2),
                minimum_due           NUMERIC(20,2),
                outstanding_principal NUMERIC(20,2),
                emi_amount            NUMERIC(20,2),
                created_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                raw_data              JSONB
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                txn_id          SERIAL PRIMARY KEY,
                user_id         VARCHAR(255) NOT NULL,
                fi_data_id      INTEGER NOT NULL REFERENCES fi_data(fi_data_id),
                setu_txn_id     VARCHAR(255),
                txn_date        TIMESTAMP,
                value_date      TIMESTAMP,
                amount          NUMERIC(20,2),
                txn_type        VARCHAR(30),
                payment_mode    VARCHAR(30),
                narration       TEXT,
                reference       VARCHAR(255),
                balance_after   NUMERIC(20,2),
                category        VARCHAR(60),
                subcategory     VARCHAR(60),
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_txn_user     ON transactions(user_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_txn_date     ON transactions(txn_date)")
        conn.commit()
    except psycopg2.Error as e:
        conn.rollback(); raise
    finally:
        cur.close(); conn.close()


def save_consent(consent_id, user_id, vua, status, data_range=None):
    conn = get_connection(); cur = conn.cursor()
    try:
        dr = data_range or {}
        cur.execute("""
            INSERT INTO consents(consent_id,user_id,vua,status,data_range_from,data_range_to)
            VALUES(%s,%s,%s,%s,%s,%s)
            ON CONFLICT(consent_id) DO UPDATE SET
                status=EXCLUDED.status, user_id=EXCLUDED.user_id,
                data_range_from=EXCLUDED.data_range_from,
                data_range_to=EXCLUDED.data_range_to
        """, (consent_id, user_id, vua, status, dr.get("from"), dr.get("to")))
        conn.commit()
    except psycopg2.Error as e:
        conn.rollback(); raise
    finally:
        cur.close(); conn.close()


def save_session(session_id, user_id, consent_id, status="PENDING"):
    conn = get_connection(); cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO sessions(session_id,user_id,consent_id,status)
            VALUES(%s,%s,%s,%s)
            ON CONFLICT(session_id) DO UPDATE SET status=EXCLUDED.status
        """, (session_id, user_id, consent_id, status))
        conn.commit()
    except psycopg2.Error as e:
        conn.rollback(); raise
    finally:
        cur.close(); conn.close()


def save_fi_data(session_id: str, user_id: str, consent_id: str, parsed_data: Dict):
    conn = get_connection(); cur = conn.cursor()
    try:
        new_txns = 0
        for fip in parsed_data.get("fips", []):
            fip_id = fip.get("fip_id")
            for account in fip.get("accounts", []):
                masked   = account.get("masked_acc") or ""
                acc_type = (account.get("fi_type") or "").upper()

                cur.execute("""
                    INSERT INTO fi_data(user_id,session_id,consent_id,fip_id,
                        link_ref_number,masked_acc_number,account_type,fi_status)
                    VALUES(%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT(user_id,masked_acc_number,account_type) DO UPDATE SET
                        session_id=EXCLUDED.session_id, consent_id=EXCLUDED.consent_id,
                        fi_status=EXCLUDED.fi_status, fip_id=EXCLUDED.fip_id
                    RETURNING fi_data_id
                """, (user_id, session_id, consent_id, fip_id,
                      account.get("link_ref"), masked, acc_type,
                      account.get("fi_status")))
                fi_data_id = cur.fetchone()[0]

                # Support both flat and nested structure (AA uses data.account)
                acc_data = account.get("data", {}).get("account", {}) or account
                
                # Profile
                p_raw = acc_data.get("profile", {})
                # AA often nests holders
                holders = p_raw.get("holders", {}).get("holder", [])
                p = holders[0] if holders else p_raw
                
                if p:
                    cur.execute("""
                        INSERT INTO profiles(user_id,fi_data_id,holder_name,holder_mobile,
                            holder_email,holder_dob,holder_pan,holder_address,
                            holder_type,ckyc_compliant,nominee)
                        VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        ON CONFLICT(fi_data_id) DO UPDATE SET
                            holder_name=EXCLUDED.holder_name,
                            holder_mobile=EXCLUDED.holder_mobile,
                            holder_email=EXCLUDED.holder_email,
                            nominee=EXCLUDED.nominee
                    """, (user_id, fi_data_id,
                          p.get("name"), p.get("mobile"), p.get("email"),
                          p.get("dob"),  p.get("pan"),    p.get("address"),
                          p_raw.get("holders", {}).get("type") or p.get("holder_type"), 
                          p.get("ckycCompliance") == "true" or p.get("ckyc") == "true",
                          p.get("nominee")))

                # Summary
                s = acc_data.get("summary", {})
                if s:
                    pending = s.get("pending", {}) or {}
                    cur.execute("""
                        INSERT INTO summaries(user_id,fi_data_id,
                            account_status, opening_date, currency,
                            current_balance, balance_datetime, branch, ifsc_code,
                            micr_code, account_type, od_limit, drawing_limit,
                            facility, pending_amount, pending_txn_type,
                            current_value, principal_amount, maturity_amount,
                            maturity_date, interest_rate, interest_computation,
                            interest_payout, compounding_frequency,
                            tenure_days, tenure_months, tenure_years,
                            recurring_amount, recurring_deposit_day,
                            credit_limit, current_dues, minimum_due,
                            outstanding_principal, emi_amount, raw_data)
                        VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                               %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        ON CONFLICT(fi_data_id) DO UPDATE SET
                            account_status       = EXCLUDED.account_status,
                            current_balance      = EXCLUDED.current_balance,
                            balance_datetime     = EXCLUDED.balance_datetime,
                            current_value        = EXCLUDED.current_value,
                            maturity_amount      = EXCLUDED.maturity_amount,
                            interest_rate        = EXCLUDED.interest_rate,
                            pending_amount       = EXCLUDED.pending_amount,
                            raw_data             = EXCLUDED.raw_data
                    """, (
                        user_id, fi_data_id,
                        s.get("status") or s.get("account_status"),
                        s.get("opening_date") or s.get("openingDate"),
                        s.get("currency", "INR"),
                        s.get("current_balance")  or s.get("currentBalance"),
                        s.get("balance_datetime") or s.get("balanceDateTime"),
                        s.get("branch"),
                        s.get("ifsc") or s.get("ifscCode"),
                        s.get("micr") or s.get("micrCode"),
                        s.get("account_type") or s.get("type"),
                        s.get("od_limit") or s.get("currentODLimit"),
                        s.get("drawing_limit") or s.get("drawingLimit"),
                        s.get("facility"),
                        float(pending.get("amount", 0) or 0),
                        pending.get("transactionType") or pending.get("txn_type"),
                        s.get("current_value") or s.get("currentValue"),
                        s.get("principal") or s.get("principalAmount"),
                        s.get("maturity_amount") or s.get("maturityAmount"),
                        s.get("maturity_date") or s.get("maturityDate"),
                        float(s.get("interest_rate", 0) or s.get("interestRate", 0) or 0),
                        s.get("interest_computation") or s.get("interestComputation"),
                        s.get("interest_payout") or s.get("interestPayout"),
                        s.get("compounding_frequency") or s.get("compoundingFrequency"),
                        int(s.get("tenure_days", 0) or s.get("tenorDays", 0) or 0),
                        int(s.get("tenure_months", 0) or 0),
                        int(s.get("tenure_years", 0) or 0),
                        float(s.get("recurring_amount", 0) or s.get("recurringAmount", 0) or 0),
                        int(s.get("recurring_deposit_day", 0) or s.get("recurringDepositDay", 0) or 0),
                        s.get("credit_limit") or s.get("creditLimit"),
                        s.get("current_dues") or s.get("currentDues"),
                        s.get("minimum_due")  or s.get("minimumDue"),
                        s.get("outstanding_principal") or s.get("outstanding"),
                        s.get("emi_amount") or s.get("emi"),
                        extras.Json(s)
                    ))

                # Transactions
                txns_obj = acc_data.get("transactions", {})
                txns_list = txns_obj.get("transaction") if isinstance(txns_obj, dict) else (account.get("transactions") or [])
                if not txns_list and isinstance(account.get("transactions"), list):
                    txns_list = account.get("transactions")

                for txn in txns_list:
                    setu_id   = txn.get("txn_id") or txn.get("txnId")
                    mode      = txn.get("mode") or txn.get("payment_mode") or ""
                    txn_type  = txn.get("type") or txn.get("txn_type") or ""
                    narration = txn.get("narration") or ""
                    balance   = txn.get("balance") or txn.get("currentBalance") or txn.get("balance_after")
                    cat, sub  = classify_transaction(mode, txn_type, narration)
                    txn_date   = _parse_dt(txn.get("date") or txn.get("transactionTimestamp") or txn.get("txn_date"))
                    value_date = _parse_dt(txn.get("valueDate") or txn.get("value_date") or txn.get("date"))
                    if txn_date is None: txn_date = value_date
                    
                    cur.execute("""
                        INSERT INTO transactions(user_id,fi_data_id,setu_txn_id,
                            txn_date,value_date,amount,txn_type,payment_mode,
                            narration,reference,balance_after,category,subcategory)
                        VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """, (user_id, fi_data_id, setu_id,
                          txn_date, value_date,
                          txn.get("amount"), txn_type, mode,
                          narration, txn.get("reference"), balance, cat, sub))
                    new_txns += 1
        conn.commit()
    except psycopg2.Error as e:
        conn.rollback(); raise
    finally:
        cur.close(); conn.close()


def get_user_accounts(user_id: str) -> List[Dict]:
    conn = get_connection(); cur = conn.cursor(cursor_factory=extras.RealDictCursor)
    try:
        cur.execute("""
            SELECT f.*, s.*, p.holder_name, p.holder_pan
            FROM fi_data f
            LEFT JOIN summaries s ON f.fi_data_id = s.fi_data_id
            LEFT JOIN profiles  p ON f.fi_data_id = p.fi_data_id
            WHERE f.user_id = %s
        """, (user_id,))
        return [dict(r) for r in cur.fetchall()]
    finally:
        cur.close(); conn.close()


def get_user_transactions(user_id: str, limit: int = None, fi_data_ids: List[int] = None) -> List[Dict]:
    conn = get_connection(); cur = conn.cursor(cursor_factory=extras.RealDictCursor)
    try:
        conditions = ["t.user_id = %s"]
        params = [user_id]
        if fi_data_ids:
            conditions.append("t.fi_data_id = ANY(%s)")
            params.append(fi_data_ids)
        where = "WHERE " + " AND ".join(conditions)
        limit_sql = f"LIMIT {limit}" if limit else ""
        cur.execute(f"""
            SELECT t.*, f.masked_acc_number, f.account_type
            FROM transactions t
            JOIN fi_data f ON t.fi_data_id = f.fi_data_id
            {where}
            ORDER BY t.txn_date DESC {limit_sql}
        """, params)
        return [dict(r) for r in cur.fetchall()]
    finally:
        cur.close(); conn.close()


def get_user_summary(user_id: str, month: int = None, year: int = None, fi_data_id: int = None) -> Dict:
    conn = get_connection(); cur = conn.cursor(cursor_factory=extras.RealDictCursor)
    try:
        # Get accounts
        if fi_data_id:
            acc_ids = [fi_data_id]
        else:
            cur.execute("SELECT fi_data_id FROM fi_data WHERE user_id = %s", (user_id,))
            acc_ids = [r['fi_data_id'] for r in cur.fetchall()]
        
        total_bal = 0
        income_types = ('CREDIT','INTEREST','OPENING','REFUND','DEPOSIT','INWARD','REVERSAL')
        expense_types = ('DEBIT','TDS','PAYMENT','INSTALLMENT','WITHDRAWAL','OUTWARD','FEES','CHARGES','TAX','OTHERS')
        
        for aid in acc_ids:
            # Try to get stored balance (Savings/Investments/FDs)
            cur.execute("""
                SELECT current_balance, current_value, principal_amount 
                FROM summaries WHERE fi_data_id = %s
            """, (aid,))
            s = cur.fetchone()
            
            bal = 0
            if s and (s['current_balance'] is not None or s['current_value'] is not None):
                bal = float(s['current_balance'] or 0) + float(s['current_value'] or 0)
            
            if bal == 0:
                # Priority 2: Latest balance_after from transactions
                cur.execute("""
                    SELECT balance_after FROM transactions 
                    WHERE fi_data_id = %s AND balance_after IS NOT NULL
                    ORDER BY txn_date DESC LIMIT 1
                """, (aid,))
                bt = cur.fetchone()
                if bt: bal = float(bt['balance_after'])
            
            if bal == 0:
                # Priority 3: Fallback calculation from history
                cur.execute("""
                    SELECT COALESCE(SUM(CASE WHEN txn_type IN %s THEN amount ELSE -amount END), 0) as net_flow
                    FROM transactions WHERE fi_data_id = %s
                """, (income_types, aid))
                bal = float(cur.fetchone()['net_flow'])
            
            total_bal += bal

        # Get total income/expenses (filtered by month/year if provided)
        query = """
            SELECT
                COALESCE(SUM(CASE WHEN txn_type IN %s THEN amount ELSE 0 END), 0) AS total_income,
                COALESCE(SUM(CASE WHEN txn_type IN %s THEN amount ELSE 0 END), 0) AS total_expenses
            FROM transactions
            WHERE user_id = %s
        """
        params = [income_types, expense_types, user_id]
        if month and year:
            query += " AND EXTRACT(MONTH FROM txn_date) = %s AND EXTRACT(YEAR FROM txn_date) = %s"
            params.extend([month, year])
        if fi_data_id:
            query += " AND fi_data_id = %s"
            params.append(fi_data_id)
        
        cur.execute(query, tuple(params))
        row = cur.fetchone()
        res = dict(row) if row else {"total_income": 0, "total_expenses": 0}

        return {
            "total_balance":  total_bal,
            "total_income":   float(res["total_income"]),
            "total_expenses": float(res["total_expenses"])
        }
    finally:
        cur.close(); conn.close()


def get_user_range_summary(user_id: str, sm: int, sy: int, em: int, ey: int, fi_data_id: int = None) -> Dict:
    """
    Sum income and expenses across a month/year range inclusive.
    """
    conn = get_connection(); cur = conn.cursor(cursor_factory=extras.RealDictCursor)
    try:
        # Range logic: convert to (year * 12 + month) for simple comparison
        start_val = sy * 12 + sm
        end_val = ey * 12 + em
        
        income_types = ('CREDIT','INTEREST','OPENING','REFUND','DEPOSIT','INWARD','REVERSAL')
        expense_types = ('DEBIT','TDS','PAYMENT','INSTALLMENT','WITHDRAWAL','OUTWARD','FEES','CHARGES','TAX','OTHERS')

        query = """
            SELECT
                COALESCE(SUM(CASE WHEN txn_type IN %s THEN amount ELSE 0 END), 0) AS total_income,
                COALESCE(SUM(CASE WHEN txn_type IN %s THEN amount ELSE 0 END), 0) AS total_expenses
            FROM transactions
            WHERE user_id = %s
              AND (EXTRACT(YEAR FROM txn_date) * 12 + EXTRACT(MONTH FROM txn_date)) >= %s
              AND (EXTRACT(YEAR FROM txn_date) * 12 + EXTRACT(MONTH FROM txn_date)) <= %s
        """
        params = [income_types, expense_types, user_id, start_val, end_val]
        if fi_data_id:
            query += " AND fi_data_id = %s"
            params.append(fi_data_id)

        cur.execute(query, tuple(params))
        row = cur.fetchone()
        res = dict(row) if row else {"total_income": 0, "total_expenses": 0}

        return {
            "total_income":   float(res["total_income"]),
            "total_expenses": float(res["total_expenses"]),
            "net_savings":    float(res["total_income"]) - float(res["total_expenses"])
        }
    finally:
        cur.close(); conn.close()


def get_category_breakdown(user_id: str, month: int = None, year: int = None) -> Dict:
    conn = get_connection(); cur = conn.cursor(cursor_factory=extras.RealDictCursor)
    try:
        expense_types = ('DEBIT','TDS','PAYMENT','INSTALLMENT','WITHDRAWAL','OUTWARD','FEES','CHARGES','TAX','OTHERS')
        query = """
            SELECT
                category,
                SUM(CASE WHEN txn_type IN %s THEN amount ELSE 0 END) AS spent
            FROM transactions
            WHERE user_id = %s
        """
        params = [user_id]
        if month and year:
            query += " AND EXTRACT(MONTH FROM txn_date) = %s AND EXTRACT(YEAR FROM txn_date) = %s"
            params.extend([month, year])
        query += " GROUP BY category ORDER BY spent DESC"
        cur.execute(query, tuple([expense_types] + params))
        rows = cur.fetchall()
        return {"breakdown": [dict(r) for r in rows]}
    finally:
        cur.close(); conn.close()
def get_category_drilldown(user_id: str, category: str, month: int = None, year: int = None) -> Dict:
    conn = get_connection(); cur = conn.cursor(cursor_factory=extras.RealDictCursor)
    try:
        expense_types = ('DEBIT','TDS','PAYMENT','INSTALLMENT','WITHDRAWAL','OUTWARD','FEES','CHARGES','TAX','OTHERS')
        query = """
            SELECT
                f.fi_data_id,
                f.masked_acc_number,
                f.account_type,
                SUM(t.amount) as spent
            FROM transactions t
            JOIN fi_data f ON t.fi_data_id = f.fi_data_id
            WHERE t.user_id = %s AND TRIM(t.category) = TRIM(%s) AND t.txn_type IN %s
        """
        params = [user_id, category, expense_types]
        if month and year:
            query += " AND EXTRACT(MONTH FROM t.txn_date) = %s AND EXTRACT(YEAR FROM t.txn_date) = %s"
            params.extend([month, year])
        
        query += " GROUP BY f.fi_data_id, f.masked_acc_number, f.account_type ORDER BY spent DESC"
        
        # DEBUG LOGGING
        with open("drilldown_debug.log", "a") as f_log:
            f_log.write(f"USER: {user_id} | CAT: '{category}' | M: {month} | Y: {year}\n")
            f_log.write(f"QUERY: {query}\n")
            f_log.write(f"PARAMS: {params}\n\n")

        cur.execute(query, tuple(params))
        rows = cur.fetchall()
        
        res = []
        for r in rows:
            row_dict = dict(r)
            row_dict['spent'] = float(row_dict['spent'])
            res.append(row_dict)
            
        return {"drilldown": res}
    finally:
        cur.close(); conn.close()


def get_six_month_trend(user_id: str) -> list[dict]:
    """Calculate the total income and expense bucketing for the last 6 months."""
    conn = get_connection(); cur = conn.cursor(cursor_factory=extras.RealDictCursor)
    try:
        income_types = ('CREDIT','INTEREST','OPENING','REFUND','DEPOSIT','INWARD','REVERSAL')
        expense_types = ('DEBIT','TDS','PAYMENT','INSTALLMENT','WITHDRAWAL','OUTWARD','FEES','CHARGES','TAX','OTHERS')
        
        query = """
            SELECT 
                TO_CHAR(txn_date, 'YYYY-MM') as month,
                SUM(CASE WHEN txn_type IN %s THEN amount ELSE 0 END) AS total_income,
                SUM(CASE WHEN txn_type IN %s THEN amount ELSE 0 END) AS total_expenses
            FROM transactions
            WHERE user_id = %s
              AND txn_date >= DATE_TRUNC('month', CURRENT_DATE) - INTERVAL '5 months'
            GROUP BY TO_CHAR(txn_date, 'YYYY-MM')
            ORDER BY month ASC
        """
        cur.execute(query, (income_types, expense_types, user_id))
        return [dict(r) for r in cur.fetchall()]
    except Exception as e:
        logger.error("Error fetching 6-month trend: %s", e)
        return []
    finally:
        cur.close(); conn.close()


def get_top_merchants(user_id: str, month: int = None, year: int = None, limit: int = 15) -> list[dict]:
    """Group by exact narration to find top spending destinations."""
    conn = get_connection(); cur = conn.cursor(cursor_factory=extras.RealDictCursor)
    try:
        expense_types = ('DEBIT','TDS','PAYMENT','INSTALLMENT','WITHDRAWAL','OUTWARD','FEES','CHARGES','TAX','OTHERS')
        query = """
            SELECT
                SUBSTRING(UPPER(TRIM(narration)) FROM 1 FOR 30) as merchant,
                COUNT(*) as txn_count,
                SUM(CASE WHEN txn_type IN %s THEN amount ELSE 0 END) AS spent
            FROM transactions
            WHERE user_id = %s AND amount > 0 AND txn_type IN %s
        """
        params = [expense_types, user_id, expense_types]
        if month and year:
            query += " AND EXTRACT(MONTH FROM txn_date) = %s AND EXTRACT(YEAR FROM txn_date) = %s"
            params.extend([month, year])
            
        query += " GROUP BY SUBSTRING(UPPER(TRIM(narration)) FROM 1 FOR 30) HAVING SUM(CASE WHEN txn_type IN %s THEN amount ELSE 0 END) > 0"
        params.insert(0, expense_types) # Since we use expense_types again in the HAVING
        query += " ORDER BY spent DESC LIMIT %s"
        params.append(limit)
        
        cur.execute(query, tuple(params))
        return [dict(r) for r in cur.fetchall()]
    except Exception as e:
        logger.error("Error fetching top merchants: %s", e)
        return []
    finally:
        cur.close(); conn.close()


def get_user_budgets(user_id: str) -> list[dict]:
    """Fetch active SPENDING_LIMIT and SAVINGS_GOAL targets from financial_goals table"""
    conn = get_connection(); cur = conn.cursor(cursor_factory=extras.RealDictCursor)
    try:
        cur.execute("""
            SELECT goal_type, title, category, target_amount, period, status
            FROM financial_goals
            WHERE user_id = %s::uuid AND status = 'ACTIVE'
        """, (user_id,))
        return [dict(r) for r in cur.fetchall()]
    except Exception as e:
        logger.warning("Error fetching financial goals (or table missing): %s", e)
        return []
    finally:
        cur.close(); conn.close()


def get_recurring_expenses(user_id: str) -> list[dict]:
    """Isolate subscriptions/EMIs by finding identical payments across distinct months."""
    conn = get_connection(); cur = conn.cursor(cursor_factory=extras.RealDictCursor)
    try:
        expense_types = ('DEBIT','TDS','PAYMENT','INSTALLMENT','WITHDRAWAL','OUTWARD','FEES','CHARGES','TAX','OTHERS')
        
        query = """
            SELECT 
                SUBSTRING(UPPER(TRIM(narration)) FROM 1 FOR 30) as merchant,
                ROUND(AVG(amount), 2) as avg_amount,
                COUNT(DISTINCT TO_CHAR(txn_date, 'YYYY-MM')) as months_active
            FROM transactions
            WHERE user_id = %s AND amount > 0 AND txn_type IN %s
            GROUP BY SUBSTRING(UPPER(TRIM(narration)) FROM 1 FOR 30)
            HAVING COUNT(DISTINCT TO_CHAR(txn_date, 'YYYY-MM')) >= 2 
               AND (MAX(amount) - MIN(amount)) / NULLIF(AVG(amount), 0) < 0.1
            ORDER BY avg_amount DESC LIMIT 10
        """
        cur.execute(query, (user_id, expense_types))
        return [dict(r) for r in cur.fetchall()]
    except Exception as e:
        logger.error("Error fetching recurring: %s", e)
        return []
    finally:
        cur.close(); conn.close()


def get_largest_transactions(user_id: str, limit: int = 5) -> list[dict]:
    """Fetch absolute largest outlier expenses of the current month."""
    conn = get_connection(); cur = conn.cursor(cursor_factory=extras.RealDictCursor)
    try:
        expense_types = ('DEBIT','TDS','PAYMENT','INSTALLMENT','WITHDRAWAL','OUTWARD','FEES','CHARGES','TAX','OTHERS')
        query = """
            SELECT txn_date, amount, category, narration
            FROM transactions
            WHERE user_id = %s AND txn_type IN %s 
              AND txn_date >= DATE_TRUNC('month', CURRENT_DATE)
            ORDER BY amount DESC LIMIT %s
        """
        cur.execute(query, (user_id, expense_types, limit))
        return [dict(r) for r in cur.fetchall()]
    except Exception as e:
        logger.error("Error fetching largest txns: %s", e)
        return []
    finally:
        cur.close(); conn.close()


