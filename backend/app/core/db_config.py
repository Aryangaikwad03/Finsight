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
    Handles:
      - Full ISO with timezone:  '2025-07-15T16:11:06+00:00'
      - Date only:               '2025-07-16'   (savings account valueDate)
      - Full ISO with Z:         '2024-04-05T23:37:41+00:00'
    Returns None if val is falsy or unparseable.
    """
    if not val:
        return None
    val = str(val).strip()
    # Try full ISO timestamp formats
    for fmt in (
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%SZ",
    ):
        try:
            return datetime.strptime(val, fmt)
        except ValueError:
            pass
    # Try date-only (savings valueDate: '2025-07-16')
    try:
        return datetime.strptime(val, "%Y-%m-%d").replace(tzinfo=timezone.utc)
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
    """
    Returns (category, subcategory) based on powerful fast local heuristic keyword matches.
    Tailored for Indian and Global contexts.
    """
    mode = (mode or "").upper()
    txn_type = (txn_type or "").upper()
    n = (narration or "").upper()

    # 1. High priority Keyword matching natively off the narration string
    
    # --- Income / Salary ---
    if txn_type == "CREDIT":
        if any(w in n for w in ["SALARY", "SAL ", "PAYROLL", "WAGE", "STIPEND", "BONUS"]):
            return ("Salary & Income", "Salary/Payroll")
        if any(w in n for w in ["REFUND", "CASHBACK", "REVERSAL", "REBATE", "CASH BACK"]):
            return ("Salary & Income", "Refund/Cashback")
        if any(w in n for w in ["FREELANCE", "CONSULTING", "INCENTIVE", "COMMISSION"]):
            return ("Salary & Income", "Freelance/Bonus")

    # --- Food & Dining ---
    if any(w in n for w in [
        "RESTAURANT", "CAFE", "COFFEE", "TEA", "STARBUCKS", "COSTA", "BARISTA", "MCDONALD", "KFC",
        "BURGER KING", "DOMINO", "PIZZA HUT", "SUBWAY", "ZOMATO", "SWIGGY", "EATSURE", "UBEREATS",
        "DOORDASH", "GRUBHUB", "DELIVEROO", "TALABAT", "FOODPANDA", "DINEOUT", "BBQ NATION",
        "HALDIRAM", "CHAI POINT", "CHAAYOS", "CAFE COFFEE DAY", "CCD", "DUNKIN", "TACO BELL",
        "NANDOS", "BAKERY", "BISTRO", "PUB", "LOUNGE", "DINER", "TAKEAWAY"
    ]):
        return ("Food & Dining", "Restaurants/Cafe")
    
    # --- Groceries ---
    if any(w in n for w in [
        "GROCERY", "SUPERMARKET", "KIRANA", "DMART", "RELIANCE FRESH", "BIG BAZAAR", "JIOMART",
        "BLINKIT", "ZEPTO", "INSTAMART", "GROFERS", "AMAZON FRESH", "SPENCER", "MORE RETAIL",
        "EASYDAY", "NATURE BASKET", "WALMART", "TESCO", "ALDI", "LIDL", "COSTCO", "WHOLE FOODS",
        "TARGET GROCERY", "VEGETABLE", "FRUITS", "MILK", "DAIRY", "BB ", "GROCERS"
    ]):
        return ("Food & Dining", "Groceries")

    # --- Transportation ---
    if any(w in n for w in [
        "PETROL", "DIESEL", "FUEL", "HPCL", "BPCL", "IOCL", "INDIAN OIL", "BHARAT PETROLEUM",
        "HINDUSTAN PETROLEUM", "SHELL", "ESSAR", "NAYARA", "CHEVRON", "EXXON", "MOBIL", "TEXACO",
        "GAS STATION"
    ]):
        return ("Transportation", "Fuel")
    
    if any(w in n for w in [
        "UBER", "OLA", "RAPIDO", "LYFT", "GRAB", "CAREEM", "TAXI", "CAB", "AUTO", "RIDESHARE",
        "TRANSIT", "METRO TAXI"
    ]):
        return ("Transportation", "Taxi/Ride Hailing")
    
    if any(w in n for w in [
        "METRO", "RAIL", "TRAIN", "IRCTC", "REDBUS", "BUS", "PARKING", "TOLL", "FASTAG",
        "NMMT", "BEST BUS", "DELHI METRO", "MUMBAI METRO", "OYSTER", "OCTOPUS"
    ]):
        return ("Transportation", "Public Transport")

    # --- Shopping & Retail ---
    if any(w in n for w in [
        "AMAZON", "FLIPKART", "MYNTRA", "AJIO", "MEESHO", "NYKAA", "TATACLIQ", "SNAPDEAL",
        "EBAY", "ETSY", "ALIEXPRESS", "TEMU", "SHEIN", "SHOPIFY"
    ]):
        return ("Shopping & Retail", "E-commerce")
    
    if any(w in n for w in [
        "ZARA", "HM", "H&M", "LIFESTYLE", "PANTALOONS", "WESTSIDE", "TRENDS", "LEVIS", "NIKE",
        "ADIDAS", "PUMA", "UNDER ARMOUR", "UNIQLO", "FOREVER21", "FASHION", "APPAREL", "GARMENT",
        "FOOTWEAR", "SHOE STORE"
    ]):
        return ("Shopping & Retail", "Clothing & Fashion")
    
    if any(w in n for w in [
        "CROMA", "RELIANCE DIGITAL", "VIJAY SALES", "APPLE", "SAMSUNG", "ONEPLUS", "XIAOMI",
        "MI STORE", "BEST BUY", "MEDIAMARKT", "GADGET", "LAPTOP", "MOBILE", "PHONE", "TABLET", "APPLIANCE"
    ]):
        return ("Shopping & Retail", "Electronics")

    # --- Bills & Utilities ---
    if any(w in n for w in [
        "ELECTRICITY", "WATER BILL", "GAS BILL", "INTERNET", "BROADBAND", "WIFI", "AIRTEL",
        "JIO", "VODAFONE", "VI ", "BSNL", "TATA PLAY", "DISH TV", "DTH", "UTILITY", "POWER",
        "TORRENT POWER", "ADANI ELECTRICITY", "BESCOM", "MSEB", "PGVCL", "BILLPAY", "RECHARGE"
    ]):
        return ("Bills & Utilities", "Utilities/BillPay")

    # --- Housing & Rent ---
    if any(w in n for w in [
        "RENT", "HOUSE RENT", "LANDLORD", "LEASE", "MAINTENANCE", "SOCIETY MAINTENANCE",
        "APARTMENT MAINTENANCE", "HOUSING", "PROPERTY TAX", "MORTGAGE", "HOME LOAN", "NESTAWAY",
        "NO BROKER", "BROKER FEE"
    ]):
        return ("Housing & Rent", "Rent/Maintenance")

    # --- Healthcare & Medical ---
    if any(w in n for w in [
        "HOSPITAL", "CLINIC", "PHARMACY", "CHEMIST", "APOLLO", "PRACTO", "NETMEDS", "1MG",
        "TATA 1MG", "MEDPLUS", "WELLNESS", "DOCTOR", "DENTIST", "PATHOLOGY", "DIAGNOSTIC",
        "LAB TEST", "HEALTHCARE", "MEDICAL"
    ]):
        return ("Healthcare & Medical", "Medical/Healthcare")

    # --- Entertainment & Leisure ---
    if any(w in n for w in [
        "NETFLIX", "PRIME VIDEO", "AMAZON PRIME", "DISNEY", "HOTSTAR", "SPOTIFY", "YOUTUBE PREMIUM",
        "GAANA", "WYNK", "JIOSAAVN", "MOVIE", "CINEMA", "PVR", "INOX", "BOOKMYSHOW", "GAMING",
        "STEAM", "PLAYSTATION", "XBOX", "NINTENDO", "HOBBY", "EVENT", "AMUSEMENT"
    ]):
        return ("Entertainment & Leisure", "Sub/Movies/Events")

    # --- Travel ---
    if any(w in n for w in [
        "HOTEL", "FLIGHT", "AIRLINE", "INDIGO", "AIR INDIA", "VISTARA", "SPICEJET", "EMIRATES",
        "QATAR AIRWAYS", "BRITISH AIRWAYS", "BOOKING.COM", "AGODA", "EXPEDIA", "trip.com",
        "MAKEYMYTRIP", "YATRA", "CLEARTRIP", "HOSTEL", "RESORT", "TOURISM"
    ]):
        return ("Travel", "Hotel/Flight")

    # --- Education ---
    if any(w in n for w in [
        "SCHOOL", "COLLEGE", "TUITION", "COACHING", "BYJU", "UNACADEMY", "COURSERA", "UDEMY",
        "EDX", "UPGRAD", "SIMPLILEARN", "SKILLSHARE", "FEES", "EXAM FEE", "CERTIFICATION", "TRAINING"
    ]):
        return ("Education", "Education/Fees")

    # --- Investments & Savings ---
    if any(w in n for w in [
        "MUTUAL FUND", "SIP", "ZERODHA", "GROWW", "UPSTOX", "ANGEL ONE", "COIN", "STOCK", "SHARE",
        "DEMAT", "DIVIDEND", "BOND", "FD ", "FIXED DEPOSIT", "RECURRING DEPOSIT", "RD ", "PPF",
        "NPS", "INVESTMENT", "CRYPTO", "BITCOIN", "COINBASE", "BINANCE", "ETF", "KITE"
    ]):
        return ("Investments & Savings", "Investments")

    # --- Insurance ---
    if any(w in n for w in [
        "INSURANCE", "LIC ", "HDFC ERGO", "ICICI LOMBARD", "STAR HEALTH", "MAX BUPA",
        "POLICY PREMIUM", "MEDICLAIM", "TERM INSURANCE", "LIFE INSURANCE"
    ]):
        return ("Insurance", "Insurance Premium")

    # --- Transfers ---
    if any(w in n for w in [
        "UPI", "NEFT", "RTGS", "IMPS", "TRANSFER", "BANK TRANSFER", "SELF TRANSFER",
        "WALLET TRANSFER", "PAYTM WALLET", "PHONEPE", "GPAY", "GOOGLE PAY", "VENMO", "PAYPAL",
        "CASH APP", "ZELLE", "REMITLY", "WISE", "WESTERN UNION"
    ]):
        return ("Transfers", "Transfers")

    # --- Taxes & Government ---
    if any(w in n for w in [
        "GST", "INCOME TAX", "TDS", "CHALLAN", "EPFO", "PF CONTRIBUTION", "ESIC",
        "GOVERNMENT FEE", "PASSPORT", "VISA", "MUNICIPAL TAX", "PROPERTY TAX", "COURT FEE",
        "PENALTY", "TRAFFIC FINE"
    ]):
        return ("Taxes & Government", "Tax/Govt Fees")

    # --- ATM / Cash Withdrawal ---
    if any(w in n for w in [
        "ATM", "CASH WITHDRAWAL", "CASH DEPOSIT", "BRANCH CASH", "WITHDRAWAL", "DEPOSIT", "CASH TXN"
    ]):
        return ("ATM / Cash Withdrawal", "Cash Transaction")

    # --- Fees & Charges ---
    if any(w in n for w in [
        "PROCESSING FEE", "LATE FEE", "BANK CHARGE", "ANNUAL FEE", "JOINING FEE",
        "CONVENIENCE FEE", "SERVICE CHARGE", "OVERDRAFT FEE", "INTEREST CHARGE"
    ]):
        return ("Fees & Charges", "Bank Fees/Penalty")

    # --- Donations & Charity ---
    if any(w in n for w in [
        "DONATION", "CHARITY", "NGO", "TEMPLE", "CHURCH", "MOSQUE", "GURUDWARA", "TRUST",
        "FUNDRAISER", "RELIEF FUND", "CROWDFUNDING"
    ]):
        return ("Donations & Charity", "Donations")

    # --- Business / Professional Expenses ---
    if any(w in n for w in [
        "OFFICE SUPPLIES", "SOFTWARE", "SAAS", "ZOOM", "SLACK", "NOTION", "ADOBE", "MICROSOFT",
        "AWS", "GOOGLE CLOUD", "HOSTING", "DOMAIN", "COWORKING", "STATIONERY", "PRINTING",
        "COURIER", "LOGISTICS"
    ]):
        return ("Business / Professional Expenses", "Business/Professional")

    # --- Subscription Services ---
    if any(w in n for w in [
        "SUBSCRIPTION", "MONTHLY PLAN", "ANNUAL PLAN", "RECURRING PAYMENT", "AUTO DEBIT",
        "RENEWAL", "MEMBERSHIP", "GYM MEMBERSHIP"
    ]):
        return ("Subscription Services", "Subscriptions")

    # 2. Type-based overrides (TD/RD specific)
    type_map = {
        "INTEREST":   ("Investments & Savings", "Interest Income"),
        "TDS":        ("Taxes & Government",    "Tax Deducted at Source"),
        "OPENING":    ("Account",               "Account Opening"),
        "REDEMPTION": ("Investments & Savings", "Redemption"),
        "RENEWAL":    ("Investments & Savings", "Renewal"),
    }
    if txn_type in type_map:
        return type_map[txn_type]

    # 3. Mode fallbacks
    mode_map = {
        "UPI":    ("Transfers", "UPI"),
        "NEFT":   ("Transfers", "NEFT"),
        "IMPS":   ("Transfers", "IMPS"),
        "FT":     ("Transfers", "Fund Transfer"),
        "RTGS":   ("Transfers", "RTGS"),
        "CARD":   ("Transfers", "Card Payment"),
        "CASH":   ("ATM / Cash Withdrawal", "Cash"),
        "ATM":    ("ATM / Cash Withdrawal", "ATM"),
        "ECS":    ("Bills & Utilities", "ECS/Auto Debit"),
        "SI":     ("Bills & Utilities", "Standing Instruction"),
        "ACH":    ("Transfers", "ACH"),
        "NACH":   ("Transfers", "NACH"),
    }
    if mode in mode_map:
        cat, sub = mode_map[mode]
        return (cat, sub)

    # 4. Final fallback
    if txn_type == "CREDIT":
        return ("Salary & Income", "Credit")
    if txn_type == "DEBIT":
        return ("Uncategorized / Unknown", "Debit")
    
    return ("Uncategorized / Unknown", "Other")


def init_database():
    conn = get_connection(); cur = conn.cursor()
    try:
        # ── consents ──────────────────────────────────────────────────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS consents (
                consent_id VARCHAR(255) PRIMARY KEY, user_id VARCHAR(255) NOT NULL,
                vua VARCHAR(255) NOT NULL, status VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                approved_at TIMESTAMP, expires_at TIMESTAMP,
                data_range_from TIMESTAMP, data_range_to TIMESTAMP
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_consents_user ON consents(user_id)")

        # ── sessions ──────────────────────────────────────────────────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id VARCHAR(255) PRIMARY KEY, user_id VARCHAR(255) NOT NULL,
                consent_id VARCHAR(255) NOT NULL REFERENCES consents(consent_id),
                status VARCHAR(50), created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id)")

        # ── fi_data — unique per user+account ─────────────────────────────────
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
        cur.execute("CREATE INDEX IF NOT EXISTS idx_fi_data_user ON fi_data(user_id)")

        # ── profiles ──────────────────────────────────────────────────────────
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

        # ── summaries — handles both DEPOSIT and TD/RD fields ─────────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS summaries (
                summary_id   SERIAL PRIMARY KEY,
                user_id      VARCHAR(255) NOT NULL,
                fi_data_id   INTEGER NOT NULL REFERENCES fi_data(fi_data_id) UNIQUE,

                -- Common fields
                account_status  VARCHAR(50),
                opening_date    DATE,
                currency        VARCHAR(10) DEFAULT 'INR',

                -- DEPOSIT (savings/current) fields
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

                -- TERM_DEPOSIT / RECURRING_DEPOSIT fields
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

                -- Credit/Loan fields (for future)
                credit_limit          NUMERIC(20,2),
                current_dues          NUMERIC(20,2),
                minimum_due           NUMERIC(20,2),
                outstanding_principal NUMERIC(20,2),
                emi_amount            NUMERIC(20,2),

                created_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                raw_data              JSONB
            )
        """)

        # ── transactions — full field set from AA data ─────────────────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                txn_id          SERIAL PRIMARY KEY,
                user_id         VARCHAR(255) NOT NULL,
                fi_data_id      INTEGER NOT NULL REFERENCES fi_data(fi_data_id),
                setu_txn_id     VARCHAR(255),

                -- Timestamps (AA provides both)
                txn_date        TIMESTAMP,    -- transactionTimestamp
                value_date      TIMESTAMP,    -- valueDate (settlement date)

                amount          NUMERIC(20,2),
                txn_type        VARCHAR(30),  -- CREDIT/DEBIT/INTEREST/TDS/OPENING/etc.
                payment_mode    VARCHAR(30),  -- UPI/NEFT/IMPS/CARD/CASH/ATM/FT/etc.
                narration       TEXT,
                reference       VARCHAR(255),
                balance_after   NUMERIC(20,2), -- currentBalance or balance field

                -- Classification (computed at insert time)
                category        VARCHAR(60),   -- e.g. "Digital Payments", "Cash"
                subcategory     VARCHAR(60),   -- e.g. "UPI Transfer (Debit)"

                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # No unique indexes on transactions — allow multiple saves per user
        # Dedup is handled at read time by filtering on user_id
        cur.execute("CREATE INDEX IF NOT EXISTS idx_txn_user     ON transactions(user_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_txn_date     ON transactions(txn_date)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_txn_category ON transactions(user_id, category)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_txn_mode     ON transactions(user_id, payment_mode)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_txn_type     ON transactions(user_id, txn_type)")

        # ── Add new columns to existing tables if upgrading ───────────────────
        for col, defn in [
            ("value_date",      "TIMESTAMP"),
            ("category",        "VARCHAR(60)"),
            ("subcategory",     "VARCHAR(60)"),
        ]:
            cur.execute(f"""
                DO $$ BEGIN
                    ALTER TABLE transactions ADD COLUMN IF NOT EXISTS {col} {defn};
                EXCEPTION WHEN others THEN NULL; END $$;
            """)

        for col, defn in [
            ("current_value",         "NUMERIC(20,2)"),
            ("principal_amount",      "NUMERIC(20,2)"),
            ("maturity_amount",       "NUMERIC(20,2)"),
            ("maturity_date",         "TIMESTAMP"),
            ("interest_rate",         "NUMERIC(8,4)"),
            ("interest_computation",  "VARCHAR(30)"),
            ("interest_payout",       "VARCHAR(30)"),
            ("compounding_frequency", "VARCHAR(30)"),
            ("tenure_days",           "INTEGER"),
            ("tenure_months",         "INTEGER"),
            ("tenure_years",          "INTEGER"),
            ("recurring_amount",      "NUMERIC(20,2)"),
            ("recurring_deposit_day", "INTEGER"),
            ("od_limit",              "NUMERIC(20,2)"),
            ("drawing_limit",         "NUMERIC(20,2)"),
            ("facility",              "VARCHAR(20)"),
            ("pending_amount",        "NUMERIC(20,2)"),
            ("pending_txn_type",      "VARCHAR(20)"),
            ("nominee",               "VARCHAR(50)"),
        ]:
            cur.execute(f"""
                DO $$ BEGIN
                    ALTER TABLE summaries ADD COLUMN IF NOT EXISTS {col} {defn};
                EXCEPTION WHEN others THEN NULL; END $$;
            """)

        # Back-fill categories for existing transactions
        cur.execute("""
            UPDATE transactions SET
                category    = CASE
                    WHEN txn_type = 'INTEREST'   THEN 'Investment Returns'
                    WHEN txn_type = 'TDS'         THEN 'Tax'
                    WHEN txn_type = 'OPENING'     THEN 'Account'
                    WHEN txn_type = 'REDEMPTION'  THEN 'Investment'
                    WHEN txn_type = 'RENEWAL'     THEN 'Investment'
                    WHEN payment_mode = 'UPI'     THEN 'Digital Payments'
                    WHEN payment_mode IN ('NEFT','IMPS','FT','RTGS') THEN 'Bank Transfer'
                    WHEN payment_mode = 'CARD'    THEN 'Card'
                    WHEN payment_mode IN ('CASH') THEN 'Cash'
                    WHEN payment_mode = 'ATM'     THEN 'Cash'
                    WHEN txn_type = 'CREDIT'      THEN 'Income'
                    WHEN txn_type = 'DEBIT'       THEN 'Expense'
                    ELSE 'Other'
                END,
                subcategory = CASE
                    WHEN txn_type = 'INTEREST'    THEN 'Interest Income'
                    WHEN txn_type = 'TDS'         THEN 'Tax Deducted at Source'
                    WHEN txn_type = 'OPENING'     THEN 'Account Opening'
                    WHEN payment_mode = 'UPI'     THEN 'UPI Transfer'
                    WHEN payment_mode = 'NEFT'    THEN 'NEFT'
                    WHEN payment_mode = 'IMPS'    THEN 'IMPS'
                    WHEN payment_mode = 'FT'      THEN 'Fund Transfer'
                    WHEN payment_mode = 'CARD'    THEN 'Card Payment'
                    WHEN payment_mode = 'CASH'    THEN 'Cash Transaction'
                    WHEN payment_mode = 'ATM'     THEN 'ATM Withdrawal'
                    ELSE txn_type
                END
            WHERE category IS NULL
        """)

        conn.commit()
        logger.info("✅ DB schema ready — all AA field types supported")

    except psycopg2.Error as e:
        conn.rollback(); logger.error("DB init error: %s", e); raise
    finally:
        cur.close(); conn.close()


# ── Write functions ───────────────────────────────────────────────────────────

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
    """
    Upsert all FI data with full deduplication.
    Handles DEPOSIT, TERM_DEPOSIT, RECURRING_DEPOSIT field differences.
    """
    conn = get_connection(); cur = conn.cursor()
    try:
        new_txns = 0

        for fip in parsed_data.get("fips", []):
            fip_id = fip.get("fip_id")
            for account in fip.get("accounts", []):
                masked   = account.get("masked_acc") or ""
                acc_type = (account.get("fi_type") or "").upper()

                # UPSERT fi_data
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
                logger.info("  fi_data_id=%s acc=%s type=%s", fi_data_id, masked, acc_type)

                # UPSERT profile — now with nominee field
                p = account.get("profile", {})
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
                          p.get("holder_type"), p.get("ckyc") == "true",
                          p.get("nominee")))

                # UPSERT summary — type-aware field mapping
                s = account.get("summary", {})
                if s:
                    is_deposit = acc_type == "DEPOSIT"
                    is_td_rd   = acc_type in ("TERM_DEPOSIT", "RECURRING_DEPOSIT")

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
                        s.get("status"),
                        s.get("openingDate") or s.get("opening_date"),
                        s.get("currency", "INR"),
                        # DEPOSIT fields
                        s.get("currentBalance")  if is_deposit else None,
                        s.get("balanceDateTime") if is_deposit else None,
                        s.get("branch"),
                        s.get("ifscCode") or s.get("ifsc"),
                        s.get("micrCode"),
                        s.get("type") or s.get("accountType"),
                        s.get("currentODLimit")  if is_deposit else None,
                        s.get("drawingLimit")    if is_deposit else None,
                        s.get("facility")        if is_deposit else None,
                        float(pending.get("amount", 0) or 0) if is_deposit else None,
                        pending.get("transactionType")       if is_deposit else None,
                        # TD/RD fields
                        s.get("currentValue")         if is_td_rd else None,
                        s.get("principalAmount")      if is_td_rd else None,
                        s.get("maturityAmount")       if is_td_rd else None,
                        s.get("maturityDate")         if is_td_rd else None,
                        float(s.get("interestRate", 0) or 0) if is_td_rd else None,
                        s.get("interestComputation")  if is_td_rd else None,
                        s.get("interestPayout")       if is_td_rd else None,
                        s.get("compoundingFrequency") if is_td_rd else None,
                        int(s.get("tenureDays", 0) or 0)    if is_td_rd else None,
                        int(s.get("tenureMonths", 0) or 0)  if is_td_rd else None,
                        int(s.get("tenureYears", 0) or 0)   if is_td_rd else None,
                        float(s.get("recurringAmount", 0) or 0) if is_td_rd else None,
                        int(s.get("recurringDepositDay", 0) or 0) if is_td_rd else None,
                        # Credit/Loan (future)
                        s.get("creditLimit"), s.get("currentDues"),
                        s.get("minimumDue"),  s.get("outstandingPrincipal"),
                        s.get("emiAmount"),
                        extras.Json(s)
                    ))

                # INSERT transactions with category classification + parsed dates
                for txn in account.get("transactions", []):
                    setu_id   = txn.get("txn_id")
                    mode      = txn.get("mode", "")
                    txn_type  = txn.get("type", "")
                    narration = txn.get("narration", "")
                    # Balance field name differs by account type
                    balance   = txn.get("currentBalance") or txn.get("balance")
                    cat, sub  = classify_transaction(mode, txn_type, narration)

                    # Parse both timestamp fields — handles all AA date formats
                    txn_date   = _parse_dt(txn.get("transactionTimestamp") or txn.get("date"))
                    value_date = _parse_dt(txn.get("valueDate") or txn.get("date"))
                    # Fallback: use whichever is available
                    if txn_date is None:
                        txn_date = value_date

                    if setu_id:
                        cur.execute("""
                            INSERT INTO transactions(user_id,fi_data_id,setu_txn_id,
                                txn_date,value_date,amount,txn_type,payment_mode,
                                narration,reference,balance_after,category,subcategory)
                            VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        """, (user_id, fi_data_id, setu_id,
                              txn_date, value_date,
                              txn.get("amount"), txn_type, mode,
                              narration, txn.get("reference"), balance, cat, sub))
                    else:
                        cur.execute("""
                            INSERT INTO transactions(user_id,fi_data_id,
                                txn_date,value_date,amount,txn_type,payment_mode,
                                narration,reference,balance_after,category,subcategory)
                            VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        """, (user_id, fi_data_id,
                              txn_date, value_date,
                              txn.get("amount"), txn_type, mode,
                              narration, txn.get("reference"), balance, cat, sub))
                    new_txns += 1

        conn.commit()
        logger.info("✅ Saved user=%s | txns_inserted=%d", user_id, new_txns)

    except psycopg2.Error as e:
        conn.rollback(); logger.error("save_fi_data failed: %s", e); raise
    finally:
        cur.close(); conn.close()


# ── Read helpers ──────────────────────────────────────────────────────────────

def get_user_accounts(user_id: str) -> List[Dict]:
    conn = get_connection()
    cur  = conn.cursor(cursor_factory=extras.RealDictCursor)
    try:
        cur.execute("""
            SELECT
                f.fi_data_id, f.user_id, f.masked_acc_number, f.account_type,
                f.fi_status, f.fip_id, f.session_id, f.consent_id,
                -- Common
                s.account_status, s.opening_date, s.currency, s.branch,
                s.ifsc_code, s.micr_code, s.account_type AS summary_acc_type,
                -- DEPOSIT specific
                s.current_balance, s.balance_datetime,
                s.od_limit, s.drawing_limit, s.facility,
                s.pending_amount, s.pending_txn_type,
                -- TD/RD specific
                s.current_value, s.principal_amount, s.maturity_amount,
                s.maturity_date, s.interest_rate, s.interest_computation,
                s.interest_payout, s.compounding_frequency,
                s.tenure_days, s.tenure_months, s.tenure_years,
                s.recurring_amount, s.recurring_deposit_day,
                -- Profile
                p.holder_name, p.holder_pan, p.holder_mobile,
                p.holder_email, p.nominee
            FROM fi_data f
            LEFT JOIN summaries s ON f.fi_data_id = s.fi_data_id
            LEFT JOIN profiles  p ON f.fi_data_id = p.fi_data_id
            WHERE f.user_id = %s
            ORDER BY f.fi_data_id ASC
        """, (user_id,))
        return [dict(r) for r in cur.fetchall()]
    finally:
        cur.close(); conn.close()


def get_user_transactions(user_id: str, limit: int = None,
                          fi_data_ids: List[int] = None) -> List[Dict]:
    conn = get_connection()
    cur  = conn.cursor(cursor_factory=extras.RealDictCursor)
    try:
        conditions = ["t.user_id = %s"]
        params     = [user_id]
        if fi_data_ids:
            conditions.append("t.fi_data_id = ANY(%s)")
            params.append(fi_data_ids)
        where = "WHERE " + " AND ".join(conditions)
        
        limit_sql = ""
        if limit:
            limit_sql = "LIMIT %s"
            params.append(limit)

        cur.execute(f"""
            SELECT
                t.txn_id, t.fi_data_id, t.setu_txn_id,
                t.txn_date, t.value_date, t.amount,
                t.txn_type, t.payment_mode, t.narration,
                t.reference, t.balance_after,
                t.category, t.subcategory,
                f.masked_acc_number, f.account_type, f.fip_id
            FROM transactions t
            JOIN fi_data f ON t.fi_data_id = f.fi_data_id
            {where}
            ORDER BY t.txn_date DESC NULLS LAST
            {limit_sql}
        """, params)
        return [dict(r) for r in cur.fetchall()]
    finally:
        cur.close(); conn.close()


def get_user_summary(user_id: str) -> Dict:
    conn = get_connection()
    cur  = conn.cursor(cursor_factory=extras.RealDictCursor)
    try:
        cur.execute("""
            SELECT
                COALESCE(SUM(DISTINCT s.current_balance), 0)  AS total_balance,
                COUNT(DISTINCT f.fi_data_id)                   AS account_count,
                COALESCE(SUM(CASE WHEN t.txn_type='CREDIT' THEN t.amount ELSE 0 END),0) AS total_income,
                COALESCE(SUM(CASE WHEN t.txn_type='DEBIT'  THEN t.amount ELSE 0 END),0) AS total_expenses
            FROM fi_data f
            LEFT JOIN summaries    s ON f.fi_data_id = s.fi_data_id
            LEFT JOIN transactions t ON f.fi_data_id = t.fi_data_id
            WHERE f.user_id = %s
        """, (user_id,))
        row = cur.fetchone()
        return dict(row) if row else {}
    finally:
        cur.close(); conn.close()


def get_category_breakdown(user_id: str, fi_data_ids: List[int] = None) -> Dict:
    """Category and mode breakdown for charts/analytics."""
    conn = get_connection()
    cur  = conn.cursor(cursor_factory=extras.RealDictCursor)
    try:
        conditions = ["t.user_id = %s"]
        params     = [user_id]
        if fi_data_ids:
            conditions.append("t.fi_data_id = ANY(%s)")
            params.append(fi_data_ids)
        where = "WHERE " + " AND ".join(conditions)

        cur.execute(f"""
            SELECT
                t.category,
                t.subcategory,
                t.txn_type,
                t.payment_mode,
                COUNT(*)          AS txn_count,
                SUM(t.amount)     AS total_amount,
                SUM(CASE WHEN t.txn_type IN ('CREDIT','INTEREST','OPENING')
                    THEN t.amount ELSE 0 END) AS credit_amount,
                SUM(CASE WHEN t.txn_type IN ('DEBIT','TDS')
                    THEN t.amount ELSE 0 END) AS debit_amount
            FROM transactions t
            {where}
            GROUP BY t.category, t.subcategory, t.txn_type, t.payment_mode
            ORDER BY total_amount DESC
        """, params)
        return {"breakdown": [dict(r) for r in cur.fetchall()]}
    finally:
        cur.close(); conn.close()