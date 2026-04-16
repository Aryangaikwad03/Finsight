"""
fi_parser.py – Utilities to extract structured data from the raw Setu AA response.

Parses all supported FI types into clean Python dicts.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("setu_aa.parser")


def parse_session_response(session_resp: Dict) -> Dict:
    """
    Top-level parser for a GET /sessions/:id response.
    Returns a normalised dict with per-FIP, per-account data.
    """
    result = {
        "session_id":   session_resp.get("id"),
        "consent_id":   session_resp.get("consentId"),
        "status":       session_resp.get("status"),
        "format":       session_resp.get("format"),
        "data_range":   session_resp.get("dataRange"),
        "fips":         [],
    }

    for fip_block in (session_resp.get("fips") or []):
        fip_entry = {
            "fip_id":   fip_block.get("fipID"),
            "accounts": [],
        }
        for acc_block in (fip_block.get("accounts") or []):
            fip_entry["accounts"].append(_parse_account(acc_block))
        result["fips"].append(fip_entry)

    return result


def _parse_account(acc_block: Dict) -> Dict:
    """Parse a single account block into a structured dict."""
    raw_fi   = acc_block.get("data", {}) or {}
    account  = raw_fi.get("account", raw_fi)   # some payloads nest under "account"
    fi_type  = account.get("type", "unknown").upper()

    parsed = {
        "link_ref":     acc_block.get("linkRefNumber"),
        "masked_acc":   acc_block.get("maskedAccNumber"),
        "fi_status":    acc_block.get("status"),
        "fi_type":      fi_type,
        "profile":      _extract_profile(account),
        "summary":      _extract_summary(account, fi_type),
        "transactions": _extract_transactions(account, fi_type),
    }
    return parsed


# ── Profile ───────────────────────────────────────────────────────────────────

def _extract_profile(account: Dict) -> Dict:
    profile_raw = account.get("profile", {})
    holders     = profile_raw.get("holders", {})
    holder_raw  = holders.get("holder", {})

    if isinstance(holder_raw, list):
        holder_raw = holder_raw[0] if holder_raw else {}

    return {
        "name":       holder_raw.get("name"),
        "mobile":     holder_raw.get("mobile"),
        "email":      holder_raw.get("email"),
        "dob":        holder_raw.get("dob"),
        "pan":        holder_raw.get("pan"),
        "address":    holder_raw.get("address"),
        "nominee":    holder_raw.get("nominee"),
        "ckyc":       holder_raw.get("ckycCompliance"),
        "holder_type": holders.get("type"),
    }


# ── Summary (type-aware) ──────────────────────────────────────────────────────

def _extract_summary(account: Dict, fi_type: str) -> Dict:
    summary = account.get("summary", {})
    if not summary:
        return {}

    base = {
        "status":        summary.get("status"),
        "opening_date":  summary.get("openingDate"),
        "currency":      summary.get("currency", "INR"),
    }

    if fi_type in ("DEPOSIT", "SAVINGS", "CURRENT", "OVERDRAFT"):
        base.update({
            "current_balance":  summary.get("currentBalance"),
            "balance_datetime": summary.get("balanceDateTime"),
            "branch":           summary.get("branch"),
            "ifsc":             summary.get("ifscCode"),
            "micr":             summary.get("micrCode"),
            "account_type":     summary.get("type"),
            "od_limit":         summary.get("currentODLimit"),
            "drawing_limit":    summary.get("drawingLimit"),
            "facility":         summary.get("facility"),
        })
    elif fi_type in ("TERM_DEPOSIT", "RECURRING_DEPOSIT"):
        base.update({
            "principal":        summary.get("principal"),
            "maturity_amount":  summary.get("maturityAmount"),
            "maturity_date":    summary.get("maturityDate"),
            "interest_rate":    summary.get("interestRate"),
            "tenor_days":       summary.get("tenorDays"),
        })
    elif fi_type in ("MUTUAL_FUNDS", "ETF"):
        base.update({
            "current_value":   summary.get("currentValue"),
            "invested_value":  summary.get("investedValue"),
            "nav":             summary.get("nav"),
            "scheme_name":     summary.get("schemeName"),
            "folio_number":    summary.get("folioNumber"),
        })
    elif fi_type == "EQUITIES":
        base.update({
            "current_value":   summary.get("currentValue"),
            "invested_value":  summary.get("investedValue"),
            "demat_id":        summary.get("dematId"),
        })
    elif fi_type == "CREDIT_CARD":
        base.update({
            "credit_limit":   summary.get("creditLimit"),
            "current_dues":   summary.get("currentDues"),
            "minimum_due":    summary.get("minimumDue"),
            "due_date":       summary.get("dueDate"),
        })
    elif fi_type == "LOAN":
        base.update({
            "outstanding":      summary.get("outstandingPrincipal"),
            "emi":              summary.get("emiAmount"),
            "next_emi_date":    summary.get("nextEmiDate"),
            "loan_start_date":  summary.get("disbursalDate"),
            "maturity_date":    summary.get("maturityDate"),
            "interest_rate":    summary.get("interestRate"),
        })
    elif fi_type in ("GST_GSTR1", "GST_GSTR2A", "GST_GSTR3B"):
        base.update({
            "gstin":              summary.get("gstin"),
            "return_period":      summary.get("retPrd"),
            "filing_date":        summary.get("filingDate"),
            "total_tax_payable":  summary.get("totalTaxPayable"),
            "total_igst":         summary.get("totalIGST"),
            "total_cgst":         summary.get("totalCGST"),
            "total_sgst":         summary.get("totalSGST"),
        })
    elif fi_type == "INSURANCE_POLICIES":
        base.update({
            "sum_assured":     summary.get("sumAssured"),
            "premium":         summary.get("premiumAmount"),
            "policy_number":   summary.get("policyNumber"),
            "maturity_date":   summary.get("maturityDate"),
            "policy_type":     summary.get("policyType"),
        })
    elif fi_type == "NPS":
        base.update({
            "pran":              summary.get("pran"),
            "tier1_balance":     summary.get("tier1Balance"),
            "tier2_balance":     summary.get("tier2Balance"),
            "nav":               summary.get("nav"),
        })
    elif fi_type == "EPFO":
        base.update({
            "uan":               summary.get("uan"),
            "employee_balance":  summary.get("employeeBalance"),
            "employer_balance":  summary.get("employerBalance"),
            "total_balance":     summary.get("totalBalance"),
        })
    elif fi_type in ("AIS", "TIS"):
        base.update({
            "pan":              summary.get("pan"),
            "gross_income":     summary.get("grossIncome"),
            "total_tax_paid":   summary.get("totalTaxPaid"),
            "assessment_year":  summary.get("assessmentYear"),
        })

    return base


# ── Transactions ──────────────────────────────────────────────────────────────

def _extract_transactions(account: Dict, fi_type: str) -> List[Dict]:
    txn_block = account.get("transactions", {})
    if not txn_block:
        return []

    raw_txns = txn_block.get("transaction", [])
    if not isinstance(raw_txns, list):
        raw_txns = [raw_txns]

    txns = []
    for t in raw_txns:
        txn: Dict[str, Any] = {
            "txn_id":    t.get("txnId"),
            "date":      t.get("transactionTimestamp") or t.get("valueDate"),
            "amount":    t.get("amount"),
            "type":      t.get("type"),      # CREDIT / DEBIT
            "mode":      t.get("mode"),      # UPI / NEFT / IMPS …
            "narration": t.get("narration"),
            "reference": t.get("reference"),
            "balance":   t.get("balance") or t.get("currentBalance"),
        }
        # Extra fields for equities / MF
        if fi_type == "EQUITIES":
            txn.update({"isin": t.get("isin"), "units": t.get("units"), "price": t.get("price")})
        elif fi_type in ("MUTUAL_FUNDS", "ETF", "SIP"):
            txn.update({"isin": t.get("isin"), "units": t.get("units"), "nav": t.get("nav")})
        elif fi_type == "INSURANCE_POLICIES":
            txn.update({"premium_type": t.get("premiumType"), "policy_number": t.get("policyNumber")})

        txns.append(txn)

    return txns


# ── Pretty-print summary ──────────────────────────────────────────────────────

def summarise(parsed: Dict) -> None:
    """Log a human-readable summary of the parsed session."""
    print(f"\n{'='*60}")
    print(f" Session : {parsed['session_id']}")
    print(f" Consent : {parsed['consent_id']}")
    print(f" Status  : {parsed['status']}")
    print(f"{'='*60}")
    for fip in parsed["fips"]:
        print(f"\n  FIP: {fip['fip_id']}")
        for acc in fip["accounts"]:
            print(f"    ├─ Acc   : {acc['masked_acc']}  [{acc['fi_type']}]  status={acc['fi_status']}")
            p = acc["profile"]
            print(f"    ├─ Name  : {p.get('name')}  PAN={p.get('pan')}")
            s = acc["summary"]
            if s.get("current_balance"):
                print(f"    ├─ Bal   : ₹{s['current_balance']}")
            print(f"    └─ Txns  : {len(acc['transactions'])} records")
    print()
