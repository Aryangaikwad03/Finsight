import React, { useState, useEffect, useMemo } from 'react'
import DashboardLayout from '../components/DashboardLayout'
import { useAuth } from '../context/AuthContext'
import { aaAPI } from '../services/api'
import { Link } from 'react-router-dom'

const fmt = (n) => n == null ? '—' : '₹' + Number(n).toLocaleString('en-IN', { maximumFractionDigits: 2 })
const fmtDate = (d) => {
  if (!d) return '—'
  try { return new Date(d).toLocaleDateString('en-IN', { day: '2-digit', month: 'short', year: 'numeric' }) }
  catch { return d }
}
const fmtPct  = (n) => n == null ? '—' : Number(n).toFixed(2) + '%'

// ── Type-aware transaction type badge ──────────────────────────────────────
const txnTypeBadge = (type) => {
  const map = {
    CREDIT:     'bg-green-100 text-green-700',
    DEBIT:      'bg-red-100 text-red-600',
    INTEREST:   'bg-blue-100 text-blue-700',
    TDS:        'bg-orange-100 text-orange-700',
    OPENING:    'bg-purple-100 text-purple-700',
    REDEMPTION: 'bg-cyan-100 text-cyan-700',
    RENEWAL:    'bg-indigo-100 text-indigo-700',
    OTHERS:     'bg-slate-100 text-slate-600',
  }
  return map[type] ?? 'bg-slate-100 text-slate-500'
}

const categoryIcon = (cat) => {
  const icons = {
    'Food & Dining': '🍴',
    'Transportation': '🚗',
    'Shopping & Retail': '🛒',
    'Bills & Utilities': '🧾',
    'Housing & Rent': '🏠',
    'Healthcare & Medical': '🏥',
    'Entertainment & Leisure': '🍿',
    'Travel': '✈️',
    'Education': '🎓',
    'Investments & Savings': '💰',
    'Insurance': '🛡️',
    'Salary & Income': '💵',
    'Transfers': '🔄',
    'Taxes & Government': '🏛️',
    'ATM / Cash Withdrawal': '🏧',
    'Fees & Charges': '⚠️',
    'Donations & Charity': '🎗️',
    'Business / Professional Expenses': '💼',
    'Subscription Services': '🔄',
    'Uncategorized / Unknown': '🔹',
  }
  return icons[cat] ?? '🔹'
}

// ── Account card: shows different fields by type ───────────────────────────
function AccountCard({ acc, isSelected, onToggle, txns }) {
  const type    = (acc.account_type ?? '').toUpperCase()
  const isTD    = type === 'TERM_DEPOSIT'
  const isRD    = type === 'RECURRING_DEPOSIT'
  const isDep   = type === 'DEPOSIT'
  const isTDRD  = isTD || isRD

  const accIncome  = txns.filter(t => ['CREDIT','INTEREST','OPENING'].includes(t.txn_type)).reduce((s,t)=>s+(Number(t.amount)||0),0)
  const accExpense = txns.filter(t => ['DEBIT','TDS'].includes(t.txn_type)).reduce((s,t)=>s+(Number(t.amount)||0),0)

  const typeLabel = { DEPOSIT: 'Savings/Current', TERM_DEPOSIT: 'Term Deposit', RECURRING_DEPOSIT: 'Recurring Deposit' }[type] ?? type

  return (
    <div
      onClick={onToggle}
      className={`rounded-xl p-4 border cursor-pointer transition-all select-none ${
        isSelected
          ? 'border-brand-500 bg-white ring-2 ring-brand-500 shadow-sm'
          : 'border-slate-200 bg-white opacity-50 hover:opacity-70'
      }`}
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-start gap-2">
          <div className={`mt-0.5 w-4 h-4 rounded border-2 flex items-center justify-center flex-shrink-0 transition-all ${
            isSelected ? 'bg-brand-600 border-brand-600' : 'border-slate-300'
          }`}>
            {isSelected && <svg className="w-2.5 h-2.5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="3" d="M5 13l4 4L19 7"/></svg>}
          </div>
          <div>
            <p className="text-xs text-slate-400 uppercase tracking-wide font-medium">{typeLabel}</p>
            <p className="text-sm font-bold text-slate-800 mt-0.5 font-mono tracking-wider">{acc.masked_acc_number ?? '••••'}</p>
          </div>
        </div>
        <span className={`text-xs px-2 py-1 rounded-full font-medium flex-shrink-0 ${
          acc.fi_status === 'ACTIVE' ? 'bg-green-100 text-green-700' : 'bg-slate-100 text-slate-500'
        }`}>{acc.fi_status ?? '—'}</span>
      </div>

      <div className="space-y-1.5">
        {/* DEPOSIT — show current balance */}
        {isDep && acc.current_balance != null && (
          <div className="flex justify-between">
            <span className="text-xs text-slate-400">Balance</span>
            <span className={`text-sm font-bold ${isSelected ? 'text-brand-700' : 'text-slate-400'}`}>{fmt(acc.current_balance)}</span>
          </div>
        )}

        {/* TD/RD — show investment summary */}
        {isTDRD && (
          <>
            {acc.principal_amount != null && (
              <div className="flex justify-between">
                <span className="text-xs text-slate-400">Principal</span>
                <span className="text-xs font-semibold text-slate-700">{fmt(acc.principal_amount)}</span>
              </div>
            )}
            {acc.current_value != null && (
              <div className="flex justify-between">
                <span className="text-xs text-slate-400">Current Value</span>
                <span className={`text-sm font-bold ${isSelected ? 'text-brand-700' : 'text-slate-400'}`}>{fmt(acc.current_value)}</span>
              </div>
            )}
            {acc.maturity_amount != null && (
              <div className="flex justify-between">
                <span className="text-xs text-slate-400">Maturity Value</span>
                <span className="text-xs font-semibold text-green-600">{fmt(acc.maturity_amount)}</span>
              </div>
            )}
            {acc.interest_rate != null && (
              <div className="flex justify-between">
                <span className="text-xs text-slate-400">Interest Rate</span>
                <span className="text-xs font-semibold text-slate-700">{fmtPct(acc.interest_rate)} p.a.</span>
              </div>
            )}
            {acc.maturity_date && (
              <div className="flex justify-between">
                <span className="text-xs text-slate-400">Matures</span>
                <span className="text-xs text-slate-600">{fmtDate(acc.maturity_date)}</span>
              </div>
            )}
            {isRD && acc.recurring_amount != null && (
              <div className="flex justify-between">
                <span className="text-xs text-slate-400">Monthly</span>
                <span className="text-xs font-semibold text-slate-700">{fmt(acc.recurring_amount)}</span>
              </div>
            )}
          </>
        )}

        {/* Common */}
        {acc.holder_name && (
          <div className="flex justify-between">
            <span className="text-xs text-slate-400">Holder</span>
            <span className="text-xs text-slate-600 truncate max-w-[150px]">{acc.holder_name}</span>
          </div>
        )}
        {acc.ifsc_code && (
          <div className="flex justify-between">
            <span className="text-xs text-slate-400">IFSC</span>
            <span className="text-xs font-mono text-slate-600">{acc.ifsc_code}</span>
          </div>
        )}
        {acc.branch && (
          <div className="flex justify-between">
            <span className="text-xs text-slate-400">Branch</span>
            <span className="text-xs text-slate-600 truncate max-w-[130px]">{acc.branch}</span>
          </div>
        )}

        {/* OD info for current accounts */}
        {isDep && acc.facility && (
          <div className="flex justify-between">
            <span className="text-xs text-slate-400">Facility</span>
            <span className="text-xs text-slate-600">{acc.facility} · {fmt(acc.od_limit)}</span>
          </div>
        )}

        {/* Mini stats */}
        <div className="pt-1.5 border-t border-slate-100 grid grid-cols-3 gap-1 text-center">
          <div>
            <p className="text-xs text-slate-400">Txns</p>
            <p className="text-xs font-semibold text-slate-700">{txns.length}</p>
          </div>
          <div>
            <p className="text-xs text-slate-400">In</p>
            <p className="text-xs font-semibold text-green-600">{fmt(accIncome)}</p>
          </div>
          <div>
            <p className="text-xs text-slate-400">Out</p>
            <p className="text-xs font-semibold text-red-500">{fmt(accExpense)}</p>
          </div>
        </div>
      </div>
    </div>
  )
}

// ── Category breakdown bar ─────────────────────────────────────────────────
function CategoryBreakdown({ transactions }) {
  const byCategory = useMemo(() => {
    const map = {}
    transactions.forEach(t => {
      const cat = t.category || 'Other'
      if (!map[cat]) map[cat] = { cat, amount: 0, count: 0 }
      map[cat].amount += Number(t.amount) || 0
      map[cat].count  += 1
    })
    return Object.values(map).sort((a,b) => b.amount - a.amount)
  }, [transactions])

  const total = byCategory.reduce((s,c) => s + c.amount, 0)
  if (!byCategory.length) return null

  const barColors = [
    'bg-brand-500', 'bg-green-500', 'bg-blue-500', 'bg-orange-400',
    'bg-purple-500', 'bg-cyan-500', 'bg-pink-500', 'bg-slate-400',
  ]

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-5">
      <h3 className="text-sm font-semibold text-slate-700 mb-4">Spending by Category</h3>

      {/* Stacked bar */}
      <div className="flex h-3 rounded-full overflow-hidden mb-5 gap-px">
        {byCategory.map((c, i) => (
          <div
            key={c.cat}
            className={`${barColors[i % barColors.length]} transition-all`}
            style={{ width: `${(c.amount / total * 100).toFixed(1)}%`, minWidth: c.amount > 0 ? '2px' : '0' }}
            title={`${c.cat}: ${fmt(c.amount)}`}
          />
        ))}
      </div>

      {/* Legend */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
        {byCategory.map((c, i) => (
          <div key={c.cat} className="flex items-start gap-2">
            <div className={`w-2.5 h-2.5 rounded-full mt-0.5 flex-shrink-0 ${barColors[i % barColors.length]}`}/>
            <div className="min-w-0">
              <p className="text-xs font-medium text-slate-700 truncate">
                {categoryIcon(c.cat)} {c.cat}
              </p>
              <p className="text-xs text-slate-400">{fmt(c.amount)} · {c.count} txns</p>
              <p className="text-xs text-slate-300">{(c.amount / total * 100).toFixed(1)}%</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}


// ── Inline category editor ─────────────────────────────────────────────────
function CategoryEditor({ txn, categories, onSave, onClose }) {
  const allCats = [
    ...categories.builtin.map(c => c.category),
    ...categories.custom.map(c => c.category),
  ]
  const [cat, setCat]     = useState(txn.category || 'Other')
  const [sub, setSub]     = useState(txn.subcategory || '')
  const [newCat, setNewCat] = useState('')
  const [saving, setSaving] = useState(false)
  const [showNew, setShowNew] = useState(false)

  const currentSubs = useMemo(() => {
    const found = [...categories.builtin, ...categories.custom].find(c => c.category === cat)
    return found?.subcategories ?? []
  }, [cat, categories])

  const handleSave = async () => {
    setSaving(true)
    try {
      const finalCat = showNew && newCat.trim() ? newCat.trim() : cat
      await onSave(txn.txn_id, finalCat, sub)
      onClose()
    } catch (e) {
      console.error('category update failed', e)
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ background: 'rgba(15,23,42,0.5)', backdropFilter: 'blur(4px)' }}
      onClick={e => e.target === e.currentTarget && onClose()}>
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-sm p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="font-semibold text-slate-900">Edit Category</h3>
          <button onClick={onClose} className="text-slate-400 hover:text-slate-600">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"/>
            </svg>
          </button>
        </div>

        <p className="text-xs text-slate-500 mb-4 truncate">{txn.narration}</p>

        {/* Category select or new */}
        <div className="space-y-3">
          <div>
            <label className="text-xs font-medium text-slate-600 mb-1 block">Category</label>
            {!showNew ? (
              <select value={cat} onChange={e => { setCat(e.target.value); setSub('') }}
                className="w-full border border-slate-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500">
                {allCats.map(c => <option key={c} value={c}>{categoryIcon(c)} {c}</option>)}
              </select>
            ) : (
              <input value={newCat} onChange={e => setNewCat(e.target.value)}
                placeholder="Enter new category name"
                className="w-full border border-brand-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500"/>
            )}
            <button onClick={() => setShowNew(p => !p)}
              className="mt-1 text-xs text-brand-600 hover:underline">
              {showNew ? '← Choose existing' : '+ Create new category'}
            </button>
          </div>

          {/* Subcategory */}
          {!showNew && currentSubs.length > 0 && (
            <div>
              <label className="text-xs font-medium text-slate-600 mb-1 block">Subcategory</label>
              <select value={sub} onChange={e => setSub(e.target.value)}
                className="w-full border border-slate-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500">
                <option value="">— none —</option>
                {currentSubs.map(s => <option key={s} value={s}>{s}</option>)}
              </select>
            </div>
          )}

          <div className="flex gap-2 pt-2">
            <button onClick={onClose}
              className="flex-1 px-4 py-2 text-sm border border-slate-200 rounded-lg hover:bg-slate-50">
              Cancel
            </button>
            <button onClick={handleSave} disabled={saving}
              className="flex-1 px-4 py-2 text-sm bg-brand-600 text-white rounded-lg hover:bg-brand-700 disabled:opacity-50">
              {saving ? 'Saving…' : 'Save'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

// ── Main page ──────────────────────────────────────────────────────────────
export default function Transactions() {
  const { user, loading: authLoading } = useAuth()
  const [status, setStatus]           = useState('loading')
  const [accounts, setAccounts]       = useState([])
  const [allTxns, setAllTxns]         = useState([])
  const [selectedIds, setSelectedIds] = useState(new Set())
  const [typeFilter, setTypeFilter]   = useState('all')
  const [search, setSearch]           = useState('')
  const [fromDate, setFromDate]       = useState('')
  const [toDate, setToDate]           = useState('')
  const [currentPage, setCurrentPage] = useState(1)
  const pageSize = 50
  const [activeTab, setActiveTab]     = useState('transactions')
  const [categories, setCategories]   = useState({ builtin: [], custom: [] })
  const [editingTxn, setEditingTxn]   = useState(null)  // txn being edited

  const loadData = async () => {
    setStatus('loading')
    try {
      const [fiRes, catRes] = await Promise.all([
        aaAPI.getFIData(),
        aaAPI.getCategories(),
      ])
      const accs = fiRes.data.accounts ?? []
      const txns = fiRes.data.transactions ?? []
      setAccounts(accs)
      setAllTxns(txns)
      setCategories(catRes.data)
      setSelectedIds(new Set(accs.map(a => a.fi_data_id)))
      setStatus(accs.length === 0 ? 'empty' : 'ok')
    } catch (e) {
      console.error('fi-data error:', e.response?.status, e.response?.data ?? e.message)
      setStatus('error')
    }
  }

  useEffect(() => {
    if (authLoading || !user) return
    loadData()
  }, [user?.id, authLoading])

  const handleCategoryUpdate = async (txnId, newCat, newSub) => {
    await aaAPI.updateCategory({ txn_id: txnId, category: newCat, subcategory: newSub })
    // Update local state instantly — no reload needed
    setAllTxns(prev => prev.map(t =>
      t.txn_id === txnId ? { ...t, category: newCat, subcategory: newSub } : t
    ))
  }

  const toggleAccount = (id) => {
    setSelectedIds(prev => {
      const next = new Set(prev)
      if (next.has(id)) { if (next.size === 1) return prev; next.delete(id) }
      else next.add(id)
      return next
    })
  }

  const activeTxns = useMemo(() =>
    allTxns.filter(t => selectedIds.has(t.fi_data_id)), [allTxns, selectedIds])

  const filtered = useMemo(() => activeTxns.filter(t => {
    // Note: txn_date is ISO format, so string comparison works securely
    const tDate = t.txn_date ? t.txn_date.slice(0, 10) : ''
    if (fromDate && tDate < fromDate) return false
    if (toDate && tDate > toDate) return false
    if (typeFilter !== 'all' && t.txn_type !== typeFilter) return false
    const q = search.toLowerCase()
    return !q ||
      (t.narration    ?? '').toLowerCase().includes(q) ||
      (t.payment_mode ?? '').toLowerCase().includes(q) ||
      (t.category     ?? '').toLowerCase().includes(q) ||
      String(t.amount ?? '').includes(q)
  }), [activeTxns, typeFilter, search, fromDate, toDate])

  // Reset pagination when any filter changes
  useEffect(() => setCurrentPage(1), [search, typeFilter, fromDate, toDate, selectedIds])
  
  const paginated = useMemo(() => filtered.slice((currentPage - 1) * pageSize, currentPage * pageSize), [filtered, currentPage])
  const maxPage = Math.max(1, Math.ceil(filtered.length / pageSize))

  const dynamicSummary = useMemo(() => {
    const income   = filtered.filter(t => ['CREDIT','INTEREST','OPENING'].includes(t.txn_type)).reduce((s,t)=>s+(Number(t.amount)||0),0)
    const expenses = filtered.filter(t => ['DEBIT','TDS'].includes(t.txn_type)).reduce((s,t)=>s+(Number(t.amount)||0),0)

    return {
      // Total balance = net flow (income - expenses) from selected accounts
      total_balance:  income - expenses,
      account_count:  selectedIds.size,
      total_income:   income,
      total_expenses: expenses,
    }
  }, [filtered, selectedIds])

  // All unique txn types present
  const txnTypes = useMemo(() => [...new Set(allTxns.map(t => t.txn_type))].filter(Boolean), [allTxns])

  if (status === 'loading') return (
    <DashboardLayout title="Transactions">
      <div className="flex items-center justify-center py-32">
        <svg className="animate-spin w-8 h-8 text-brand-400" fill="none" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
        </svg>
      </div>
    </DashboardLayout>
  )
  if (status === 'error') return (
    <DashboardLayout title="Transactions">
      <div className="flex flex-col items-center justify-center py-32 text-center">
        <h3 className="text-lg font-semibold text-slate-900 mb-2">Could Not Load Data</h3>
        <div className="flex gap-3 mt-4">
          <button onClick={loadData} className="px-4 py-2 text-sm border rounded-lg hover:bg-slate-50">Retry</button>
          <Link to="/consent" className="px-5 py-2.5 rounded-full bg-gradient-to-r from-brand-500 to-brand-700 text-white text-sm">Go to Consent</Link>
        </div>
      </div>
    </DashboardLayout>
  )
  if (status === 'empty') return (
    <DashboardLayout title="Transactions">
      <div className="flex flex-col items-center justify-center py-32 text-center">
        <h3 className="text-lg font-semibold text-slate-900 mb-2">No Data Yet</h3>
        <Link to="/consent" className="mt-4 px-5 py-2.5 rounded-full bg-gradient-to-r from-brand-500 to-brand-700 text-white text-sm">Create Consent</Link>
      </div>
    </DashboardLayout>
  )

  return (
    <DashboardLayout title="Transactions">
      <div className="max-w-6xl mx-auto space-y-6">

        {/* Dynamic Summary */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          {[
            { label: 'Total Balance',  value: fmt(dynamicSummary.total_balance),
              sub: `${dynamicSummary.account_count} account${dynamicSummary.account_count!==1?'s':''} selected`,
              color: 'text-brand-600', bg: 'bg-brand-50', border: 'border-brand-100' },
            { label: 'Transactions',  value: filtered.length,
              sub: `Out of ${activeTxns.length} total`,
              color: 'text-slate-700', bg: 'bg-slate-50', border: 'border-slate-100' },
            { label: 'Net Income',  value: fmt(dynamicSummary.total_income),
              sub: `${filtered.filter(t=>['CREDIT','INTEREST'].includes(t.txn_type)).length} credits`,
              color: 'text-green-600', bg: 'bg-green-50', border: 'border-green-100' },
            { label: 'Net Expenses',value: fmt(dynamicSummary.total_expenses),
              sub: `${filtered.filter(t=>['DEBIT','TDS'].includes(t.txn_type)).length} debits`,
              color: 'text-red-500',   bg: 'bg-red-50',   border: 'border-red-100' },
          ].map(({ label, value, sub, color, bg, border }) => (
            <div key={label} className={`${bg} rounded-xl p-4 border ${border}`}>
              <p className="text-xs text-slate-500 mb-1">{label}</p>
              <p className={`text-lg font-bold ${color}`}>{value}</p>
              <p className="text-xs text-slate-400 mt-1">{sub}</p>
            </div>
          ))}
        </div>

        {/* Account multi-select */}
        <div>
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-sm font-semibold text-slate-700">
              Your Accounts
              <span className="ml-2 text-xs font-normal text-slate-400">click to include/exclude</span>
            </h2>
            {selectedIds.size < accounts.length && (
              <button onClick={() => setSelectedIds(new Set(accounts.map(a=>a.fi_data_id)))}
                className="text-xs text-brand-600 hover:underline">Select all</button>
            )}
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {accounts.map(acc => (
              <AccountCard
                key={acc.fi_data_id}
                acc={acc}
                isSelected={selectedIds.has(acc.fi_data_id)}
                onToggle={() => toggleAccount(acc.fi_data_id)}
                txns={allTxns.filter(t => t.fi_data_id === acc.fi_data_id)}
              />
            ))}
          </div>
        </div>

        {/* Tabs: Transactions | Analytics */}
        <div className="border-b border-slate-200">
          <div className="flex gap-6">
            {['transactions','analytics'].map(tab => (
              <button key={tab} onClick={() => setActiveTab(tab)}
                className={`pb-2 text-sm font-medium border-b-2 transition-colors capitalize ${
                  activeTab === tab
                    ? 'border-brand-500 text-brand-600'
                    : 'border-transparent text-slate-500 hover:text-slate-700'
                }`}>
                {tab === 'transactions' ? `Transactions (${filtered.length})` : 'Analytics'}
              </button>
            ))}
          </div>
        </div>

        {activeTab === 'analytics' && (
          <CategoryBreakdown transactions={filtered} />
        )}

        {activeTab === 'transactions' && (
          <div>
            {/* Filters */}
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
              <div className="flex flex-wrap gap-2 items-center">
                {/* Type filter — shows all types present in data */}
                <button onClick={() => setTypeFilter('all')}
                  className={`text-xs px-3 py-1.5 rounded-full border font-medium ${
                    typeFilter === 'all' ? 'bg-brand-100 border-brand-300 text-brand-700' : 'border-slate-200 text-slate-500 bg-white'
                  }`}>All</button>
                {txnTypes.map(t => (
                  <button key={t} onClick={() => setTypeFilter(t)}
                    className={`text-xs px-3 py-1.5 rounded-full border font-medium transition-colors ${
                      typeFilter === t
                        ? `${txnTypeBadge(t)} border-current`
                        : 'border-slate-200 text-slate-500 bg-white hover:border-slate-300'
                    }`}>
                    {t}
                  </button>
                ))}
              </div>
              <div className="flex gap-2 flex-wrap items-center">
                <input type="date" value={fromDate} onChange={e => setFromDate(e.target.value)} title="From Date" className="py-1.5 px-3 text-xs border border-slate-200 rounded-full text-slate-500 max-w-[120px]"/>
                <span className="text-xs text-slate-400">to</span>
                <input type="date" value={toDate} onChange={e => setToDate(e.target.value)} title="To Date" className="py-1.5 px-3 text-xs border border-slate-200 rounded-full text-slate-500 max-w-[120px]"/>
               
                <div className="relative">
                  <svg className="w-3.5 h-3.5 text-slate-400 absolute left-2.5 top-1/2 -translate-y-1/2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/>
                  </svg>
                  <input value={search} onChange={e => setSearch(e.target.value)}
                    placeholder="Search narations..."
                    className="pl-8 pr-3 py-1.5 text-xs border border-slate-200 rounded-full focus:outline-none focus:ring-2 focus:ring-brand-500 w-36 sm:w-48"/>
                </div>
              </div>
            </div>

            {filtered.length === 0 ? (
              <div className="text-center py-12 border border-dashed border-slate-200 rounded-xl">
                <p className="text-slate-400 text-sm">No transactions match your selection.</p>
              </div>
            ) : (
              <div className="border border-slate-200 rounded-xl overflow-hidden shadow-sm">
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="bg-slate-50 border-b border-slate-200">
                        <th className="text-left px-4 py-3 text-xs font-semibold text-slate-500 whitespace-nowrap">Date</th>
                        <th className="text-left px-4 py-3 text-xs font-semibold text-slate-500">Narration</th>
                        <th className="text-left px-4 py-3 text-xs font-semibold text-slate-500 hidden sm:table-cell">Type / Mode</th>
                        <th className="text-left px-4 py-3 text-xs font-semibold text-slate-500 hidden md:table-cell">Category</th>
                        <th className="text-left px-4 py-3 text-xs font-semibold text-slate-500 hidden md:table-cell">Account</th>
                        <th className="text-right px-4 py-3 text-xs font-semibold text-slate-500">Amount</th>
                        <th className="text-right px-4 py-3 text-xs font-semibold text-slate-500 hidden lg:table-cell">Balance</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                      {paginated.map((txn, i) => (
                        <tr key={txn.txn_id ?? i} className="hover:bg-slate-50 transition-colors">
                          <td className="px-4 py-3 text-xs text-slate-500 whitespace-nowrap">
                            <p>{fmtDate(txn.txn_date)}</p>
                            {txn.value_date && txn.value_date !== txn.txn_date && (
                              <p className="text-slate-300 text-xs">Val: {fmtDate(txn.value_date)}</p>
                            )}
                          </td>
                          <td className="px-4 py-3 max-w-[200px]">
                            <p className="text-xs text-slate-800 truncate">{txn.narration ?? '—'}</p>
                            {txn.reference && <p className="text-xs text-slate-400 font-mono truncate mt-0.5">{txn.reference}</p>}
                          </td>
                          <td className="px-4 py-3 hidden sm:table-cell">
                            <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${txnTypeBadge(txn.txn_type)}`}>
                              {txn.txn_type}
                            </span>
                            {txn.payment_mode && (
                              <p className="text-xs text-slate-400 mt-1">{txn.payment_mode}</p>
                            )}
                          </td>
                          <td className="px-4 py-3 hidden md:table-cell">
                            <button
                              onClick={e => { e.stopPropagation(); setEditingTxn(txn) }}
                              className="group flex items-start gap-1 text-left hover:bg-slate-100 rounded-lg px-1 py-0.5 -mx-1 transition-colors"
                              title="Click to change category"
                            >
                              <div>
                                <p className="text-xs text-slate-600 group-hover:text-brand-600 transition-colors">
                                  {categoryIcon(txn.category)} {txn.category ?? '—'}
                                </p>
                                <p className="text-xs text-slate-400 mt-0.5">{txn.subcategory ?? ''}</p>
                              </div>
                              <svg className="w-3 h-3 text-slate-300 group-hover:text-brand-400 flex-shrink-0 mt-0.5 opacity-0 group-hover:opacity-100 transition-opacity" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"/>
                              </svg>
                            </button>
                          </td>
                          <td className="px-4 py-3 hidden md:table-cell">
                            <p className="text-xs font-mono text-slate-500">{txn.masked_acc_number ?? '—'}</p>
                            <p className="text-xs text-slate-400 mt-0.5">{txn.account_type ?? ''}</p>
                          </td>
                          <td className="px-4 py-3 text-right whitespace-nowrap">
                            <span className={`text-sm font-bold ${
                              ['CREDIT','INTEREST','OPENING'].includes(txn.txn_type) ? 'text-green-600' : 'text-red-500'
                            }`}>
                              {['CREDIT','INTEREST','OPENING'].includes(txn.txn_type) ? '+' : '−'}{fmt(txn.amount)}
                            </span>
                          </td>
                          <td className="px-4 py-3 text-right hidden lg:table-cell">
                            <span className="text-xs text-slate-500">{fmt(txn.balance_after)}</span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div className="px-5 py-4 bg-slate-50 border-t border-slate-200 flex flex-col sm:flex-row justify-between items-center gap-4">
                  <div className="flex items-center gap-3">
                    <p className="text-xs font-semibold text-slate-600">{filtered.length} total txns</p>
                    <span className="text-slate-300">|</span>
                    <p className="text-xs text-slate-500">
                      Net Flow: <span className={`font-bold ml-1 ${
                        filtered.reduce((s,t)=>s+(['CREDIT','INTEREST','OPENING'].includes(t.txn_type)?1:-1)*(Number(t.amount)||0),0)>=0
                          ?'text-green-600':'text-red-500'
                      }`}>
                        {fmt(Math.abs(filtered.reduce((s,t)=>s+(['CREDIT','INTEREST','OPENING'].includes(t.txn_type)?1:-1)*(Number(t.amount)||0),0)))}
                      </span>
                    </p>
                  </div>
                  
                  <div className="flex items-center gap-3">
                    <button 
                      onClick={() => setCurrentPage(p => Math.max(1, p - 1))} 
                      disabled={currentPage === 1}
                      className="text-xs font-medium px-4 py-2 bg-white border border-slate-200 rounded-full disabled:opacity-40 hover:bg-slate-100 transition-colors"
                    >
                      Previous
                    </button>
                    <span className="text-xs font-medium text-slate-500">Page {currentPage} of {maxPage}</span>
                    <button 
                      onClick={() => setCurrentPage(p => Math.min(maxPage, p + 1))} 
                      disabled={currentPage === maxPage}
                      className="text-xs font-medium px-4 py-2 bg-white border border-slate-200 rounded-full disabled:opacity-40 hover:bg-slate-100 transition-colors"
                    >
                      Next
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Category editor modal */}
      {editingTxn && (
        <CategoryEditor
          txn={editingTxn}
          categories={categories}
          onSave={handleCategoryUpdate}
          onClose={() => setEditingTxn(null)}
        />
      )}
    </DashboardLayout>
  )
}