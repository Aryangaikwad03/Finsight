import React, { useState, useEffect, useMemo } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import DashboardLayout from '../components/DashboardLayout'
import { useAuth } from '../context/AuthContext'
import { aaAPI, goalsAPI } from '../services/api'
import {
  Chart as ChartJS,
  ArcElement,
  Tooltip,
  Legend,
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  LineElement,
  Title,
  Filler,
} from 'chart.js'
import { Doughnut, Bar } from 'react-chartjs-2'

ChartJS.register(
  ArcElement,
  Tooltip,
  Legend,
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  LineElement,
  Title,
  Filler
)

const fmt = (n) => n == null ? '—' : '₹' + Number(n).toLocaleString('en-IN', { maximumFractionDigits: 0 })
const fmtDate = (d) => {
  if (!d) return '—'
  try { return new Date(d).toLocaleDateString('en-IN', { day: '2-digit', month: 'short' }) }
  catch { return d }
}

const CATEGORY_ICONS = {
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

const CHART_COLORS = [
  '#6366f1', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', 
  '#ec4899', '#06b6d4', '#f43f5e', '#14b8a6', '#f97316'
]

export default function Dashboard() {
  const { user, loading: authLoading } = useAuth()
  const navigate = useNavigate()
  const [status, setStatus] = useState('loading')
  const [accounts, setAccounts] = useState([])
  const [allTxns, setAllTxns] = useState([])
  const [dbBreakdown, setDbBreakdown] = useState([])
  const [dbSummary, setDbSummary] = useState({})
  const [goalsSummary, setGoalsSummary] = useState({ total_active: 0, spending_on_track: 0, spending_exceeded: 0 })
  const [userGoals, setUserGoals] = useState([])
  const [selectedAccount, setSelectedAccount] = useState('all')
  const [month, setMonth] = useState(new Date().getMonth() + 1)
  const [year, setYear] = useState(new Date().getFullYear())

  const loadData = async () => {
    setStatus('loading')
    try {
      const res = await aaAPI.getFIData({ month, year })
      setAccounts(res.data.accounts ?? [])
      setAllTxns(res.data.transactions ?? [])
      setDbBreakdown(res.data.breakdown?.breakdown ?? [])
      setDbSummary(res.data.summary ?? {})
      
      const gSum = await goalsAPI.getSummary()
      setGoalsSummary(gSum.data)
      const gList = await goalsAPI.getGoals()
      setUserGoals(gList.data.goals)

      setStatus(res.data.accounts?.length === 0 ? 'empty' : 'ok')
    } catch (e) {
      console.error('Dashboard load failed:', e)
      setStatus('error')
    }
  }

  useEffect(() => {
    if (!authLoading && user) loadData()
  }, [user?.id, authLoading, month, year])

  const handleCategoryClick = (category) => {
    if (selectedAccount === 'all') {
      // Navigate to all transactions for this category
      navigateToTxns(category)
    } else {
      // Navigate to transactions for this category and account
      navigateToTxns(category, selectedAccount)
    }
  }

  const navigateToTxns = (category, fiDataId = null) => {
    navigate('/transactions', { state: { category, fi_data_id: fiDataId } })
  }

  // --- Computed Data ---
  const activeData = useMemo(() => {
    const isAll = selectedAccount === 'all'
    const incomeTypes = ['CREDIT', 'INTEREST', 'OPENING', 'REFUND', 'DEPOSIT', 'INWARD', 'REVERSAL']
    const expenseTypes = ['DEBIT', 'TDS', 'PAYMENT', 'INSTALLMENT', 'WITHDRAWAL', 'OUTWARD', 'FEES', 'CHARGES', 'TAX', 'OTHERS']
    
    // Filter transactions for both summary and trend
    const txnsForView = isAll ? allTxns : allTxns.filter(t => String(t.fi_data_id) === selectedAccount)

    // 1. Calculate Trend (last 6 months)
    const months = {}
    const now = new Date()
    for (let i = 5; i >= 0; i--) {
      const d = new Date(now.getFullYear(), now.getMonth() - i, 1)
      const key = d.toLocaleString('default', { month: 'short' })
      months[key] = { income: 0, expense: 0, byAccount: {} }
    }

    txnsForView.forEach(t => {
      const d = new Date(t.txn_date)
      const key = d.toLocaleString('default', { month: 'short' })
      if (months[key]) {
        const amt = Number(t.amount) || 0
        if (incomeTypes.includes(t.txn_type)) {
          months[key].income += amt
          months[key].byAccount[t.fi_data_id] = (months[key].byAccount[t.fi_data_id] || { inc: 0, exp: 0 })
          months[key].byAccount[t.fi_data_id].inc += amt
        }
        if (expenseTypes.includes(t.txn_type)) {
          months[key].expense += amt
          months[key].byAccount[t.fi_data_id] = (months[key].byAccount[t.fi_data_id] || { inc: 0, exp: 0 })
          months[key].byAccount[t.fi_data_id].exp += amt
        }
      }
    })

    // 2. Calculate Spending by Account (for current month)
    const accSpending = {}
    accounts.forEach(a => {
      accSpending[a.fi_data_id] = { 
        id: a.fi_data_id, 
        name: a.masked_acc_number, 
        type: a.account_type, 
        spent: 0 
      }
    })
    
    allTxns.filter(t => {
      const d = new Date(t.txn_date)
      return (d.getMonth() + 1) === month && d.getFullYear() === year && expenseTypes.includes(t.txn_type)
    }).forEach(t => {
      if (accSpending[t.fi_data_id]) accSpending[t.fi_data_id].spent += Number(t.amount) || 0
    })

    const sortedAccSpending = Object.values(accSpending)
      .filter(a => a.spent > 0)
      .sort((a, b) => b.spent - a.spent)

    if (isAll) {
      const income  = Number(dbSummary.total_income) || 0
      const expense = Number(dbSummary.total_expenses) || 0
      
      const sortedCats = dbBreakdown
        .filter(b => Number(b.spent) > 0)
        .map(b => [b.category, Number(b.spent)])
        .sort((a, b) => b[1] - a[1])

      return {
        balance: Number(dbSummary.total_balance) || 0,
        income,
        expense,
        savings: income - expense,
        savingsRate: income > 0 ? ((income - expense) / income * 100) : 0,
        categories: sortedCats,
        accSpending: sortedAccSpending,
        trend: months,
        recent: allTxns.slice(0, 5)
      }
    }

    // Account-Specific
    const income = txnsForView
      .filter(t => incomeTypes.includes(t.txn_type))
      .filter(t => {
        const d = new Date(t.txn_date)
        return (d.getMonth() + 1) === month && d.getFullYear() === year
      })
      .reduce((s, t) => s + (Number(t.amount) || 0), 0)

    const expense = txnsForView
      .filter(t => expenseTypes.includes(t.txn_type))
      .filter(t => {
        const d = new Date(t.txn_date)
        return (d.getMonth() + 1) === month && d.getFullYear() === year
      })
      .reduce((s, t) => s + (Number(t.amount) || 0), 0)

    const catMap = {}
    txnsForView
      .filter(t => expenseTypes.includes(t.txn_type))
      .filter(t => {
        const d = new Date(t.txn_date)
        return (d.getMonth() + 1) === month && d.getFullYear() === year
      })
      .forEach(t => {
        catMap[t.category] = (catMap[t.category] || 0) + Number(t.amount)
      })
    
    const sortedCats = Object.entries(catMap)
      .sort((a, b) => b[1] - a[1])

    const acc = accounts.find(a => String(a.fi_data_id) === selectedAccount)
    let balance = 0
    if (acc) {
      balance = (Number(acc.current_balance) || 0) + (Number(acc.current_value) || 0)
      if (balance === 0) {
        const sorted = [...txnsForView].sort((a,b) => new Date(b.txn_date) - new Date(a.txn_date))
        balance = sorted.find(t => t.balance_after != null)?.balance_after || 0
      }
    }

    return {
      balance,
      income,
      expense,
      savings: income - expense,
      savingsRate: income > 0 ? ((income - expense) / income * 100) : 0,
      categories: sortedCats,
      accSpending: sortedAccSpending.filter(a => String(a.id) === selectedAccount),
      trend: months,
      recent: txnsForView.slice(0, 5)
    }
  }, [selectedAccount, dbSummary, dbBreakdown, allTxns, month, year, accounts])

  const doughnutData = {
    labels: activeData.categories.map(c => c[0]),
    datasets: [{
      data: activeData.categories.map(c => c[1]),
      backgroundColor: CHART_COLORS,
      borderWidth: 0,
      hoverOffset: 10
    }]
  }

  const trendKeys = Object.keys(activeData.trend)
  // Stacked chart datasets
  const barDatasets = []
  if (selectedAccount === 'all') {
    accounts.forEach((acc, i) => {
      // Income bars
      barDatasets.push({
        label: `${acc.masked_acc_number} (In)`,
        data: trendKeys.map(k => activeData.trend[k].byAccount[acc.fi_data_id]?.inc || 0),
        backgroundColor: `rgba(16, 185, 129, ${1 - i * 0.2})`, // Green shades
        stack: 'Stack 0',
        borderRadius: 4,
      })
      // Expense bars
      barDatasets.push({
        label: `${acc.masked_acc_number} (Out)`,
        data: trendKeys.map(k => activeData.trend[k].byAccount[acc.fi_data_id]?.exp || 0),
        backgroundColor: `rgba(239, 68, 68, ${1 - i * 0.2})`, // Red shades
        stack: 'Stack 1',
        borderRadius: 4,
      })
    })
  } else {
    barDatasets.push({
      label: 'Income',
      data: trendKeys.map(k => activeData.trend[k].income),
      backgroundColor: '#10b981',
      borderRadius: 4,
    })
    barDatasets.push({
      label: 'Expense',
      data: trendKeys.map(k => activeData.trend[k].expense),
      backgroundColor: '#ef4444',
      borderRadius: 4,
    })
  }

  const barData = {
    labels: trendKeys,
    datasets: barDatasets
  }

  if (status === 'loading') return (
    <DashboardLayout title="Insights">
      <div className="flex items-center justify-center py-32">
        <svg className="animate-spin w-8 h-8 text-brand-500" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/></svg>
      </div>
    </DashboardLayout>
  )

  return (
    <DashboardLayout title="Insights">
      <div className="max-w-6xl mx-auto space-y-8">
        
        {/* View Selector */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="flex bg-slate-100 p-1 rounded-xl w-fit">
            <button 
              onClick={() => setSelectedAccount('all')}
              className={`px-4 py-2 text-xs font-bold rounded-lg transition-all ${selectedAccount === 'all' ? 'bg-white text-brand-600 shadow-sm' : 'text-slate-500 hover:text-slate-700'}`}
            >
              Consolidated
            </button>
            <button 
              onClick={() => setSelectedAccount(accounts[0]?.fi_data_id ? String(accounts[0].fi_data_id) : 'all')}
              className={`px-4 py-2 text-xs font-bold rounded-lg transition-all ${selectedAccount !== 'all' ? 'bg-white text-brand-600 shadow-sm' : 'text-slate-500 hover:text-slate-700'}`}
            >
              Granulated
            </button>
          </div>

          {selectedAccount !== 'all' && (
            <select 
              value={selectedAccount}
              onChange={(e) => setSelectedAccount(e.target.value)}
              className="bg-white border border-slate-200 text-xs font-bold text-slate-700 px-3 py-2 rounded-xl focus:outline-none focus:ring-2 focus:ring-brand-500"
            >
              {accounts.map(acc => (
                <option key={acc.fi_data_id} value={acc.fi_data_id}>
                  {acc.masked_acc_number} ({acc.account_type})
                </option>
              ))}
            </select>
          )}

          <div className="flex gap-2">
            <select 
              value={month} 
              onChange={(e) => setMonth(Number(e.target.value))}
              className="bg-white border border-slate-200 text-xs font-bold text-slate-700 px-3 py-2 rounded-xl focus:outline-none focus:ring-2 focus:ring-brand-500"
            >
              {Array.from({length: 12}, (_, i) => (
                <option key={i+1} value={i+1}>{new Date(2000, i).toLocaleString('default', { month: 'long' })}</option>
              ))}
            </select>
            <select 
              value={year} 
              onChange={(e) => setYear(Number(e.target.value))}
              className="bg-white border border-slate-200 text-xs font-bold text-slate-700 px-3 py-2 rounded-xl focus:outline-none focus:ring-2 focus:ring-brand-500"
            >
              {[2024, 2025, 2026].map(y => (
                <option key={y} value={y}>{y}</option>
              ))}
            </select>
          </div>
        </div>

        {/* Header Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white p-6 rounded-2xl border border-slate-100 shadow-sm">
            <p className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-1">Net Worth</p>
            <h2 className="text-2xl font-black text-slate-800 tracking-tight">{fmt(activeData.balance)}</h2>
            <div className="mt-2 flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-brand-500"></span>
              <p className="text-[10px] font-bold text-slate-400">Consolidated across accounts</p>
            </div>
          </div>
          <div className="bg-white p-6 rounded-2xl border border-slate-100 shadow-sm">
            <p className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-1">Monthly Income</p>
            <h2 className="text-2xl font-black text-green-600 tracking-tight">{fmt(activeData.income)}</h2>
            <p className="text-[10px] font-bold text-slate-400 mt-2 italic">Expected inflow for {new Date().toLocaleString('default', { month: 'long' })}</p>
          </div>
          <div className="bg-white p-6 rounded-2xl border border-slate-100 shadow-sm">
            <p className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-1">Monthly Spent</p>
            <h2 className="text-2xl font-black text-red-500 tracking-tight">{fmt(activeData.expense)}</h2>
            <div className="mt-2 h-1 bg-slate-100 rounded-full overflow-hidden">
              <div className="h-full bg-red-400" style={{ width: `${Math.min(100, (activeData.expense/activeData.income)*100)}%` }}></div>
            </div>
          </div>
          <div className="bg-white p-6 rounded-2xl border border-slate-100 shadow-sm">
            <p className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-1">Savings Rate</p>
            <h2 className="text-2xl font-black text-brand-600 tracking-tight">{activeData.savingsRate.toFixed(0)}%</h2>
            <p className="text-[10px] font-bold text-slate-400 mt-2">Target: 30% or more</p>
          </div>
        </div>

        {/* Goals Summary Card */}
        {goalsSummary.total_active > 0 && (
          <div className="bg-brand-600 p-6 rounded-2xl shadow-xl shadow-brand-100 flex flex-col md:flex-row items-center justify-between gap-6 text-white overflow-hidden relative">
            <div className="absolute -right-8 -bottom-8 w-32 h-32 bg-white/10 rounded-full blur-2xl"></div>
            <div className="relative z-10">
              <div className="flex items-center gap-2 mb-2">
                <span className="text-xl">🎯</span>
                <h3 className="text-lg font-black tracking-tight">Active Financial Goals</h3>
              </div>
              <p className="text-sm font-medium text-brand-100">
                You have <span className="text-white font-bold">{goalsSummary.total_active}</span> tracking goals. 
                {goalsSummary.spending_exceeded > 0 ? (
                  <> Stay alert, <span className="text-white font-bold underline decoration-red-400">{goalsSummary.spending_exceeded} limits</span> have been exceeded.</>
                ) : (
                  <> Great job! All your spending limits are currently on track.</>
                )}
              </p>
            </div>
            <Link 
              to="/goals"
              className="relative z-10 bg-white text-brand-600 font-bold px-6 py-2.5 rounded-xl text-xs hover:bg-brand-50 transition-colors shrink-0"
            >
              Analyze Goals →
            </Link>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Categorized Spending */}
          <div className="bg-white p-6 rounded-2xl border border-slate-100 shadow-sm">
            <h3 className="text-sm font-bold text-slate-800 mb-6 flex items-center gap-2">
              <span className="w-1.5 h-4 bg-brand-500 rounded-full"></span>
              Expense Breakdown
            </h3>
            <div className="flex flex-col items-center justify-center mb-6">
              <span className="text-[10px] uppercase tracking-widest text-slate-400 font-black mb-1">Total Monthly Spent</span>
              <span className="text-2xl font-black text-slate-800 tracking-tight leading-none">{fmt(activeData.expense)}</span>
            </div>

            <div className="relative h-48 mb-8 flex justify-center">
              {activeData.categories.length > 0 ? (
                <Doughnut 
                  data={doughnutData} 
                  options={{ 
                    cutout: '75%', 
                    plugins: { legend: { display: false } },
                    maintainAspectRatio: false,
                    onClick: (evt, elements) => {
                      if (elements.length > 0) {
                        const idx = elements[0].index
                        handleCategoryClick(activeData.categories[idx][0])
                      }
                    }
                  }} 
                />
              ) : (
                <div className="text-slate-300 text-sm italic py-12">No data to display</div>
              )}
            </div>
            <div className="mt-6 space-y-2">
              {activeData.categories.map((cat, i) => {
                const goal = userGoals.find(g => g.category === cat[0])
                const isExceeded = goal && (cat[1] > goal.target_amount)
                
                return (
                  <div 
                    key={`${cat[0]}-${i}`} 
                    onClick={() => handleCategoryClick(cat[0])}
                    className="group flex flex-col p-2 rounded-lg hover:bg-slate-50 cursor-pointer transition-colors"
                  >
                    <div className="flex items-center justify-between w-full mb-1">
                      <div className="flex items-center gap-2 overflow-hidden">
                        <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ backgroundColor: CHART_COLORS[i % CHART_COLORS.length] }}></span>
                        <span className="text-xs text-slate-600 truncate">{CATEGORY_ICONS[cat[0]] || '🔹'} {cat[0]}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        {goal && <span className={`text-[9px] font-black uppercase ${isExceeded ? 'text-red-500' : 'text-slate-400'}`}>{isExceeded ? 'Exceeded' : 'Limit set'}</span>}
                        <span className="text-xs font-bold text-slate-700">{fmt(cat[1])}</span>
                      </div>
                    </div>
                    {goal && (
                      <div className="ml-4 h-1 bg-slate-100 rounded-full overflow-hidden">
                        <div 
                          className={`h-full transition-all duration-500 ${isExceeded ? 'bg-red-500' : 'bg-brand-300'}`}
                          style={{ width: `${Math.min(100, (cat[1] / goal.target_amount) * 100)}%` }}
                        ></div>
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          </div>

          {/* Cash Flow Trend */}
          <div className="lg:col-span-2 bg-white p-6 rounded-2xl border border-slate-100 shadow-sm">
            <h3 className="text-sm font-bold text-slate-800 mb-6 flex items-center gap-2">
              <span className="w-1.5 h-4 bg-green-500 rounded-full"></span>
              Monthly Cash Flow Breakdown
            </h3>
            <div className="h-[300px]">
              <Bar 
                data={barData} 
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  scales: {
                    x: { 
                      stacked: selectedAccount === 'all',
                      grid: { display: false } 
                    },
                    y: { 
                      stacked: selectedAccount === 'all',
                      beginAtZero: true,
                      grid: { color: '#f1f5f9' },
                      ticks: { callback: (v) => '₹' + v / 1000 + 'k' }
                    }
                  },
                  plugins: {
                    legend: { 
                      position: 'top', 
                      align: 'start', 
                      labels: { boxWidth: 8, usePointStyle: true, pointStyle: 'circle', font: { size: 10 } } 
                    }
                  }
                }} 
              />
            </div>
          </div>
        </div>

        {/* Recent Activity */}
        <div className="bg-white rounded-2xl border border-slate-100 shadow-sm overflow-hidden">
          <div className="px-6 py-4 border-b border-slate-50 flex items-center justify-between">
            <h3 className="text-sm font-bold text-slate-800">Recent Activity</h3>
            <Link to="/transactions" className="text-xs font-bold text-brand-600 hover:text-brand-700">View All Transactions →</Link>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left">
              <thead>
                <tr className="bg-slate-50/50">
                  <th className="px-6 py-3 text-[10px] font-bold text-slate-400 uppercase tracking-widest">Date</th>
                  <th className="px-6 py-3 text-[10px] font-bold text-slate-400 uppercase tracking-widest">Narration</th>
                  <th className="px-6 py-3 text-[10px] font-black text-slate-400 uppercase tracking-widest text-right">Amount</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-50">
                {activeData.recent.map((t, i) => (
                  <tr key={i} className="hover:bg-slate-50/50 transition-colors">
                    <td className="px-6 py-4 text-xs text-slate-500 whitespace-nowrap">{fmtDate(t.txn_date)}</td>
                    <td className="px-6 py-4">
                      <p className="text-xs font-semibold text-slate-700 truncate max-w-xs">{t.narration}</p>
                      <p className="text-[10px] text-slate-400 mt-0.5">{CATEGORY_ICONS[t.category] || '🔹'} {t.category || 'Other'}</p>
                    </td>
                    <td className={`px-6 py-4 text-sm font-bold text-right whitespace-nowrap ${['CREDIT', 'INTEREST'].includes(t.txn_type) ? 'text-green-600' : 'text-slate-700'}`}>
                      {['CREDIT', 'INTEREST'].includes(t.txn_type) ? '+' : '-'}{fmt(t.amount)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </DashboardLayout>
  )
}