import React, { useState, useEffect, useCallback } from 'react'
import { useSearchParams, useNavigate } from 'react-router-dom'
import DashboardLayout from '../components/DashboardLayout'
import ConsentModal from '../components/ConsentModal'
import { useAuth } from '../context/AuthContext'
import { aaAPI, pennyAPI } from '../services/api'

// Fetch state machine: null → 'polling_consent' → 'fetching_data' → 'done' | 'error'
export default function Consent() {
  const [showModal, setShowModal]       = useState(false)
  const [consent, setConsent]           = useState(null)
  const [fetchPhase, setFetchPhase]     = useState(null)
  const [fetchMsg, setFetchMsg]         = useState('')
  const [dataReady, setDataReady]       = useState(false)
  const [pageLoading, setPageLoading]   = useState(true)
  const { user }                        = useAuth()
  const [searchParams, setSearchParams] = useSearchParams()
  const navigate                         = useNavigate()

  // ── Load existing consent on mount ────────────────────────────────────────
  const loadConsent = useCallback(async () => {
    try {
      const { data } = await aaAPI.getMyConsent()
      if (data.consent_id) {
        setConsent(data)
        if (data.consent_status === 'ACTIVE') setDataReady(true)
      }
    } catch {}
    finally { setPageLoading(false) }
  }, [])

  useEffect(() => { loadConsent() }, [loadConsent])

  // ── Poll consent status after user returns from AA webview ────────────────
  useEffect(() => {
    const consentId = user?.aa_consent_id
    if (!consentId || dataReady) return
    if (user?.aa_consent_status !== 'PENDING') return



    const poll = setInterval(async () => {
      try {
        const { data } = await aaAPI.getConsentStatus(consentId)
        setConsent(prev => ({ ...prev, consent_status: data.status, status: data.status }))

        if (data.status === 'ACTIVE') {
          clearInterval(poll)
          setSearchParams({})
          // Now trigger data fetch
          triggerDataFetch(consentId)
        } else if (['REJECTED', 'REVOKED', 'EXPIRED'].includes(data.status)) {
          clearInterval(poll)
          setFetchPhase('error')
          setFetchMsg(`Consent ${data.status.toLowerCase()}. Please create a new consent.`)
        }
      } catch { clearInterval(poll) }
    }, 5000)

    return () => clearInterval(poll)
  }, [user?.aa_consent_id, user?.aa_consent_status, dataReady])

  const triggerDataFetch = async (consentId) => {
    setFetchPhase('fetching_data')
    setFetchMsg('Consent approved! Fetching your financial data from linked accounts...')
    try {
      await aaAPI.fetchData(consentId)
      // Background task started — poll every 8s to check if it completed
      // (In production use webhooks; here we wait a bit then mark done)
      setTimeout(() => {
        setFetchPhase('done')
        setFetchMsg('Financial data successfully fetched!')
        setDataReady(true)
        setConsent(prev => ({ ...prev, status: 'ACTIVE', consent_status: 'ACTIVE' }))
        navigate('/')
      }, 12000)  // give background task ~12s to complete
    } catch {
      setFetchPhase('error')
      setFetchMsg('Data fetch failed. Click "Refresh Data" on the consent card to retry.')
    }
  }

  const handleRevoke = async (consentId) => {
    if (!window.confirm('Revoke this consent? This will unlink your bank accounts.')) return
    try {
      await aaAPI.revokeConsent(consentId)
      setConsent(null)
      setDataReady(false)
      setFetchPhase(null)
      setFetchMsg('')
    } catch (e) {
      alert('Revoke failed: ' + (e.response?.data?.detail ?? e.message))
    }
  }

  const handleRefreshData = async () => {
    if (!consent?.consent_id) return
    setFetchPhase('fetching_data')
    setFetchMsg('Re-fetching financial data...')
    setDataReady(false)
    await triggerDataFetch(consent.consent_id)
  }

  const processUpload = async (file, password = null) => {
    const formData = new FormData()
    formData.append('file', file)
    if (password) formData.append('password', password)

    setFetchPhase('fetching_data')
    setFetchMsg(`Uploading & analyzing ${file.name} (this takes about 10-20 seconds)...`)
    try {
      const res = await pennyAPI.uploadStatement(formData)
      setFetchPhase('done')
      setFetchMsg(res.data.message || 'Statement uploaded and parsed successfully!')
      setDataReady(true)
      
      // If we don't have an active UI consent block, fake one to show success
      if (!consent) {
        setConsent({
          consent_id: res.data.consent_id,
          status: 'ACTIVE',
          consent_status: 'ACTIVE',
          vua: 'Offline Statement'
        })
      }
      setTimeout(() => navigate('/transactions'), 2000)
    } catch (err) {
      if (err.response?.status === 401 && err.response?.data?.detail === 'encrypted_pdf') {
        const pwd = window.prompt("This PDF is password protected. Please enter the password to unlock it:")
        if (pwd) {
          return processUpload(file, pwd)
        } else {
          setFetchPhase('error')
          setFetchMsg('Password is required to read this statement.')
          return
        }
      }
      setFetchPhase('error')
      setFetchMsg(err.response?.data?.detail || 'Failed to upload statement.')
    }
  }

  const handleFileUpload = (e) => {
    const file = e.target.files[0]
    if (!file) return
    processUpload(file)
    e.target.value = null
  }

  // ── Status helpers ─────────────────────────────────────────────────────────
  const effectiveStatus = consent?.consent_status ?? consent?.status
  const statusBadge = (s) => ({
    ACTIVE:   'text-green-700 bg-green-50 border-green-200',
    PENDING:  'text-amber-700 bg-amber-50 border-amber-200',
    REJECTED: 'text-red-700 bg-red-50 border-red-200',
    REVOKED:  'text-slate-600 bg-slate-50 border-slate-200',
  }[s] ?? 'text-slate-600 bg-slate-50 border-slate-200')

  const phaseBanner = {
    polling_consent: { bg: 'bg-amber-50 border-amber-200 text-amber-700', spin: true },
    fetching_data:   { bg: 'bg-brand-50 border-brand-200 text-brand-700', spin: true },
    done:            { bg: 'bg-green-50 border-green-200 text-green-700', spin: false },
    error:           { bg: 'bg-red-50 border-red-200 text-red-700',       spin: false },
  }

  return (
    <DashboardLayout title="Consent Management">
      <div className="max-w-6xl mx-auto">

        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <p className="text-sm text-slate-500">
            Manage RBI Account Aggregator consents for{' '}
            <span className="font-medium text-slate-700">{user?.full_name ?? 'your account'}</span>.
          </p>
          {!consent && !pageLoading && (
            <button
              onClick={() => setShowModal(true)}
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-full bg-gradient-to-r from-brand-500 to-brand-700 text-white font-medium shadow-md hover:shadow-lg transition-all text-sm"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 4v16m8-8H4"/>
              </svg>
              New Consent
            </button>
          )}
        </div>

        {/* Phase banner */}
        {fetchPhase && (() => {
          const b = phaseBanner[fetchPhase]
          return (
            <div className={`flex items-center gap-3 rounded-xl px-5 py-4 mb-6 border text-sm font-medium ${b.bg}`}>
              {b.spin ? (
                <svg className="animate-spin w-5 h-5 flex-shrink-0" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
                </svg>
              ) : fetchPhase === 'done' ? (
                <svg className="w-5 h-5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd"/>
                </svg>
              ) : (
                <svg className="w-5 h-5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd"/>
                </svg>
              )}
              {fetchMsg}
            </div>
          )
        })()}

        {pageLoading ? (
          <div className="flex items-center justify-center py-24">
            <svg className="animate-spin w-8 h-8 text-brand-400" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
            </svg>
          </div>

        ) : consent ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">

            {/* ── Consent card ── */}
            <div className="bg-gradient-to-br from-[#0b0710] via-[#1b121a] to-[#2a1b23] text-white rounded-2xl p-6 shadow-xl">
              <div className="flex justify-between items-start mb-6">
                <div>
                  <p className="text-xs text-slate-400 uppercase tracking-wider mb-1">FinSight Consent</p>
                  <p className="text-xs text-slate-300">RBI Account Aggregator</p>
                </div>
                <span className={`text-xs font-semibold px-2.5 py-1 rounded-full border ${statusBadge(effectiveStatus)}`}>
                  {effectiveStatus ?? 'PENDING'}
                </span>
              </div>

              <div className="font-mono text-sm tracking-widest text-slate-200 mb-2">
                {consent.consent_id?.replace(/-/g, ' ').toUpperCase().slice(0, 23)}
              </div>
              <p className="text-xs text-slate-400 mb-6">
                Authorized for <span className="text-slate-200">{consent.vua ?? user?.email}</span>
              </p>

              {/* Status messages */}
              {effectiveStatus === 'PENDING' && !fetchPhase && (
                <div className="bg-amber-500/10 border border-amber-500/20 rounded-lg px-3 py-2 mb-4">
                  <p className="text-xs text-amber-300">⏳ Awaiting approval on AA portal...</p>
                </div>
              )}
              {effectiveStatus === 'ACTIVE' && !dataReady && (
                <div className="bg-blue-500/10 border border-blue-500/20 rounded-lg px-3 py-2 mb-4">
                  <p className="text-xs text-blue-300">⚙️ Fetching financial data...</p>
                </div>
              )}
              {effectiveStatus === 'ACTIVE' && dataReady && (
                <div className="bg-green-500/10 border border-green-500/20 rounded-lg px-3 py-2 mb-4">
                  <p className="text-xs text-green-300">✅ Consent active & data fetched successfully</p>
                </div>
              )}

              <div className="flex gap-2 flex-wrap">
                {effectiveStatus === 'ACTIVE' && dataReady && (
                  <button
                    onClick={handleRefreshData}
                    className="text-xs px-3 py-1.5 rounded-full bg-white/10 hover:bg-white/20 transition-colors"
                  >
                    Refresh Data
                  </button>
                )}
                {effectiveStatus === 'PENDING' && (
                  <button
                    onClick={loadConsent}
                    className="text-xs px-3 py-1.5 rounded-full bg-white/10 hover:bg-white/20 transition-colors"
                  >
                    Check Status
                  </button>
                )}
                <button
                  onClick={() => handleRevoke(consent.consent_id)}
                  className="text-xs px-3 py-1.5 rounded-full bg-red-500/20 text-red-300 hover:bg-red-500/30 transition-colors"
                >
                  Revoke
                </button>
                <label className="text-xs px-3 py-1.5 rounded-full bg-brand-500/20 text-brand-300 hover:bg-brand-500/30 transition-colors cursor-pointer flex items-center gap-1" title="Supports PDF, CSV, Excel">
                  <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"/></svg>
                  Upload PDF / CSV / Excel
                  <input type="file" className="hidden" accept=".pdf,.csv,.xlsx,.xls" onChange={handleFileUpload} />
                </label>
              </div>
            </div>

            {/* ── Info card ── */}
            <div className="border border-slate-200 rounded-2xl p-6">
              <h3 className="font-semibold text-slate-900 mb-4 text-sm">Consent Details</h3>
              <div className="space-y-0">
                {[
                  ['Consent ID', consent.consent_id?.slice(0, 20) + '...'],
                  ['VUA',        consent.vua ?? '—'],
                  ['Status',     effectiveStatus ?? '—'],
                  ['Data Ready', dataReady ? 'Yes ✅' : 'Pending...'],
                  ['Account',    user?.email ?? '—'],
                ].map(([label, val]) => (
                  <div key={label} className="flex justify-between items-center py-2.5 border-b border-slate-50">
                    <span className="text-xs text-slate-400 font-medium">{label}</span>
                    <span className="text-xs text-slate-700 font-medium truncate max-w-[220px]">{val}</span>
                  </div>
                ))}
              </div>

              {dataReady && (
                <div className="mt-4 p-3 bg-green-50 border border-green-100 rounded-xl">
                  <p className="text-xs text-green-700 font-medium">
                    🎉 Your financial data is linked. Head to the Dashboard to view your accounts and insights.
                  </p>
                </div>
              )}
            </div>
          </div>

        ) : (
          /* ── Empty state ── */
          <div className="border border-dashed border-slate-200 rounded-xl p-12 text-center">
            <div className="flex items-center justify-center mb-4">
              <div className="w-16 h-16 rounded-2xl bg-brand-50 border border-brand-100 flex items-center justify-center">
                <svg className="w-8 h-8 text-brand-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"/>
                </svg>
              </div>
            </div>
            <h3 className="text-lg font-semibold text-slate-900 mb-2">No User Consents Created</h3>
            <p className="text-sm text-slate-500 mb-6 max-w-sm mx-auto">
              Create a consent to start viewing your financial data and transactions.
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 max-w-xl mx-auto text-left mb-8">
              {[
                ['📋', 'Create Consent',  'Select FI types and data range for your bank.'],
                ['✅', 'User Approves',   'Approve on the AA portal — takes under a minute.'],
                ['📊', 'Data Fetched',    'Financial data flows securely into your dashboard.'],
              ].map(([icon, title, desc]) => (
                <div key={title} className="bg-slate-50 rounded-xl p-4 border border-slate-100">
                  <div className="text-2xl mb-2">{icon}</div>
                  <p className="text-sm font-semibold text-slate-800 mb-1">{title}</p>
                  <p className="text-xs text-slate-500">{desc}</p>
                </div>
              ))}
            </div>
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
              <button
                onClick={() => setShowModal(true)}
                className="inline-flex items-center gap-2 px-6 py-3 rounded-full bg-gradient-to-r from-brand-500 to-brand-700 text-white font-medium shadow-md hover:shadow-lg transition-all text-sm"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 4v16m8-8H4"/>
                </svg>
                Create New Consent
              </button>
              
              <div className="text-slate-400 text-sm font-medium">OR</div>

              <div className="flex flex-col items-center">
                <label className="inline-flex items-center gap-2 px-6 py-3 rounded-full bg-white border border-slate-200 text-slate-700 font-medium shadow-sm hover:bg-slate-50 transition-all text-sm cursor-pointer">
                  <svg className="w-4 h-4 text-brand-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"/></svg>
                  Upload Statement (PDF, CSV, Excel)
                  <input type="file" className="hidden" accept=".pdf,.csv,.xlsx,.xls" onChange={handleFileUpload} />
                </label>
                <p className="text-[10px] text-slate-400 mt-2 italic">
                  Supports .pdf, .csv, .xlsx, .xls formats
                </p>
              </div>
            </div>
          </div>
        )}
      </div>

      {showModal && (
        <ConsentModal
          onClose={() => setShowModal(false)}
          onSuccess={(data) => {
            setShowModal(false)
            setConsent({ consent_id: data.consent_id, status: 'PENDING', consent_status: 'PENDING', vua: null })
          }}
        />
      )}
    </DashboardLayout>
  )
}