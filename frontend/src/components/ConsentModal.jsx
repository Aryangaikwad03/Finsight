import { useState } from 'react'
import { aaAPI } from '../services/api'




export default function ConsentModal({ onClose, onSuccess }) {
  const [step, setStep]       = useState(1)   // 1=form, 2=loading, 3=redirecting
  const [mobile, setMobile]   = useState('')

  const [error, setError]     = useState('')

  const handleCreate = async () => {
    setError('')
    const digits = mobile.replace(/\D/g, '')
    if (digits.length < 10) { setError('Enter a valid 10-digit mobile number.'); return }

    setStep(2)
    try {
      const { data } = await aaAPI.createConsent({ mobile: digits, aa_handle: 'onemoney', preset: 'banking' })

      // Step 3 — show redirect message briefly then open AA webview
      setStep(3)
      setTimeout(() => {
        // Redirect current tab to Setu AA webview
        // Setu will redirect back to /consent on completion
        window.location.href = data.webview_url
      }, 1500)

    } catch (err) {
      const msg = err.response?.data?.detail
      setError(typeof msg === 'string' ? msg : 'Failed to create consent. Please try again.')
      setStep(1)
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ background: 'rgba(15,23,42,0.6)', backdropFilter: 'blur(8px)' }}
      onClick={e => e.target === e.currentTarget && onClose()}
    >
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-lg overflow-hidden">

        {/* Header */}
        <div className="bg-gradient-to-r from-brand-600 to-purple-600 px-8 pt-7 pb-6 text-white relative">
          <button
            onClick={onClose}
            className="absolute top-4 right-4 w-7 h-7 rounded-full bg-white/20 hover:bg-white/30 flex items-center justify-center"
          >
            <svg className="w-3.5 h-3.5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"/>
            </svg>
          </button>
          <div className="flex items-center gap-3 mb-3">
            <div className="w-9 h-9 rounded-lg bg-white/20 flex items-center justify-center">
              <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"/>
              </svg>
            </div>
            <span className="font-bold text-lg">New AA Consent</span>
          </div>
          <p className="text-white/75 text-sm">
            Link your bank accounts via RBI's Account Aggregator framework.
          </p>
        </div>

        {/* Body */}
        <div className="px-8 py-6">

          {/* Step 1 — Form */}
          {step === 1 && (
            <div className="space-y-5">

              {error && (
                <div className="flex items-center gap-2 bg-red-50 border border-red-200 rounded-xl px-4 py-3 text-sm text-red-700">
                  <svg className="w-4 h-4 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd"/>
                  </svg>
                  {error}
                </div>
              )}

              {/* Mobile number */}
              <div>
                <label className="block text-sm font-semibold text-slate-700 mb-2">
                  Mobile Number
                  <span className="text-slate-400 font-normal ml-1">(linked to your bank)</span>
                </label>
                <div className="flex">
                  <div className="flex items-center px-3 bg-slate-50 border border-r-0 border-slate-200 rounded-l-xl text-sm text-slate-500 font-medium">
                    +91
                  </div>
                  <input
                    type="tel"
                    placeholder="1234567890@onemoney"
                    value={mobile}
                    onChange={e => { setMobile(e.target.value); setError('') }}
                    maxLength={20}
                    className="flex-1 border border-slate-200 rounded-r-xl px-4 py-3 text-sm text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-brand-500 focus:border-transparent"
                  />
                </div>
                <p className="text-xs text-slate-400 mt-1.5 mb-4">
                  This should be the mobile number registered with your bank.
                </p>

                {/* Visual indicator for OneMoney */}
                <div>
                  <label className="block text-sm font-semibold text-slate-700 mb-2">Account Aggregator</label>
                  <div className="p-3 rounded-xl border border-brand-500 bg-brand-50 ring-1 ring-brand-500 text-left">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm font-semibold text-brand-700">OneMoney</p>
                        <p className="text-xs text-brand-600/80 mt-0.5">Selected provider</p>
                      </div>
                      <svg className="w-5 h-5 text-brand-600" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd"/>
                      </svg>
                    </div>
                  </div>
                </div>
              </div>

              <button
                onClick={handleCreate}
                className="w-full bg-brand-600 hover:bg-brand-700 text-white font-semibold py-3.5 rounded-xl transition-all shadow-lg shadow-brand-500/25 flex items-center justify-center gap-2 text-sm"
              >
                Create Consent & Link Accounts
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M14 5l7 7m0 0l-7 7m7-7H3"/>
                </svg>
              </button>
            </div>
          )}

          {/* Step 2 — Creating consent */}
          {step === 2 && (
            <div className="text-center py-8">
              <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-brand-50 flex items-center justify-center">
                <svg className="animate-spin w-8 h-8 text-brand-600" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
                </svg>
              </div>
              <h3 className="text-lg font-semibold text-slate-900 mb-2">Creating Consent...</h3>
              <p className="text-sm text-slate-500">Setting up your consent request with Setu AA</p>
            </div>
          )}

          {/* Step 3 — Redirecting */}
          {step === 3 && (
            <div className="text-center py-8">
              <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-green-50 flex items-center justify-center">
                <svg className="w-8 h-8 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
                </svg>
              </div>
              <h3 className="text-lg font-semibold text-slate-900 mb-2">Consent Created!</h3>
              <p className="text-sm text-slate-500 mb-4">
                Redirecting you to the AA portal to approve...
              </p>
              <div className="flex justify-center gap-1">
                {[0,1,2].map(i => (
                  <div
                    key={i}
                    className="w-2 h-2 rounded-full bg-brand-400 animate-bounce"
                    style={{ animationDelay: `${i * 0.15}s` }}
                  />
                ))}
              </div>
            </div>
          )}

        </div>
      </div>
    </div>
  )
}
