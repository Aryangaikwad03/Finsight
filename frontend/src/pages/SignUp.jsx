import React, { useState, useEffect } from 'react'
import Nav from '../components/Nav'
import { Link, useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'

export default function Signup() {
  const [form, setForm] = useState({
    firstName: '',
    lastName:  '',
    email:     '',
    mobile:    '',
    password:  '',
    agree:     false,
  })
  const [errors, setErrors]   = useState({})
  const [apiError, setApiError] = useState('')
  const [loading, setLoading] = useState(false)
  const { signup, user }      = useAuth()
  const navigate              = useNavigate()

  // Already logged in → go to dashboard
  useEffect(() => {
    if (user) navigate('/dashboard', { replace: true })
  }, [user, navigate])

  const validate = (v) => {
    const e = {}
    if (!v.firstName.trim()) e.firstName = 'First name is required.'
    if (!v.lastName.trim())  e.lastName  = 'Last name is required.'
    if (!v.email.trim())     e.email     = 'Email is required.'
    else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(v.email)) e.email = 'Enter a valid email.'
    if (!v.mobile.trim())    e.mobile    = 'Mobile number is required.'
    else if (v.mobile.replace(/\D/g, '').length < 10) e.mobile = 'Enter a valid 10-digit mobile number.'
    if (!v.password)         e.password  = 'Password is required.'
    else if (v.password.length < 8) e.password = 'Password must be at least 8 characters.'
    if (!v.agree)            e.agree     = 'You must accept the terms.'
    return e
  }

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target
    setForm(prev => ({ ...prev, [name]: type === 'checkbox' ? checked : value }))
    setErrors(prev => ({ ...prev, [name]: undefined }))
    setApiError('')
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    const v = validate(form)
    setErrors(v)
    if (Object.keys(v).length > 0) return

    setLoading(true)
    setApiError('')
    try {
      // Backend expects: full_name, email, mobile, password
      await signup({
        full_name: `${form.firstName.trim()} ${form.lastName.trim()}`,
        email:     form.email.trim(),
        mobile:    form.mobile.replace(/\D/g, ''),
        password:  form.password,
      })
      navigate('/dashboard')
    } catch (err) {
      const detail = err.response?.data?.detail
      if (Array.isArray(detail)) {
        // Pydantic validation errors
        const msgs = detail.map(d => d.msg).join(', ')
        setApiError(msgs)
      } else {
        setApiError(typeof detail === 'string' ? detail : 'Sign up failed. Please try again.')
      }
    } finally {
      setLoading(false)
    }
  }

  const isValid = Object.keys(validate(form)).length === 0

  return (
    <div className="min-h-screen bg-white font-sans text-slate-800">
      <Nav />
      <main className="pt-24">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-start py-24">

            {/* Left — app preview */}
            <div className="flex items-center justify-center">
              <div className="w-80 h-[520px] bg-gradient-to-br from-brand-50 to-purple-50 rounded-3xl border border-slate-200 shadow-2xl flex flex-col items-center justify-center gap-6 p-8">
                <div className="w-16 h-16 rounded-2xl bg-brand-600 flex items-center justify-center shadow-lg shadow-brand-500/30">
                  <svg className="w-9 h-9 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
                  </svg>
                </div>
                <div className="text-center">
                  <p className="font-bold text-slate-900 text-lg mb-1">FinSight</p>
                  <p className="text-slate-500 text-sm">Your Finance Copilot</p>
                </div>
                <div className="w-full space-y-3">
                  {[['Link your bank accounts'],['View all transactions'],['Track spending by category'],['AI-powered insights']].map(([t]) => (
                    <div key={t} className="bg-white rounded-xl px-4 py-2.5 border border-slate-100 flex items-center gap-2">
                      <div className="w-1.5 h-1.5 rounded-full bg-brand-500 flex-shrink-0"/>
                      <span className="text-xs text-slate-600">{t}</span>
                    </div>
                  ))}
                </div>
                <p className="text-xs text-slate-400 text-center">RBI licensed · Bank-grade security</p>
              </div>
            </div>

            {/* Right — form */}
            <div>
              <h2 className="text-2xl font-bold text-slate-900 mb-2">Sign Up</h2>
              <p className="text-sm text-slate-500 mb-6">Start managing your finances faster & seamlessly</p>

              {apiError && (
                <div className="flex items-center gap-2 bg-red-50 border border-red-200 rounded-lg px-4 py-3 mb-4">
                  <svg className="w-4 h-4 text-red-500 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd"/>
                  </svg>
                  <p className="text-red-700 text-sm">{apiError}</p>
                </div>
              )}

              <form className="space-y-4 max-w-md" onSubmit={handleSubmit} noValidate>

                {/* First + Last name */}
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <input
                      name="firstName" value={form.firstName} onChange={handleChange}
                      placeholder="First Name"
                      className={`w-full border rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500 ${errors.firstName ? 'border-red-400' : 'border-slate-200'}`}
                    />
                    {errors.firstName && <p className="text-red-500 text-xs mt-1">{errors.firstName}</p>}
                  </div>
                  <div>
                    <input
                      name="lastName" value={form.lastName} onChange={handleChange}
                      placeholder="Last Name"
                      className={`w-full border rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500 ${errors.lastName ? 'border-red-400' : 'border-slate-200'}`}
                    />
                    {errors.lastName && <p className="text-red-500 text-xs mt-1">{errors.lastName}</p>}
                  </div>
                </div>

                {/* Email */}
                <div>
                  <input
                    name="email" type="email" value={form.email} onChange={handleChange}
                    placeholder="Email address"
                    className={`w-full border rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500 ${errors.email ? 'border-red-400' : 'border-slate-200'}`}
                  />
                  {errors.email && <p className="text-red-500 text-xs mt-1">{errors.email}</p>}
                </div>

                {/* Mobile */}
                <div>
                  <div className="flex">
                    <div className="flex items-center px-3 bg-slate-50 border border-r-0 border-slate-200 rounded-l-md text-sm text-slate-500">
                      +91
                    </div>
                    <input
                      name="mobile" type="tel" value={form.mobile} onChange={handleChange}
                      placeholder="10-digit mobile number"
                      maxLength={10}
                      className={`flex-1 border rounded-r-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500 ${errors.mobile ? 'border-red-400' : 'border-slate-200'}`}
                    />
                  </div>
                  {errors.mobile && <p className="text-red-500 text-xs mt-1">{errors.mobile}</p>}
                </div>

                {/* Password */}
                <div>
                  <input
                    name="password" type="password" value={form.password} onChange={handleChange}
                    placeholder="Password (min. 8 characters)"
                    className={`w-full border rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500 ${errors.password ? 'border-red-400' : 'border-slate-200'}`}
                  />
                  {errors.password && <p className="text-red-500 text-xs mt-1">{errors.password}</p>}
                </div>

                {/* Terms */}
                <label className="flex items-start gap-3 text-sm text-slate-600 cursor-pointer">
                  <input name="agree" type="checkbox" checked={form.agree} onChange={handleChange} className="mt-1"/>
                  <span>I accept the <a href="#" className="text-brand-600 hover:underline">terms & conditions</a></span>
                </label>
                {errors.agree && <p className="text-red-500 text-xs">{errors.agree}</p>}

                <button
                  type="submit"
                  disabled={loading}
                  className={`w-full px-6 py-3 rounded-md font-semibold transition flex items-center justify-center gap-2 ${
                    !loading ? 'bg-black hover:bg-slate-800 text-white shadow-lg' : 'bg-slate-200 text-slate-500 cursor-not-allowed'
                  }`}
                >
                  {loading ? (
                    <>
                      <svg className="animate-spin w-4 h-4" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
                      </svg>
                      Creating account…
                    </>
                  ) : 'Sign Up'}
                </button>

                <div className="text-center text-sm text-slate-600">
                  Already have an account?{' '}
                  <Link to="/signin" className="text-brand-600 font-medium hover:underline">Sign in</Link>
                </div>
              </form>
            </div>

          </div>
        </div>
      </main>
    </div>
  )
}