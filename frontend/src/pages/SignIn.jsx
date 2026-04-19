import React, { useState, useEffect } from 'react'
import Nav from '../components/Nav'
import { Link, useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'

export default function Signin() {
  const [form, setForm]         = useState({ email: '', password: '', remember: false })
  const [errors, setErrors]     = useState({})
  const [apiError, setApiError] = useState('')
  const [loading, setLoading]   = useState(false)
  const { login, user }         = useAuth()
  const navigate                = useNavigate()

  // If already logged in, go straight to transactions
  useEffect(() => {
    if (user) navigate('/transactions', { replace: true })
  }, [user, navigate])

  const validate = (values) => {
    const e = {}
    if (!values.email.trim())    e.email    = 'Email is required.'
    if (!values.password)        e.password = 'Password is required.'
    else if (values.password.length < 8) e.password = 'Password must be at least 8 characters.'
    return e
  }

  const handleChange = (ev) => {
    const { name, value, type, checked } = ev.target
    setForm(prev => ({ ...prev, [name]: type === 'checkbox' ? checked : value }))
    // Clear field error on change
    setErrors(prev => ({ ...prev, [name]: undefined }))
    setApiError('')
  }

  const handleSubmit = async (ev) => {
    ev.preventDefault()
    const v = validate(form)
    setErrors(v)
    if (Object.keys(v).length > 0) return

    setLoading(true)
    setApiError('')
    try {
      await login({ email: form.email, password: form.password })
      navigate('/transactions')          // ← redirect to transactions on success
    } catch (err) {
      const msg = err.response?.data?.detail
      setApiError(typeof msg === 'string' ? msg : 'Incorrect email or password.')
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

            {/* Left — App preview */}
            <div className="flex items-center justify-center lg:justify-center">
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
                {/* Mini mock dashboard */}
                <div className="w-full space-y-3">
                  {[['Total Balance','₹4,82,340','text-brand-600'],['Income','₹95,000','text-green-600'],['Expenses','₹42,560','text-red-500']].map(([label, val, cls]) => (
                    <div key={label} className="bg-white rounded-xl px-4 py-3 border border-slate-100 flex justify-between items-center">
                      <span className="text-xs text-slate-500 font-medium">{label}</span>
                      <span className={`text-sm font-bold ${cls}`}>{val}</span>
                    </div>
                  ))}
                </div>
                <div className="w-full bg-white rounded-xl p-3 border border-slate-100">
                  <p className="text-xs text-slate-400 mb-2">Spending</p>
                  <div className="flex gap-1 h-8 items-end">
                    {[40,65,45,80,55,70,50].map((h, i) => (
                      <div key={i} className="flex-1 bg-brand-200 rounded-sm" style={{ height: `${h}%` }}></div>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* Right — Sign in form */}
            <div>
              <h2 className="text-2xl font-bold text-slate-900 mb-2">Sign In</h2>
              <p className="text-sm text-slate-500 mb-6">Welcome back! Please enter your details.</p>

              {/* API error */}
              {apiError && (
                <div className="flex items-center gap-2 bg-red-50 border border-red-200 rounded-lg px-4 py-3 mb-4">
                  <svg className="w-4 h-4 text-red-500 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd"/>
                  </svg>
                  <p className="text-red-700 text-sm">{apiError}</p>
                </div>
              )}

              <form className="space-y-4 max-w-md" onSubmit={handleSubmit} noValidate>
                <div>
                  <label className="block text-xs text-slate-600 mb-2">Email</label>
                  <input
                    name="email"
                    type="email"
                    value={form.email}
                    onChange={handleChange}
                    className={`w-full border rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500 transition-all ${errors.email ? 'border-red-400' : 'border-slate-200'}`}
                    placeholder="Enter your email"
                  />
                  {errors.email && <div className="text-red-500 text-xs mt-1">{errors.email}</div>}
                </div>

                <div>
                  <label className="block text-xs text-slate-600 mb-2">Password</label>
                  <input
                    name="password"
                    type="password"
                    value={form.password}
                    onChange={handleChange}
                    className={`w-full border rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500 transition-all ${errors.password ? 'border-red-400' : 'border-slate-200'}`}
                    placeholder="Enter your password"
                  />
                  {errors.password && <div className="text-red-500 text-xs mt-1">{errors.password}</div>}
                </div>

                <div className="flex items-center justify-between text-sm">
                  <label className="flex items-center gap-2 text-slate-600 cursor-pointer">
                    <input
                      name="remember"
                      type="checkbox"
                      checked={form.remember}
                      onChange={handleChange}
                      className="rounded"
                    />
                    <span>Remember me</span>
                  </label>
                  <a href="#" className="text-brand-600 hover:underline">Forgot password?</a>
                </div>

                <div>
                  <button
                    type="submit"
                    disabled={!isValid || loading}
                    className={`w-full px-6 py-3 rounded-md font-semibold transition flex items-center justify-center gap-2 ${
                      isValid && !loading
                        ? 'bg-black hover:bg-slate-800 text-white'
                        : 'bg-slate-200 text-slate-500 cursor-not-allowed'
                    }`}
                  >
                    {loading ? (
                      <>
                        <svg className="animate-spin w-4 h-4" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
                        </svg>
                        Signing in…
                      </>
                    ) : 'Sign In'}
                  </button>
                </div>

                <div className="text-center text-sm text-slate-600">
                  Don't have an account?{' '}
                  <Link to="/signup" className="text-brand-600 font-medium hover:underline">Sign up</Link>
                </div>
              </form>
            </div>

          </div>
        </div>
      </main>
    </div>
  )
}