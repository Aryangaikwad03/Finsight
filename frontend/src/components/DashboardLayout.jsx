import React, { useState } from 'react'
import { Link, useLocation, useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import PennyChatPanel from './PennyChatPanel'

export default function DashboardLayout({ title = 'Dashboard', rightElement = null, children }) {
  const { user, logout } = useAuth()
  const navigate         = useNavigate()
  const location         = useLocation()
  const [menuOpen, setMenuOpen] = useState(false)

  const handleLogout = async () => {
    await logout()
    navigate('/')
  }

  const initials    = user?.full_name
    ? user.full_name.split(' ').map(n => n[0]).join('').toUpperCase().slice(0, 2)
    : 'U'
  const displayName = user?.email?.split('@')[0] ?? 'user'

  const navLinks = [
    { to: '/dashboard',     label: 'Dashboard',    icon: 'M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6' },
    { to: '/transactions',  label: 'Transactions', icon: 'M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2' },
    { to: '/consent',       label: 'Consent',      icon: 'M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z' },
    { to: '/assistant',     label: 'AI Assistant', icon: 'M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z' },
    { to: '/goals',         label: 'Goals',        icon: 'M13 10V3L4 14h7v7l9-11h-7z' },
  ]

  return (
    <div className="min-h-screen bg-white font-sans text-slate-800">
      <div className="flex">

        {/* ── Sidebar ── */}
        <aside className="w-60 hidden lg:block border-r border-slate-100 px-6 py-6 h-screen sticky top-0 bg-white">
          <div className="space-y-6">

            {/* Logo */}
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-brand-600 flex items-center justify-center relative overflow-hidden">
                <div className="absolute inset-0 bg-gradient-to-br from-brand-400 to-brand-700"></div>
                <svg className="w-5 h-5 text-white relative z-10" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
                </svg>
              </div>
              <div>
                <div className="text-sm font-bold text-slate-900">FinSight</div>
                <div className="text-xs text-slate-400">Finance Copilot</div>
              </div>
            </div>

            {/* Nav links — active state based on current route */}
            <nav className="mt-4 space-y-1">
              {navLinks.map(({ to, label, icon }) => {
                const active = location.pathname === to
                return (
                  <Link
                    key={to}
                    to={to}
                    className={`flex items-center gap-3 px-3 py-2 rounded-md text-sm transition-colors ${
                      active
                        ? 'bg-brand-50 text-brand-700 font-medium'
                        : 'text-slate-600 hover:bg-slate-50'
                    }`}
                  >
                    <svg className="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d={icon}/>
                    </svg>
                    {label}
                  </Link>
                )
              })}
            </nav>

            <div className="pt-6 border-t border-slate-100 space-y-1">
              <Link to="#" className="flex items-center gap-3 px-3 py-2 rounded-md text-slate-600 hover:bg-slate-50 text-sm">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"/><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/>
                </svg>
                Settings
              </Link>
              <button
                onClick={handleLogout}
                className="flex items-center gap-3 px-3 py-2 rounded-md text-red-500 hover:bg-red-50 text-sm w-full text-left"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1"/>
                </svg>
                Log Out
              </button>
            </div>
          </div>
        </aside>

        {/* ── Main content ── */}
        <main className="flex-1 px-6 py-6">

          {/* Top bar */}
          <div className="flex items-start justify-between gap-6 mb-8">
            <div>
              <h1 className="text-2xl font-bold text-slate-900">{title}</h1>
              <p className="text-sm text-slate-500">Manage your financial data & consents securely.</p>
            </div>

            <div className="flex items-center gap-4">
              {rightElement}

              {/* User dropdown */}
              <div className="relative">
                <button
                  onClick={() => setMenuOpen(s => !s)}
                  className="flex items-center gap-2 px-3 py-2 border rounded-md border-slate-200 bg-white hover:border-slate-300 transition-colors"
                >
                  <div className="w-8 h-8 rounded-full bg-brand-600 text-white flex items-center justify-center text-xs font-bold">
                    {initials}
                  </div>
                  <div className="text-sm text-slate-700">{displayName}</div>
                  <svg className="w-4 h-4 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"/>
                  </svg>
                </button>

                {menuOpen && (
                  <div className="absolute right-0 mt-2 w-52 bg-white border border-slate-100 rounded-md shadow-lg py-2 z-50">
                    <div className="px-4 py-2 border-b border-slate-100 mb-1">
                      <p className="text-xs font-semibold text-slate-900 truncate">{user?.full_name}</p>
                      <p className="text-xs text-slate-400 truncate">{user?.email}</p>
                    </div>
                    <a href="#" className="block px-4 py-2 text-sm text-slate-700 hover:bg-slate-50">Profile</a>
                    <a href="#" className="block px-4 py-2 text-sm text-slate-700 hover:bg-slate-50">Settings</a>
                    <button
                      onClick={handleLogout}
                      className="block w-full text-left px-4 py-2 text-sm text-red-600 hover:bg-slate-50"
                    >
                      Log out
                    </button>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Page content */}
          {children}
        </main>
      </div>
      {/* Penny — floats on every page */}
      <PennyChatPanel />
    </div>
  )
}