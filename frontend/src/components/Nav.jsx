import React from 'react'
import { Link } from 'react-router-dom'

export default function Nav() {
  return (
    <nav className="fixed w-full top-0 z-50 bg-white/80 backdrop-blur-md border-b border-slate-100">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-20">
          <div className="flex-shrink-0 flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-brand-600 flex items-center justify-center relative overflow-hidden">
              <div className="absolute inset-0 bg-gradient-to-br from-brand-400 to-brand-700 animate-pulse"></div>
              <svg className="w-5 h-5 text-white relative z-10" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
            </div>
            <span className="font-bold text-xl tracking-tight text-slate-900">FinSight</span>
          </div>

          <div className="flex items-center space-x-4">
            <Link to="/signin" className="text-slate-600 hover:text-slate-900 font-medium transition-colors">Login</Link>
            <Link to="/signup" className="bg-brand-600 hover:bg-brand-700 text-white px-5 py-2.5 rounded-full font-medium transition-all shadow-lg shadow-brand-500/30 hover:shadow-brand-500/50">Sign Up</Link>
          </div>
        </div>
      </div>
    </nav>
  )
}
