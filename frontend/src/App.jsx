import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { AuthProvider } from './context/AuthContext'
import ProtectedRoute from './components/ProtectedRoute'
import Landing      from './pages/Landing'
import SignIn       from './pages/SignIn'
import SignUp       from './pages/SignUp'
import Dashboard    from './pages/Dashboard'
import Consent      from './pages/Consent'
import Transactions from './pages/Transactions'
import Goals        from './pages/Goals'

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <Routes>
          <Route path="/"             element={<Landing />} />
          <Route path="/signin"       element={<SignIn />} />
          <Route path="/signup"       element={<SignUp />} />
          <Route path="/dashboard"    element={<ProtectedRoute><Dashboard /></ProtectedRoute>} />
          <Route path="/consent"      element={<ProtectedRoute><Consent /></ProtectedRoute>} />
          <Route path="/transactions" element={<ProtectedRoute><Transactions /></ProtectedRoute>} />
          <Route path="/goals"        element={<ProtectedRoute><Goals /></ProtectedRoute>} />
        </Routes>
      </AuthProvider>
    </BrowserRouter>
  )
}