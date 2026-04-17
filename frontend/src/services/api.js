import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  headers: { 'Content-Type': 'application/json' },
})

api.interceptors.request.use((config) => {
  const token = localStorage.getItem('access_token')
  if (token) config.headers.Authorization = `Bearer ${token}`
  return config
})

api.interceptors.response.use(
  (response) => response,
  async (error) => {
    const original = error.config
    if (error.response?.status === 401 && !original._retry) {
      original._retry = true
      const refreshToken = localStorage.getItem('refresh_token')
      if (refreshToken) {
        try {
          const { data } = await axios.post('/api/auth/refresh', { refresh_token: refreshToken })
          localStorage.setItem('access_token',  data.access_token)
          localStorage.setItem('refresh_token', data.refresh_token)
          original.headers.Authorization = `Bearer ${data.access_token}`
          return api(original)
        } catch {
          localStorage.removeItem('access_token')
          localStorage.removeItem('refresh_token')
          window.location.href = '/signin'
        }
      }
    }
    return Promise.reject(error)
  }
)

export const authAPI = {
  signup:  (data)  => api.post('/auth/signup', data),
  login:   (data)  => api.post('/auth/login', data),
  me:      ()      => api.get('/auth/me'),
  logout:  ()      => api.post('/auth/logout'),
  refresh: (token) => api.post('/auth/refresh', { refresh_token: token }),
}

export const aaAPI = {
  createConsent:     (data)       => api.post('/aa/create-consent', data),
  getConsentStatus:  (consentId)  => api.get(`/aa/consent-status/${consentId}`),
  fetchData:         (consentId)  => api.post(`/aa/fetch-data/${consentId}`),
  getMyConsent:      ()           => api.get('/aa/my-consent'),
  revokeConsent:     (consentId)  => api.delete(`/aa/revoke-consent/${consentId}`),
  getFIData:         (params)     => api.get('/aa/fi-data', { params }),
  getCategories:     ()           => api.get('/aa/categories'),
  updateCategory:    (data)       => api.patch('/aa/transaction/category', data),
  getUserSummary:    (month, year) => api.get('/aa/user-summary', { params: { month, year } }),
  getCategoryBreakdown: (month, year) => api.get('/aa/category-breakdown', { params: { month, year } }),
  getCategoryDrilldown: (category, month, year) => api.get('/aa/category-drilldown', { params: { category, month, year } }),
}

export const pennyAPI = {
  // SSE streaming chat — returns a fetch Response for manual stream reading
  chatStream: async (question, history = []) => {
    const token = localStorage.getItem('access_token')
    return fetch('/api/penny/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`,
      },
      body: JSON.stringify({ question, messages: history }),
    })
  },
  getInsights:     ()       => api.get('/penny/insights'),
  autoCategorize:  (data)   => api.post('/penny/auto-categorize', data),
  uploadStatement: (formData) => api.post('/penny/upload-statement', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  }),
  getHistory:      ()       => api.get('/penny/history'),
  clearHistory:    ()       => api.delete('/penny/history'),
  submitFeedback:  (data)   => api.post('/penny/feedback', data),
  refreshVectors:  ()       => api.post('/penny/refresh-vectors'),
}

export const goalsAPI = {
  getGoals:      ()        => api.get('/goals'),
  createGoal:    (data)    => api.post('/goals', data),
  deleteGoal:    (id)      => api.delete(`/goals/${id}`),
  getSummary:    ()        => api.get('/goals/summary'),
}

export default api