import React, { useState, useEffect, useRef, useCallback } from 'react'
import { pennyAPI } from '../services/api'
import { useAuth } from '../context/AuthContext'

const QUICK_CHIPS = [
  'How much did I spend this month?',
  'Which category did I spend the most on?',
  'Show me spending for all my accounts',
  'Show me my food transactions',
  'Find my latest transactions',
  'What are my spending patterns?',
  'Which categories are rising this month?',
  'Am I a weekend or weekday spender?',
]

const INTENT_LABELS = {
  spending_summary:     '💳 Spending',
  category_spending:    '📊 Categories',
  merchant_spending:    '🏪 Merchants',
  budget_status:        '🎯 Budget',
  goal_progress:        '🏆 Goals',
  savings_analysis:     '💰 Savings',
  income_analysis:      '💵 Income',
  account_balance:      '🏦 Balance',
  recurring_payments:   '🔄 Recurring',
  unusual_transaction:  '⚠️ Anomalies',
  fd_rd_query:          '📈 FD/RD',
  comparison_query:     '📅 Trends',
  financial_health:     '❤️ Health',
  pattern_analysis:     '🧠 Patterns',
  transaction_lookup:   '🔍 Transactions',
  account_transactions: '🏦 Account History',
  general:              '💬 General',
}


function TypingDots() {
  return (
    <div className="flex items-center gap-1 px-4 py-3">
      {[0, 1, 2].map(i => (
        <span
          key={i}
          className="w-2 h-2 rounded-full bg-purple-400"
          style={{ animation: `bounce 1.2s ease-in-out ${i * 0.2}s infinite` }}
        />
      ))}
    </div>
  )
}

function MessageBubble({ msg, onFeedback }) {
  const isUser = msg.role === 'user'
  const [feedbackGiven, setFeedbackGiven] = useState(null)

  const handleFeedback = async (helpful) => {
    if (feedbackGiven !== null || !msg.id) return
    setFeedbackGiven(helpful)
    try {
      await pennyAPI.submitFeedback({ message_id: msg.id, helpful })
    } catch (e) {
      console.error('Feedback error:', e)
    }
  }

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-3 group`}>
      {!isUser && (
        <div className="w-7 h-7 rounded-full bg-gradient-to-br from-purple-500 to-indigo-600 flex items-center justify-center text-white text-xs font-black mr-2 mt-1 flex-shrink-0 shadow-lg">
          P
        </div>
      )}
      <div className={`flex flex-col ${isUser ? 'items-end' : 'items-start'} max-w-[85%]`}>
        <div
          className={`px-4 py-3 rounded-2xl text-sm leading-relaxed ${
            isUser
              ? 'bg-gradient-to-br from-purple-600 to-indigo-600 text-white rounded-br-md shadow-md'
              : 'bg-white text-slate-700 rounded-bl-md shadow-sm border border-slate-100'
          }`}
          style={{ wordBreak: 'break-word', whiteSpace: 'pre-wrap' }}
        >
          {msg.content}
        </div>

        {/* Intent Badge + Feedback for assistant */}
        {!isUser && msg.id && (
          <div className="flex items-center gap-2 mt-1.5 opacity-0 group-hover:opacity-100 transition-opacity">
            {msg.intent && INTENT_LABELS[msg.intent] && (
              <span className="text-[10px] text-slate-400 bg-slate-100 rounded-full px-2 py-0.5">
                {INTENT_LABELS[msg.intent]}
              </span>
            )}
            <button
              onClick={() => handleFeedback(true)}
              className={`text-xs rounded-full px-2 py-0.5 transition-colors ${
                feedbackGiven === true
                  ? 'bg-green-100 text-green-600'
                  : 'text-slate-400 hover:text-green-500 hover:bg-green-50'
              }`}
              title="Helpful"
            >
              👍
            </button>
            <button
              onClick={() => handleFeedback(false)}
              className={`text-xs rounded-full px-2 py-0.5 transition-colors ${
                feedbackGiven === false
                  ? 'bg-red-100 text-red-500'
                  : 'text-slate-400 hover:text-red-500 hover:bg-red-50'
              }`}
              title="Not helpful"
            >
              👎
            </button>
          </div>
        )}
        <span className="text-[10px] text-slate-400 mt-1">
          {msg.time || new Date().toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' })}
        </span>
      </div>
    </div>
  )
}

export default function PennyChatPanel() {
  const { user } = useAuth()
  const [isOpen, setIsOpen] = useState(false)
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [isStreaming, setIsStreaming] = useState(false)
  const [historyLoaded, setHistoryLoaded] = useState(false)
  const [showClearConfirm, setShowClearConfirm] = useState(false)
  const messagesEndRef = useRef(null)
  const textareaRef = useRef(null)
  const abortRef = useRef(null)

  // Load chat history from DB when panel first opens
  useEffect(() => {
    if (isOpen && !historyLoaded && user) {
      pennyAPI.getHistory()
        .then(res => {
          const hist = res.data.history || []
          setMessages(hist.map(h => ({
            id: h.id,
            role: h.role,
            content: h.content,
            intent: h.intent,
            time: new Date(h.created_at).toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' }),
          })))
          setHistoryLoaded(true)
        })
        .catch(() => setHistoryLoaded(true))
    }
  }, [isOpen, historyLoaded, user])

  // Auto-scroll to bottom
  useEffect(() => {
    if (isOpen) {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }
  }, [messages, isOpen])

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 100) + 'px'
    }
  }, [input])

  const buildHistory = () =>
    messages.slice(-6).map(m => ({ role: m.role, content: m.content }))

  const sendMessage = useCallback(async (text) => {
    const question = text.trim()
    if (!question || isStreaming) return
    setInput('')

    const userMsg = {
      role: 'user',
      content: question,
      time: new Date().toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' }),
    }
    setMessages(prev => [...prev, userMsg])
    setIsStreaming(true)

    // Placeholder streaming message
    const streamingId = Date.now()
    setMessages(prev => [...prev, {
      id: null,
      role: 'assistant',
      content: '',
      _streaming: true,
      _streamingId: streamingId,
    }])

    try {
      const response = await pennyAPI.chatStream(question, buildHistory())
      if (!response.ok) throw new Error('Stream failed')

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let finalMsgId = null
      let finalIntent = 'general'

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() // keep incomplete line

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          try {
            const parsed = JSON.parse(line.slice(6))
            if (parsed.done) {
              finalMsgId = parsed.message_id
              finalIntent = parsed.intent || 'general'
            } else if (parsed.content) {
              setMessages(prev => prev.map(m =>
                m._streamingId === streamingId
                  ? { ...m, content: m.content + parsed.content }
                  : m
              ))
            }
          } catch { /* skip bad lines */ }
        }
      }

      // Finalize the streaming message
      setMessages(prev => prev.map(m =>
        m._streamingId === streamingId
          ? { ...m, id: finalMsgId, intent: finalIntent, _streaming: false, _streamingId: undefined }
          : m
      ))
    } catch (e) {
      console.error('Stream error:', e)
      setMessages(prev => prev.map(m =>
        m._streamingId === streamingId
          ? { ...m, content: 'Sorry, I ran into an issue. Please try again.', _streaming: false }
          : m
      ))
    } finally {
      setIsStreaming(false)
    }
  }, [isStreaming, messages])

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage(input)
    }
  }

  const handleClearHistory = async () => {
    try {
      await pennyAPI.clearHistory()
      setMessages([])
      setShowClearConfirm(false)
    } catch (e) {
      console.error('Clear error:', e)
    }
  }

  const isEmpty = messages.length === 0

  return (
    <>
      {/* Floating Trigger Button */}
      <button
        onClick={() => setIsOpen(o => !o)}
        className="fixed bottom-6 right-6 z-50 w-14 h-14 rounded-full bg-gradient-to-br from-purple-600 to-indigo-600 shadow-2xl shadow-purple-400/40 flex items-center justify-center text-white transition-all duration-300 hover:scale-110"
        style={{ animation: !isOpen ? 'pennyPulse 2.5s ease-in-out infinite' : 'none' }}
        title="Chat with Penny"
        id="penny-chat-toggle"
      >
        {isOpen ? (
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M6 18L18 6M6 6l12 12" />
          </svg>
        ) : (
          <span className="text-xl font-black" style={{ fontFamily: 'serif' }}>P</span>
        )}
        {!isOpen && messages.length > 0 && (
          <span className="absolute -top-1 -right-1 w-4 h-4 bg-green-400 rounded-full border-2 border-white" />
        )}
      </button>

      {/* Chat Panel */}
      <div
        className={`fixed bottom-24 right-6 z-50 w-[420px] h-[600px] bg-white rounded-3xl shadow-2xl shadow-slate-300/50 border border-slate-100 flex flex-col overflow-hidden transition-all duration-300 origin-bottom-right ${
          isOpen ? 'opacity-100 scale-100 pointer-events-auto' : 'opacity-0 scale-95 pointer-events-none'
        }`}
      >
        {/* Header */}
        <div className="bg-gradient-to-r from-purple-600 to-indigo-600 px-5 py-4 flex items-center justify-between flex-shrink-0">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-full bg-white/20 flex items-center justify-center text-white font-black text-lg border border-white/30">
              P
            </div>
            <div>
              <p className="text-white font-bold text-sm leading-none">Penny</p>
              <div className="flex items-center gap-1.5 mt-0.5">
                <span className="w-1.5 h-1.5 rounded-full bg-green-400" />
                <p className="text-purple-200 text-[10px]">AI Financial Assistant</p>
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {messages.length > 0 && (
              <button
                onClick={() => setShowClearConfirm(true)}
                className="text-purple-200 hover:text-white text-xs px-2 py-1 rounded-lg hover:bg-white/10 transition-colors"
                title="Clear chat"
              >
                Clear
              </button>
            )}
            <button
              onClick={() => setIsOpen(false)}
              className="text-purple-200 hover:text-white w-7 h-7 rounded-lg hover:bg-white/10 flex items-center justify-center transition-colors"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M19 9l-7 7-7-7" />
              </svg>
            </button>
          </div>
        </div>

        {/* Clear Confirmation */}
        {showClearConfirm && (
          <div className="bg-amber-50 border-b border-amber-100 px-4 py-3 flex items-center justify-between">
            <p className="text-xs text-amber-700 font-medium">Clear all chat history?</p>
            <div className="flex gap-2">
              <button onClick={() => setShowClearConfirm(false)} className="text-xs text-slate-500 hover:text-slate-700">Cancel</button>
              <button onClick={handleClearHistory} className="text-xs text-red-600 font-bold hover:text-red-700">Clear</button>
            </div>
          </div>
        )}

        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto px-4 py-4 space-y-1 bg-slate-50/50">
          {isEmpty && !isStreaming ? (
            <div className="flex flex-col items-center justify-center h-full text-center px-4">
              <div className="w-16 h-16 rounded-full bg-gradient-to-br from-purple-100 to-indigo-100 flex items-center justify-center mb-4">
                <span className="text-3xl">💬</span>
              </div>
              <h3 className="font-bold text-slate-700 text-sm mb-1">Ask Penny anything</h3>
              <p className="text-slate-400 text-xs mb-6">Your AI financial advisor powered by real data</p>
              <div className="flex flex-wrap gap-2 justify-center">
                {QUICK_CHIPS.map(chip => (
                  <button
                    key={chip}
                    onClick={() => sendMessage(chip)}
                    className="text-[11px] bg-white text-purple-600 border border-purple-200 rounded-full px-3 py-1.5 hover:bg-purple-50 hover:border-purple-400 transition-all font-medium shadow-sm"
                  >
                    {chip}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <>
              {messages.map((msg, i) => (
                <div key={i}>
                  {msg._streaming && msg.content === '' ? (
                    <div className="flex justify-start mb-3">
                      <div className="w-7 h-7 rounded-full bg-gradient-to-br from-purple-500 to-indigo-600 flex items-center justify-center text-white text-xs font-black mr-2 mt-1 flex-shrink-0">P</div>
                      <div className="bg-white rounded-2xl rounded-bl-md shadow-sm border border-slate-100">
                        <TypingDots />
                      </div>
                    </div>
                  ) : (
                    <MessageBubble msg={msg} />
                  )}
                </div>
              ))}
              {/* Quick chips after AI response if no more streaming */}
              {!isStreaming && messages.length > 0 && messages.length <= 2 && (
                <div className="flex flex-wrap gap-1.5 pt-2 justify-center">
                  {QUICK_CHIPS.slice(0, 3).map(chip => (
                    <button
                      key={chip}
                      onClick={() => sendMessage(chip)}
                      className="text-[10px] bg-white text-purple-500 border border-purple-100 rounded-full px-2.5 py-1 hover:bg-purple-50 transition-all"
                    >
                      {chip}
                    </button>
                  ))}
                </div>
              )}
            </>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="px-4 py-3 border-t border-slate-100 bg-white flex-shrink-0">
          <div className="flex items-end gap-2 bg-slate-50 border border-slate-200 rounded-2xl px-3 py-2 focus-within:border-purple-400 focus-within:bg-white transition-all">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask Penny about your finances..."
              disabled={isStreaming}
              rows={1}
              className="flex-1 bg-transparent text-sm text-slate-700 placeholder-slate-400 resize-none outline-none py-0.5 max-h-24 leading-relaxed disabled:opacity-60"
              id="penny-chat-input"
            />
            <button
              onClick={() => sendMessage(input)}
              disabled={!input.trim() || isStreaming}
              className="w-8 h-8 rounded-xl bg-gradient-to-br from-purple-600 to-indigo-600 flex items-center justify-center text-white flex-shrink-0 disabled:opacity-40 disabled:cursor-not-allowed hover:opacity-90 transition-all shadow-md"
            >
              {isStreaming ? (
                <svg className="w-3 h-3 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
              ) : (
                <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                </svg>
              )}
            </button>
          </div>
          <p className="text-[10px] text-slate-400 mt-1.5 text-center">
            Penny uses your real financial data · Never shares advice
          </p>
        </div>
      </div>

      {/* Global keyframe styles */}
      <style>{`
        @keyframes pennyPulse {
          0%, 100% { box-shadow: 0 0 0 0 rgba(139, 92, 246, 0.4), 0 20px 60px rgba(139, 92, 246, 0.4); }
          50% { box-shadow: 0 0 0 12px rgba(139, 92, 246, 0), 0 20px 60px rgba(139, 92, 246, 0.4); }
        }
        @keyframes bounce {
          0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; }
          40% { transform: scale(1); opacity: 1; }
        }
      `}</style>
    </>
  )
}
