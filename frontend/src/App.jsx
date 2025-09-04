// frontend/src/App.jsx — Single-model UI with "New Chat", Top-K UI removed

import React, { useState, useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import { saveAs } from "file-saver";
import "highlight.js/styles/github-dark.css";
import "./App.css";

/* API URL via .env: prefer VITE_API_BASE, fallback to VITE_API_URL, then default */
const API_BASE =
  import.meta.env.VITE_API_BASE ||
  import.meta.env.VITE_API_URL ||
  "http://einstein.neurology.emory.edu:8000";

/* Helpers */
const safeJSON = (v, fallback) => { try { return JSON.parse(v); } catch { return fallback; } };

/* Smart title from history: prefer explicit title, else first user line, else timestamp */
const deriveTitle = (explicitTitle, messages) => {
  if (explicitTitle && explicitTitle.trim()) return explicitTitle.trim();
  const firstUser = (messages || []).find(m => m.role === "user");
  if (firstUser && firstUser.content) {
    const line = firstUser.content.split("\n")[0].trim();
    if (line) return line.slice(0, 60);
  }
  return `Chat @ ${new Date().toLocaleString()}`;
};

export default function App() {
  /* State */
  const [sessions, setSessions] = useState(() => {
    const saved = localStorage.getItem("chatSessions");
    return saved ? safeJSON(saved, []) : [];
  });

  const [history, setHistory] = useState(() => {
    const draft = localStorage.getItem("currentChat");
    if (draft) {
      const parsed = safeJSON(draft, {});
      if (Array.isArray(parsed.messages)) return parsed.messages;
    }
    const saved = localStorage.getItem("chatSessions");
    if (saved) {
      const arr = safeJSON(saved, []);
      const last = arr[arr.length - 1];
      return last ? last.messages : [];
    }
    return [];
  });

  const [chatTitle, setChatTitle] = useState(() => {
    const draft = localStorage.getItem("currentChat");
    if (draft) {
      const parsed = safeJSON(draft, {});
      return parsed?.name || "";
    }
    const saved = localStorage.getItem("chatSessions");
    if (saved) {
      const arr = safeJSON(saved, []);
      const last = arr[arr.length - 1];
      return last?.name || "";
    }
    return "";
  });

  const [query, setQuery] = useState("");
  const [darkMode, setDarkMode] = useState(false);
  const [loading, setLoading] = useState(false);

  // FAQ modal
  const [showFAQ, setShowFAQ] = useState(false);
  const [faq, setFaq] = useState([]);

  const chatRef = useRef(null);
  const abortRef = useRef(null);

  /* Persist */
  useEffect(() => { localStorage.setItem("chatSessions", JSON.stringify(sessions)); }, [sessions]);
  useEffect(() => {
    const draft = { name: chatTitle || "", messages: history || [] };
    localStorage.setItem("currentChat", JSON.stringify(draft));
  }, [history, chatTitle]);

  /* UI */
  useEffect(() => { document.body.className = darkMode ? "dark-mode" : ""; }, [darkMode]);
  useEffect(() => { chatRef.current?.scrollTo({ top: chatRef.current.scrollHeight, behavior: "smooth" }); }, [history, loading]);

  /* FAQ helpers */
  const openFAQ = async () => {
    try {
      const res = await fetch(`${API_BASE}/faq`);
      const data = await res.json();
      setFaq(Array.isArray(data) ? data : []);
      setShowFAQ(true);
    } catch {
      setFaq([]);
      setShowFAQ(true);
    }
  };

  const refreshFAQ = async () => {
    try {
      const res = await fetch(`${API_BASE}/faq`);
      const data = await res.json();
      setFaq(Array.isArray(data) ? data : []);
    } catch {}
  };

  const saveQAtoServer = async (q, a) => {
    try {
      await fetch(`${API_BASE}/save`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q, answer: a }),
      });
    } catch {}
  };

  const deleteFAQItem = async (id) => {
    try {
      const res = await fetch(`${API_BASE}/faq/${id}`, { method: "DELETE" });
      if (!res.ok) {
        await fetch(`${API_BASE}/faq/item?id=${encodeURIComponent(id)}`, { method: "DELETE" });
      }
      await refreshFAQ();
    } catch {
      alert("Failed to delete item.");
    }
  };

  const clearAllFAQ = async () => {
    try {
      const res = await fetch(`${API_BASE}/faq`, { method: "DELETE" });
      if (!res.ok) {
        await fetch(`${API_BASE}/faq/clear`, { method: "POST" });
      }
      await refreshFAQ();
    } catch {
      alert("Failed to clear FAQ.");
    }
  };

  const exportFAQ = async (fmt) => {
    try {
      const res = await fetch(`${API_BASE}/faq/export/${fmt}`);
      if (!res.ok) {
        const res2 = await fetch(`${API_BASE}/export/${fmt}`);
        if (!res2.ok) throw new Error("Export failed");
        const blob2 = await res2.blob();
        saveAs(blob2, `faq.${fmt === "word" ? "docx" : fmt}`);
        return;
      }
      const blob = await res.blob();
      saveAs(blob, `faq.${fmt === "word" ? "docx" : fmt}`);
    } catch {
      alert("Export failed.");
    }
  };

  /* Ask / Stop */
  const askQuestion = async () => {
    if (!query.trim() || loading) return;

    const userMessage = { role: "user", content: query };
    setHistory((prev) => [...prev, userMessage]);

    setLoading(true);
    const controller = new AbortController();
    abortRef.current = controller;

    const currentQuery = query;
    setQuery("");

    const recentHistory = [...history, userMessage].slice(-20);

    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify({
          question: currentQuery,
          history: recentHistory,
          // No Top-K knob sent from UI; server defaults handle advanced retrieval.
        }),
      });

      const text = await res.text();
      let answer = "⚠️ Unexpected response format.";
      try {
        const data = JSON.parse(text);
        answer = data?.response ?? data?.answer ?? data?.message ?? data?.text ?? answer;
      } catch {
        if (text && typeof text === "string") answer = text;
      }

      setHistory((prev) => [...prev, { role: "assistant", content: answer }]);
    } catch (err) {
      if (err.name === "AbortError") {
        setHistory((prev) => [...prev, { role: "assistant", content: "⏹️ Stopped." }]);
      } else {
        setHistory((prev) => [...prev, { role: "assistant", content: "❌ Network error: " + err.message }]);
      }
    } finally {
      setLoading(false);
      abortRef.current = null;
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      askQuestion();
    }
  };

  /* Sessions */
  const saveSession = () => {
    if (history.length === 0) return;
    const title = deriveTitle(chatTitle, history);
    const session = { name: title, messages: history };
    setSessions((prev) => [...prev, session]);
    setHistory([]);
    setChatTitle("");
    localStorage.removeItem("currentChat");
  };

  const deleteSession = (index) => {
    setSessions(sessions.filter((_, i) => i !== index));
  };

  // ChatGPT-like "New Chat": auto-saves current chat, then clears composer/history
  const newChat = () => {
    if (history.length > 0) {
      const title = deriveTitle(chatTitle, history);
      const session = { name: title, messages: history };
      setSessions((prev) => [...prev, session]);
    }
    setHistory([]);
    setChatTitle("");
    setQuery("");
    localStorage.removeItem("currentChat");
  };

  // Per-answer saves (FAQ)
  const onSaveClick = async (answerIndex) => {
    let q = "";
    for (let i = answerIndex - 1; i >= 0; i--) {
      if (history[i]?.role === "user") { q = history[i].content; break; }
    }
    const a = history[answerIndex]?.content || "";
    await saveQAtoServer(q, a);
  };

  const saveQAasMarkdown = (q, a) => {
    const content = `# Q\n${q}\n\n# A\n${a}\n`;
    const name = `qa-${Date.now()}.md`;
    const blob = new Blob([content], { type: "text/markdown;charset=utf-8" });
    saveAs(blob, name);
  };

  const onSaveMdClick = (answerIndex) => {
    let q = "";
    for (let i = answerIndex - 1; i >= 0; i--) {
      if (history[i]?.role === "user") { q = history[i].content; break; }
    }
    const a = history[answerIndex]?.content || "";
    saveQAasMarkdown(q, a);
  };

  return (
    <div className={`chat-ui ${darkMode ? "dark" : ""}`}>
      {/* Sidebar */}
      <aside className="sidebar open">
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8 }}>
          <h2 title="Saved Chat Sessions" style={{ margin: 0 }}>💬 Chats</h2>
          <button
            className="icon-btn"
            title="New Chat (auto-saves current, then clears)"
            onClick={newChat}
          >
            ➕ New Chat
          </button>
        </div>

        <ul>
          {sessions.map((s, i) => (
            <li key={i}>
              <span
                title="Load chat"
                onClick={() => {
                  setHistory(s.messages);
                  setChatTitle(s.name);
                  localStorage.setItem("currentChat", JSON.stringify({ name: s.name, messages: s.messages }));
                }}
              >
                {s.name}
              </span>
              <div style={{ display: "flex", gap: 6 }}>
                <button
                  title="Rename chat"
                  onClick={() => {
                    const newName = prompt("Rename chat:", s.name);
                    if (newName) {
                      const updated = [...sessions];
                      updated[i].name = newName;
                      setSessions(updated);
                      if (chatTitle === s.name) setChatTitle(newName);
                    }
                  }}
                >✏️</button>
                <button title="Delete chat" onClick={() => deleteSession(i)}>🗑️</button>
              </div>
            </li>
          ))}
        </ul>

        <div className="bottom-controls">
          {/* (Top-K control removed) */}

          <input
            value={chatTitle}
            onChange={(e) => setChatTitle(e.target.value)}
            placeholder="Chat title..."
            title="Enter chat title"
          />

          <button onClick={saveSession} title="Save this chat">💾 Save</button>
          <button onClick={openFAQ} title="Show saved Q&A (FAQ)">📚 FAQ</button>
          <button onClick={() => setDarkMode(!darkMode)} title="Toggle Light/Dark mode">
            {darkMode ? "☀️ Light" : "🌙 Dark"}
          </button>
        </div>
      </aside>

      {/* Main */}
      <main>
        <div className="header">
          <h1>GitHub Chatbot</h1>
          {/* (Top-K indicator removed) */}
        </div>

        <div className="chat-box" ref={chatRef}>
          {history.map((msg, i) => (
            <div key={i} className={`chat-line ${msg.role}`}>
              <div className="bubble">
                <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[[rehypeHighlight, { detect: true }]]}>
                  {msg.content}
                </ReactMarkdown>

                <div className="bubble-actions">
                  {msg.role === "assistant" ? (
                    <>
                      <button className="icon-btn" title="Save this Q&A to FAQ (server)" onClick={() => onSaveClick(i)}>
                        Save
                      </button>
                      <button className="icon-btn" title="Download this Q&A as .md" onClick={() => onSaveMdClick(i)}>
                        Save .md
                      </button>
                    </>
                  ) : null}
                </div>
              </div>
            </div>
          ))}

          {loading && (
            <div className="chat-line assistant typing">
              <div className="bubble">
                <span className="typing-dots"><span>•</span><span>•</span><span>•</span></span>
              </div>
            </div>
          )}
        </div>

        <div className="input-area">
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask something…"
            title="Type your question"
            disabled={loading}
          />
          <button
            onClick={() => {
              if (loading && abortRef.current) abortRef.current.abort();
              else askQuestion();
            }}
            title={loading ? "Stop" : "Send"}
          >
            {loading ? "◼" : "↑"}
          </button>
        </div>
      </main>

      {/* FAQ Modal */}
      {showFAQ && (
        <div className="modal-backdrop" onClick={() => setShowFAQ(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Saved Q&A (FAQ)</h3>
              <div style={{ display: "flex", gap: 8 }}>
                <button className="icon-btn" onClick={() => exportFAQ("json")}>Export .json</button>
                <button className="icon-btn" onClick={() => exportFAQ("md")}>Export .md</button>
                <button className="icon-btn" onClick={() => exportFAQ("html")}>Export .html</button>
                <button className="icon-btn" onClick={() => exportFAQ("word")}>Export .word</button>
                <button className="icon-btn" onClick={clearAllFAQ}>Clear all</button>
                <button className="icon-btn" onClick={() => setShowFAQ(false)}>Close</button>
              </div>
            </div>
            <div className="faq-list">
              {faq.length === 0 ? (
                <div className="faq-empty">No saved answers yet.</div>
              ) : (
                faq
                  .slice()
                  .reverse()
                  .map((item, idx) => (
                    <div key={item.id || idx} className="faq-item">
                      <div className="faq-item-head">
                        <div className="faq-q">Q: {item.question}</div>
                        <div className="faq-item-actions">
                          <button className="icon-btn" title="Delete this FAQ" onClick={() => deleteFAQItem(item.id)}>
                            Delete
                          </button>
                        </div>
                      </div>
                      <div className="faq-a">
                        <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[[rehypeHighlight, { detect: true }]]}>
                          {item.answer || ""}
                        </ReactMarkdown>
                      </div>
                      {item.sources && item.sources.length > 0 && (
                        <div className="faq-sources">
                          Sources: {item.sources.map((s) => s.source).filter(Boolean).join(", ")}
                        </div>
                      )}
                    </div>
                  ))
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}