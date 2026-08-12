import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const DEFAULT_API_BASE =
  typeof window !== "undefined"
    ? `${window.location.protocol}//${window.location.hostname}:8000`
    : "http://127.0.0.1:8000";
const API_BASE = import.meta.env.VITE_API_BASE_URL || DEFAULT_API_BASE;

const QUICK_PROMPTS = [
  "Summarize key obligations for high-risk AI systems under the EU AI Act.",
  "Give me a compliance checklist for an AI provider launching in the EU.",
  "What are the latest AI regulation developments this month?",
  "Compare prohibited AI practices and high-risk obligations.",
];

function createSessionId() {
  const stamp = new Date().toISOString().replace(/[-:.TZ]/g, "").slice(0, 14);
  const rand = Math.random().toString(16).slice(2, 8);
  return `chat_${stamp}_${rand}`;
}

function sanitizeNumArticles(value) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    return 5;
  }
  return Math.max(1, Math.min(10, Math.floor(parsed)));
}

const VALID_MODES = ["all", "single", "parallel", "legal_news_parallel", "legal_first", "verify_only"];
function sanitizeExpertMode(value) {
  return VALID_MODES.includes(value) ? value : "all";
}

function extractSseData(packet) {
  const lines = packet.split(/\r?\n/);
  const dataLines = [];

  for (const line of lines) {
    if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trimStart());
    }
  }

  return dataLines.join("\n");
}

function downloadFile(filename, content, mimeType) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

function mapPersistedMessagesToUi(messages) {
  if (!Array.isArray(messages)) {
    return [];
  }

  const stamp = Date.now();
  return messages
    .filter((item) => item && typeof item === "object")
    .filter((item) => item.role === "user" || item.role === "assistant")
    .map((item, index) => ({
      id: `persisted_${stamp}_${index}`,
      role: item.role,
      content: String(item.content || ""),
      pending: false,
      metadata: item.metadata || null,
    }));
}

function normalizeImportedMessages(payload) {
  if (!payload || typeof payload !== "object") {
    return [];
  }

  if (!Array.isArray(payload.messages)) {
    return [];
  }

  const stamp = Date.now();
  return payload.messages
    .filter((item) => item && typeof item === "object")
    .filter((item) => item.role === "user" || item.role === "assistant")
    .map((item, index) => ({
      id: `imported_${stamp}_${index}`,
      role: item.role,
      content: String(item.content || ""),
      pending: false,
      metadata: item.metadata || null,
    }));
}

function formatAgentTimings(timings) {
  if (!timings || typeof timings !== "object") {
    return "";
  }

  const entries = Object.entries(timings)
    .map(([name, value]) => [name, Number(value)]);

  const validEntries = entries
    .filter(([, value]) => Number.isFinite(value) && value >= 0)
    .sort((a, b) => b[1] - a[1]);

  if (validEntries.length === 0) {
    return "";
  }

  return validEntries
    .map(([name, value]) => `${name}: ${value.toFixed(1)} ms`)
    .join(" | ");
}

function App() {
  const [sessionId, setSessionId] = useState(() => createSessionId());
  const [messages, setMessages] = useState([]);
  const [draft, setDraft] = useState("");
  const [lastUserPrompt, setLastUserPrompt] = useState("");
  const [activity, setActivity] = useState([]);
  const [health, setHealth] = useState(null);
  const [runtimeConfig, setRuntimeConfig] = useState(null);
  const [readiness, setReadiness] = useState(null);
  const [sources, setSources] = useState({ articles: [], documents: [], total_articles: 0 });
  const [sessions, setSessions] = useState([]);
  const [sessionsTotal, setSessionsTotal] = useState(0);
  const [fetchNews, setFetchNews] = useState(true);
  const [numArticles, setNumArticles] = useState(5);
  const [expertExecutionMode, setExpertExecutionMode] = useState("all");
  const [sourceLimit, setSourceLimit] = useState(12);
  const [showActivity, setShowActivity] = useState(true);
  const [isSending, setIsSending] = useState(false);
  const [isClearing, setIsClearing] = useState(false);
  const [isClearingSessions, setIsClearingSessions] = useState(false);
  const [isLoadingSources, setIsLoadingSources] = useState(false);
  const [isLoadingSessions, setIsLoadingSessions] = useState(false);

  const endRef = useRef(null);
  const activityStepRef = useRef(0);
  const fileInputRef = useRef(null);
  const requestControllerRef = useRef(null);
  const expertModeInitializedRef = useRef(false);

  const lastAssistantMessage = useMemo(() => {
    for (let index = messages.length - 1; index >= 0; index -= 1) {
      const message = messages[index];
      if (message.role === "assistant" && !message.pending) {
        return message;
      }
    }
    return null;
  }, [messages]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, activity]);

  useEffect(() => {
    const loadInitialState = async () => {
      await Promise.all([
        loadHealth(),
        loadReadiness(),
        loadRuntimeConfig(),
        loadSources(sourceLimit),
        loadSessions(),
      ]);
    };

    loadInitialState();

    const timer = setInterval(() => {
      loadHealth();
      loadReadiness();
    }, 20000);

    return () => {
      clearInterval(timer);
      if (requestControllerRef.current) {
        requestControllerRef.current.abort();
      }
    };
  }, [sourceLimit]);

  async function loadHealth() {
    try {
      const response = await fetch(`${API_BASE}/health`);
      if (!response.ok) {
        return;
      }
      const payload = await response.json();
      setHealth(payload);
    } catch {
      // Keep the UI responsive even if health checks fail.
    }
  }

  async function loadRuntimeConfig() {
    try {
      const response = await fetch(`${API_BASE}/runtime`);
      if (!response.ok) {
        return;
      }
      const payload = await response.json();
      setRuntimeConfig(payload);

      if (!expertModeInitializedRef.current) {
        setExpertExecutionMode(sanitizeExpertMode(payload.expert_execution_mode));
        expertModeInitializedRef.current = true;
      }
    } catch {
      // Optional metadata.
    }
  }

  async function loadReadiness() {
    try {
      const response = await fetch(`${API_BASE}/readiness`);
      if (!response.ok) {
        return;
      }
      const payload = await response.json();
      setReadiness(payload);
    } catch {
      // Optional metadata.
    }
  }

  async function loadSources(limit = sourceLimit) {
    setIsLoadingSources(true);
    try {
      const response = await fetch(`${API_BASE}/sources?limit=${limit}`);
      if (!response.ok) {
        throw new Error(`Failed to load sources (${response.status})`);
      }
      const payload = await response.json();
      setSources({
        articles: Array.isArray(payload.articles) ? payload.articles : [],
        documents: Array.isArray(payload.documents) ? payload.documents : [],
        total_articles: Number(payload.total_articles || 0),
      });
    } catch (error) {
      addActivity(`Source load issue: ${error.message}`, false);
    } finally {
      setIsLoadingSources(false);
    }
  }

  async function loadSessions() {
    setIsLoadingSessions(true);
    try {
      const response = await fetch(`${API_BASE}/sessions?limit=40`);
      if (!response.ok) {
        throw new Error(`Failed to load sessions (${response.status})`);
      }
      const payload = await response.json();
      setSessions(Array.isArray(payload.sessions) ? payload.sessions : []);
      setSessionsTotal(Number(payload.total || 0));
    } catch (error) {
      addActivity(`Session load issue: ${error.message}`, false);
    } finally {
      setIsLoadingSessions(false);
    }
  }

  async function loadSessionById(targetSessionId) {
    try {
      const response = await fetch(`${API_BASE}/sessions/${encodeURIComponent(targetSessionId)}`);
      if (!response.ok) {
        throw new Error(`Failed to load session (${response.status})`);
      }

      const payload = await response.json();
      setSessionId(payload.session_id || targetSessionId);
      setMessages(mapPersistedMessagesToUi(payload.messages));
      setActivity([]);
      activityStepRef.current = 0;
      addActivity(`Loaded session ${targetSessionId}`, false);
    } catch (error) {
      addActivity(`Load session failed: ${error.message}`, false);
    }
  }

  async function deleteSessionById(targetSessionId) {
    const confirmed = window.confirm(`Delete saved session ${targetSessionId}?`);
    if (!confirmed) {
      return;
    }

    try {
      const response = await fetch(`${API_BASE}/sessions/${encodeURIComponent(targetSessionId)}`, {
        method: "DELETE",
      });
      if (!response.ok) {
        throw new Error(`Delete failed (${response.status})`);
      }

      addActivity(`Deleted session ${targetSessionId}`, false);
      if (targetSessionId === sessionId) {
        startNewChat();
      }
      await loadSessions();
    } catch (error) {
      addActivity(`Delete session failed: ${error.message}`, false);
    }
  }

  async function clearSavedSessions() {
    if (!window.confirm("Delete all saved sessions?")) {
      return;
    }

    setIsClearingSessions(true);
    try {
      const response = await fetch(`${API_BASE}/sessions`, { method: "DELETE" });
      const payload = await response.json();
      addActivity(payload.message || "Sessions cleared.", false);
      await loadSessions();
    } catch (error) {
      addActivity(`Clear sessions failed: ${error.message}`, false);
    } finally {
      setIsClearingSessions(false);
    }
  }

  const statusText = useMemo(() => {
    if (!health) {
      return "Connecting to backend...";
    }
    const ollamaStatus = health.ollama_reachable ? "Online" : "Offline";
    return `Docs ${health.documents} | Articles ${health.articles} | Ollama ${ollamaStatus}`;
  }, [health]);

  const readinessText = useMemo(() => {
    if (!readiness) {
      return "Readiness check unavailable";
    }
    return readiness.status === "ready" ? "Ready" : "Degraded";
  }, [readiness]);

  function addActivity(line, withStep = true) {
    if (!line) {
      return;
    }

    const formatted = withStep
      ? `Step ${++activityStepRef.current}: ${line}`
      : line;

    setActivity((prev) => {
      return [...prev.slice(-59), formatted];
    });
  }

  function updateAssistantMessage(messageId, patch) {
    setMessages((prev) =>
      prev.map((message) => {
        if (message.id !== messageId) {
          return message;
        }
        return { ...message, ...patch };
      })
    );
  }

  function startNewChat() {
    setSessionId(createSessionId());
    setMessages([]);
    setActivity([]);
    setLastUserPrompt("");
    activityStepRef.current = 0;
  }

  function clearActivityFeed() {
    setActivity([]);
    activityStepRef.current = 0;
  }

  async function clearData() {
    if (!window.confirm("Clear all local articles and vector-store data?")) {
      return;
    }

    setIsClearing(true);
    try {
      const response = await fetch(`${API_BASE}/admin/clear`, { method: "POST" });
      const payload = await response.json();
      addActivity(payload.message || "Data cleared.", false);
      startNewChat();
      await Promise.all([loadHealth(), loadSources(sourceLimit)]);
    } catch (error) {
      addActivity(`Clear failed: ${error.message}`, false);
    } finally {
      setIsClearing(false);
    }
  }

  async function copyLastAssistant() {
    if (!lastAssistantMessage) {
      addActivity("No assistant response available to copy.", false);
      return;
    }

    try {
      await navigator.clipboard.writeText(lastAssistantMessage.content);
      addActivity("Copied last answer to clipboard.", false);
    } catch {
      addActivity("Clipboard copy failed in this browser context.", false);
    }
  }

  function exportChatAsJson() {
    const payload = {
      exported_at: new Date().toISOString(),
      session_id: sessionId,
      fetch_news: fetchNews,
      num_articles: numArticles,
      expert_execution_mode: expertExecutionMode,
      messages: messages.map((message) => ({
        role: message.role,
        content: message.content,
        metadata: message.metadata || null,
      })),
    };

    const fileName = `legalai_${sessionId}.json`;
    downloadFile(fileName, JSON.stringify(payload, null, 2), "application/json");
    addActivity(`Exported ${messages.length} messages as JSON.`, false);
  }

  function exportChatAsMarkdown() {
    const lines = [
      `# Legal AI Chat Export`,
      ``,
      `- Session: ${sessionId}`,
      `- Exported at: ${new Date().toISOString()}`,
      ``,
    ];

    messages.forEach((message, index) => {
      const title = message.role === "assistant" ? "Assistant" : "User";
      lines.push(`## ${index + 1}. ${title}`);
      lines.push("");
      lines.push(message.content || "(empty)");
      lines.push("");
    });

    const fileName = `legalai_${sessionId}.md`;
    downloadFile(fileName, lines.join("\n"), "text/markdown");
    addActivity(`Exported ${messages.length} messages as Markdown.`, false);
  }

  function openImportDialog() {
    fileInputRef.current?.click();
  }

  function importChatFromFile(event) {
    const file = event.target.files?.[0];
    event.target.value = "";

    if (!file) {
      return;
    }

    const reader = new FileReader();
    reader.onload = () => {
      try {
        const parsed = JSON.parse(String(reader.result || "{}"));
        const importedMessages = normalizeImportedMessages(parsed);
        if (importedMessages.length === 0) {
          throw new Error("No valid messages found in the import file");
        }

        setSessionId(parsed.session_id || createSessionId());
        setMessages(importedMessages);
        setFetchNews(Boolean(parsed.fetch_news ?? true));
        setNumArticles(sanitizeNumArticles(parsed.num_articles ?? 5));
        setExpertExecutionMode(sanitizeExpertMode(parsed.expert_execution_mode));
        clearActivityFeed();
        addActivity(`Imported ${importedMessages.length} messages from file.`, false);
      } catch (error) {
        addActivity(`Import failed: ${error.message}`, false);
      }
    };

    reader.onerror = () => {
      addActivity("Import failed: could not read selected file.", false);
    };

    reader.readAsText(file);
  }

  function applyQuickPrompt(prompt) {
    setDraft(prompt);
  }

  function retryLastPrompt() {
    if (!lastUserPrompt) {
      addActivity("No previous prompt to retry.", false);
      return;
    }

    sendMessage({ promptOverride: lastUserPrompt, appendUser: true });
  }

  function regenerateLastResponse() {
    if (!lastUserPrompt) {
      addActivity("No previous prompt to regenerate.", false);
      return;
    }

    sendMessage({ promptOverride: lastUserPrompt, appendUser: false, isRegeneration: true });
  }

  function abortCurrentRequest() {
    if (!requestControllerRef.current) {
      return;
    }
    requestControllerRef.current.abort();
    requestControllerRef.current = null;
    addActivity("Request aborted by user.", false);
  }

  function handleStreamEvent(event, assistantId) {
    switch (event.type) {
      case "status":
        addActivity(event.message || "Working...");
        return false;
      case "route":
        addActivity(`Route selected: ${(event.route || "general").toUpperCase()}`);
        return false;
      case "fetch_progress":
        if (event.current_article_title) {
          addActivity(`${event.current_action || "Fetching sources..."} (${event.current_article_title})`);
        } else {
          addActivity(event.current_action || "Fetching sources...");
        }
        return false;
      case "thinking": {
        const step = (event.step || "step").replaceAll("_", " ");
        const details = event.details ? ` - ${event.details}` : "";
        addActivity(`${step}${details}`);
        return false;
      }
      case "final":
        updateAssistantMessage(assistantId, {
          content: event.response || "No response generated.",
          pending: false,
          metadata: {
            route: event.route || null,
            fetched: Boolean(event.fetched),
            articles_count: Number(event.articles_count || 0),
            fetch_error: event.fetch_error || null,
            expert_execution_mode: sanitizeExpertMode(event.expert_execution_mode),
            workflow_elapsed_ms: Number(event.workflow_elapsed_ms || 0),
            agent_timings_ms: event.agent_timings_ms || {},
          },
        });
        if (event.session_id) {
          setSessionId(event.session_id);
        }
        if (Number(event.workflow_elapsed_ms || 0) > 0) {
          addActivity(`Workflow latency: ${Number(event.workflow_elapsed_ms).toFixed(1)} ms`, false);
        }
        const timingsSummary = formatAgentTimings(event.agent_timings_ms || {});
        if (timingsSummary) {
          addActivity(`Node timings: ${timingsSummary}`, false);
        }
        addActivity("Response completed.");
        return true;
      case "error":
        updateAssistantMessage(assistantId, {
          content: `Error: ${event.message || "Unknown error"}`,
          pending: false,
        });
        addActivity(`Error: ${event.message || "Unknown error"}`);
        return true;
      default:
        return false;
    }
  }

  async function sendMessage({ promptOverride = null, appendUser = true, isRegeneration = false } = {}) {
    const trimmed = (promptOverride ?? draft).trim();
    if (!trimmed || isSending) {
      return;
    }

    setLastUserPrompt(trimmed);

    if (requestControllerRef.current) {
      requestControllerRef.current.abort();
    }

    const controller = new AbortController();
    requestControllerRef.current = controller;

    const userMessage = {
      id: `${Date.now()}_u`,
      role: "user",
      content: trimmed,
    };
    const assistantId = `${Date.now()}_a`;

    setMessages((prev) => {
      const next = [...prev];
      if (appendUser) {
        next.push(userMessage);
      }
      next.push({
        id: assistantId,
        role: "assistant",
        content: isRegeneration ? "Regenerating response..." : "Working on your request...",
        pending: true,
      });
      return next;
    });

    if (promptOverride === null) {
      setDraft("");
    }
    setIsSending(true);
    setActivity([]);
    activityStepRef.current = 0;

    const requestPayload = {
      message: trimmed,
      session_id: sessionId,
      fetch_news: fetchNews,
      num_articles: sanitizeNumArticles(numArticles),
      expert_execution_mode: sanitizeExpertMode(expertExecutionMode),
    };

    const applyFinalPayload = (payload) => {
      updateAssistantMessage(assistantId, {
        content: payload.response || "No response generated.",
        pending: false,
        metadata: {
          route: payload.route || null,
          fetched: Boolean(payload.fetched),
          articles_count: Number(payload.articles_count || 0),
          fetch_error: payload.fetch_error || null,
          expert_execution_mode: sanitizeExpertMode(payload.expert_execution_mode),
          workflow_elapsed_ms: Number(payload.workflow_elapsed_ms || 0),
          agent_timings_ms: payload.agent_timings_ms || {},
        },
      });
      if (payload.session_id) {
        setSessionId(payload.session_id);
      }

      if (Number(payload.workflow_elapsed_ms || 0) > 0) {
        addActivity(`Workflow latency: ${Number(payload.workflow_elapsed_ms).toFixed(1)} ms`, false);
      }
      const timingsSummary = formatAgentTimings(payload.agent_timings_ms || {});
      if (timingsSummary) {
        addActivity(`Node timings: ${timingsSummary}`, false);
      }
    };

    const fallbackToNonStream = async () => {
      addActivity("Streaming unavailable. Falling back to standard response endpoint.");

      const fallbackResponse = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestPayload),
        signal: controller.signal,
      });

      if (!fallbackResponse.ok) {
        throw new Error(`Fallback request failed (${fallbackResponse.status})`);
      }

      const fallbackPayload = await fallbackResponse.json();
      applyFinalPayload(fallbackPayload);
      addActivity("Fallback response completed.");
    };

    try {
      addActivity("Request sent to backend.");

      const response = await fetch(`${API_BASE}/chat/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestPayload),
        signal: controller.signal,
      });

      if (!response.ok) {
        throw new Error(`Request failed (${response.status})`);
      }

      if (!response.body || typeof response.body.getReader !== "function") {
        await fallbackToNonStream();
        const healthResponse = await fetch(`${API_BASE}/health`);
        if (healthResponse.ok) {
          setHealth(await healthResponse.json());
        }
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let terminalEventSeen = false;

      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }

        buffer += decoder.decode(value, { stream: true });
        const packets = buffer.split(/\r?\n\r?\n/);
        buffer = packets.pop() || "";

        for (const packet of packets) {
          const dataLine = extractSseData(packet);

          if (!dataLine) {
            continue;
          }

          try {
            const event = JSON.parse(dataLine);
            const isTerminal = handleStreamEvent(event, assistantId);
            if (isTerminal) {
              terminalEventSeen = true;
            }
          } catch {
            // Ignore malformed event chunks and continue streaming.
          }
        }
      }

      buffer += decoder.decode();

      if (buffer.trim()) {
        try {
          const trailing = extractSseData(buffer);
          if (trailing) {
            const trailingEvent = JSON.parse(trailing);
            const isTerminal = handleStreamEvent(trailingEvent, assistantId);
            if (isTerminal) {
              terminalEventSeen = true;
            }
          }
        } catch {
          // Ignore trailing parse errors.
        }
      }

      if (!terminalEventSeen) {
        updateAssistantMessage(assistantId, {
          content: "The stream ended before a final response was emitted. Please retry.",
          pending: false,
        });
        addActivity("Stream ended before final output.");
      }

      await Promise.all([loadHealth(), loadReadiness(), loadSources(sourceLimit), loadSessions()]);
    } catch (error) {
      if (error?.name === "AbortError") {
        updateAssistantMessage(assistantId, {
          content: "Request cancelled.",
          pending: false,
        });
        return;
      }

      try {
        await fallbackToNonStream();
        await Promise.all([loadHealth(), loadSources(sourceLimit), loadSessions()]);
      } catch (fallbackError) {
        updateAssistantMessage(assistantId, {
          content: `Error: ${fallbackError.message}`,
          pending: false,
        });
        addActivity(`Request failed: ${fallbackError.message}`);
      }
    } finally {
      setIsSending(false);
      requestControllerRef.current = null;
    }
  }

  function onComposerSubmit(event) {
    event.preventDefault();
    sendMessage();
  }

  function renderAssistantMeta(message) {
    if (!message.metadata || typeof message.metadata !== "object") {
      return null;
    }

    const route = message.metadata.route ? String(message.metadata.route).toUpperCase() : null;
    const mode = message.metadata.expert_execution_mode
      ? String(message.metadata.expert_execution_mode).toUpperCase()
      : null;
    const fetched = message.metadata.fetched ? "Fetched" : "Cached";
    const articlesCount = Number(message.metadata.articles_count || 0);
    const workflowMs = Number(message.metadata.workflow_elapsed_ms || 0);

    const metaParts = [route ? `Route ${route}` : "Route unknown", fetched];
    if (mode) {
      metaParts.push(`Mode ${mode}`);
    }
    if (articlesCount > 0) {
      metaParts.push(`${articlesCount} article(s)`);
    }
    if (workflowMs > 0) {
      metaParts.push(`${workflowMs.toFixed(1)} ms`);
    }

    return (
      <p className="message-meta">{metaParts.join(" | ")}</p>
    );
  }

  return (
    <div className="app-shell">
      <aside className="side-panel">
        <h1>Legal AI</h1>
        <p className="tagline">Thesis-grade legal intelligence with a multi-agent backend.</p>

        <input
          ref={fileInputRef}
          type="file"
          accept="application/json"
          className="hidden-file-input"
          onChange={importChatFromFile}
        />

        <section className="panel-card">
          <h2>Session Controls</h2>
          <p className="muted">{sessionId}</p>

          <div className="button-grid">
            <button type="button" className="ghost-button" onClick={startNewChat}>
              New Chat
            </button>
            <button
              type="button"
              className="ghost-button"
              onClick={retryLastPrompt}
              disabled={isSending || isClearing || !lastUserPrompt}
            >
              Retry Prompt
            </button>
            <button
              type="button"
              className="ghost-button"
              onClick={regenerateLastResponse}
              disabled={isSending || isClearing || !lastUserPrompt}
            >
              Regenerate
            </button>
            <button type="button" className="ghost-button" onClick={copyLastAssistant}>
              Copy Last Answer
            </button>
            <button type="button" className="ghost-button" onClick={exportChatAsJson} disabled={messages.length === 0}>
              Export JSON
            </button>
            <button type="button" className="ghost-button" onClick={exportChatAsMarkdown} disabled={messages.length === 0}>
              Export Markdown
            </button>
            <button type="button" className="ghost-button" onClick={openImportDialog}>
              Import Chat
            </button>
            <button type="button" className="ghost-button" onClick={clearActivityFeed}>
              Clear Workflow
            </button>
          </div>
        </section>

        <section className="panel-card">
          <h2>Status</h2>
          <p>{statusText}</p>
          <p className="muted">Readiness: {readinessText}</p>
          <p className="muted">API: {API_BASE}</p>
          {runtimeConfig && (
            <p className="muted small">
              Model {runtimeConfig.chat_model} | Embedding {runtimeConfig.embedding_model} | Default Mode {String(runtimeConfig.expert_execution_mode || "all").toUpperCase()} | Request Mode {String(expertExecutionMode).toUpperCase()}
            </p>
          )}
          <div className="button-row">
            <button type="button" className="ghost-button" onClick={loadHealth}>
              Refresh Health
            </button>
            <button type="button" className="ghost-button" onClick={loadReadiness}>
              Refresh Readiness
            </button>
          </div>
        </section>

        <section className="panel-card">
          <h2>Fetch Settings</h2>
          <label className="toggle-row">
            <input
              type="checkbox"
              checked={fetchNews}
              onChange={(event) => setFetchNews(event.target.checked)}
            />
            <span>Allow fresh news fetching</span>
          </label>

          <label className="field-label" htmlFor="articles-count">
            Articles per fetch
          </label>
          <input
            id="articles-count"
            className="number-input"
            type="number"
            min={1}
            max={10}
            value={numArticles}
            onChange={(event) => setNumArticles(sanitizeNumArticles(event.target.value))}
          />

          <div className="button-row">
            <button
              type="button"
              className="warn-button"
              onClick={clearData}
              disabled={isClearing || isSending}
            >
              {isClearing ? "Clearing..." : "Clear Local Data"}
            </button>
            <button
              type="button"
              className="warn-button"
              onClick={clearSavedSessions}
              disabled={isClearingSessions || isSending}
            >
              {isClearingSessions ? "Clearing..." : "Clear Sessions"}
            </button>
          </div>
        </section>

        <section className="panel-card">
          <h2>Saved Sessions</h2>
          <p className="muted">Total: {sessionsTotal}</p>
          <div className="button-row">
            <button type="button" className="ghost-button" onClick={loadSessions} disabled={isLoadingSessions}>
              {isLoadingSessions ? "Loading..." : "Refresh Sessions"}
            </button>
          </div>
          <div className="session-list">
            {sessions.length === 0 ? (
              <p className="muted">No saved sessions yet.</p>
            ) : (
              sessions.map((item) => (
                <article key={item.session_id} className="session-item">
                  <p className="session-id">{item.session_id}</p>
                  <p className="muted small">Messages: {item.message_count}</p>
                  {item.last_user_message && <p className="session-preview">{item.last_user_message}</p>}
                  <div className="button-row compact">
                    <button type="button" className="ghost-button" onClick={() => loadSessionById(item.session_id)}>
                      Load
                    </button>
                    <button type="button" className="warn-button" onClick={() => deleteSessionById(item.session_id)}>
                      Delete
                    </button>
                  </div>
                </article>
              ))
            )}
          </div>
        </section>

        <section className="panel-card">
          <h2>Sources</h2>
          <label className="field-label" htmlFor="source-limit">
            Source preview limit
          </label>
          <input
            id="source-limit"
            className="number-input"
            type="number"
            min={3}
            max={30}
            value={sourceLimit}
            onChange={(event) => setSourceLimit(Math.max(3, Math.min(30, Number(event.target.value || 12))))}
          />
          <button
            type="button"
            className="ghost-button"
            onClick={() => loadSources(sourceLimit)}
            disabled={isLoadingSources}
          >
            {isLoadingSources ? "Loading..." : "Refresh Sources"}
          </button>
          <p className="muted">Total articles tracked: {sources.total_articles}</p>
          <div className="source-list">
            {(sources.articles || []).slice(0, 5).map((article, index) => (
              <article key={`${article.url || article.title || "source"}_${index}`} className="source-item">
                <p className="source-title">{article.title || "Untitled source"}</p>
                <p className="muted small">{article.source || "Unknown source"}</p>
              </article>
            ))}
          </div>
        </section>
      </aside>

      <main className="chat-panel">
        <header className="chat-header">
          <h2>Assistant</h2>
          <p>Ask legal, compliance, and AI regulation questions.</p>
          <div className="quick-prompts">
            {QUICK_PROMPTS.map((prompt) => (
              <button
                key={prompt}
                type="button"
                className="quick-prompt"
                onClick={() => applyQuickPrompt(prompt)}
                disabled={isSending || isClearing}
              >
                {prompt}
              </button>
            ))}
          </div>
          <div className="header-links">
            <a href={`${API_BASE}/docs`} target="_blank" rel="noreferrer">
              API Docs
            </a>
            <button
              type="button"
              className="ghost-button"
              onClick={() => setShowActivity((prev) => !prev)}
            >
              {showActivity ? "Hide Workflow" : "Show Workflow"}
            </button>
          </div>
        </header>

        <section className="messages" aria-live="polite">
          {messages.length === 0 && (
            <article className="message assistant intro">
              <p>
                Ask about the EU AI Act, recent regulatory updates, or compliance obligations.
              </p>
            </article>
          )}

          {messages.map((message) => (
            <article key={message.id} className={`message ${message.role} ${message.pending ? "pending" : ""}`}>
              {message.role === "assistant" ? (
                <div className="markdown">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
                  {renderAssistantMeta(message)}
                </div>
              ) : (
                <p>{message.content}</p>
              )}
            </article>
          ))}

          <div ref={endRef} />
        </section>

        {showActivity && (
          <section className="activity-feed">
            <h3>Live Workflow</h3>
            <div className="activity-list">
              {activity.length === 0 ? (
                <p className="muted">No workflow events yet.</p>
              ) : (
                activity.map((line, index) => (
                  <p key={`${line}_${index}`} className="activity-item">
                    {line}
                  </p>
                ))
              )}
            </div>
          </section>
        )}

        <form className="composer" onSubmit={onComposerSubmit}>
          <textarea
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            placeholder="Ask a legal AI question..."
            rows={3}
            disabled={isSending || isClearing}
          />
          <div className="composer-buttons">
            <label className="composer-mode" htmlFor="expert-mode-composer">
              <span>Mode</span>
              <select
                id="expert-mode-composer"
                className="mode-select"
                value={expertExecutionMode}
                onChange={(event) => setExpertExecutionMode(sanitizeExpertMode(event.target.value))}
                disabled={isSending || isClearing}
              >
                <option value="all">All Experts (Sequential)</option>
                <option value="single">Single Expert (Routed)</option>
                <option value="parallel">Parallel Experts (All Concurrently)</option>
                <option value="legal_news_parallel">Parallel Legal & News</option>
                <option value="legal_first">Sequential Legal ➔ News</option>
                <option value="verify_only">Retrieval Only (Bypass Experts)</option>
              </select>
            </label>
            <button type="submit" disabled={isSending || isClearing || !draft.trim()}>
              {isSending ? "Sending..." : "Send"}
            </button>
            <button
              type="button"
              className="warn-button"
              disabled={!isSending}
              onClick={abortCurrentRequest}
            >
              Stop
            </button>
          </div>
        </form>
      </main>
    </div>
  );
}

export default App;
