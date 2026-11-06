(() => {
  "use strict";

  const STORAGE_KEYS = {
    settings: "clean-history-rag.settings.v1",
    legacyWorkspace: "clean-history-rag.workspace.v1",
    scopedWorkspacePrefix: "clean-history-rag.workspace.v2.",
    legacyWorkspaceMigration: "clean-history-rag.workspace.v2.migrated-from-v1",
  };
  function uiText(english, hebrew) {
    if (window.ProjectXUI?.text) return window.ProjectXUI.text(english, hebrew);
    return window.ProjectXUI?.language === "he" ? hebrew : english;
  }

  function uiIsHebrew() {
    return window.ProjectXUI?.language === "he";
  }

  const DEFAULT_API_BASE_URL =
    window.location && /^https?:$/.test(window.location.protocol)
      ? window.location.origin
      : "http://localhost:8000";
  const DEFAULT_SETTINGS = {
    apiBaseUrl: DEFAULT_API_BASE_URL,
    tenantId: "",
    theme: "light",
    processingMotion: "animated",
    requestTimeoutSeconds: 1200,
  };
  const MAX_HISTORY_TURNS = 12;
  const MAX_HISTORY_TURN_CHARACTERS = 6000;
  const MAX_HISTORY_TOTAL_CHARACTERS = 24000;
  const MAX_QUERY_CHARACTERS = 4000;
  const MAX_API_RESPONSE_BYTES = 2 * 1024 * 1024;
  const MIN_REQUEST_TIMEOUT_SECONDS = 1200;
  const MAX_REQUEST_TIMEOUT_SECONDS = 1200;
  const TENANT_ID_PATTERN = /^[A-Za-z0-9](?:[A-Za-z0-9_-]{0,62}[A-Za-z0-9])?$/;
  const SOURCE_ID_PATTERN = /^[A-Za-z0-9_-]{43}$/;
  const CONVERSATION_ID_PATTERN = /^[A-Za-z0-9_-]{8,64}$/;
  const SUPPORTED_MODEL_BACKENDS = new Set(["gemma", "llama_cpp"]);
  const frontendPolicy = window.ProjectXFrontendPolicy;
  if (!frontendPolicy) throw new Error("Frontend security policy failed to load.");

  const elements = {
    appShell: document.getElementById("app-shell"),
    main: document.querySelector("main"),
    sidebar: document.getElementById("sidebar"),
    sidebarBackdrop: document.getElementById("sidebar-backdrop"),
    mobileMenuButton: document.getElementById("mobile-menu-btn"),
    newThreadButton: document.getElementById("new-thread-btn"),
    threadSearch: document.getElementById("thread-search"),
    threadSearchClear: document.getElementById("thread-search-clear"),
    threadList: document.getElementById("thread-list"),
    threadTitle: document.getElementById("thread-title"),
    chatHeader: document.getElementById("chat-header"),
    chatHistory: document.getElementById("chat-history"),
    scrollToLatestButton: document.getElementById("scroll-to-latest-btn"),
    starterPrompts: document.getElementById("starter-prompts"),
    composerShortcuts: document.getElementById("composer-shortcuts"),
    chatInput: document.getElementById("chat-input"),
    composerHelp: document.getElementById("composer-help"),
    composerStatusCopy: document.getElementById("composer-status-copy"),
    sendButton: document.getElementById("send-btn"),
    sendButtonIcon: document.getElementById("send-btn-icon"),
    connectionStatusButton: document.getElementById("connection-status-button"),
    connectionStatusLabel: document.getElementById("connection-status-label"),
    exportThreadButton: document.getElementById("export-thread-btn"),
    sourcesToggleButton: document.getElementById("sources-toggle-btn"),
    sourcesCountBadge: document.getElementById("sources-count-badge"),
    sourcePanel: document.getElementById("source-panel"),
    sourcePanelBackdrop: document.getElementById("source-panel-backdrop"),
    sourcePanelCloseButton: document.getElementById("source-panel-close"),
    sourceUniqueCount: document.getElementById("source-unique-count"),
    sourceReferenceCount: document.getElementById("source-reference-count"),
    sourceSearch: document.getElementById("source-search"),
    sourceSearchClear: document.getElementById("source-search-clear"),
    sourcePanelContent: document.getElementById("source-panel-content"),
    tenantDisplay: document.getElementById("tenant-display"),
    documentInventory: document.getElementById("document-inventory"),
    chatBackend: document.getElementById("chat-backend"),
    answerModelName: document.getElementById("answer-model-name"),
    modelState: document.getElementById("model-state"),
    modelRetry: document.getElementById("model-retry"),
    settingsOverlay: document.getElementById("settings-overlay"),
    settingsForm: document.getElementById("settings-dialog"),
    settingsOpenButton: document.getElementById("settings-open-btn"),
    settingsCloseButton: document.getElementById("settings-close-btn"),
    settingsCancelButton: document.getElementById("settings-cancel-btn"),
    settingsSaveButton: document.getElementById("settings-save-btn"),
    apiBaseUrlInput: document.getElementById("api-base-url"),
    tenantIdInput: document.getElementById("tenant-id"),
    requestTimeoutInput: document.getElementById("request-timeout-seconds"),
    themeToggleButton: document.getElementById("theme-toggle-btn"),
    themeToggleKnob: document.getElementById("theme-toggle-knob"),
    processingMotionInput: document.getElementById("processing-motion"),
    settingsError: document.getElementById("settings-error"),
    settingsTestConnectionButton: document.getElementById("settings-test-connection-btn"),
    settingsConnectionStatus: document.getElementById("settings-connection-status"),
    settingsConnectionLabel: document.getElementById("settings-connection-label"),
    settingsConnectionDetail: document.getElementById("settings-connection-detail"),
    threadActionOverlay: document.getElementById("thread-action-overlay"),
    threadActionDialog: document.getElementById("thread-action-dialog"),
    threadActionTitle: document.getElementById("thread-action-title"),
    threadActionDescription: document.getElementById("thread-action-description"),
    threadActionVisual: document.getElementById("thread-action-visual"),
    threadActionField: document.getElementById("thread-action-field"),
    threadActionLabel: document.getElementById("thread-action-label"),
    threadActionInput: document.getElementById("thread-action-input"),
    threadActionError: document.getElementById("thread-action-error"),
    threadActionCloseButton: document.getElementById("thread-action-close-btn"),
    threadActionCancelButton: document.getElementById("thread-action-cancel-btn"),
    threadActionConfirmButton: document.getElementById("thread-action-confirm-btn"),
    helpOverlay: document.getElementById("help-overlay"),
    helpDialog: document.getElementById("help-dialog"),
    helpOpenButton: document.getElementById("help-open-btn"),
    helpCloseButton: document.getElementById("help-close-btn"),
    helpDoneButton: document.getElementById("help-done-btn"),
    appStatus: document.getElementById("app-status"),
    toastRegion: document.getElementById("toast-region"),
  };

  let settings = loadSettings();
  let activeWorkspaceKey = scopedWorkspaceKey(settings);
  let workspace = loadWorkspace(settings, { migrateLegacy: true });
  const pendingRequests = new Map();
  const threadErrors = new Map();
  const citationHighlightTimers = new WeakMap();
  const citationDisclosureState = new Map();
  let settingsThemeDraft = settings.theme;
  let themeBeforeDialog = settings.theme;
  let lastFocusedElement = null;
  let sidebarReturnFocus = null;
  let sourcePanelReturnFocus = null;
  let threadSearchQuery = "";
  let sourceSearchQuery = "";
  let inventoryController = null;
  let activeDocumentSourceIds = null;
  let inventoryBackendIdentity = "";
  let healthCheckController = null;
  let settingsTestController = null;
  let statusAnnouncementTimer = null;
  let draftPersistTimer = null;
  let workspaceGeneration = 0;
  let threadActionState = null;
  let threadActionReturnFocus = null;

  function makeId(prefix) {
    if (window.crypto && typeof window.crypto.randomUUID === "function") {
      return `${prefix}-${window.crypto.randomUUID()}`;
    }
    return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  }

  function newThread() {
    const now = new Date().toISOString();
    return {
      id: makeId("thread"),
      title: "Untitled thread",
      createdAt: now,
      updatedAt: now,
      messages: [],
      draft: "",
      selectedSourceId: "",
      conversationId: "",
    };
  }

  function safeParse(value) {
    try {
      return JSON.parse(value);
    } catch (_error) {
      return null;
    }
  }

  async function readBoundedResponseText(
    response,
    maxBytes = MAX_API_RESPONSE_BYTES,
  ) {
    const declaredValue = response?.headers?.get?.("Content-Length")?.trim() || "";
    if (/^\d+$/.test(declaredValue) && Number(declaredValue) > maxBytes) {
      throw new Error(`The server response exceeded the ${maxBytes}-byte safety limit.`);
    }

    const stream = response?.body;
    if (!stream || typeof stream.getReader !== "function") {
      const text = await response.text();
      if (new Blob([text]).size > maxBytes) {
        throw new Error(`The server response exceeded the ${maxBytes}-byte safety limit.`);
      }
      return text;
    }

    const reader = stream.getReader();
    const decoder = new TextDecoder("utf-8");
    const parts = [];
    let receivedBytes = 0;
    try {
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        receivedBytes += value?.byteLength || 0;
        if (receivedBytes > maxBytes) {
          await reader.cancel("Response safety limit exceeded");
          throw new Error(`The server response exceeded the ${maxBytes}-byte safety limit.`);
        }
        parts.push(decoder.decode(value, { stream: true }));
      }
      parts.push(decoder.decode());
      return parts.join("");
    } finally {
      reader.releaseLock();
    }
  }

  async function readBoundedResponseJson(response) {
    const text = await readBoundedResponseText(response);
    if (!text.trim()) throw new Error("The server returned an empty JSON response.");
    return JSON.parse(text);
  }

  function normalizeApiBaseUrl(value) {
    const raw = String(value || "").trim();
    try {
      const parsed = new URL(raw || DEFAULT_API_BASE_URL);
      if (parsed.protocol !== "http:" && parsed.protocol !== "https:") throw new Error();
      parsed.hash = "";
      parsed.search = "";
      parsed.pathname = parsed.pathname.replace(/\/+$/, "") || "/";
      return parsed.href.replace(/\/+$/, "");
    } catch (_error) {
      return (raw || DEFAULT_API_BASE_URL).replace(/\/+$/, "");
    }
  }

  function normalizeTenantId(value) {
    return typeof value === "string" ? value.trim() : "";
  }

  function normalizeSourceOpenUrl(value, sourceId, effectiveTenant) {
    return frontendPolicy.normalizeSourceOpenUrl(
      value,
      sourceId,
      effectiveTenant,
      settings?.apiBaseUrl || DEFAULT_API_BASE_URL,
    );
  }

  function connectionIdentity(value) {
    return JSON.stringify([
      normalizeApiBaseUrl(value?.apiBaseUrl),
      normalizeTenantId(value?.tenantId),
    ]);
  }

  function scopedWorkspaceKey(value) {
    return `${STORAGE_KEYS.scopedWorkspacePrefix}${encodeURIComponent(connectionIdentity(value))}`;
  }

  function readStorage(key) {
    try {
      return localStorage.getItem(key);
    } catch (error) {
      console.warn("Could not read browser storage", error);
      return null;
    }
  }

  function writeStorage(key, value) {
    try {
      localStorage.setItem(key, value);
      return true;
    } catch (error) {
      console.warn("Could not write browser storage", error);
      return false;
    }
  }

  function loadSettings() {
    const saved = safeParse(readStorage(STORAGE_KEYS.settings));
    const next = saved && typeof saved === "object" && !Array.isArray(saved) ? saved : {};
    // Existing system defaults predate the lab owner's explicit animation choice.
    // Migrate only this preference once; preserve static and all unrelated fields.
    if (saved && next === saved && !(Number.isInteger(next.processingMotionRevision) && next.processingMotionRevision >= 1)) {
      const processingMotion = ["animated", "static"].includes(next.processingMotion)
        ? next.processingMotion : DEFAULT_SETTINGS.processingMotion;
      const migrated = { ...next, processingMotion, processingMotionRevision: 1 };
      writeStorage(STORAGE_KEYS.settings, JSON.stringify(migrated));
      Object.assign(next, migrated);
    }
    const savedTimeout = Number(next.requestTimeoutSeconds);
    return {
      apiBaseUrl:
        typeof next.apiBaseUrl === "string" && next.apiBaseUrl.trim()
          ? normalizeApiBaseUrl(next.apiBaseUrl)
          : normalizeApiBaseUrl(DEFAULT_SETTINGS.apiBaseUrl),
      tenantId: normalizeTenantId(next.tenantId),
      theme: next.theme === "dark" ? "dark" : "light",
      processingMotion: ["system", "animated", "static"].includes(next.processingMotion)
        ? next.processingMotion : DEFAULT_SETTINGS.processingMotion,
      requestTimeoutSeconds:
        Number.isInteger(savedTimeout) &&
        savedTimeout >= MIN_REQUEST_TIMEOUT_SECONDS &&
        savedTimeout <= MAX_REQUEST_TIMEOUT_SECONDS
          ? savedTimeout
          : DEFAULT_SETTINGS.requestTimeoutSeconds,
    };
  }

  function normalizeCitation(citation, index, effectiveTenant) {
    const value = citation && typeof citation === "object" ? citation : {};
    const chunkId = value.chunk_id ?? value.chunkId;
    const chunkIndex = value.chunk_index ?? value.chunkIndex;
    const sourceIdValue = value.source_id ?? value.sourceId;
    const sourceId = sourceIdValue == null ? "" : String(sourceIdValue);
    const pageValue = value.page;
    const page = Number.isInteger(pageValue) && pageValue > 0 ? pageValue : null;
    const evidencePageValue = value.evidence_page ?? value.evidencePage;
    const evidencePage =
      Number.isInteger(evidencePageValue) && evidencePageValue > 0
        ? evidencePageValue
        : null;
    const evidenceQuoteValue = value.evidence_quote ?? value.evidenceQuote;
    const evidenceQuote =
      typeof evidenceQuoteValue === "string" &&
      Array.from(evidenceQuoteValue).length <= 820
        ? evidenceQuoteValue
        : "";
    const evidenceLanguageValue = String(
      value.evidence_quote_language ?? value.evidenceQuoteLanguage ?? "",
    );
    const evidenceQuoteLanguage = ["he", "en", "mixed", "und"].includes(
      evidenceLanguageValue,
    )
      ? evidenceLanguageValue
      : "";
    const evidenceQuoteDirection = frontendPolicy.normalizeTextDirection(
      value.evidence_quote_direction ?? value.evidenceQuoteDirection,
    );
    const evidenceStatusValue = String(
      value.evidence_quote_status ?? value.evidenceQuoteStatus ?? "quote_unavailable",
    );
    const evidenceQuoteStatus = [
      "verified",
      "verified_duplicate",
      "quote_unavailable",
      "omitted_answer_limit",
    ].includes(evidenceStatusValue)
      ? evidenceStatusValue
      : "quote_unavailable";
    const evidenceChunkIdValue = String(
      value.evidence_chunk_id ?? value.evidenceChunkId ?? "",
    );
    const evidenceDuplicateValue =
      value.evidence_duplicate_of ?? value.evidenceDuplicateOf;
    const displayNameValue = frontendPolicy.safeCitationDisplayName(
      value.display_name ?? value.displayName ?? value.file,
    );
    const sourceRelativePath = frontendPolicy.safeCitationRelativePath(
      value.source_relative_path ?? value.sourceRelativePath,
    );
    return {
      id: value.id == null ? String(index + 1) : String(value.id),
      citationNumber:
        Number.isInteger(value.citation_number) && value.citation_number > 0
          ? value.citation_number
          : index + 1,
      file: sourceRelativePath || displayNameValue,
      displayName: sourceRelativePath || displayNameValue,
      sourceRelativePath,
      location: value.location == null ? "" : String(value.location),
      score: value.score == null ? "" : String(value.score),
      snippet: value.snippet == null ? "" : String(value.snippet),
      chunkId: chunkId == null ? "" : String(chunkId),
      chunkIndex:
        typeof chunkIndex === "number" && Number.isFinite(chunkIndex)
          ? chunkIndex
          : null,
      sourceId: SOURCE_ID_PATTERN.test(sourceId) ? sourceId : "",
      mimeType: value.mime_type == null && value.mimeType == null
        ? "" : String(value.mime_type ?? value.mimeType),
      page,
      slide: Number.isInteger(value.slide) && value.slide > 0 ? value.slide : null,
      sheet: value.sheet == null ? "" : String(value.sheet),
      row: Number.isInteger(value.row) && value.row > 0 ? value.row : null,
      cell: value.cell == null ? "" : String(value.cell),
      paragraph: Number.isInteger(value.paragraph) && value.paragraph > 0 ? value.paragraph : null,
      section: value.section == null ? "" : String(value.section),
      presentationGroupId: /^[a-f0-9]{64}$/.test(
        String(value.presentation_group_id ?? value.presentationGroupId ?? ""),
      ) ? String(value.presentation_group_id ?? value.presentationGroupId) : "",
      evidenceQuote:
        ["verified", "verified_duplicate"].includes(evidenceQuoteStatus) ? evidenceQuote : "",
      evidenceQuoteLanguage,
      evidenceQuoteDirection,
      evidencePage,
      evidenceLocationLabel:
        value.evidence_location_label == null &&
        value.evidenceLocationLabel == null
          ? ""
          : String(value.evidence_location_label ?? value.evidenceLocationLabel),
      evidenceQuoteTruncated:
        (value.evidence_quote_truncated ?? value.evidenceQuoteTruncated) === true,
      evidenceChunkId: /^[A-Za-z0-9_-]{24}$/.test(evidenceChunkIdValue)
        ? evidenceChunkIdValue
        : "",
      evidenceQuoteStatus,
      evidenceDuplicateOf:
        evidenceDuplicateValue == null ? "" : String(evidenceDuplicateValue),
      openUrl: normalizeSourceOpenUrl(
        value.open_url ?? value.openUrl,
        sourceId,
        effectiveTenant,
      ),
    };
  }

  function normalizeAnswerCompleteness(value) {
    return ["complete", "partial", "unavailable"].includes(value) ? value : null;
  }

  function normalizePartialReasons(value) {
    if (!Array.isArray(value)) return [];
    return [...new Set(value.filter((reason) =>
      ["source_context_partial", "source_coverage_partial"].includes(reason)))];
  }

  function normalizeMessage(message) {
    if (!message || (message.role !== "user" && message.role !== "assistant")) {
      return null;
    }
    if (typeof message.content !== "string") return null;
    const effectiveTenant = message.role === "assistant"
      ? normalizeTenantId(
          message.effective_tenant ??
          message.effectiveTenant ??
          "",
        )
      : "";
    const citations = Array.isArray(message.citations)
      ? message.citations.map((citation, index) =>
          normalizeCitation(citation, index, effectiveTenant))
      : [];
    return {
      id: typeof message.id === "string" ? message.id : makeId("message"),
      role: message.role,
      content: message.content,
      answerDirection: frontendPolicy.normalizeTextDirection(
        message.answer_direction ?? message.answerDirection,
      ),
      citations,
      effectiveTenant,
      answerCompleteness: message.role === "assistant"
        ? normalizeAnswerCompleteness(message.answer_completeness ?? message.answerCompleteness)
        : null,
      responseReasonCodes: message.role === "assistant"
        ? normalizePartialReasons(message.response_reason_codes ?? message.responseReasonCodes)
        : [],
      referenceRecovery: message.role === "assistant" &&
        (message.reference_recovery ?? message.referenceRecovery) === "new_search"
        ? "new_search" : null,
      backend: typeof message.backend === "string" ? message.backend : "gemma",
      retiredCorpus: message.retiredCorpus === true,
      selectedSourceId:
        message.role === "user" &&
        SOURCE_ID_PATTERN.test(String(message.selectedSourceId || ""))
          ? String(message.selectedSourceId)
          : "",
      createdAt:
        typeof message.createdAt === "string"
          ? message.createdAt
          : new Date().toISOString(),
      durationMs:
        typeof message.durationMs === "number" &&
        Number.isFinite(message.durationMs) &&
        message.durationMs >= 0
          ? message.durationMs
          : null,
    };
  }

  function normalizeThread(thread) {
    if (!thread || typeof thread !== "object" || typeof thread.id !== "string") {
      return null;
    }
    const messages = Array.isArray(thread.messages)
      ? thread.messages.map(normalizeMessage).filter(Boolean)
      : [];
    const now = new Date().toISOString();
    return {
      id: thread.id,
      title:
        typeof thread.title === "string" && thread.title.trim()
          ? thread.title.trim()
          : "Untitled thread",
      createdAt: typeof thread.createdAt === "string" ? thread.createdAt : now,
      updatedAt: typeof thread.updatedAt === "string" ? thread.updatedAt : now,
      messages,
      draft: typeof thread.draft === "string"
        ? thread.draft.slice(0, MAX_QUERY_CHARACTERS)
        : "",
      selectedSourceId: SOURCE_ID_PATTERN.test(
        String(thread.selectedSourceId || ""),
      )
        ? String(thread.selectedSourceId)
        : "",
      conversationId: CONVERSATION_ID_PATTERN.test(
        String(thread.conversationId || ""),
      )
        ? String(thread.conversationId)
        : "",
    };
  }

  function normalizeWorkspace(saved) {
    const threads =
      saved && Array.isArray(saved.threads)
        ? saved.threads.map(normalizeThread).filter(Boolean)
        : [];
    if (!threads.length) threads.push(newThread());
    const activeThreadId = threads.some((thread) => thread.id === saved?.activeThreadId)
      ? saved.activeThreadId
      : threads[0].id;
    return { activeThreadId, threads };
  }

  function loadWorkspace(workspaceSettings, { migrateLegacy = false } = {}) {
    const scopeKey = scopedWorkspaceKey(workspaceSettings);
    const scoped = safeParse(readStorage(scopeKey));
    if (scoped && typeof scoped === "object") return normalizeWorkspace(scoped);

    const migrationRecorded = Boolean(readStorage(STORAGE_KEYS.legacyWorkspaceMigration));
    if (migrateLegacy && !migrationRecorded) {
      const legacy = safeParse(readStorage(STORAGE_KEYS.legacyWorkspace));
      if (legacy && typeof legacy === "object") {
        const migrated = normalizeWorkspace(legacy);
        if (writeStorage(scopeKey, JSON.stringify(migrated))) {
          writeStorage(STORAGE_KEYS.legacyWorkspaceMigration, connectionIdentity(workspaceSettings));
        }
        return migrated;
      }
    }
    return normalizeWorkspace(null);
  }

  function persistSettings(value = settings) {
    return writeStorage(STORAGE_KEYS.settings, JSON.stringify(value));
  }

  function persistWorkspace() {
    return writeStorage(activeWorkspaceKey, JSON.stringify(workspace));
  }

  function getActiveThread() {
    return workspace.threads.find((thread) => thread.id === workspace.activeThreadId);
  }

  function getThread(threadId) {
    return workspace.threads.find((thread) => thread.id === threadId);
  }

  function textElement(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    node.textContent = text;
    return node;
  }

  function icon(name, size = "18px") {
    const paths = {
      chat_bubble: "M5.5 5.5h13v9h-7l-4 3v-3h-2z",
      edit: "M5 19h3.2L18.7 8.5l-3.2-3.2L5 15.8V19Zm8.9-12.1 3.2 3.2",
      delete: "M6.5 7.5h11M9 7.5V5h6v2.5M8 9.5l.7 9h6.6l.7-9",
      content_copy: "M8 8h10v11H8zM6 16H4V5h10v2",
      refresh: "M19 8a7 7 0 1 0 1 7M19 4v4h-4",
      stop: "M8 8h8v8H8z",
      arrow_up_right: "M8 16 16 8M9 8h7v7",
      sparkles: "m12 4 1.2 3.3L16.5 8.5l-3.3 1.2L12 13l-1.2-3.3-3.3-1.2 3.3-1.2L12 4Zm6 10 .7 1.8 1.8.7-1.8.7L18 19l-.7-1.8-1.8-.7 1.8-.7L18 14Z",
    };
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 24 24");
    svg.setAttribute("aria-hidden", "true");
    svg.classList.add("ui-icon");
    svg.style.width = size;
    svg.style.height = size;
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("d", paths[name] || paths.chat_bubble);
    svg.append(path);
    return svg;
  }

  function announce(message) {
    if (!elements.appStatus) return;
    window.clearTimeout(statusAnnouncementTimer);
    elements.appStatus.textContent = "";
    statusAnnouncementTimer = window.setTimeout(() => {
      elements.appStatus.textContent = String(message || "");
    }, 30);
  }

  function showToast(message) {
    if (!elements.toastRegion || !message) return;
    const toast = textElement("div", "toast", message);
    toast.setAttribute("role", "status");
    elements.toastRegion.append(toast);
    window.setTimeout(() => toast.remove(), 3200);
  }

  function normalizedSearchText(value) {
    return String(value || "")
      .normalize("NFKC")
      .toLocaleLowerCase()
      .replace(/\s+/g, " ")
      .trim();
  }

  function usesHebrew(text) {
    return frontendPolicy.usesHebrew(text);
  }

  function compactSourceName(value) {
    const compact = String(value || "Unknown source").replace(/\s+/g, " ").trim();
    const characters = Array.from(compact);
    return characters.length > 120
      ? `${characters.slice(0, 119).join("")}…`
      : compact;
  }

  function compactAnswerExcerpt(value) {
    const compact = String(value || "").replace(/\s+/g, " ").trim();
    const characters = Array.from(compact);
    return characters.length > 180
      ? `${characters.slice(0, 179).join("")}…`
      : compact;
  }

  function bidiIsolate(value) {
    return `\u2068${value}\u2069`;
  }

  function cancelDraftPersistTimer() {
    if (draftPersistTimer == null) return;
    window.clearTimeout(draftPersistTimer);
    draftPersistTimer = null;
  }

  function syncActiveDraftFromComposer({ persist = false } = {}) {
    const thread = getActiveThread();
    if (!thread) return true;
    thread.draft = elements.chatInput.value.slice(0, MAX_QUERY_CHARACTERS);
    cancelDraftPersistTimer();
    if (persist) {
      return persistWorkspace();
    }
    draftPersistTimer = window.setTimeout(() => {
      draftPersistTimer = null;
      persistWorkspace();
    }, 300);
    return true;
  }

  function flushActiveDraft() {
    return syncActiveDraftFromComposer({ persist: true });
  }

  function applyComposerDraft(nextDraft) {
    elements.chatInput.value = nextDraft;
    syncActiveDraftFromComposer({ persist: true });
    resizeInput();
    elements.chatInput.focus();
    const end = elements.chatInput.value.length;
    if (typeof elements.chatInput.setSelectionRange === "function") {
      elements.chatInput.setSelectionRange(end, end);
    }
    if (typeof elements.chatInput.scrollIntoView === "function") {
      elements.chatInput.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }

  function prefillComposer(prompt, { selectedSourceId = "" } = {}) {
    const thread = getActiveThread();
    if (!thread) return;
    const nextDraft = String(prompt || "").slice(0, MAX_QUERY_CHARACTERS);
    const nextSourceId = SOURCE_ID_PATTERN.test(String(selectedSourceId || ""))
      ? String(selectedSourceId)
      : "";
    const applyPrefill = () => {
      thread.selectedSourceId = nextSourceId;
      applyComposerDraft(nextDraft);
    };
    const currentDraft = elements.chatInput.value;
    if (currentDraft.trim() && currentDraft !== nextDraft) {
      openThreadActionDialog({
        kind: "replace-draft",
        title: "Replace your current draft?",
        description: "The text you already entered will be replaced with the selected suggestion.",
        confirmLabel: "Replace draft",
        onConfirm: applyPrefill,
      });
      return;
    }
    applyPrefill();
  }

  function promptButton(
    label,
    prompt,
    action,
    className = "prompt-chip",
    selectedSourceId = "",
  ) {
    const button = textElement("button", className, label);
    button.type = "button";
    button.dir = "auto";
    button.dataset.action = action;
    button.addEventListener("click", () =>
      prefillComposer(prompt, { selectedSourceId }));
    return button;
  }

  function sourceQuestion(citation, answer, hebrew) {
    const source = bidiIsolate(compactSourceName(citation.file));
    const answerExcerpt = bidiIsolate(compactAnswerExcerpt(answer));
    return hebrew
      ? `בהסתמך על המקור \"${source}\" בלבד, בדוק את התמיכה בתשובה \"${answerExcerpt}\" וציין הסתייגויות או פערים.`
      : `Using only the source "${source}", assess support for the answer "${answerExcerpt}" and identify any caveats or gaps.`;
  }

  function citationLocationLabel(citation, hebrew = false) {
    const page = citation?.evidencePage || citation?.page;
    if (page) return hebrew ? `עמ׳ ${page}` : `Page ${page}`;
    if (citation?.sheet) {
      return hebrew ? `גיליון: ${citation.sheet}` : `Sheet: ${citation.sheet}`;
    }
    return citation?.evidenceLocationLabel || citation?.section || citation?.location || "";
  }

  function sourceOpenLabel(citation) {
    const name = compactSourceName(citation.displayName || citation.file);
    const evidenceLocation = citationLocationLabel(citation, uiIsHebrew());
    const location = evidenceLocation ? `, ${evidenceLocation}` : "";
    return uiText(`Open source: ${name}${location}`, `פתיחת מקור: ${name}${location}`);
  }

  function sourceOpenAnchor(citation, className, label = uiText("Open original", "פתיחת המקור")) {
    if (!citation?.openUrl) return null;
    const anchor = textElement("a", className, label);
    anchor.href = citation.openUrl;
    anchor.target = "_blank";
    anchor.rel = "noopener noreferrer";
    anchor.setAttribute("aria-label", sourceOpenLabel(citation));
    anchor.title = sourceOpenLabel(citation);
    return anchor;
  }

  function followUpSuggestions(message) {
    const hebrew = uiIsHebrew();
    const hasNumbers = /\d/.test(message.content);
    const secondLabel = message.citations.length > 1
      ? "Compare the sources"
      : message.citations.length === 1
        ? "Show the strongest evidence"
        : hasNumbers
          ? "Explain the numbers"
          : "Give a concrete example";
    const englishPrompts = [
      "Summarize the answer in three short points",
      message.citations.length > 1
        ? "Where do the sources agree or differ?"
        : message.citations.length === 1
          ? "What is the strongest evidence in the source?"
          : hasNumbers
            ? "Explain the numbers and assumptions"
            : "Give a concrete example from the available documents",
      "What remains uncertain or needs verification?",
    ];
    const hebrewPrompts = [
      "סכם את התשובה בשלוש נקודות קצרות",
      message.citations.length > 1
        ? "במה המקורות מסכימים ובמה הם שונים?"
        : message.citations.length === 1
          ? "איזו ראיה במקור תומכת הכי חזק במסקנה?"
          : hasNumbers
            ? "הסבר את המספרים וההנחות שמאחוריהם"
            : "תן דוגמה מעשית מתוך המסמכים הזמינים",
      "מה עדיין לא ודאי או דורש בדיקה נוספת?",
    ];
    const labels = [
      uiText("Summarize in 3 points", "סיכום בשלוש נקודות"),
      hebrew ? (message.citations.length > 1 ? "השוואת המקורות" : message.citations.length === 1 ? "הצגת הראיה המרכזית" : hasNumbers ? "הסבר המספרים" : "דוגמה מעשית") : secondLabel,
      uiText("What remains uncertain?", "מה עדיין אינו ודאי?"),
    ];
    const prompts = hebrew ? hebrewPrompts : englishPrompts;
    return labels.map((label, index) => ({ label, prompt: prompts[index] }));
  }

  function fallbackCopyText(value) {
    const textarea = document.createElement("textarea");
    textarea.value = value;
    textarea.setAttribute("readonly", "");
    textarea.style.position = "fixed";
    textarea.style.opacity = "0";
    textarea.style.pointerEvents = "none";
    document.body.append(textarea);
    textarea.select();
    let copied = false;
    try {
      copied = typeof document.execCommand === "function" && document.execCommand("copy");
    } finally {
      textarea.remove();
    }
    if (!copied) throw new Error("Copy is unavailable in this browser.");
  }

  async function writeClipboard(value) {
    if (navigator.clipboard && typeof navigator.clipboard.writeText === "function") {
      try {
        await navigator.clipboard.writeText(value);
        return;
      } catch (_error) {
        // The fallback supports local HTTP pages where Clipboard API permission is denied.
      }
    }
    fallbackCopyText(value);
  }

  function answerWithSourcesText(message) {
    const presentation = citationPresentation(message);
    const sections = [presentation.text];
    if (!message.citations.length) return message.content;
    const hebrew = uiIsHebrew();
    sections.push("", uiText("Sources", "מקורות"));
    presentation.groups.forEach((target) => {
      const citation = target.citation;
      const details = citationLocationLabel(citation, hebrew);
      sections.push("[" + target.displayNumber + "] " +
        (citation.file || uiText("Unknown source", "מקור לא ידוע")) + (details ? " — " + details : ""));
      if (citation.openUrl) sections.push("  " + citation.openUrl);
      const seenQuotes = new Set();
      target.members.forEach(({ citation: member }) => {
        if (["verified", "verified_duplicate"].includes(member.evidenceQuoteStatus) && member.evidenceQuote && !seenQuotes.has(member.evidenceQuote)) {
          seenQuotes.add(member.evidenceQuote);
          sections.push("  " + uiText("Authenticated source excerpt", "ציטוט מאומת מהמקור") +
            ": “" + member.evidenceQuote + "”");
        } else if (member.evidenceQuoteStatus === "verified_duplicate" && member.evidenceDuplicateOf) {
          const original = presentation.targets.get(member.evidenceDuplicateOf);
          if (original && original !== target) sections.push("  " + uiText("Same authenticated excerpt as", "אותו ציטוט מאומת כמו") +
            " [" + original.displayNumber + "]");
        }
      });
    });
    sections.push("", uiText("AI-generated answer · check the original sources",
      "תשובה שנוצרה ב-AI · מומלץ לבדוק את המקורות"));
    if (message.citations.some((citation) => !["verified", "verified_duplicate"].includes(citation.evidenceQuoteStatus))) {
      sections.push(uiText("Authenticated verbatim excerpts are unavailable for some sources.",
        "ציטוטים מילוליים מאומתים אינם זמינים לחלק מהמקורות."));
    }
    return sections.join("\n");
  }

  async function copyAssistantAnswer(
    button,
    label,
    message,
    { includeSources = false } = {},
  ) {
    const defaultLabel = includeSources ? uiText("Copy with sources", "העתקה עם מקורות") : uiText("Copy answer", "העתקת התשובה");
    const defaultAriaLabel = includeSources
      ? uiText("Copy assistant answer with sources", "העתקת התשובה עם המקורות")
      : uiText("Copy assistant answer", "העתקת התשובה");
    button.disabled = true;
    try {
      await writeClipboard(
        includeSources ? answerWithSourcesText(message) : citationPresentation(message).text,
      );
      label.textContent = uiText("Copied", "הועתק");
      button.classList.add("success");
      button.setAttribute(
        "aria-label",
        includeSources ? "Answer and sources copied" : "Answer copied",
      );
      announce(
        includeSources
          ? "Answer and sources copied to the clipboard"
          : "Answer copied to the clipboard",
      );
    } catch (_error) {
      label.textContent = uiText("Copy failed", "ההעתקה נכשלה");
      button.classList.add("failed");
      button.setAttribute("aria-label", uiText("Copy failed", "ההעתקה נכשלה"));
      announce("The answer could not be copied");
    }
    window.setTimeout(() => {
      label.textContent = defaultLabel;
      button.classList.remove("success", "failed");
      button.setAttribute("aria-label", defaultAriaLabel);
      button.disabled = false;
    }, 1800);
  }

  function formatMessageTime(value) {
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return "";
    return new Intl.DateTimeFormat(uiIsHebrew() ? "he-IL" : "en-GB", {
      hour: "2-digit",
      minute: "2-digit",
    }).format(date);
  }

  function formatDuration(milliseconds) {
    const totalSeconds = Math.max(0, Math.round(Number(milliseconds) / 1000));
    if (totalSeconds < 60) return `${totalSeconds}s`;
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    return seconds ? `${minutes}m ${seconds}s` : `${minutes}m`;
  }

  function conversationSourceIndex(thread = getActiveThread()) {
    if (!thread) return [];
    const grouped = new Map();
    thread.messages.forEach((message) => {
      if (message.role !== "assistant" || !message.citations.length) return;
      message.citations.forEach((citation) => {
        const file = compactSourceName(citation.file);
        const key = citation.sourceId || normalizedSearchText(file.replace(/\\/g, "/"));
        if (!grouped.has(key)) {
          grouped.set(key, {
            file,
            citation,
            references: [],
            locations: new Set(),
            snippets: [],
          });
        }
        const source = grouped.get(key);
        source.references.push({ citation, message });
        const location = citationLocationLabel(
          citation,
          usesHebrew(message.content),
        );
        if (location) source.locations.add(location);
        if (
          ["verified", "verified_duplicate"].includes(citation.evidenceQuoteStatus) &&
          citation.evidenceQuote &&
          !source.snippets.includes(citation.evidenceQuote)
        ) {
          source.snippets.push(citation.evidenceQuote);
        }
      });
    });
    return [...grouped.values()].sort((first, second) =>
      second.references.length - first.references.length ||
      first.file.localeCompare(second.file),
    );
  }

  function resetSourceFilter() {
    sourceSearchQuery = "";
    elements.sourceSearch.value = "";
    elements.sourceSearchClear.classList.add("hidden");
  }

  function jumpToConversationMessage(messageId) {
    closeSourcePanel({ restoreFocus: false });
    window.requestAnimationFrame(() => {
      const row = [...elements.chatHistory.querySelectorAll("[data-message-id]")]
        .find((candidate) => candidate.dataset.messageId === messageId);
      if (!row) return;
      row.classList.add("message-targeted");
      const reduceMotion =
        typeof window.matchMedia === "function" &&
        window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      row.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "center" });
      if (typeof row.focus === "function") {
        row.tabIndex = -1;
        try {
          row.focus({ preventScroll: true });
        } catch (_error) {
          row.focus();
        }
      }
      window.setTimeout(() => row.classList.remove("message-targeted"), 2200);
    });
  }

  function renderSourcePanel(sources = conversationSourceIndex()) {
    const referenceCount = sources.reduce(
      (total, source) => total + source.references.length,
      0,
    );
    elements.sourceUniqueCount.textContent = String(sources.length);
    elements.sourceReferenceCount.textContent = String(referenceCount);
    elements.sourcePanelContent.replaceChildren();

    const filtered = sourceSearchQuery
      ? sources.filter((source) => {
          const searchable = [
            source.file,
            ...source.locations,
            ...source.snippets,
          ].join(" ");
          return normalizedSearchText(searchable).includes(sourceSearchQuery);
        })
      : sources;

    if (!filtered.length) {
      const copy = sources.length
        ? "No sources match this filter. Try another word or clear the filter."
        : "Sources will appear here after Hebrew-English RAG Application answers with evidence from your documents.";
      elements.sourcePanelContent.append(textElement("div", "source-panel-empty", copy));
      return;
    }

    filtered.forEach((source) => {
      const card = document.createElement("article");
      card.className = "source-index-card";
      const heading = document.createElement("div");
      heading.className = "source-index-heading";
      const name = textElement("div", "source-index-name", source.file);
      name.dir = "auto";
      name.title = source.file;
      const count = textElement(
        "span",
        "source-index-count",
        `${source.references.length} ${source.references.length === 1 ? "ref" : "refs"}`,
      );
      heading.append(name, count);
      card.append(heading);

      const details = [];
      if (source.locations.size) {
        details.push([...source.locations].slice(0, 3).join(" · "));
      }
      const answerCount = new Set(
        source.references.map((reference) => reference.message.id),
      ).size;
      details.push(`Used in ${answerCount} ${answerCount === 1 ? "answer" : "answers"}`);
      const meta = textElement("div", "source-index-meta", details.join(" · "));
      meta.dir = "auto";
      card.append(meta);

      if (source.snippets.length) {
        const snippet = textElement("div", "source-index-snippet", source.snippets[0]);
        snippet.dir = "auto";
        card.append(snippet);
      }

      const actions = document.createElement("div");
      actions.className = "source-index-actions";
      const openLink = sourceOpenAnchor(
        source.citation,
        "source-index-action",
        "Open original",
      );
      if (openLink) {
        openLink.replaceChildren(
          icon("arrow_up_right", "13px"),
          document.createTextNode("Open original"),
        );
      }
      const jumpButton = textElement("button", "source-index-action", "");
      jumpButton.type = "button";
      jumpButton.append(icon("arrow_up_right", "13px"), document.createTextNode("Jump to answer"));
      jumpButton.addEventListener("click", () =>
        jumpToConversationMessage(source.references[0].message.id),
      );

      const askButton = textElement("button", "source-index-action", "");
      askButton.type = "button";
      askButton.append(icon("sparkles", "13px"), document.createTextNode("Ask this source"));
      askButton.addEventListener("click", () => {
        const reference = source.references[0];
        const hebrew = usesHebrew(reference.message.content);
        closeSourcePanel({ restoreFocus: false });
        prefillComposer(
          sourceQuestion(reference.citation, reference.message.content, hebrew),
          { selectedSourceId: reference.citation.sourceId },
        );
      });
      actions.append(...[openLink, jumpButton, askButton].filter(Boolean));
      card.append(actions);
      elements.sourcePanelContent.append(card);
    });
  }

  function openSourcePanel() {
    if (!conversationSourceIndex().length) return;
    closeSidebar();
    sourcePanelReturnFocus = document.activeElement;
    elements.appShell.classList.add("sources-open");
    elements.sourcePanel.setAttribute("aria-hidden", "false");
    elements.sourcesToggleButton.setAttribute("aria-expanded", "true");
    elements.sourcesToggleButton.title = "Close conversation sources";
    elements.main.inert = true;
    elements.sidebar.inert = true;
    window.setTimeout(() => elements.sourceSearch.focus(), 180);
  }

  function closeSourcePanel({ restoreFocus = true } = {}) {
    const wasOpen = elements.appShell.classList.contains("sources-open");
    elements.appShell.classList.remove("sources-open");
    elements.sourcePanel.setAttribute("aria-hidden", "true");
    elements.sourcesToggleButton.setAttribute("aria-expanded", "false");
    elements.sourcesToggleButton.title = "Open conversation sources";
    elements.main.inert = false;
    elements.sidebar.inert = false;
    if (
      wasOpen &&
      restoreFocus &&
      sourcePanelReturnFocus &&
      !sourcePanelReturnFocus.disabled &&
      typeof sourcePanelReturnFocus.focus === "function"
    ) {
      sourcePanelReturnFocus.focus();
    }
    sourcePanelReturnFocus = null;
  }

  function conversationAsMarkdown(thread) {
    const exportedAt = new Intl.DateTimeFormat(uiIsHebrew() ? "he-IL" : "en-GB", {
      dateStyle: "medium",
      timeStyle: "short",
    }).format(new Date());
    const sections = [
      `# ${thread.title}`,
      "",
      `> Exported from Hebrew-English RAG Application on ${exportedAt}`,
      "",
    ];
    thread.messages.forEach((message) => {
      const role = message.role === "user" ? "You" : "Hebrew-English RAG Application";
      const timestamp = formatMessageTime(message.createdAt);
      const presentation = message.role === "assistant" ? citationPresentation(message) : null;
      sections.push(`## ${role}${timestamp ? ` · ${timestamp}` : ""}`, "", presentation?.text ?? message.content, "");
      if (message.role === "assistant" && message.citations.length) {
        sections.push("### Sources", "");
        const hebrew = uiIsHebrew();
        presentation.groups.forEach((target) => {
          const citation = target.citation;
          const details = citationLocationLabel(citation, hebrew);
          sections.push(
            `- [${target.displayNumber}] ${citation.file || "Unknown source"}${details ? ` — ${details}` : ""}`,
          );
          const seenQuotes = new Set();
          target.members.forEach(({ citation: member }) => {
            if (["verified", "verified_duplicate"].includes(member.evidenceQuoteStatus) && member.evidenceQuote && !seenQuotes.has(member.evidenceQuote)) {
              seenQuotes.add(member.evidenceQuote);
              sections.push(`  > ${member.evidenceQuote.replace(/\s+/g, " ").trim()}`);
            } else if (member.evidenceQuoteStatus === "verified_duplicate" && member.evidenceDuplicateOf) {
              const original = presentation.targets.get(member.evidenceDuplicateOf);
              if (original && original !== target) sections.push(`  - Same verified evidence as [${original.displayNumber}]`);
            }
          });
        });
        sections.push("");
      }
      sections.push("---", "");
    });
    return `${sections.join("\n").trim()}\n`;
  }

  function exportActiveThread() {
    const thread = getActiveThread();
    if (!thread || !thread.messages.length) return;
    const markdown = conversationAsMarkdown(thread);
    const blob = new Blob([markdown], { type: "text/markdown;charset=utf-8" });
    const objectUrl = URL.createObjectURL(blob);
    const download = document.createElement("a");
    const safeTitle = thread.title
      .replace(/[<>:"/\\|?*\u0000-\u001f]/g, "-")
      .replace(/\s+/g, " ")
      .trim()
      .slice(0, 80) || "hebrew-english-rag-conversation";
    download.href = objectUrl;
    download.download = `${safeTitle}.md`;
    document.body.append(download);
    download.click();
    download.remove();
    window.setTimeout(() => URL.revokeObjectURL(objectUrl), 0);
    showToast(uiText("Conversation exported as Markdown", "השיחה יוצאה כקובץ Markdown"));
    announce(uiText("Conversation exported as Markdown", "השיחה יוצאה כקובץ Markdown"));
  }

  function updateConversationUtilities(thread = getActiveThread()) {
    const sources = conversationSourceIndex(thread);
    const hasMessages = Boolean(thread?.messages.length);

    elements.exportThreadButton.disabled = !hasMessages;
    elements.sourcesToggleButton.disabled = true;
    elements.sourcesCountBadge.textContent = "";
    elements.sourcesCountBadge.classList.add("hidden");
    elements.sourcesToggleButton.setAttribute(
      "aria-label",
      sources.length
        ? `${elements.appShell.classList.contains("sources-open") ? "Close" : "Open"} ${sources.length} conversation ${sources.length === 1 ? "source" : "sources"}`
        : "No conversation sources yet",
    );
    if (!sources.length && elements.appShell.classList.contains("sources-open")) {
      closeSourcePanel({ restoreFocus: false });
    }
  }

  function renderThreadList() {
    elements.threadList.replaceChildren();
    const orderedThreads = [...workspace.threads].sort((a, b) =>
      b.updatedAt.localeCompare(a.updatedAt),
    );
    const filteredThreads = threadSearchQuery
      ? orderedThreads.filter((thread) => {
          const searchable = [
            thread.title,
            ...thread.messages.slice(-12).map((message) => message.content),
          ].join(" ");
          return normalizedSearchText(searchable).includes(threadSearchQuery);
        })
      : orderedThreads;

    if (threadSearchQuery) {
      elements.threadList.append(textElement("div", "thread-section-header",
        uiText("Search results", "תוצאות חיפוש")));
    }

    if (!filteredThreads.length) {
      const empty = textElement(
        "div",
        "thread-empty-state",
        uiText("No matching conversations. Try a different word or clear the search.", "לא נמצאו שיחות מתאימות. נסו מילה אחרת או נקו את החיפוש."),
      );
      elements.threadList.append(empty);
      return;
    }

    filteredThreads.forEach((thread) => {
      const item = document.createElement("div");
      item.className = `thread-item${
        thread.id === workspace.activeThreadId ? " active" : ""
      }`;
      item.dataset.threadId = thread.id;

      const displayTitle = thread.title === "Untitled thread"
        ? uiText("New conversation", "שיחה חדשה") : thread.title;
      const selectButton = document.createElement("button");
      selectButton.type = "button";
      selectButton.className = "thread-select";
      selectButton.setAttribute("aria-current", thread.id === workspace.activeThreadId ? "true" : "false");
      selectButton.setAttribute("aria-label", uiText("Open conversation: ", "פתיחת שיחה: ") + displayTitle);
      selectButton.append(icon("chat_bubble", "17px"));
      const threadCopy = document.createElement("span");
      threadCopy.className = "thread-copy";
      const titleRow = document.createElement("span");
      titleRow.className = "thread-title-row";
      const title = textElement("span", "thread-title-text", displayTitle);
      title.title = displayTitle;
      title.dir = "auto";
      titleRow.append(title);
      threadCopy.append(titleRow);
      selectButton.append(threadCopy);
      selectButton.addEventListener("click", () => selectThread(thread.id));

      const actions = document.createElement("div");
      actions.className = "action-icons";

      const renameButton = document.createElement("button");
      renameButton.type = "button";
      renameButton.className = "thread-action";
      renameButton.title = uiText("Rename conversation", "שינוי שם השיחה");
      renameButton.setAttribute("aria-label", uiText("Rename conversation: ", "שינוי שם השיחה: ") + displayTitle);
      renameButton.append(icon("edit", "16px"));
      renameButton.addEventListener("click", () => renameThread(thread.id));

      const deleteButton = document.createElement("button");
      deleteButton.type = "button";
      deleteButton.className = "thread-action";
      deleteButton.title = uiText("Delete conversation", "מחיקת השיחה");
      deleteButton.setAttribute("aria-label", uiText("Delete conversation: ", "מחיקת השיחה: ") + displayTitle);
      deleteButton.append(icon("delete", "16px"));
      deleteButton.addEventListener("click", () => deleteThread(thread.id));

      actions.append(renameButton, deleteButton);
      item.append(selectButton, actions);
      elements.threadList.append(item);
    });
  }

  function numericCitationId(citation) {
    return frontendPolicy.numericCitationId(citation);
  }

  function createCitationTargets(citations) {
    const targets = new Map();
    const groups = new Map();
    citations.forEach((citation, index) => {
      const number = numericCitationId(citation);
      if (!number || targets.has(number)) return;
      const groupKey = citationPresentationKey(citation);
      let target = groupKey ? groups.get(groupKey) : null;
      if (!target) {
        const displayNumber = String(new Set(targets.values()).size + 1);
        target = {
          citation,
          index,
          displayNumber,
          elementId: makeId(`citation-${displayNumber}`),
          card: null,
          used: false,
          members: [],
        };
        if (groupKey) groups.set(groupKey, target);
      }
      target.members.push({ citation, index, number });
      targets.set(number, target);
    });
    return targets;
  }

  function citationPresentationKey(citation) {
    // The server binds this opaque key to the verified source version and
    // location. Retain each citation/span; only their displayed card is shared.
    if (!/^[a-f0-9]{64}$/.test(citation.presentationGroupId || "") ||
        !SOURCE_ID_PATTERN.test(citation.sourceId || "") || !citation.openUrl ||
        !["verified", "verified_duplicate"].includes(citation.evidenceQuoteStatus)) return "";
    return JSON.stringify([
      citation.presentationGroupId, citation.sourceId, citation.openUrl,
      citation.sourceRelativePath, citation.mimeType, citation.page, citation.slide,
      citation.sheet, citation.row, citation.cell, citation.paragraph,
      citation.section, citation.location,
    ]);
  }

  function citationPresentation(message) {
    const targets = createCitationTargets(message.citations || []);
    const value = String(message.content || "");
    const pattern = /`[^`\n]*`|\[(\d{1,3})\]/g;
    let text = "";
    let cursor = 0;
    let previousTarget = null;
    let match;
    while ((match = pattern.exec(value)) !== null) {
      const target = targets.get(match[1]);
      const before = match.index > 0 ? value[match.index - 1] : "";
      const after = pattern.lastIndex < value.length ? value[pattern.lastIndex] : "";
      if (!target?.citation?.openUrl || before === "[" || after === "]" ||
          isCitationBoundaryWord(before) || isCitationBoundaryWord(after)) continue;
      target.used = true;
      if (target !== previousTarget || value.slice(cursor, match.index).trim()) {
        text += value.slice(cursor, match.index) + `[${target.displayNumber}]`;
      }
      cursor = pattern.lastIndex;
      previousTarget = target;
    }
    text += value.slice(cursor);
    return { text, targets, groups: [...new Set(targets.values())].filter(target => target.used) };
  }

  function isCitationBoundaryWord(character) {
    return frontendPolicy.isCitationBoundaryWord(character);
  }

  function revealCitation(target, targets) {
    const targetCard = target && target.card;
    if (!targetCard) return;

    if (target.disclosure && !target.disclosure.open) {
      target.disclosure.open = true;
    }

    targets.forEach((candidate) => {
      if (!candidate.card) return;
      const timer = citationHighlightTimers.get(candidate.card);
      if (timer) window.clearTimeout(timer);
      citationHighlightTimers.delete(candidate.card);
      candidate.card.classList.remove("citation-card-targeted");
    });

    targetCard.classList.add("citation-card-targeted");
    const reduceMotion =
      typeof window.matchMedia === "function" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (typeof targetCard.scrollIntoView === "function") {
      targetCard.scrollIntoView({
        behavior: reduceMotion ? "auto" : "smooth",
        block: "nearest",
      });
    }
    if (typeof targetCard.focus === "function") {
      try {
        targetCard.focus({ preventScroll: true });
      } catch (_error) {
        targetCard.focus();
      }
    }

    const timer = window.setTimeout(() => {
      targetCard.classList.remove("citation-card-targeted");
      citationHighlightTimers.delete(targetCard);
    }, 2200);
    citationHighlightTimers.set(targetCard, timer);
  }

  function appendCitationAwareText(container, value, citationTargets, hebrew) {
    const markerPattern = /\[(\d{1,3})\]/g;
    let cursor = 0;
    let match;
    let previousTarget = null;

    while ((match = markerPattern.exec(value)) !== null) {
      const markerStart = match.index;
      const markerEnd = markerPattern.lastIndex;
      const before = markerStart > 0 ? value[markerStart - 1] : "";
      const after = markerEnd < value.length ? value[markerEnd] : "";
      const target = citationTargets.get(match[1]);
      const hasValidBoundaries =
        before !== "[" &&
        after !== "]" &&
        !isCitationBoundaryWord(before) &&
        !isCitationBoundaryWord(after);

      if (!target?.citation?.openUrl || !hasValidBoundaries) continue;

      // Consecutive aliases for one page need one visible marker. A later
      // claim still receives its own marker and retains its original binding.
      if (target === previousTarget && !value.slice(cursor, markerStart).trim()) {
        cursor = markerEnd;
        continue;
      }

      if (markerStart > cursor) {
        container.append(document.createTextNode(value.slice(cursor, markerStart)));
      }

      target.used = true;
      const marker = sourceOpenAnchor(target.citation, "citation-marker", `[${target.displayNumber}]`);
      marker.dir = "ltr";
      marker.dataset.citationNumber = target.displayNumber;
      marker.setAttribute("aria-controls", target.elementId);
      marker.addEventListener("click", () => revealCitation(target, citationTargets));
      container.append(marker);
      cursor = markerEnd;
      previousTarget = target;
    }

    if (cursor < value.length) {
      container.append(document.createTextNode(value.slice(cursor)));
    }
  }

  function appendAnswerInline(container, value, citationTargets, hebrew) {
    const inlinePattern = /(\*\*[^*\n]+\*\*|`[^`\n]+`)/g;
    let cursor = 0;
    let match;
    while ((match = inlinePattern.exec(value)) !== null) {
      if (match.index > cursor) {
        appendCitationAwareText(
          container,
          value.slice(cursor, match.index),
          citationTargets,
          hebrew,
        );
      }
      const token = match[0];
      if (token.startsWith("**")) {
        const strong = document.createElement("strong");
        appendCitationAwareText(strong, token.slice(2, -2), citationTargets, hebrew);
        container.append(strong);
      } else {
        const code = textElement("code", "answer-inline-code", token.slice(1, -1));
        container.append(code);
      }
      cursor = inlinePattern.lastIndex;
    }
    if (cursor < value.length) {
      appendCitationAwareText(container, value.slice(cursor), citationTargets, hebrew);
    }
  }

  function splitAnswerTableRow(value) {
    return frontendPolicy.splitAnswerTableRow(value);
  }

  function isAnswerTableSeparator(value) {
    return frontendPolicy.isAnswerTableSeparator(value);
  }

  function answerListMatch(value) {
    return frontendPolicy.answerListMatch(value);
  }

  function isStructuredAnswerStart(lines, index) {
    const value = lines[index] || "";
    if (/^\s*#{1,3}\s+\S/.test(value) || answerListMatch(value)) return true;
    return value.includes("|") && isAnswerTableSeparator(lines[index + 1] || "");
  }

  function createAnswerContent(message, citationTargets, hebrew) {
    const content = document.createElement("div");
    content.className = "message-text answer-content";
    content.dir = frontendPolicy.normalizeTextDirection(message.answerDirection);

    const lines = String(message.content || "").replace(/\r\n?/g, "\n").split("\n");
    let index = 0;
    while (index < lines.length) {
      if (!lines[index].trim()) {
        index += 1;
        continue;
      }

      const heading = lines[index].match(/^\s*(#{1,3})\s+(.+)$/);
      if (heading) {
        const level = Math.min(3, heading[1].length + 1);
        const node = document.createElement(`h${level}`);
        node.className = `answer-heading level-${level}`;
        node.dir = "auto";
        appendAnswerInline(node, heading[2].trim(), citationTargets, hebrew);
        content.append(node);
        index += 1;
        continue;
      }

      if (lines[index].includes("|") && isAnswerTableSeparator(lines[index + 1] || "")) {
        const headers = splitAnswerTableRow(lines[index]);
        const wrap = document.createElement("div");
        wrap.className = "answer-table-wrap";
        wrap.tabIndex = 0;
        wrap.setAttribute("role", "region");
        wrap.setAttribute("aria-label", uiText("Answer table", "טבלה בתשובה"));
        const table = document.createElement("table");
        table.className = "answer-table";
        const thead = document.createElement("thead");
        const headerRow = document.createElement("tr");
        headers.forEach((header) => {
          const cell = document.createElement("th");
          cell.scope = "col";
          cell.dir = "auto";
          appendAnswerInline(cell, header, citationTargets, hebrew);
          headerRow.append(cell);
        });
        thead.append(headerRow);
        table.append(thead);
        const tbody = document.createElement("tbody");
        index += 2;
        while (index < lines.length && lines[index].trim() && lines[index].includes("|")) {
          const values = splitAnswerTableRow(lines[index]);
          const row = document.createElement("tr");
          headers.forEach((_header, cellIndex) => {
            const cell = document.createElement("td");
            cell.dir = "auto";
            appendAnswerInline(cell, values[cellIndex] || "", citationTargets, hebrew);
            row.append(cell);
          });
          tbody.append(row);
          index += 1;
        }
        table.append(tbody);
        wrap.append(table);
        content.append(wrap);
        continue;
      }

      const listItem = answerListMatch(lines[index]);
      if (listItem) {
        const list = document.createElement(listItem.ordered ? "ol" : "ul");
        list.className = "answer-list";
        list.dir = "auto";
        while (index < lines.length) {
          const candidate = answerListMatch(lines[index]);
          if (!candidate || candidate.ordered !== listItem.ordered) break;
          const item = document.createElement("li");
          if (candidate.ordered) {
            const ordinal = lines[index].match(/^\s*(\d+)[.)]\s+/);
            if (ordinal) item.value = Number(ordinal[1]);
          }
          item.dir = "auto";
          appendAnswerInline(item, candidate.content, citationTargets, hebrew);
          list.append(item);
          index += 1;
        }
        content.append(list);
        continue;
      }

      const paragraphLines = [];
      while (
        index < lines.length &&
        lines[index].trim() &&
        !isStructuredAnswerStart(lines, index)
      ) {
        paragraphLines.push(lines[index].trim());
        index += 1;
      }
      if (!paragraphLines.length) {
        paragraphLines.push(lines[index].trim());
        index += 1;
      }
      const paragraph = document.createElement("p");
      paragraph.className = "answer-paragraph";
      paragraph.dir = "auto";
      appendAnswerInline(
        paragraph,
        paragraphLines.join(" "),
        citationTargets,
        hebrew,
      );
      content.append(paragraph);
    }
    return content;
  }

  function createInlineEvidence(message, citationTargets, hebrew) {
    if (!message.citations.length) return null;
    const cited = [];
    const includedTargets = new Set();
    message.citations.forEach((citation, index) => {
      const target = citationTargets.get(numericCitationId(citation));
      if (target?.used && !includedTargets.has(target)) {
        includedTargets.add(target);
        cited.push({ citation: target.citation || citation, index: target.index ?? index, target });
      }
    });
    if (!cited.length) return null;
    const section = document.createElement("details");
    section.className = "inline-evidence answer-sources";
    section.setAttribute("aria-label", uiText("Answer sources", "מקורות התשובה"));
    section.open = citationDisclosureState.get(message.id) === true;
    section.addEventListener("toggle", () => citationDisclosureState.set(message.id, section.open));
    const summary = document.createElement("summary");
    summary.className = "sources-summary";
    summary.append(
      textElement("span", "sources-summary-label", uiText("Cited sources", "מקורות שצוטטו")),
      textElement("span", "source-count", String(cited.length)),
    );
    section.append(summary);
    const list = document.createElement("div");
    list.className = "inline-evidence-list source-list";
    cited.forEach(({ citation, index, target }) => {
      target.disclosure = section;
      list.append(createCitationElement(citation, index, message.content, hebrew, target));
    });
    section.append(list);
    return section;
  }

  function createCitationElement(citation, index, answer, hebrew, target = null) {
    const displayNumber = target?.displayNumber || citation.id || index + 1;
    const card = document.createElement("article");
    card.className = "citation-card";
    if (target) {
      card.id = target.elementId;
      card.tabIndex = -1;
      target.card = card;
    }
    if (target?.used) {
      card.classList.add("citation-card-used");
      card.dataset.usedInAnswer = "true";
    }
    card.setAttribute("aria-label", uiText(
      `Source ${displayNumber}${target?.used ? ", cited in answer" : ""}`,
      `מקור ${displayNumber}${target?.used ? ", צוטט בתשובה" : ""}`));
    const heading = document.createElement("div");
    heading.className = "citation-heading";
    const number = textElement("button", "citation-number", `[${displayNumber}]`);
    number.type = "button";
    number.dir = "ltr";
    if (target?.used) {
      number.setAttribute("aria-label", uiText("Find citation in answer", "מעבר לציטוט בתשובה"));
      number.addEventListener("click", () => {
        const marker = [...card.closest(".message-card").querySelectorAll(".citation-marker")]
          .find((item) => item.dataset.citationNumber === String(displayNumber));
        if (!marker) return;
        marker.scrollIntoView({ block: "nearest", behavior:
          window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
        marker.focus({ preventScroll: true });
      });
    } else {
      number.disabled = true;
    }
    heading.append(number);
    const file = textElement("bdi", "citation-file", citation.file || uiText("Unknown source", "מקור לא ידוע"));
    file.dir = "auto";
    file.title = citation.file || "";
    heading.append(file);
    const location = citationLocationLabel(citation, hebrew);
    if (location) {
      const meta = textElement("span", "citation-meta", location);
      meta.dir = "auto";
      heading.append(meta);
    }
    const openSourceLink = sourceOpenAnchor(citation, "citation-source-action",
      uiText("Open original", "פתיחת המקור"));
    if (openSourceLink) heading.append(openSourceLink);
    card.append(heading);
    // Keep each verified source span separate, even when several citations
    // share one presentation card. Never render unverified snippets as quotes.
    const members = target?.members || [{ citation }];
    const shown = new Set();
    members.forEach(({ citation: member }) => {
      if (!["verified", "verified_duplicate"].includes(member.evidenceQuoteStatus) || !member.evidenceQuote ||
          shown.has(member.evidenceQuote)) return;
      shown.add(member.evidenceQuote);
      const quote = document.createElement("blockquote");
      quote.className = "citation-evidence-quote";
      quote.dir = frontendPolicy.normalizeTextDirection(member.evidenceQuoteDirection);
      if (["he", "en"].includes(member.evidenceQuoteLanguage)) {
        quote.lang = member.evidenceQuoteLanguage;
      }
      quote.textContent = member.evidenceQuote;
      card.append(quote);
    });
    return card;
  }

  function createMessageElement(
    message,
    { showFollowUps = false, allowQuestionEdit = false } = {},
  ) {
    const row = document.createElement("article");
    row.className = `message-row ${message.role}`;
    row.dataset.messageId = message.id;
    const hebrew = uiIsHebrew();
    row.setAttribute(
      "aria-label",
      message.role === "assistant" ? uiText("Hebrew-English RAG Application answer", "תשובת Hebrew-English RAG Application") : uiText("Your question", "השאלה שלך"),
    );

    const card = document.createElement("div");
    card.className = "message-card";
    const messageHeading = document.createElement("div");
    messageHeading.className = "message-heading";
    if (message.role === "assistant") {
      messageHeading.append(textElement("div", "message-label", "Hebrew-English RAG Application"));
    }
    const messageMeta = document.createElement("div");
    messageMeta.className = "message-heading-meta";
    if (message.role === "assistant" && message.durationMs != null) {
      const duration = textElement(
        "span",
        "message-duration",
        formatDuration(message.durationMs),
      );
      duration.title = uiText("Time to answer", "זמן יצירת התשובה");
      messageMeta.append(duration);
    }
    if (message.role === "assistant" && message.retiredCorpus) {
      messageMeta.append(textElement("span", "message-history-status",
        uiText("Historical answer", "תשובה היסטורית")));
    }
    if (message.role === "assistant" && message.answerCompleteness === "partial") {
      messageMeta.append(textElement("span", "message-history-status",
        uiText("Partial answer", "תשובה חלקית")));
    }
    const timestamp = textElement("time", "message-time", formatMessageTime(message.createdAt));
    timestamp.dateTime = message.createdAt;
    messageMeta.append(timestamp);
    messageHeading.append(messageMeta);
    card.append(messageHeading);
    if (message.role === "assistant" && message.referenceRecovery === "new_search") {
      const notice = textElement("div", "message-history-status", uiText(
        "Searched again using the previous request; the results may not be the exact source mentioned before.",
        "חיפשתי שוב לפי הבקשה הקודמת; אין ודאות שזהו המקור שהוזכר קודם."));
      notice.dir = "auto";
      card.append(notice);
    }
    const citationTargets =
      message.role === "assistant" ? createCitationTargets(message.citations) : new Map();
    const content =
      message.role === "assistant"
        ? createAnswerContent(message, citationTargets, hebrew)
        : textElement("div", "message-text", message.content);
    if (message.role !== "assistant") content.dir = "auto";
    card.append(content);
    if (message.role === "assistant" && message.citations.length) {
      const inlineEvidence = createInlineEvidence(
        message,
        citationTargets,
        hebrew,
      );
      if (inlineEvidence) card.append(inlineEvidence);
    }

    if (message.role === "assistant") {
      const toolbar = document.createElement("div");
      toolbar.className = "message-toolbar";
      toolbar.dir = uiIsHebrew() ? "rtl" : "ltr";
      const copyButton = document.createElement("button");
      copyButton.type = "button";
      copyButton.className = "message-action";
      copyButton.dataset.action = "copy-answer";
      copyButton.setAttribute("aria-label", uiText("Copy assistant answer", "העתקת התשובה"));
      copyButton.append(icon("content_copy", "15px"));
      const copyLabel = textElement("span", "message-action-label", uiText("Copy answer", "העתקת התשובה"));
      copyButton.append(copyLabel);
      copyButton.addEventListener("click", () => {
        void copyAssistantAnswer(copyButton, copyLabel, message);
      });
      toolbar.append(copyButton);
      if (message.citations.length) {
        const copySourcesButton = document.createElement("button");
        copySourcesButton.type = "button";
        copySourcesButton.className = "message-action";
        copySourcesButton.dataset.action = "copy-answer-with-sources";
        copySourcesButton.setAttribute(
          "aria-label",
          uiText("Copy assistant answer with sources", "העתקת התשובה עם המקורות"),
        );
        copySourcesButton.append(icon("content_copy", "15px"));
        const copySourcesLabel = textElement(
          "span",
          "message-action-label",
          uiText("Copy with sources", "העתקה עם מקורות"),
        );
        copySourcesButton.append(copySourcesLabel);
        copySourcesButton.addEventListener("click", () => {
          void copyAssistantAnswer(
            copySourcesButton,
            copySourcesLabel,
            message,
            { includeSources: true },
          );
        });
        toolbar.append(copySourcesButton);
      }
      if (showFollowUps) {
        const regenerateButton = document.createElement("button");
        regenerateButton.type = "button";
        regenerateButton.className = "message-action";
        regenerateButton.dataset.action = "regenerate-answer";
        regenerateButton.setAttribute(
          "aria-label",
          uiText("Regenerate the latest answer", "יצירה מחדש של התשובה האחרונה"),
        );
        regenerateButton.append(icon("refresh", "15px"));
        regenerateButton.append(
          textElement(
            "span",
            "message-action-label",
            uiText("Regenerate", "יצירת תשובה מחדש"),
          ),
        );
        regenerateButton.addEventListener("click", () => regenerateAnswer(message.id));
        toolbar.append(regenerateButton);
      }
      card.append(toolbar);
    } else if (allowQuestionEdit) {
      const toolbar = document.createElement("div");
      toolbar.className = "message-toolbar user-message-toolbar";
      toolbar.dir = uiIsHebrew() ? "rtl" : "ltr";
      const editButton = document.createElement("button");
      editButton.type = "button";
      editButton.className = "message-action";
      editButton.dataset.action = "edit-answered-question";
      editButton.setAttribute(
        "aria-label",
        uiText("Edit the latest question and remove its answer", "עריכת השאלה האחרונה והסרת התשובה שלה"),
      );
      editButton.append(
        icon("edit", "15px"),
        textElement("span", "message-action-label", uiText("Edit question", "עריכת השאלה")),
      );
      editButton.addEventListener("click", () => editAnsweredQuestion(message.id));
      toolbar.append(editButton);
      card.append(toolbar);
    }

    if (message.role === "assistant" && showFollowUps) {
      const suggestions = document.createElement("section");
      suggestions.className = "follow-ups";
      suggestions.dir = uiIsHebrew() ? "rtl" : "ltr";
      suggestions.setAttribute("aria-label", uiText("Suggested follow-up questions", "הצעות לשאלות המשך"));
      suggestions.append(
        textElement("div", "follow-up-title", uiText("Continue from here", "אפשר להמשיך מכאן")),
      );
      const list = document.createElement("div");
      list.className = "follow-up-list";
      followUpSuggestions(message).forEach(({ label, prompt }) => {
        list.append(promptButton(label, prompt, "follow-up", "follow-up-chip"));
      });
      suggestions.append(list);
      card.append(suggestions);
    }
    row.append(card);
    return row;
  }

  function createLoadingElement(request, threadId) {
    const regenerating = Boolean(request?.replaceMessageId);
    const row = document.createElement("div");
    row.className = "message-row assistant";
    const card = document.createElement("div");
    card.className = "loading-indicator";
    card.dir = uiIsHebrew() ? "rtl" : "ltr";
    const orb = document.createElement("div");
    orb.className = "loading-orb";
    orb.setAttribute("aria-hidden", "true");
    const copy = document.createElement("span");
    copy.className = "sr-only";
    copy.setAttribute("role", "status");
    copy.setAttribute("aria-atomic", "true");
    copy.textContent = regenerating
      ? uiText("Preparing a new answer", "מכינים תשובה חדשה")
      : uiText("Preparing your answer", "מכינים את התשובה");
    const stopButton = document.createElement("button");
    stopButton.type = "button";
    stopButton.className = "loading-stop-button";
    stopButton.setAttribute(
      "aria-label",
      uiText("Stop waiting for the answer", "עצירת ההמתנה לתשובה"),
    );
    stopButton.append(
      icon("stop", "14px"),
      textElement("span", "", uiText("Stop", "עצירה")),
    );
    stopButton.addEventListener("click", () => abortThreadRequest(threadId, "user"));
    card.append(orb, copy, stopButton);
    row.append(card);
    return row;
  }

  function createErrorElement(threadId, failure) {
    const thread = getThread(threadId);
    const message = typeof failure === "string" ? failure : failure?.message || "";
    const canEditQuestion =
      failure?.allowEdit !== false &&
      !failure?.replaceMessageId &&
      thread?.messages[thread.messages.length - 1]?.role === "user";
    const canRetry = failure?.retryable !== false;
    const row = document.createElement("div");
    row.className = "message-row error";
    row.setAttribute("role", "alert");
    const card = document.createElement("div");
    card.className = "message-card";
    card.dir = uiIsHebrew() ? "rtl" : "ltr";
    card.append(
      textElement("div", "message-label", failure?.title || uiText("Request failed", "לא ניתן להשלים את הבקשה")),
    );
    const content = textElement("div", "message-text", message);
    content.dir = "auto";
    card.append(content);
    if (failure?.hint) {
      card.append(textElement("div", "error-hint", failure.hint));
    }
    if (failure?.diagnostic) {
      const diagnostics = document.createElement("details");
      diagnostics.className = "error-diagnostics";
      diagnostics.append(textElement("summary", "", uiText("Technical details", "פרטים טכניים")));
      const detail = textElement("pre", "", String(failure.diagnostic).slice(0, 250));
      detail.dir = "ltr";
      diagnostics.append(detail);
      card.append(diagnostics);
    }
    const actions = document.createElement("div");
    actions.className = "error-actions";
    const appendEditButton = () => {
      const editButton = textElement(
        "button",
        "edit-question-button",
        uiText("Edit question", "עריכת השאלה"),
      );
      editButton.type = "button";
      editButton.addEventListener("click", () => editFailedQuestion(threadId));
      actions.append(editButton);
    };
    if (canEditQuestion && !canRetry) appendEditButton();
    if (canRetry) {
      const retryButton = textElement(
        "button",
        "retry-button",
        uiText("Retry question", "שליחת השאלה שוב"),
      );
      retryButton.type = "button";
      retryButton.addEventListener("click", () => retryThread(threadId));
      actions.append(retryButton);
    }
    if (canEditQuestion && canRetry) appendEditButton();
    if (failure?.openSettings) {
      const settingsButton = textElement(
        "button",
        "settings-error-button",
        uiText("Open settings", "פתיחת הגדרות"),
      );
      settingsButton.type = "button";
      settingsButton.addEventListener("click", openSettings);
      actions.append(settingsButton);
    }
    if (actions.childElementCount) card.append(actions);
    row.append(card);
    return row;
  }

  function isConversationNearBottom(threshold = 72) {
    const distanceFromBottom =
      elements.chatHistory.scrollHeight -
      elements.chatHistory.scrollTop -
      elements.chatHistory.clientHeight;
    return distanceFromBottom <= threshold;
  }

  function updateScrollToLatestButton() {
    const thread = getActiveThread();
    const request = thread ? pendingRequests.get(thread.id) : null;
    const hasConversationContent = Boolean(
      thread &&
        (thread.messages.length || request || threadErrors.has(thread.id)),
    );
    const canScroll =
      elements.chatHistory.scrollHeight > elements.chatHistory.clientHeight + 1;
    const hidden =
      !hasConversationContent || !canScroll || isConversationNearBottom();
    elements.scrollToLatestButton.classList.toggle("hidden", hidden);
    if (hidden || !thread) return;

    const label = elements.scrollToLatestButton.querySelector("span");
    if (label) label.textContent = uiText("Latest", "ההודעה האחרונה");
    elements.scrollToLatestButton.setAttribute(
      "aria-label",
      uiText("Scroll to the latest message", "גלילה להודעה האחרונה"),
    );
  }

  function scrollConversationToLatest() {
    const reduceMotion =
      typeof window.matchMedia === "function" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    elements.chatHistory.scrollTo({
      top: elements.chatHistory.scrollHeight,
      behavior: reduceMotion ? "auto" : "smooth",
    });
  }

  function renderConversation({ scroll = true, forceScroll = false } = {}) {
    const thread = getActiveThread();
    if (!thread) {
      elements.scrollToLatestButton.classList.add("hidden");
      return;
    }
    const wasNearBottom = isConversationNearBottom();
    const previousScrollTop = elements.chatHistory.scrollTop;
    // Keep the live indicator attached while rebuilding messages so unrelated
    // rerenders do not restart its animation.
    const request = pendingRequests.get(thread.id);
    const loadingRow = request?.loadingElement;
    const keepLoadingRow = loadingRow?.parentElement === elements.chatHistory;
    Array.from(elements.chatHistory.children).forEach((child) => {
      if (!keepLoadingRow || child !== loadingRow) child.remove();
    });
    const appendContent = (element) => {
      if (keepLoadingRow) elements.chatHistory.insertBefore(element, loadingRow);
      else elements.chatHistory.append(element);
    };
    const lastMessage = thread.messages[thread.messages.length - 1];
    const latestAssistantId =
      lastMessage?.role === "assistant" && !pendingRequests.has(thread.id)
        ? lastMessage.id
        : null;
    const latestEditableQuestionId =
      latestAssistantId &&
      thread.messages.length >= 2 &&
      thread.messages[thread.messages.length - 2]?.role === "user"
        ? thread.messages[thread.messages.length - 2].id
        : null;
    thread.messages.forEach((message) => {
      appendContent(
        createMessageElement(message, {
          showFollowUps: message.id === latestAssistantId,
          allowQuestionEdit: message.id === latestEditableQuestionId,
        }),
      );
    });
    if (threadErrors.has(thread.id)) {
      appendContent(createErrorElement(thread.id, threadErrors.get(thread.id)));
    }
    if (request && !keepLoadingRow) {
      request.loadingElement = createLoadingElement(request, thread.id);
      elements.chatHistory.append(request.loadingElement);
    } else if (keepLoadingRow) {
      const indicator = loadingRow.querySelector(".loading-indicator");
      const status = loadingRow.querySelector("[role=status]");
      const button = loadingRow.querySelector(".loading-stop-button");
      indicator.dir = uiIsHebrew() ? "rtl" : "ltr";
      status.textContent = request.replaceMessageId
        ? uiText("Preparing a new answer", "מכינים תשובה חדשה")
        : uiText("Preparing your answer", "מכינים את התשובה");
      button.setAttribute("aria-label", uiText("Stop waiting for the answer", "עצירת ההמתנה לתשובה"));
      button.querySelector("span").textContent = uiText("Stop", "עצירה");
    }

    const hasMessages = thread.messages.length > 0;
    elements.main.classList.toggle("has-messages", hasMessages);
    elements.chatHeader.hidden = hasMessages;
    elements.threadTitle.textContent = hasMessages ? thread.title : uiText("New conversation", "שיחה חדשה");
    elements.threadTitle.title = elements.threadTitle.textContent;
    elements.threadTitle.dir = "auto";
    elements.chatHistory.setAttribute("aria-busy", pendingRequests.has(thread.id) ? "true" : "false");
    updateConversationUtilities(thread);
    const restoredDraft = typeof thread.draft === "string" ? thread.draft : "";
    if (elements.chatInput.value !== restoredDraft) elements.chatInput.value = restoredDraft;
    resizeInput();
    window.requestAnimationFrame(() => {
      if (forceScroll || (scroll && wasNearBottom)) {
        elements.chatHistory.scrollTop = elements.chatHistory.scrollHeight;
      } else {
        const maximumScrollTop = Math.max(
          0,
          elements.chatHistory.scrollHeight - elements.chatHistory.clientHeight,
        );
        elements.chatHistory.scrollTop = Math.min(previousScrollTop, maximumScrollTop);
      }
      updateScrollToLatestButton();
    });
  }

  function renderAll(options) {
    renderThreadList();
    renderConversation(options);
    updateWorkspaceSummary();
  }

  let tenantCatalog = new Map();
  let tenantCatalogBaseUrl = "";

  async function refreshTenantChoices(baseUrl = settings.apiBaseUrl) {
    const normalizedBaseUrl = normalizeApiBaseUrl(baseUrl);
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 6000);
    try {
      const response = await fetch(`${normalizedBaseUrl}/api/tenants`, {
        cache: "no-store", signal: controller.signal,
      });
      const data = await readBoundedResponseJson(response);
      if (!response.ok || !Array.isArray(data?.tenants) || !data.tenants.length) {
        throw new Error("Authorized knowledge bases were unavailable.");
      }
      const catalog = new Map();
      for (const item of data.tenants) {
        const id = normalizeTenantId(item?.tenant_id);
        if (!TENANT_ID_PATTERN.test(id) || typeof item?.display_name !== "string") continue;
        catalog.set(id, item.display_name.slice(0, 128) || id);
      }
      // A response for an earlier URL must not replace the active draft.
      const draftBaseUrl = elements.settingsOverlay.classList.contains("hidden")
        ? settings.apiBaseUrl : elements.apiBaseUrlInput.value;
      if (normalizeApiBaseUrl(draftBaseUrl) !== normalizedBaseUrl) return;
      tenantCatalog = catalog;
      tenantCatalogBaseUrl = normalizedBaseUrl;
      const prior = elements.tenantIdInput.value;
      elements.tenantIdInput.replaceChildren();
      const defaultOption = document.createElement("option");
      defaultOption.value = "";
      const defaultName = catalog.get(normalizeTenantId(data.default_tenant));
      defaultOption.textContent = defaultName
        ? uiText("Default: ", "ברירת מחדל: ") + defaultName
        : uiText("Select a knowledge base", "בחרו מאגר ידע");
      elements.tenantIdInput.append(defaultOption);
      for (const [id, displayName] of catalog) {
        const option = document.createElement("option");
        option.value = id;
        option.textContent = displayName === id ? id : `${displayName} (${id})`;
        elements.tenantIdInput.append(option);
      }
      if (prior && !catalog.has(prior)) {
        const option = document.createElement("option");
        option.value = prior;
        option.textContent = prior;
        elements.tenantIdInput.append(option);
      }
      elements.tenantIdInput.value = prior;
      updateWorkspaceSummary();
      if (elements.settingsError.textContent === uiText(
        "Knowledge bases could not be loaded. Test connection to retry.",
        "לא ניתן לטעון את מאגרי הידע. בדקו את החיבור כדי לנסות שוב.",
      )) showSettingsError("");
      return true;
    } catch (error) {
      // Preserve the current selection and make a failed discovery retryable.
      if (!elements.settingsOverlay.classList.contains("hidden") &&
          normalizeApiBaseUrl(elements.apiBaseUrlInput.value) === normalizedBaseUrl) {
        showSettingsError(uiText(
          "Knowledge bases could not be loaded. Test connection to retry.",
          "לא ניתן לטעון את מאגרי הידע. בדקו את החיבור כדי לנסות שוב.",
        ));
      }
      console.warn("Authorized knowledge bases could not be loaded.", error);
      return false;
    } finally {
      window.clearTimeout(timeout);
    }
  }

  function updateWorkspaceSummary() {
    const cachedTenant = lastInventoryData?.identity === connectionIdentity(settings)
      ? normalizeTenantId(lastInventoryData.effectiveTenant) : "";
    const tenant = settings.tenantId ||
      (TENANT_ID_PATTERN.test(cachedTenant) ? cachedTenant : "") ||
      uiText("Default workspace", "סביבת העבודה");
    const label = tenantCatalogBaseUrl === normalizeApiBaseUrl(settings.apiBaseUrl)
      ? tenantCatalog.get(tenant) || tenant : tenant;
    elements.tenantDisplay.textContent = label;
    elements.tenantDisplay.title = tenant;
  }

  let lastConnectionState = null;
  function setConnectionStatus(status, label, detail = "") {
    lastConnectionState = { status, label, detail };
    const labels = {
      "Connected": uiText("API connected", "API מחובר"),
      "API connected": uiText("API connected", "API מחובר"),
      "Checking…": uiText("Checking API…", "בודקים חיבור API…"),
      "Reconnecting…": uiText("Reconnecting…", "מתחברים מחדש…"),
      "Not ready": uiText("API connected", "API מחובר"),
      "Offline": uiText("Service unreachable", "השירות אינו נגיש"),
    };
    const visibleLabel = labels[label] || label;
    elements.connectionStatusButton.dataset.status = status;
    elements.connectionStatusLabel.textContent = visibleLabel;
    // Infrastructure addresses and internal errors belong in Settings, not the customer header.
    const description = status === "offline"
      ? uiText("The app service could not be reached. Check that the local service and WSL are running.",
        "לא ניתן להתחבר לשירות. יש לבדוק שהשירות המקומי ו-WSL פועלים.")
      : status === "degraded"
        ? uiText("The API is reachable. The selected model or document workspace is not ready.",
          "ה-API זמין. המודל שנבחר או סביבת המסמכים עדיין אינם מוכנים.")
        : uiText("Check API connection", "בדיקת חיבור API");
    elements.connectionStatusButton.title = description;
    elements.connectionStatusButton.setAttribute("aria-label", visibleLabel + ". " + description);
  }

  function readinessReasonMessage(reason, tenantId = "") {
    const knowledgeBase = tenantId ? `Knowledge base “${tenantId}”` : "The knowledge base";
    switch (reason) {
      case "model_unavailable":
        return "The answer model is still loading or unavailable.";
      case "tenant_unresolved":
        return `${knowledgeBase} could not be resolved. Check its ID and your access.`;
      case "retriever_empty":
        return `${knowledgeBase} has no searchable documents yet.`;
      case "retriever_unavailable":
        return `${knowledgeBase} is not ready for search.`;
      case "source_index_rebuild_required":
        return `${knowledgeBase} must be rebuilt before source documents can be opened safely.`;
      default:
        return `${knowledgeBase} is not ready yet.`;
    }
  }

  function retireStaleSourceState(sourceIds) {
    let changed = false;
    let activeRetired = false;
    for (const thread of workspace.threads) {
      const stale = (thread.selectedSourceId && !sourceIds.has(thread.selectedSourceId)) ||
        thread.messages.some((message) =>
          (message.selectedSourceId && !sourceIds.has(message.selectedSourceId)) ||
          (message.role === "assistant" && message.citations.some((citation) =>
            citation.sourceId && !sourceIds.has(citation.sourceId))));
      if (!stale) continue;
      thread.conversationId = "";
      thread.selectedSourceId = "";
      for (const message of thread.messages) {
        message.retiredCorpus = true;
        message.citations = [];
        message.selectedSourceId = "";
      }
      changed = true;
      if (thread.id === workspace.activeThreadId) activeRetired = true;
    }
    if (activeRetired && !pendingRequests.size) {
      const fresh = newThread();
      workspace.threads.unshift(fresh);
      workspace.activeThreadId = fresh.id;
    }
    if (changed) {
      persistWorkspace();
      renderAll({ forceScroll: activeRetired });
      showToast("The document corpus changed. A fresh conversation is ready; retired source links were removed.");
    }
  }

  let modelSwitchPending = false;
  let modelSelectionIntent = null;
  let modelStatusRevision = 0;

  let lastModelData = null;
  function renderModelState(data) {
    lastModelData = data;
    const ownerBackend = data?.backend || data?.selected_backend;
    const reportedState = data?.state || data?.model_state || "error";
    const loaded = data?.loaded === true || data?.model_loaded === true;
    const ready = data?.ready === true || data?.model_ready === true || data?.model_inference_ready === true;
    const unconfirmed = (reportedState === "ready" && (!loaded || !ready)) || (reportedState === "busy" && !loaded);
    const supportedBackend = SUPPORTED_MODEL_BACKENDS.has(ownerBackend);
    const ownerState = supportedBackend && !unconfirmed ? reportedState : "error";
    const ownerError = data?.error || data?.model_error || (unconfirmed ? "model_unavailable" : ownerBackend && !supportedBackend ? "unsupported_model" : "");
    if (modelSelectionIntent?.identity !== connectionIdentity(settings)) modelSelectionIntent = null;
    if (!modelSwitchPending && modelSelectionIntent?.backend === ownerBackend && data?.ready === true) {
      modelSelectionIntent = null;
    }
    const backend = modelSelectionIntent?.backend || ownerBackend;
    const mismatch = modelSelectionIntent && ownerBackend !== backend;
    const state = mismatch ? (modelSelectionIntent.error ? "error" : "initializing") : ownerState;
    const error = mismatch ? modelSelectionIntent.error : ownerError || modelSelectionIntent?.error || "";
    const admissionFailed = /resource_admission|resource_floor|insufficient|admission.failed/i.test(String(error));
    const reportedModel = data?.selected || data?.selected_model;
    const label = typeof reportedModel === "string" && reportedModel.trim()
      ? reportedModel.trim().slice(0, 128)
      : ownerBackend === "gemma" ? "Gemma 4 31B" : uiText("Local model", "מודל מקומי");
    if (elements.answerModelName) elements.answerModelName.textContent = label;
    let visualState;
    let statusLabel;
    if (data?.unreachable) {
      visualState = "unreachable";
      statusLabel = uiText("Model status unavailable", "מצב המודל אינו זמין");
    } else if (admissionFailed) {
      visualState = "admission-failed";
      statusLabel = uiText("Resources unavailable", "אין מספיק משאבים");
    } else if (state === "ready") {
      visualState = "ready";
      statusLabel = uiText("Model ready", "המודל מוכן");
    } else if (state === "busy") {
      visualState = "busy";
      statusLabel = uiText("Generating answer", "יוצרים תשובה");
    } else if (state === "error") {
      visualState = "error";
      statusLabel = /model_start|before_ready|model_identity|model_readiness/.test(String(error))
        ? uiText("Model startup failed", "טעינת המודל נכשלה")
        : uiText("Model unavailable", "המודל אינו זמין");
    } else {
      visualState = "loading";
      statusLabel = uiText("Loading model…", "טוענים את המודל…");
    }
    if (SUPPORTED_MODEL_BACKENDS.has(backend)) elements.chatBackend.value = backend;
    elements.modelState.textContent = statusLabel;
    elements.modelState.dataset.state = visualState;
    elements.modelState.title = label + " · " + statusLabel;
    elements.modelRetry.textContent = uiText("Retry model", "ניסיון נוסף למודל");
    elements.modelRetry.hidden = visualState !== "error" && visualState !== "admission-failed";
    elements.modelRetry.disabled = modelSwitchPending;
    const diagnostics = document.getElementById("model-diagnostics");
    const diagnosticDetails = document.getElementById("model-diagnostic-details");
    if (diagnostics) {
      const safeCode = /^[a-z0-9_]{1,100}$/.test(String(error)) ? String(error) : "";
      diagnostics.textContent = label + " · " + state + (safeCode ? " · " + safeCode : "");
      if (diagnosticDetails) {
        diagnosticDetails.hidden = !error && !data?.unreachable;
        if (diagnosticDetails.hidden) diagnosticDetails.open = false;
        const summary = diagnosticDetails.querySelector("summary");
        if (summary) summary.textContent = uiText("Technical details", "פרטים טכניים");
      }
    }
    if (["error", "admission-failed", "loading", "unreachable"].includes(visualState) && lastConnectionState?.status === "online") {
      setConnectionStatus("degraded", "API connected");
    }
    updateComposerState();
  }

  async function switchAnswerModel() {
    if (modelSwitchPending) return;
    const backend = elements.chatBackend.value;
    if (!SUPPORTED_MODEL_BACKENDS.has(backend)) {
      showToast(uiText("Choose a workspace and check the connection first.", "בחרו סביבת עבודה ובדקו את החיבור תחילה."));
      return;
    }
    const generation = workspaceGeneration;
    const baseUrl = normalizeApiBaseUrl(settings.apiBaseUrl);
    const tenant = normalizeTenantId(settings.tenantId);
    // In-flight checks for the previous worker must not overwrite this selection.
    const statusRevision = ++modelStatusRevision;
    modelSwitchPending = true;
    modelSelectionIntent = {backend, identity:connectionIdentity(settings), error:""};
    elements.chatBackend.disabled = true;
    renderModelState({backend, state:"initializing"});
    const controller = new AbortController();
    const timer = window.setTimeout(() => controller.abort(), 3600000);
    const progress = window.setInterval(async () => {
      try {
        const response = await fetch(`${baseUrl}/api/model`, {
          headers: tenant ? {"X-Tenant-ID":tenant} : {}, cache:"no-store",
        });
        const data = await readBoundedResponseJson(response);
        if (generation === workspaceGeneration && statusRevision === modelStatusRevision && modelSwitchPending && response.ok) renderModelState(data);
      } catch (_) { /* The switch response remains the authority for completion. */ }
    }, 3000);
    try {
      const response = await fetch(`${baseUrl}/api/model`, {
        method:"POST", headers:{"Content-Type":"application/json", ...(tenant ? {"X-Tenant-ID":tenant} : {})},
        body:JSON.stringify({backend}), signal:controller.signal,
      });
      const data = await readBoundedResponseJson(response);
      if (generation !== workspaceGeneration) return;
      if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "The selected model could not become ready.");
      if (data.backend !== backend || typeof data.selected !== "string" || !data.selected || data.ready !== true || data.loaded !== true) throw new Error("Model readiness was not confirmed.");
      renderModelState(data);
      showToast(uiText("The model is ready.", "המודל מוכן."));
    } catch (error) {
      if (generation === workspaceGeneration) {
        const detail = controller.signal.aborted ? "Model switch timed out. Check model status before retrying." : error.message;
        modelSelectionIntent = {backend, identity:connectionIdentity(settings), error:detail};
        renderModelState({backend, state:"error", error:detail});
        showToast(uiText("The model is unavailable. Check its terminal or log, then retry.", "המודל אינו זמין. בדקו את חלון ההפעלה או את היומן ונסו שוב."));
      }
    } finally {
      window.clearTimeout(timer);
      window.clearInterval(progress);
      modelSwitchPending = false;
      elements.chatBackend.disabled = activeDocumentSourceIds === null;
      elements.modelRetry.disabled = false;
      if (generation === workspaceGeneration) void checkBackendHealth();
    }
  }

  let modelStatusChecking = false;
  async function refreshSelectedModelState() {
    if (modelStatusChecking || activeDocumentSourceIds === null) return;
    modelStatusChecking = true;
    const generation = workspaceGeneration;
    const statusRevision = modelStatusRevision;
    const tenant = normalizeTenantId(settings.tenantId);
    const controller = new AbortController();
    const timer = window.setTimeout(() => controller.abort(), 5000);
    try {
      const response = await fetch(`${normalizeApiBaseUrl(settings.apiBaseUrl)}/api/model`, {
        headers:tenant ? {"X-Tenant-ID":tenant} : {}, cache:"no-store", signal:controller.signal,
      });
      const data = await readBoundedResponseJson(response);
      if (generation !== workspaceGeneration || statusRevision !== modelStatusRevision || !response.ok) return;
      // A successful authenticated status request proves the API is reachable,
      // even while a model generation is in progress. Model readiness remains
      // a separate state and can still mark the connection as degraded below.
      if (lastConnectionState?.status === "offline") {
        setConnectionStatus("online", "API connected", "Model status endpoint responded.");
      }
      renderModelState(data);
    } catch (_) {
      if (generation === workspaceGeneration && statusRevision === modelStatusRevision && !modelSwitchPending) {
        setConnectionStatus("offline", "Offline");
        renderModelState({backend:elements.chatBackend.value, state:"error", unreachable:true});
      }
    } finally {
      window.clearTimeout(timer);
      modelStatusChecking = false;
    }
  }

  let lastInventoryData = null;
  function renderDocumentInventory(data, effectiveTenant) {
    const sourceIds = new Set();
    const fragment = document.createDocumentFragment();
    for (const item of data.documents) {
      if (!SOURCE_ID_PATTERN.test(String(item.source_id || ""))) throw new Error("The document inventory has an invalid source identity.");
      sourceIds.add(item.source_id);
      const entry = textElement("div", "inventory-document", "");
      const url = normalizeSourceOpenUrl(item.open_url, item.source_id, effectiveTenant);
      const sourceLabel = frontendPolicy.safeCitationRelativePath(item.source_relative_path)
        || String(item.display_name || uiText("Document", "מסמך"));
      const name = textElement(url ? "a" : "span", "inventory-name", sourceLabel);
      name.dir = "auto";
      name.title = sourceLabel;
      if (url) { name.href = url; name.target = "_blank"; name.rel = "noopener noreferrer"; }
      entry.append(name);
      const context = document.createElement("div");
      context.className = "inventory-context";
      if (Number.isInteger(item.pdf_page_count)) {
        context.append(textElement("span", "inventory-pages",
          uiText(item.pdf_page_count + " pages", item.pdf_page_count + " עמודים")));
      }
      if (Array.isArray(item.indexed_slides) && item.indexed_slides.length) {
        context.append(textElement("span", "inventory-slides",
          uiText("Slides: ", "שקופיות: ") + item.indexed_slides.join(", ")));
      }
      if (Array.isArray(item.indexed_sheets) && item.indexed_sheets.length) {
        context.append(textElement("span", "inventory-sheets",
          uiText("Sheets: ", "גיליונות: ") + item.indexed_sheets.join(", ")));
      }
      if (url) {
        const open = textElement("a", "inventory-open", uiText("Open original", "פתיחת המקור"));
        open.href = url; open.target = "_blank"; open.rel = "noopener noreferrer";
        open.setAttribute("aria-label", uiText("Open original: ", "פתיחת מקור: ") + sourceLabel);
        context.append(open);
      }
      entry.append(context);
      const technical = document.createElement("details");
      technical.className = "inventory-technical";
      technical.append(textElement("summary", "", uiText("Details", "פרטים")));
      technical.append(textElement("div", "inventory-detail",
        uiText((Number(item.chunk_count) || 0) + " searchable chunks", (Number(item.chunk_count) || 0) + " מקטעים לחיפוש")));
      const checksum = textElement("div", "inventory-detail inventory-checksum", "SHA-256: " + String(item.source_sha256 || ""));
      checksum.dir = "ltr";
      technical.append(checksum);
      if (item.text_extraction_limitations) technical.append(textElement("div", "inventory-detail", String(item.text_extraction_limitations)));
      entry.append(technical);
      fragment.append(entry);
    }
    elements.documentInventory.replaceChildren(fragment);
    if (!data.documents.length) elements.documentInventory.textContent = uiText("No active indexed documents.", "אין מסמכים מאונדקסים פעילים.");
    return sourceIds;
  }

  async function refreshDocumentInventory() {
    if (inventoryController) inventoryController.abort();
    const controller = new AbortController();
    inventoryController = controller;
    const generation = workspaceGeneration;
    const statusRevision = modelStatusRevision;
    const tenant = normalizeTenantId(settings.tenantId);
    activeDocumentSourceIds = null;
    elements.documentInventory.textContent = uiText("Loading documents…", "טוענים מסמכים…");
    elements.chatBackend.disabled = true;
    const timeout = window.setTimeout(() => controller.abort(), 15000);
    try {
      const response = await fetch(`${normalizeApiBaseUrl(settings.apiBaseUrl)}/api/documents`, {
        headers: tenant ? { "X-Tenant-ID": tenant } : {},
        cache: "no-store", signal: controller.signal,
      });
      const data = await readBoundedResponseJson(response);
      if (!response.ok) throw new Error("Active documents are unavailable. Check tenant access and connection.");
      if (inventoryController !== controller || generation !== workspaceGeneration) return;
      const effectiveTenant = normalizeTenantId(data.tenant_id);
      if (!TENANT_ID_PATTERN.test(effectiveTenant) || (tenant && tenant !== effectiveTenant) || !Array.isArray(data.documents)) {
        throw new Error("The document inventory has an invalid tenant identity.");
      }
      const sourceIds = renderDocumentInventory(data, effectiveTenant);
      lastInventoryData = { data, effectiveTenant, identity: connectionIdentity(settings) };
      updateWorkspaceSummary();
      activeDocumentSourceIds = sourceIds;
      retireStaleSourceState(sourceIds);
      if (!Array.isArray(data.backend_options) || data.backend_options.length !== 1 ||
          !SUPPORTED_MODEL_BACKENDS.has(data.backend_options[0]) ||
          data.selected_backend !== data.backend_options[0]) {
        throw new Error("The configured model backend was not confirmed.");
      }
      elements.chatBackend.value = data.backend_options[0];
      elements.chatBackend.disabled = modelSwitchPending;
      if (statusRevision === modelStatusRevision) renderModelState(data);
    } catch (error) {
      if (inventoryController === controller && generation === workspaceGeneration) {
        activeDocumentSourceIds = null;
        renderModelState({backend:elements.chatBackend.value, state:"error", error:"model_configuration_unavailable"});
        elements.documentInventory.textContent = error?.message || "Active documents could not be loaded.";
      }
    } finally {
      window.clearTimeout(timeout);
      if (inventoryController === controller) inventoryController = null;
    }
  }

  async function checkBackendHealth({ announceResult = false } = {}) {
    void refreshTenantChoices();
    void refreshDocumentInventory();
    if (healthCheckController) healthCheckController.abort();
    const controller = new AbortController();
    healthCheckController = controller;
    const baseUrl = normalizeApiBaseUrl(settings.apiBaseUrl);
    const tenantId = normalizeTenantId(settings.tenantId);
    const healthWorkspaceGeneration = workspaceGeneration;
    const statusRevision = modelStatusRevision;
    setConnectionStatus("checking", "Checking…", `Checking ${baseUrl}`);
    const timeoutId = window.setTimeout(() => controller.abort(), 6000);
    try {
      const response = await fetch(`${baseUrl}/readyz`, {
        method: "GET",
        headers: tenantId ? { "X-Tenant-ID": tenantId } : {},
        cache: "no-store",
        signal: controller.signal,
      });
      let data;
      try {
        data = await readBoundedResponseJson(response);
      } catch (_error) {
        const invalidResponse = new Error("The readiness endpoint returned invalid JSON.");
        invalidResponse.name = "BackendNotReady";
        throw invalidResponse;
      }
      if (healthWorkspaceGeneration === workspaceGeneration && statusRevision === modelStatusRevision && data?.model_state) renderModelState(data);
      if (!response.ok) {
        const rejected = new Error(
          `The readiness check returned HTTP ${response.status}. ${readinessReasonMessage(data?.reason, tenantId)}`,
        );
        rejected.name = "BackendNotReady";
        throw rejected;
      }
      const ready =
        data?.ready === true &&
        data?.model_loaded === true &&
        (!tenantId || data?.retriever_ready === true);
      if (!ready) {
        const reason =
          typeof data?.reason === "string"
            ? data.reason
            : data?.model_loaded !== true
              ? "model_unavailable"
              : tenantId && data?.retriever_ready !== true
                ? "retriever_unavailable"
                : "";
        const unavailable = new Error(
          `The backend is reachable. ${readinessReasonMessage(reason, tenantId)}`,
        );
        unavailable.name = "BackendNotReady";
        throw unavailable;
      }
      if (
        healthCheckController !== controller ||
        healthWorkspaceGeneration !== workspaceGeneration ||
        statusRevision !== modelStatusRevision
      ) return;
      setConnectionStatus("online", "Connected", `Backend ready at ${baseUrl}`);
      if (announceResult) {
        announce("Backend connected and ready");
        showToast("Backend connected and ready");
      }
    } catch (error) {
      if (healthCheckController !== controller || statusRevision !== modelStatusRevision) return;
      const notReady = error?.name === "BackendNotReady";
      if (!notReady) renderModelState({backend:elements.chatBackend.value, state:"error", unreachable:true});
      const detail = controller.signal.aborted
        ? `Connection check timed out for ${baseUrl}`
        : notReady
          ? error.message
          : `Could not reach ${baseUrl}${error?.message ? ` (${error.message})` : ""}`;
      setConnectionStatus(notReady ? "degraded" : "offline", notReady ? "API connected" : "Offline", detail);
      if (announceResult) {
        const message = notReady
          ? "The backend is connected, but the selected knowledge base is not ready."
          : "The backend could not be reached. Open Settings to verify the API URL.";
        announce(message);
        showToast(notReady ? "Knowledge base not ready" : "Backend unavailable — check Settings");
      }
    } finally {
      window.clearTimeout(timeoutId);
      if (healthCheckController === controller) healthCheckController = null;
    }
  }

  function createAndSelectThread() {
    flushActiveDraft();
    abortThreadRequest(workspace.activeThreadId, "conversation_changed");
    closeSourcePanel({ restoreFocus: false });
    const thread = newThread();
    workspace.threads.unshift(thread);
    workspace.activeThreadId = thread.id;
    threadSearchQuery = "";
    elements.threadSearch.value = "";
    elements.threadSearchClear.classList.add("hidden");
    resetSourceFilter();
    persistWorkspace();
    closeSidebar();
    renderAll({ forceScroll: true });
    elements.chatInput.focus();
  }

  function selectThread(threadId) {
    if (!getThread(threadId)) return;
    if (threadId === workspace.activeThreadId) {
      closeSidebar();
      elements.chatInput.focus();
      return;
    }
    flushActiveDraft();
    closeSourcePanel({ restoreFocus: false });
    abortThreadRequest(workspace.activeThreadId, "conversation_changed");
    workspace.activeThreadId = threadId;
    resetSourceFilter();
    persistWorkspace();
    closeSidebar();
    renderAll({ forceScroll: true });
    elements.chatInput.focus();
  }

  function showThreadActionError(message) {
    elements.threadActionError.textContent = message;
    elements.threadActionError.classList.toggle("hidden", !message);
    elements.threadActionInput.setAttribute("aria-invalid", message ? "true" : "false");
  }

  function openThreadActionDialog({
    kind,
    title,
    description,
    confirmLabel,
    inputLabel = "",
    initialValue = "",
    danger = false,
    onConfirm,
  }) {
    closeSourcePanel({ restoreFocus: false });
    threadActionReturnFocus = document.activeElement;
    threadActionState = { kind, inputLabel, onConfirm };
    elements.threadActionTitle.textContent = title;
    elements.threadActionDescription.textContent = description;
    elements.threadActionField.classList.toggle("hidden", !inputLabel);
    elements.threadActionLabel.textContent = inputLabel || "Value";
    elements.threadActionInput.value = initialValue;
    elements.threadActionConfirmButton.textContent = confirmLabel;
    elements.threadActionConfirmButton.className = danger ? "danger-button" : "primary-button";
    elements.threadActionVisual.classList.toggle("danger", danger);
    elements.threadActionVisual.replaceChildren(
      icon(danger ? "delete" : kind === "replace-draft" ? "sparkles" : "edit", "20px"),
    );
    showThreadActionError("");
    setAppInert(true);
    elements.threadActionOverlay.classList.remove("hidden");
    elements.threadActionOverlay.classList.add("flex");
    elements.threadActionOverlay.setAttribute("aria-hidden", "false");
    window.setTimeout(() => {
      if (inputLabel) {
        elements.threadActionInput.focus();
        elements.threadActionInput.select();
      } else {
        elements.threadActionConfirmButton.focus();
      }
    }, 0);
  }

  function closeThreadActionDialog({ restoreFocus = true } = {}) {
    if (elements.threadActionOverlay.classList.contains("hidden")) return;
    elements.threadActionOverlay.classList.add("hidden");
    elements.threadActionOverlay.classList.remove("flex");
    elements.threadActionOverlay.setAttribute("aria-hidden", "true");
    setAppInert(false);
    showThreadActionError("");
    threadActionState = null;
    if (
      restoreFocus &&
      threadActionReturnFocus &&
      typeof threadActionReturnFocus.focus === "function"
    ) {
      threadActionReturnFocus.focus();
    }
    threadActionReturnFocus = null;
  }

  function submitThreadActionDialog() {
    if (!threadActionState || typeof threadActionState.onConfirm !== "function") return;
    let value = "";
    if (threadActionState.inputLabel) {
      value = elements.threadActionInput.value.trim();
      if (!value) {
        showThreadActionError("Enter a conversation name before saving.");
        elements.threadActionInput.focus();
        return;
      }
      value = Array.from(value).slice(0, 80).join("");
    }
    const confirm = threadActionState.onConfirm;
    closeThreadActionDialog({ restoreFocus: false });
    confirm(value);
  }

  function renameThread(threadId) {
    const thread = getThread(threadId);
    if (!thread) return;
    openThreadActionDialog({
      kind: "rename",
      title: "Rename conversation",
      description: "Choose a short name that will be easy to recognize later.",
      confirmLabel: "Save name",
      inputLabel: "Conversation name",
      initialValue: thread.title,
      onConfirm: (nextTitle) => {
        thread.title = nextTitle;
        thread.updatedAt = new Date().toISOString();
        persistWorkspace();
        renderAll({ scroll: false });
        showToast("Conversation renamed");
        announce("Conversation renamed");
        elements.chatInput.focus();
      },
    });
  }

  function deleteThread(threadId) {
    const thread = getThread(threadId);
    if (!thread) return;
    openThreadActionDialog({
      kind: "delete",
      title: "Delete conversation?",
      description: `“${thread.title}” will be removed from this browser. Your knowledge-base documents are not affected.`,
      confirmLabel: "Delete conversation",
      danger: true,
      onConfirm: () => performDeleteThread(threadId),
    });
  }

  function performDeleteThread(threadId) {
    const thread = getThread(threadId);
    if (!thread) return;
    const deletingActiveThread = workspace.activeThreadId === threadId;
    thread.messages.forEach((message) => citationDisclosureState.delete(message.id));
    const pending = pendingRequests.get(threadId);
    if (pending) {
      pending.abortReason = "deleted";
      clearRequestTimer(pending);
      pending.controller.abort();
    }
    pendingRequests.delete(threadId);
    threadErrors.delete(threadId);
    workspace.threads = workspace.threads.filter((item) => item.id !== threadId);
    if (!workspace.threads.length) workspace.threads.push(newThread());
    if (workspace.activeThreadId === threadId) {
      workspace.activeThreadId = workspace.threads[0].id;
    }
    persistWorkspace();
    renderAll(deletingActiveThread ? { forceScroll: true } : { scroll: false });
    showToast("Conversation deleted");
    announce("Conversation deleted");
    elements.chatInput.focus();
  }

  function makeAutomaticTitle(query) {
    const compact = query.replace(/\s+/g, " ").trim();
    const characters = Array.from(compact);
    return characters.length > 46
      ? `${characters.slice(0, 46).join("")}…`
      : compact || "Untitled thread";
  }

  function editAnsweredQuestion(messageId) {
    const thread = getActiveThread();
    if (!thread || pendingRequests.has(thread.id)) return;
    const userIndex = thread.messages.findIndex(
      (message) => message.id === messageId && message.role === "user",
    );
    const answer = thread.messages[userIndex + 1];
    if (
      userIndex < 0 ||
      userIndex !== thread.messages.length - 2 ||
      answer?.role !== "assistant"
    ) return;

    const hasDraft = Boolean(elements.chatInput.value.trim());
    openThreadActionDialog({
      kind: "edit-question",
      title: uiText("Edit the latest question?", "לערוך את השאלה האחרונה?"),
      description: hasDraft
        ? uiText("The latest answer will be removed and your current draft will be replaced. Your source documents are unchanged.", "התשובה האחרונה תוסר והטיוטה הנוכחית תוחלף. מסמכי המקור יישארו ללא שינוי.")
        : uiText("The latest answer will be removed and the question will return to the composer. Your source documents are unchanged.", "התשובה האחרונה תוסר והשאלה תחזור לאזור הכתיבה. מסמכי המקור יישארו ללא שינוי."),
      confirmLabel: uiText("Edit question", "עריכת השאלה"),
      onConfirm: () => restoreAnsweredQuestionForEditing(thread.id, messageId),
    });
  }

  function restoreAnsweredQuestionForEditing(threadId, messageId) {
    const thread = getThread(threadId);
    if (!thread || pendingRequests.has(threadId)) return;
    const userIndex = thread.messages.findIndex(
      (message) => message.id === messageId && message.role === "user",
    );
    const answer = thread.messages[userIndex + 1];
    if (
      userIndex < 0 ||
      userIndex !== thread.messages.length - 2 ||
      answer?.role !== "assistant"
    ) return;
    citationDisclosureState.delete(answer.id);
    const questionMessage = thread.messages[userIndex];
    const question = questionMessage.content;
    thread.messages.splice(userIndex, 2);
    thread.draft = question;
    thread.selectedSourceId = questionMessage.selectedSourceId || "";
    if (!thread.messages.length) thread.title = "Untitled thread";
    thread.updatedAt = new Date().toISOString();
    threadErrors.delete(threadId);
    persistWorkspace();
    renderAll({ forceScroll: true });
    window.requestAnimationFrame(() => {
      elements.chatInput.focus();
      const end = elements.chatInput.value.length;
      if (typeof elements.chatInput.setSelectionRange === "function") {
        elements.chatInput.setSelectionRange(end, end);
      }
    });
    showToast("Question ready to edit");
    announce("The latest question is ready to edit");
  }

  function recentHistory(messages) {
    const candidates = messages
      .filter((message) => !message.retiredCorpus && (message.role === "user" || message.role === "assistant"))
      .slice(-MAX_HISTORY_TURNS)
      .map((message) => {
        const turn = {
          role: message.role,
          content: String(message.content || "").slice(0, MAX_HISTORY_TURN_CHARACTERS),
        };
        if (message.role === "assistant") {
          const citations = Array.isArray(message.citations)
            ? message.citations.slice(0, 8)
            : [];
          turn.source_ids = [
            ...new Set(
              citations
                .map((citation) => String(citation.sourceId || ""))
                .filter((value) => SOURCE_ID_PATTERN.test(value)),
            ),
          ].slice(0, 8);
          turn.evidence_chunk_ids = [
            ...new Set(
              citations
                .map((citation) => String(citation.evidenceChunkId || ""))
                .filter((value) => /^[A-Za-z0-9_-]{24}$/.test(value)),
            ),
          ].slice(0, 8);
          turn.pages = [
            ...new Set(
              citations
                .map((citation) => citation.evidencePage || citation.page)
                .filter((value) => Number.isInteger(value) && value > 0),
            ),
          ].slice(0, 8);
        }
        return turn;
      })
      .filter((message) => message.content.trim());
    let remaining = MAX_HISTORY_TOTAL_CHARACTERS;
    let turns = [];
    for (let index = candidates.length - 1; index >= 0; index -= 1) {
      const candidate = candidates[index];
      if (candidate.content.length > remaining) break;
      turns.unshift(candidate);
      remaining -= candidate.content.length;
    }
    while (turns.length && turns[0].role === "assistant") turns = turns.slice(1);
    return turns;
  }

  function parseErrorDetail(data, status) {
    if (data && typeof data.detail === "string") return data.detail;
    if (data && Array.isArray(data.detail)) {
      return data.detail
        .map((item) => (item && item.msg ? String(item.msg) : "Invalid request"))
        .join("; ");
    }
    if (data && typeof data.message === "string") return data.message;
    return `The server returned HTTP ${status}.`;
  }

  function errorWithRequestReference(message, response) {
    const value = response?.headers?.get?.("X-Request-ID")?.trim() || "";
    if (!/^[A-Za-z0-9._:-]{1,128}$/.test(value)) return message;
    return `${message} Request reference: ${value}`;
  }

  function retryAfterSeconds(response) {
    const raw = response?.headers?.get?.("Retry-After")?.trim() || "";
    if (/^\d{1,5}$/.test(raw)) return Math.min(3600, Number(raw));
    const retryAt = Date.parse(raw);
    if (!Number.isFinite(retryAt)) return null;
    return Math.min(3600, Math.max(1, Math.ceil((retryAt - Date.now()) / 1000)));
  }

  function requestFailureFromResponse(data, response) {
    const status = Number(response?.status) || 0;
    const detail = parseErrorDetail(data, status);
    const detailLower = detail.toLowerCase();
    const resourceAdmissionFailed = /resource_admission|resource_floor/.test(detailLower);
    const retryAfter = retryAfterSeconds(response);
    let title = uiText("Request failed", "לא ניתן להשלים את הבקשה");
    let hint = uiText("Review the question, then try again.", "בדקו את השאלה ולאחר מכן נסו שוב.");
    let retryable = status >= 500;
    let allowEdit = true;
    let openSettings = false;

    if (status === 400) {
      title = uiText("Request needs attention", "נדרשת בדיקה של הבקשה");
      hint = uiText("Review the question and connection details before sending it again.", "בדקו את השאלה ופרטי החיבור לפני שליחה חוזרת.");
      retryable = false;
    } else if (status === 413 || status === 422) {
      title = uiText("Question needs editing", "נדרשת עריכה של השאלה");
      hint = uiText("Shorten or revise the question, then send it again.", "קצרו או ערכו את השאלה ולאחר מכן שלחו אותה שוב.");
      retryable = false;
    } else if (status === 401 || status === 403) {
      title = uiText("Access needs attention", "נדרשת בדיקת הרשאות");
      hint = uiText("Check the API address and knowledge-base access in Settings.", "בדקו בהגדרות את כתובת ה-API ואת ההרשאות לסביבת המסמכים.");
      retryable = false;
      allowEdit = false;
      openSettings = true;
    } else if (status === 404) {
      title = uiText("Knowledge base not found", "סביבת המסמכים לא נמצאה");
      hint = uiText("Check the selected knowledge-base ID in Settings.", "בדקו בהגדרות את מזהה סביבת המסמכים.");
      retryable = false;
      allowEdit = false;
      openSettings = true;
    } else if (status === 429) {
      title = uiText("Service is busy", "השירות עסוק");
      hint = retryAfter == null
        ? uiText("Wait briefly, then try again.", "המתינו מעט ונסו שוב.")
        : `Try again in about ${retryAfter} ${retryAfter === 1 ? "second" : "seconds"}.`;
      retryable = true;
      allowEdit = false;
    } else if (status === 502) {
      const answerNeedsNarrowing = /context window|output limit|reliable answer|finish_reason|incomplete/.test(
        detailLower,
      );
      title = answerNeedsNarrowing
        ? uiText("Answer could not be verified", "לא ניתן לאמת את התשובה")
        : uiText("Answer service unavailable", "שירות התשובות אינו זמין");
      hint = answerNeedsNarrowing
        ? uiText("Ask a narrower question or request fewer details so every claim can be checked.", "נסחו שאלה ממוקדת יותר או בקשו פחות פרטים כדי לאפשר את בדיקת הטענות.")
        : uiText("The model service failed temporarily. Try again.", "שירות המודל נכשל זמנית. אפשר לנסות שוב.");
      retryable = !answerNeedsNarrowing;
      allowEdit = answerNeedsNarrowing;
    } else if (status === 503) {
      title = detailLower.includes("retrieval")
        ? uiText("Sources temporarily unavailable", "המקורות אינם זמינים כרגע")
        : uiText("Service temporarily unavailable", "השירות אינו זמין כרגע");
      hint = uiText("The service is not ready yet. Wait briefly, then try again.", "השירות עדיין אינו מוכן. המתינו מעט ונסו שוב.");
      retryable = true;
      allowEdit = false;
    } else if (status === 504) {
      title = uiText("Answer timed out", "תם הזמן להמתנה לתשובה");
      hint = uiText("Try again, or ask a narrower question if this keeps happening.", "נסו שוב, או נסחו שאלה ממוקדת יותר אם הבעיה נמשכת.");
      retryable = true;
    } else if (status >= 500) {
      title = uiText("Server error", "שגיאה בשירות");
      hint = uiText("The server could not complete the answer. Try again.", "השירות לא הצליח להשלים את התשובה. אפשר לנסות שוב.");
      retryable = true;
      allowEdit = false;
    }

    // Editing only restores local text; transient service failures do not block it.
    if (status === 429 || status >= 500) allowEdit = true;
    if (resourceAdmissionFailed) {
      title = uiText("Model resources unavailable", "משאבי המודל אינם זמינים");
      hint = uiText("Retry the selected model explicitly. Once it is ready, retry or edit this question.",
        "אפשר לנסות שוב לטעון את המודל שנבחר. כשהוא מוכן, אפשר לשלוח שוב את השאלה או לערוך אותה.");
    }

    const publicMessage = status === 401 || status === 403
      ? uiText("This workspace could not be accessed.", "לא ניתן לגשת לסביבת העבודה הזו.")
      : status === 404
        ? uiText("The selected document workspace was not found.", "סביבת המסמכים שנבחרה לא נמצאה.")
        : status === 429
          ? uiText("The answer service is busy.", "שירות התשובות עסוק כרגע.")
          : status === 413 || status === 422 || status === 400
            ? uiText("The question could not be accepted. Review it before sending again.", "לא ניתן לקבל את השאלה. בדקו אותה לפני שליחה נוספת.")
            : resourceAdmissionFailed
              ? uiText("The selected model cannot answer right now because resources are unavailable.", "המודל שנבחר אינו יכול לענות כרגע משום שאין מספיק משאבים.")
              : uiText("The answer could not be completed. Your question is saved.", "לא ניתן להשלים את התשובה. השאלה שלך נשמרה.");
    const failure = new Error(publicMessage);
    Object.assign(failure, {
      name: "RequestFailure",
      status,
      resourceAdmissionFailed,
      title,
      hint,
      diagnostic: errorWithRequestReference("HTTP " + status, response),
      retryable,
      allowEdit,
      openSettings,
    });
    return failure;
  }

  function networkRequestFailure(baseUrl) {
    const failure = new Error(uiText(
      "The app service could not be reached. Your question is saved.",
      "לא ניתן להתחבר לשירות. השאלה שלך נשמרה."));
    Object.assign(failure, {
      name: "NetworkRequestFailure",
      title: uiText("Service unreachable", "השירות אינו נגיש"),
      hint: uiText("Check that the local service and WSL are running, then retry the question.",
        "בדקו שהשירות המקומי ו-WSL פועלים, ולאחר מכן נסו לשלוח שוב את השאלה."),
      diagnostic: "API connection failed",
      retryable: true,
      allowEdit: true,
      openSettings: true,
    });
    return failure;
  }

  function clearRequestTimer(request) {
    if (!request || request.timeoutId == null) return;
    window.clearTimeout(request.timeoutId);
    request.timeoutId = null;
  }

  function abortThreadRequest(threadId, reason = "user") {
    const request = pendingRequests.get(threadId);
    if (!request || request.controller.signal.aborted) return false;
    request.abortReason = reason;
    clearRequestTimer(request);
    request.controller.abort();
    // Transport abort need not settle immediately. Retire ownership now so a
    // delayed body reader cannot restore a canceled answer or waiting indicator.
    pendingRequests.delete(threadId);
    const ownsWorkspace =
      request.workspaceKey === activeWorkspaceKey &&
      request.workspaceGeneration === workspaceGeneration;
    const failure = abortFailure(request);
    if (ownsWorkspace && failure && getThread(threadId)) {
      threadErrors.set(threadId, { ...failure, replaceMessageId: request.replaceMessageId });
      announce(failure.message);
    }
    if (ownsWorkspace) {
      renderThreadList();
      if (workspace.activeThreadId === threadId) {
        renderConversation();
        if (reason === "user") elements.chatInput.focus();
      }
    }
    return true;
  }

  function abortRequestsForConnectionChange() {
    if (healthCheckController) {
      healthCheckController.abort();
      healthCheckController = null;
    }
    pendingRequests.forEach((request) => {
      request.abortReason = "connection_changed";
      clearRequestTimer(request);
      request.controller.abort();
    });
    pendingRequests.clear();
    threadErrors.clear();
  }

  function abortFailure(request) {
    if (request.abortReason === "timeout") {
      return {
        message: `The request timed out after ${request.timeoutSeconds} seconds.`,
        title: uiText("Answer timed out", "תם הזמן להמתנה לתשובה"),
        hint: uiText("Try again, or ask a narrower question if this keeps happening.", "נסו שוב, או נסחו שאלה ממוקדת יותר אם הבעיה נמשכת."),
        retryable: true,
        allowEdit: true,
      };
    }
    if (request.abortReason === "user") {
      return {
        message: uiText("You stopped waiting for the answer.", "עצרתם את ההמתנה לתשובה."),
        title: uiText("Answer stopped", "ההמתנה לתשובה נעצרה"),
        hint: uiText("The server may still finish processing. You can retry or edit the question.", "ייתכן שהשרת עדיין מסיים לעבד את הבקשה. אפשר לנסות שוב או לערוך את השאלה."),
        retryable: true,
        allowEdit: true,
      };
    }
    return null;
  }

  function throwIfRequestAborted(request) {
    if (!request.controller.signal.aborted) return;
    const error = new Error("The request was aborted.");
    error.name = "AbortError";
    throw error;
  }

  async function requestAssistant(
    threadId,
    query,
    history,
    { replaceMessageId = null, selectedSourceId = "" } = {},
  ) {
    if (pendingRequests.has(threadId)) return;
    if (selectedModelBlocksQuestions()) {
      showToast(uiText("The selected model is not ready. Your question has not been retried.",
        "המודל שנבחר אינו מוכן. השאלה לא נשלחה שוב."));
      return;
    }
    const controller = new AbortController();
    const selectedBackend = elements.chatBackend.value;
    if (!SUPPORTED_MODEL_BACKENDS.has(selectedBackend)) {
      showToast(uiText("Choose a workspace and check the connection first.", "בחרו סביבת עבודה ובדקו את החיבור תחילה."));
      return;
    }
    const request = {
      controller,
      query,
      abortReason: "",
      timeoutId: null,
      timeoutSeconds: 3600,
      startedAt: Date.now(),
      requestId: makeId("request"),
      replaceMessageId,
      selectedSourceId: SOURCE_ID_PATTERN.test(String(selectedSourceId || ""))
        ? String(selectedSourceId)
        : "",
      conversationId: CONVERSATION_ID_PATTERN.test(
        String(getThread(threadId)?.conversationId || ""),
      )
        ? String(getThread(threadId).conversationId)
        : "",
      backend: selectedBackend,
      baseUrl: normalizeApiBaseUrl(settings.apiBaseUrl),
      tenantId: normalizeTenantId(settings.tenantId),
      workspaceKey: activeWorkspaceKey,
      workspaceGeneration,
    };
    pendingRequests.set(threadId, request);
    announce(
      replaceMessageId
        ? uiText("Preparing a new answer", "מכינים תשובה חדשה")
        : uiText("Preparing your answer", "מכינים את התשובה"),
    );
    request.timeoutId = window.setTimeout(() => {
      if (pendingRequests.get(threadId) !== request) return;
      abortThreadRequest(threadId, "timeout");
    }, request.timeoutSeconds * 1000);
    threadErrors.delete(threadId);
    renderThreadList();
    if (workspace.activeThreadId === threadId) renderConversation();

    const baseUrl = request.baseUrl;
    try {
      const response = await fetch(`${baseUrl}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Tenant-ID": request.tenantId,
          "X-Request-ID": request.requestId,
        },
        body: JSON.stringify({
          query,
          history,
          backend: request.backend,
          ...(request.conversationId
            ? { conversation_id: request.conversationId }
            : {}),
          ...(request.selectedSourceId
            ? { selected_source_id: request.selectedSourceId }
            : {}),
        }),
        signal: controller.signal,
      });
      const responseText = await readBoundedResponseText(response);
      throwIfRequestAborted(request);
      if (
        pendingRequests.get(threadId) !== request ||
        request.workspaceKey !== activeWorkspaceKey ||
        request.workspaceGeneration !== workspaceGeneration
      ) return;
      let data = null;
      if (responseText) {
        try {
          data = JSON.parse(responseText);
        } catch (_error) {
          if (response.ok) throw new Error("The server returned an invalid JSON response.");
        }
      }
      throwIfRequestAborted(request);
      if (!response.ok) {
        throw requestFailureFromResponse(data, response);
      }
      if (!data || typeof data.answer !== "string") {
        throw new Error("The server response did not include an answer.");
      }
      if (data.backend !== request.backend) {
        throw new Error("The response did not come from the selected model backend.");
      }

      const thread = getThread(threadId);
      if (!thread) return;
      const conversationId = String(data.conversation_id || "");
      if (!CONVERSATION_ID_PATTERN.test(conversationId)) {
        throw new Error("The server returned an invalid conversation identifier.");
      }
      thread.conversationId = conversationId;
      const effectiveTenant = normalizeTenantId(data.effective_tenant);
      if (
        !TENANT_ID_PATTERN.test(effectiveTenant) ||
        (request.tenantId && effectiveTenant !== request.tenantId)
      ) {
        throw new Error("The server returned an invalid effective tenant.");
      }
      const citations = Array.isArray(data.citations)
        ? data.citations.map((citation, index) =>
            normalizeCitation(citation, index, effectiveTenant))
        : [];
      const nextMessage = {
        id: replaceMessageId || makeId("message"),
        role: "assistant",
        content: data.answer.trim() || "No answer was returned.",
        answerDirection: frontendPolicy.normalizeTextDirection(
          data.answer_direction,
        ),
        citations,
        effectiveTenant,
        answerCompleteness: normalizeAnswerCompleteness(data.answer_completeness),
        responseReasonCodes: normalizePartialReasons(data.response_reason_codes),
        referenceRecovery: data.reference_recovery === "new_search" ? "new_search" : null,
        backend: request.backend,
        retiredCorpus: false,
        createdAt: new Date().toISOString(),
        durationMs: Date.now() - request.startedAt,
      };
      if (replaceMessageId) {
        const answerIndex = thread.messages.findIndex(
          (message) => message.id === replaceMessageId && message.role === "assistant",
        );
        if (answerIndex < 0) return;
        thread.messages.splice(answerIndex, 1, nextMessage);
        citationDisclosureState.delete(replaceMessageId);
      } else {
        thread.messages.push(nextMessage);
      }
      thread.updatedAt = new Date().toISOString();
      threadErrors.delete(threadId);
      persistWorkspace();
      setConnectionStatus("online", "Connected", `Backend answered at ${baseUrl}`);
      announce("Answer ready");
    } catch (error) {
      const ownsActiveWorkspace =
        pendingRequests.get(threadId) === request &&
        request.workspaceKey === activeWorkspaceKey &&
        request.workspaceGeneration === workspaceGeneration;
      if (controller.signal.aborted || (error && error.name === "AbortError")) {
        const failure = abortFailure(request);
        if (ownsActiveWorkspace && failure && getThread(threadId)) {
          threadErrors.set(threadId, { ...failure, replaceMessageId });
        }
        if (ownsActiveWorkspace && failure) announce(failure.message);
        return;
      }
      if (!ownsActiveWorkspace) return;
      const failure = error instanceof TypeError
        ? networkRequestFailure(baseUrl)
        : {
            message: error?.name === "RequestFailure" ? error.message : uiText(
              "The answer could not be completed. Your question is saved.",
              "לא ניתן להשלים את התשובה. השאלה שלך נשמרה."),
            diagnostic: error?.diagnostic || "Unexpected response format",
            status: error?.status || 0,
            resourceAdmissionFailed: error?.resourceAdmissionFailed === true,
            title: error?.title || uiText("Request failed", "לא ניתן להשלים את הבקשה"),
            hint: error?.hint || uiText("Review the question, then try again.", "בדקו את השאלה ולאחר מכן נסו שוב."),
            retryable: error?.retryable !== false,
            allowEdit: error?.allowEdit !== false,
            openSettings: error?.openSettings === true,
          };
      if (error instanceof TypeError) {
        modelStatusRevision += 1;
        setConnectionStatus("offline", "Offline");
        renderModelState({backend:request.backend, state:"error", unreachable:true});
      } else if (error?.resourceAdmissionFailed === true) {
        // A failed generation is a newer readiness event than any earlier status response.
        // Only a subsequent actual status check may report this selected model ready again.
        modelStatusRevision += 1;
        setConnectionStatus("degraded", "API connected");
        renderModelState({backend:request.backend, state:"error", error:"resource_admission"});
      }
      if (getThread(threadId)) {
        threadErrors.set(threadId, { ...failure, message: failure.message, replaceMessageId });
      }
      announce(failure.message);
    } finally {
      clearRequestTimer(request);
      if (pendingRequests.get(threadId) === request) pendingRequests.delete(threadId);
      if (
        request.workspaceKey === activeWorkspaceKey &&
        request.workspaceGeneration === workspaceGeneration
      ) {
        renderThreadList();
        if (workspace.activeThreadId === threadId) renderConversation();
      }
    }
  }

  function sendMessage() {
    const thread = getActiveThread();
    if (!thread || pendingRequests.has(thread.id)) return;
    const query = elements.chatInput.value.trim();
    if (!query || query.length > MAX_QUERY_CHARACTERS) return;

    if (activeDocumentSourceIds === null) {
      showToast("Wait for the authenticated document inventory before asking a question.");
      void refreshDocumentInventory();
      return;
    }
    if (selectedModelBlocksQuestions()) {
      showToast(uiText("Wait for the selected model to be ready. Your draft is saved.",
        "יש להמתין עד שהמודל שנבחר יהיה מוכן. הטיוטה שלך נשמרה."));
      return;
    }
    const history = recentHistory(thread.messages);
    const selectedSourceId = SOURCE_ID_PATTERN.test(
      String(thread.selectedSourceId || ""),
    )
      ? String(thread.selectedSourceId)
      : "";
    thread.messages.push({
      id: makeId("message"),
      role: "user",
      content: query,
      citations: [],
      selectedSourceId,
      createdAt: new Date().toISOString(),
    });
    if (thread.messages.length === 1 && thread.title === "Untitled thread") {
      thread.title = makeAutomaticTitle(query);
    }
    thread.updatedAt = new Date().toISOString();
    thread.draft = "";
    thread.selectedSourceId = "";
    threadErrors.delete(thread.id);
    cancelDraftPersistTimer();
    elements.chatInput.value = "";
    resizeInput();
    persistWorkspace();
    renderAll({ forceScroll: true });
    void requestAssistant(thread.id, query, history, { selectedSourceId });
  }

  function handleComposerAction() {
    const thread = getActiveThread();
    if (!thread) return;
    if (pendingRequests.has(thread.id)) {
      abortThreadRequest(thread.id, "user");
      return;
    }
    sendMessage();
  }

  function regenerateAnswer(messageId) {
    const thread = getActiveThread();
    if (!thread || pendingRequests.has(thread.id)) return;
    const answerIndex = thread.messages.findIndex(
      (message) => message.id === messageId && message.role === "assistant",
    );
    if (answerIndex !== thread.messages.length - 1 || answerIndex < 1) return;
    const userIndex = answerIndex - 1;
    const userMessage = thread.messages[userIndex];
    if (userMessage.role !== "user") return;
    const history = recentHistory(thread.messages.slice(0, userIndex));
    void requestAssistant(thread.id, userMessage.content, history, {
      replaceMessageId: messageId,
      selectedSourceId: userMessage.selectedSourceId || "",
    });
  }

  function editFailedQuestion(threadId) {
    const thread = getThread(threadId);
    const failure = threadErrors.get(threadId);
    if (
      !thread ||
      pendingRequests.has(threadId) ||
      failure?.replaceMessageId ||
      thread.messages[thread.messages.length - 1]?.role !== "user"
    ) return;

    if (elements.chatInput.value.trim()) {
      openThreadActionDialog({
        kind: "edit-failed-question",
        title: uiText("Edit the unanswered question?", "לערוך את השאלה שלא נענתה?"),
        description: uiText("The unanswered question will return to the composer and replace your current draft.", "השאלה שלא נענתה תחזור לאזור הכתיבה ותחליף את הטיוטה הנוכחית."),
        confirmLabel: uiText("Edit question", "עריכת השאלה"),
        onConfirm: () => restoreFailedQuestionForEditing(threadId),
      });
      return;
    }
    restoreFailedQuestionForEditing(threadId);
  }

  function restoreFailedQuestionForEditing(threadId) {
    const thread = getThread(threadId);
    const failure = threadErrors.get(threadId);
    if (
      !thread ||
      pendingRequests.has(threadId) ||
      failure?.replaceMessageId ||
      thread.messages[thread.messages.length - 1]?.role !== "user"
    ) return;
    const questionMessage = thread.messages.pop();
    const question = questionMessage.content;
    if (!thread.messages.length) thread.title = "Untitled thread";
    thread.draft = question;
    thread.selectedSourceId = questionMessage.selectedSourceId || "";
    thread.updatedAt = new Date().toISOString();
    threadErrors.delete(threadId);
    persistWorkspace();
    renderAll({ forceScroll: true });
    window.requestAnimationFrame(() => {
      elements.chatInput.focus();
      const end = elements.chatInput.value.length;
      if (typeof elements.chatInput.setSelectionRange === "function") {
        elements.chatInput.setSelectionRange(end, end);
      }
    });
    const restored = uiText("Question restored for editing", "השאלה הוחזרה לעריכה");
    announce(restored);
    showToast(restored);
  }

  function retryThread(threadId) {
    const thread = getThread(threadId);
    if (!thread || pendingRequests.has(threadId)) return;
    const failure = threadErrors.get(threadId);
    if (failure?.replaceMessageId) {
      regenerateAnswer(failure.replaceMessageId);
      return;
    }
    let userIndex = -1;
    for (let index = thread.messages.length - 1; index >= 0; index -= 1) {
      if (thread.messages[index].role === "user") {
        userIndex = index;
        break;
      }
    }
    if (userIndex < 0) return;
    const query = thread.messages[userIndex].content;
    const history = recentHistory(thread.messages.slice(0, userIndex));
    void requestAssistant(threadId, query, history, {
      selectedSourceId: thread.messages[userIndex].selectedSourceId || "",
    });
  }

  function resizeInput() {
    elements.chatInput.style.height = "auto";
    elements.chatInput.style.height = `${Math.min(elements.chatInput.scrollHeight, 192)}px`;
    elements.chatInput.style.overflowY =
      elements.chatInput.scrollHeight > 192 ? "auto" : "hidden";
    updateComposerState();
  }

  function selectedModelBlocksQuestions() {
    return modelSwitchPending || ["loading", "error", "admission-failed", "unreachable"]
      .includes(elements.modelState.dataset.state);
  }

  function updateComposerState() {
    const thread = getActiveThread();
    const pending = Boolean(thread && pendingRequests.has(thread.id));
    const query = elements.chatInput.value.trim();
    elements.sendButton.disabled =
      !thread || (!pending && (!query || query.length > MAX_QUERY_CHARACTERS ||
        selectedModelBlocksQuestions()));
    elements.sendButton.classList.toggle("is-stop", pending);
    elements.sendButton.dataset.mode = pending ? "stop" : "send";
    elements.sendButton.setAttribute(
      "aria-label",
      pending ? uiText("Stop waiting for the answer", "עצירת ההמתנה לתשובה") : uiText("Send message", "שליחת הודעה"),
    );
    elements.sendButton.title = pending ? uiText("Stop waiting for the answer", "עצירת ההמתנה לתשובה") : uiText("Send message", "שליחת הודעה");
    // A long-running answer should not block the user's next thought. Keep the
    // composer available as a per-thread draft while the stop control remains
    // on the send button.
    elements.chatInput.disabled = !thread;
    elements.chatInput.removeAttribute("placeholder");
    elements.composerHelp.classList.toggle("working", pending);
    elements.composerStatusCopy.textContent = pending
      ? uiText("Answer in progress · you can draft the next question", "התשובה נוצרת · אפשר לנסח את השאלה הבאה")
      : "";
    elements.composerShortcuts
      .querySelectorAll("button[data-composer-prompt]")
      .forEach((button) => {
        button.disabled = !thread;
      });
  }

  function applyTheme(theme) {
    const dark = theme === "dark";
    document.documentElement.classList.toggle("dark", dark);
    document.documentElement.style.colorScheme = dark ? "dark" : "light";
    elements.themeToggleButton.setAttribute("aria-checked", String(dark));
    elements.themeToggleButton.setAttribute(
      "aria-label",
      dark ? uiText("Use light theme", "שימוש בערכת נושא בהירה") : uiText("Use dark theme", "שימוש בערכת נושא כהה"),
    );
    elements.themeToggleKnob.style.transform = dark ? "translateX(20px)" : "translateX(0)";
    elements.themeToggleButton.style.backgroundColor = dark ? "#a94608" : "#d5dce7";
  }

  function showSettingsError(message) {
    elements.settingsError.textContent = message;
    elements.settingsError.classList.toggle("hidden", !message);
  }

  function setAppInert(value) {
    if (!elements.appShell) return;
    elements.appShell.inert = Boolean(value);
  }

  function focusableElements(container) {
    return [...container.querySelectorAll(
      'button:not([disabled]), input:not([disabled]), textarea:not([disabled]), select:not([disabled]), a[href], [tabindex]:not([tabindex="-1"])',
    )].filter((element) => !element.hidden && element.getClientRects().length > 0);
  }

  function trapFocus(container, event) {
    if (event.key !== "Tab") return;
    const focusable = focusableElements(container);
    if (!focusable.length) return;
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  }

  function setSettingsConnectionStatus(status, label, detail) {
    elements.settingsConnectionStatus.dataset.status = status;
    elements.settingsConnectionLabel.textContent = label;
    elements.settingsConnectionDetail.textContent = detail;
  }

  function abortSettingsConnectionTest() {
    if (!settingsTestController) return;
    settingsTestController.abort();
    settingsTestController = null;
  }

  function markSettingsConnectionUntested() {
    abortSettingsConnectionTest();
    elements.settingsTestConnectionButton.disabled = false;
    elements.settingsTestConnectionButton.textContent = "Test connection";
    setSettingsConnectionStatus(
      "idle",
      "Connection not tested",
      "Test these details before saving.",
    );
  }

  function readConnectionDraft() {
    [elements.apiBaseUrlInput, elements.tenantIdInput].forEach(
      (input) => input.setAttribute("aria-invalid", "false"),
    );
    showSettingsError("");
    const rawBaseUrl = elements.apiBaseUrlInput.value.trim().replace(/\/+$/, "");
    if (!rawBaseUrl) {
      showSettingsError("Enter the API base URL, including http:// or https://.");
      elements.apiBaseUrlInput.setAttribute("aria-invalid", "true");
      elements.apiBaseUrlInput.focus();
      return null;
    }
    let apiBaseUrl;
    try {
      const parsed = new URL(rawBaseUrl);
      if (parsed.protocol !== "http:" && parsed.protocol !== "https:") throw new Error();
      apiBaseUrl = normalizeApiBaseUrl(parsed.href);
    } catch (_error) {
      showSettingsError("Use a valid http:// or https:// API base URL.");
      elements.apiBaseUrlInput.setAttribute("aria-invalid", "true");
      elements.apiBaseUrlInput.focus();
      return null;
    }
    const tenantId = normalizeTenantId(elements.tenantIdInput.value);
    if (tenantId && !TENANT_ID_PATTERN.test(tenantId)) {
      showSettingsError(
        "Knowledge-base ID must be 1–64 ASCII letters, numbers, hyphens, or underscores, and must begin and end with a letter or number.",
      );
      elements.tenantIdInput.setAttribute("aria-invalid", "true");
      elements.tenantIdInput.focus();
      return null;
    }
    return { apiBaseUrl, tenantId };
  }

  async function testSettingsConnection() {
    abortSettingsConnectionTest();
    const draft = readConnectionDraft();
    if (!draft) return;
    const controller = new AbortController();
    settingsTestController = controller;
    elements.settingsTestConnectionButton.disabled = true;
    elements.settingsTestConnectionButton.textContent = "Testing…";
    setSettingsConnectionStatus(
      "checking",
      "Testing connection…",
      `Contacting ${draft.apiBaseUrl}`,
    );
    const timeoutId = window.setTimeout(() => controller.abort(), 6000);
    // Tenant selection remains available while the answer model is unloaded.
    // Retry discovery independently of the model readiness result.
    const catalogRefresh = refreshTenantChoices(draft.apiBaseUrl);
    try {
      const response = await fetch(`${draft.apiBaseUrl}/readyz`, {
        method: "GET",
        headers: draft.tenantId ? { "X-Tenant-ID": draft.tenantId } : {},
        cache: "no-store",
        signal: controller.signal,
      });
      let data;
      try {
        data = await readBoundedResponseJson(response);
      } catch (_error) {
        throw new Error("The server responded, but readiness data was invalid.");
      }
      const ready =
        response.ok &&
        data?.ready === true &&
        data?.model_loaded === true &&
        (!draft.tenantId || data?.retriever_ready === true);
      if (!ready) {
        const reason =
          typeof data?.reason === "string"
            ? data.reason
            : data?.model_loaded !== true
              ? "model_unavailable"
              : draft.tenantId && data?.retriever_ready !== true
                ? "retriever_unavailable"
                : "";
        throw new Error(readinessReasonMessage(reason, draft.tenantId));
      }
      window.clearTimeout(timeoutId);
      await catalogRefresh;
      if (settingsTestController !== controller) return;
      setSettingsConnectionStatus(
        "ready",
        "Connected and ready",
        draft.tenantId
          ? `Knowledge base “${draft.tenantId}” is available.`
          : "The default knowledge base is available.",
      );
      announce("Connection test succeeded");
    } catch (error) {
      if (settingsTestController !== controller) return;
      const detail = controller.signal.aborted
        ? "The connection test timed out after 6 seconds."
        : error instanceof TypeError
          ? "The server could not be reached. Check the URL, network, and CORS settings."
          : error?.message || "The selected knowledge base is not ready.";
      setSettingsConnectionStatus("failed", "Connection needs attention", detail);
      announce("Connection test failed");
    } finally {
      window.clearTimeout(timeoutId);
      await catalogRefresh;
      if (settingsTestController === controller) {
        settingsTestController = null;
        elements.settingsTestConnectionButton.disabled = false;
        elements.settingsTestConnectionButton.textContent = "Test again";
      }
    }
  }

  function openSettings() {
    closeSourcePanel({ restoreFocus: false });
    closeSidebar();
    lastFocusedElement = document.activeElement;
    themeBeforeDialog = settings.theme;
    settingsThemeDraft = settings.theme;
    elements.apiBaseUrlInput.value = settings.apiBaseUrl;
    if (settings.tenantId && !Array.from(elements.tenantIdInput.options).some((item) => item.value === settings.tenantId)) {
      const option = document.createElement("option");
      option.value = settings.tenantId;
      option.textContent = settings.tenantId;
      elements.tenantIdInput.append(option);
    }
    elements.tenantIdInput.value = settings.tenantId;
    void refreshTenantChoices();
    elements.requestTimeoutInput.value = String(settings.requestTimeoutSeconds);
    elements.processingMotionInput.value = settings.processingMotion;
    [elements.apiBaseUrlInput, elements.tenantIdInput, elements.requestTimeoutInput].forEach(
      (input) => input.setAttribute("aria-invalid", "false"),
    );
    showSettingsError("");
    markSettingsConnectionUntested();
    applyTheme(settingsThemeDraft);
    setAppInert(true);
    elements.settingsOverlay.classList.remove("hidden");
    elements.settingsOverlay.classList.add("flex");
    elements.settingsOverlay.setAttribute("aria-hidden", "false");
    window.setTimeout(() => elements.apiBaseUrlInput.focus(), 0);
  }

  function closeSettings({ restoreTheme = true } = {}) {
    if (elements.settingsOverlay.classList.contains("hidden")) return;
    abortSettingsConnectionTest();
    if (restoreTheme) applyTheme(themeBeforeDialog);
    elements.settingsOverlay.classList.add("hidden");
    elements.settingsOverlay.classList.remove("flex");
    elements.settingsOverlay.setAttribute("aria-hidden", "true");
    setAppInert(false);
    showSettingsError("");
    if (lastFocusedElement && typeof lastFocusedElement.focus === "function") {
      lastFocusedElement.focus();
    }
  }

  function saveSettings() {
    [elements.apiBaseUrlInput, elements.tenantIdInput, elements.requestTimeoutInput].forEach(
      (input) => input.setAttribute("aria-invalid", "false"),
    );
    showSettingsError("");
    const connectionDraft = readConnectionDraft();
    if (!connectionDraft) return;
    const { apiBaseUrl, tenantId } = connectionDraft;
    const requestTimeoutSeconds = Number(elements.requestTimeoutInput.value);
    if (
      !Number.isInteger(requestTimeoutSeconds) ||
      requestTimeoutSeconds < MIN_REQUEST_TIMEOUT_SECONDS ||
      requestTimeoutSeconds > MAX_REQUEST_TIMEOUT_SECONDS
    ) {
      showSettingsError(
        "Answer timeout must remain 1,200 seconds (20 minutes).",
      );
      elements.requestTimeoutInput.setAttribute("aria-invalid", "true");
      elements.requestTimeoutInput.focus();
      elements.requestTimeoutInput.select();
      return;
    }

    const nextSettings = {
      apiBaseUrl,
      tenantId,
      theme: settingsThemeDraft,
      processingMotion: ["system", "animated", "static"].includes(elements.processingMotionInput.value)
        ? elements.processingMotionInput.value : DEFAULT_SETTINGS.processingMotion,
      processingMotionRevision: 1,
      requestTimeoutSeconds,
    };
    const nextWorkspaceKey = scopedWorkspaceKey(nextSettings);
    const connectionChanged = nextWorkspaceKey !== activeWorkspaceKey;
    if (connectionChanged && !flushActiveDraft()) {
      showSettingsError(
        "The current conversation could not be saved, so the knowledge base was not changed.",
      );
      return;
    }
    if (!persistSettings(nextSettings)) {
      showSettingsError("Settings could not be saved in this browser.");
      return;
    }
    if (connectionChanged) {
      abortRequestsForConnectionChange();
      workspaceGeneration += 1;
      activeWorkspaceKey = nextWorkspaceKey;
      workspace = loadWorkspace(nextSettings);
      threadSearchQuery = "";
      elements.threadSearch.value = "";
      elements.threadSearchClear.classList.add("hidden");
      threadErrors.clear();
    }
    settings = nextSettings;
    if (connectionChanged) persistWorkspace();
    applyTheme(settings.theme);
    applyProcessingMotion(settings.processingMotion);
    closeSettings({ restoreTheme: false });
    renderAll(connectionChanged ? { forceScroll: true } : { scroll: false });
    if (connectionChanged) {
      setConnectionStatus("unknown", "Not checked", "Settings changed; check the backend connection.");
    }
    showToast("Settings saved");
    announce("Settings saved");
    if (connectionChanged) void checkBackendHealth();
  }

  function openHelp() {
    closeSourcePanel({ restoreFocus: false });
    closeSidebar();
    lastFocusedElement = document.activeElement;
    setAppInert(true);
    elements.helpOverlay.classList.remove("hidden");
    elements.helpOverlay.classList.add("flex");
    elements.helpOverlay.setAttribute("aria-hidden", "false");
    window.setTimeout(() => elements.helpCloseButton.focus(), 0);
  }

  function closeHelp() {
    if (elements.helpOverlay.classList.contains("hidden")) return;
    elements.helpOverlay.classList.add("hidden");
    elements.helpOverlay.classList.remove("flex");
    elements.helpOverlay.setAttribute("aria-hidden", "true");
    setAppInert(false);
    if (lastFocusedElement && typeof lastFocusedElement.focus === "function") {
      lastFocusedElement.focus();
    }
  }

  function updateSidebarAccessibility() {
    const mobile = window.innerWidth <= 900;
    const open = elements.sidebar.classList.contains("mobile-open");
    const collapsed = !mobile && elements.appShell.classList.contains("sidebar-collapsed");
    elements.sidebar.inert = (mobile && !open) || collapsed;
    elements.sidebar.setAttribute("aria-hidden", String((mobile && !open) || collapsed));
    elements.mobileMenuButton.setAttribute("aria-expanded", String(mobile ? open : !collapsed));
    elements.mobileMenuButton.setAttribute("aria-label", uiText(
      mobile && open ? "Close navigation" : "Open navigation",
      mobile && open ? "סגירת הניווט" : "פתיחת הניווט"));
  }

  function openSidebar() {
    closeSourcePanel({ restoreFocus: false });
    if (window.innerWidth > 900) {
      elements.appShell.classList.remove("sidebar-collapsed");
      updateSidebarAccessibility();
      elements.main.inert = false;
      return;
    }
    sidebarReturnFocus = document.activeElement;
    elements.sidebar.classList.add("mobile-open");
    elements.sidebarBackdrop.classList.add("mobile-open");
    updateSidebarAccessibility();
    elements.main.inert = true;
    window.setTimeout(() => {
      if (elements.sidebar.classList.contains("mobile-open")) elements.newThreadButton.focus();
    }, 0);
  }

  function focusConversationSearch() {
    const needsSidebar = window.innerWidth <= 900 && !elements.sidebar.classList.contains("mobile-open");
    if (needsSidebar || elements.appShell.classList.contains("sidebar-collapsed")) openSidebar();
    window.setTimeout(() => {
      elements.threadSearch.focus();
      if (typeof elements.threadSearch.select === "function") elements.threadSearch.select();
    }, 0);
  }

  function closeSidebar() {
    const wasOpen = elements.sidebar.classList.contains("mobile-open");
    elements.sidebar.classList.remove("mobile-open");
    elements.sidebarBackdrop.classList.remove("mobile-open");
    updateSidebarAccessibility();
    elements.main.inert = elements.appShell.classList.contains("sources-open");
    if (wasOpen && sidebarReturnFocus && typeof sidebarReturnFocus.focus === "function") sidebarReturnFocus.focus();
    sidebarReturnFocus = null;
  }

  elements.newThreadButton.addEventListener("click", createAndSelectThread);
  elements.threadSearch.addEventListener("input", () => {
    threadSearchQuery = normalizedSearchText(elements.threadSearch.value);
    elements.threadSearchClear.classList.toggle("hidden", !threadSearchQuery);
    renderThreadList();
  });
  elements.threadSearchClear.addEventListener("click", () => {
    elements.threadSearch.value = "";
    threadSearchQuery = "";
    elements.threadSearchClear.classList.add("hidden");
    renderThreadList();
    elements.threadSearch.focus();
  });
  elements.starterPrompts.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-prompt]");
    if (!button) return;
    prefillComposer(button.dataset.prompt || "");
  });
  elements.composerShortcuts.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-composer-prompt]");
    if (!button || button.disabled) return;
    prefillComposer(button.dataset.composerPrompt || "");
  });
  elements.modelRetry.addEventListener("click", () => void switchAnswerModel());
  elements.sendButton.addEventListener("click", handleComposerAction);
  elements.chatHistory.addEventListener("scroll", updateScrollToLatestButton, {
    passive: true,
  });
  elements.scrollToLatestButton.addEventListener("click", scrollConversationToLatest);
  elements.chatInput.addEventListener("input", () => {
    syncActiveDraftFromComposer();
    resizeInput();
  });
  elements.chatInput.addEventListener("keydown", (event) => {
    const thread = getActiveThread();
    const pending = Boolean(thread && pendingRequests.has(thread.id));
    if (
      event.key === "Enter" &&
      !event.shiftKey &&
      !event.isComposing &&
      !pending
    ) {
      event.preventDefault();
      handleComposerAction();
    }
  });
  elements.mobileMenuButton.addEventListener("click", () => {
    if (elements.sidebar.classList.contains("mobile-open")) closeSidebar();
    else openSidebar();
  });
  elements.sidebarBackdrop.addEventListener("click", closeSidebar);
  document.getElementById("sidebar-collapse-btn")?.addEventListener("click", () => {
    if (window.innerWidth <= 900) { closeSidebar(); return; }
    elements.appShell.classList.add("sidebar-collapsed");
    updateSidebarAccessibility();
    elements.mobileMenuButton.focus();
  });
  document.getElementById("sidebar-close-btn")?.addEventListener("click", closeSidebar);
  elements.connectionStatusButton.addEventListener("click", () => {
    void checkBackendHealth({ announceResult: true });
  });
  elements.exportThreadButton.addEventListener("click", exportActiveThread);
  elements.sourcesToggleButton.addEventListener("click", () => {
    if (elements.appShell.classList.contains("sources-open")) closeSourcePanel();
    else openSourcePanel();
  });
  elements.sourcePanelCloseButton.addEventListener("click", () => closeSourcePanel());
  elements.sourcePanelBackdrop.addEventListener("click", () => closeSourcePanel());
  elements.sourceSearch.addEventListener("input", () => {
    sourceSearchQuery = normalizedSearchText(elements.sourceSearch.value);
    elements.sourceSearchClear.classList.toggle("hidden", !sourceSearchQuery);
    renderSourcePanel();
  });
  elements.sourceSearchClear.addEventListener("click", () => {
    resetSourceFilter();
    renderSourcePanel();
    elements.sourceSearch.focus();
  });
  elements.settingsOpenButton.addEventListener("click", openSettings);
  elements.settingsCloseButton.addEventListener("click", () => closeSettings());
  elements.settingsCancelButton.addEventListener("click", () => closeSettings());
  elements.settingsForm.addEventListener("submit", (event) => {
    event.preventDefault();
    saveSettings();
  });
  elements.settingsTestConnectionButton.addEventListener("click", () => {
    void testSettingsConnection();
  });
  [elements.apiBaseUrlInput, elements.tenantIdInput].forEach((input) => {
    input.addEventListener("input", markSettingsConnectionUntested);
  });
  elements.apiBaseUrlInput.addEventListener("change", () => {
    void refreshTenantChoices(elements.apiBaseUrlInput.value);
  });
  elements.tenantIdInput.addEventListener("change", markSettingsConnectionUntested);

  elements.themeToggleButton.addEventListener("click", () => {
    settingsThemeDraft = settingsThemeDraft === "dark" ? "light" : "dark";
    applyTheme(settingsThemeDraft);
  });
  elements.settingsOverlay.addEventListener("click", (event) => {
    if (event.target === elements.settingsOverlay) closeSettings();
  });
  elements.threadActionDialog.addEventListener("submit", (event) => {
    event.preventDefault();
    submitThreadActionDialog();
  });
  elements.threadActionCloseButton.addEventListener("click", () => closeThreadActionDialog());
  elements.threadActionCancelButton.addEventListener("click", () => closeThreadActionDialog());
  elements.threadActionOverlay.addEventListener("click", (event) => {
    if (event.target === elements.threadActionOverlay) closeThreadActionDialog();
  });
  elements.helpOpenButton.addEventListener("click", openHelp);
  elements.helpCloseButton.addEventListener("click", closeHelp);
  elements.helpDoneButton.addEventListener("click", closeHelp);
  elements.helpOverlay.addEventListener("click", (event) => {
    if (event.target === elements.helpOverlay) closeHelp();
  });
  document.addEventListener("keydown", (event) => {
    const settingsOpen = !elements.settingsOverlay.classList.contains("hidden");
    const threadActionOpen = !elements.threadActionOverlay.classList.contains("hidden");
    const helpOpen = !elements.helpOverlay.classList.contains("hidden");
    const sourcesOpen = elements.appShell.classList.contains("sources-open");

    if (event.key === "Escape") {
      if (settingsOpen) closeSettings();
      else if (threadActionOpen) closeThreadActionDialog();
      else if (helpOpen) closeHelp();
      else if (sourcesOpen) closeSourcePanel();
      else closeSidebar();
      return;
    }

    if (settingsOpen) {
      trapFocus(elements.settingsForm, event);
      return;
    }
    if (threadActionOpen) {
      trapFocus(elements.threadActionDialog, event);
      return;
    }
    if (helpOpen) {
      trapFocus(elements.helpDialog, event);
      return;
    }
    if (sourcesOpen) {
      trapFocus(elements.sourcePanel, event);
      return;
    }
    if (elements.sidebar.classList.contains("mobile-open")) {
      trapFocus(elements.sidebar, event);
      return;
    }

    if ((event.ctrlKey || event.metaKey) && event.key.toLocaleLowerCase() === "k") {
      event.preventDefault();
      focusConversationSearch();
      return;
    }

    const target = event.target;
    const isTyping = target instanceof Element && Boolean(target.closest("input, textarea, [contenteditable='true']"));
    if (event.key === "/" && !event.ctrlKey && !event.metaKey && !event.altKey && !isTyping) {
      event.preventDefault();
      elements.chatInput.focus();
    }
  });
  window.addEventListener("resize", () => {
    if (window.innerWidth > 900) closeSidebar();
    updateSidebarAccessibility();
    updateScrollToLatestButton();
  });
  window.addEventListener("offline", () => {
    setConnectionStatus(
      "offline",
      "Offline",
      "This browser is offline. Pending answers may resume only if the connection returns.",
    );
    announce("Browser connection lost");
  });
  window.addEventListener("online", () => {
    setConnectionStatus("checking", "Reconnecting…", "Browser connection restored; checking the backend.");
    announce("Browser connection restored");
    void checkBackendHealth();
  });
  window.addEventListener("pagehide", flushActiveDraft);

  window.addEventListener("projectx-language-change", () => {
    flushActiveDraft();
    renderAll({ scroll: false });
    if (lastInventoryData?.identity === connectionIdentity(settings)) {
      renderDocumentInventory(lastInventoryData.data, lastInventoryData.effectiveTenant);
    }
    if (lastModelData) renderModelState(lastModelData);
    if (lastConnectionState) setConnectionStatus(lastConnectionState.status, lastConnectionState.label, lastConnectionState.detail);
    updateSidebarAccessibility();
  });

  function applyProcessingMotion(value) {
    document.documentElement.classList.toggle("processing-motion-animated", value === "animated");
    document.documentElement.classList.toggle("processing-motion-static", value === "static");
  }

  updateSidebarAccessibility();
  applyTheme(settings.theme);
  applyProcessingMotion(settings.processingMotion);
  persistWorkspace();
  renderAll({ scroll: false });
  resizeInput();
  window.setInterval(() => void refreshSelectedModelState(), 5000);
  window.setTimeout(() => void checkBackendHealth(), 450);
})();
