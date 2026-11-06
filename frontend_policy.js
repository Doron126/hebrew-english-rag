(function installProjectXFrontendPolicy(root, factory) {
  "use strict";
  const policy = factory();
  if (typeof module === "object" && module && module.exports) {
    module.exports = policy;
  }
  root.ProjectXFrontendPolicy = policy;
})(typeof globalThis === "object" ? globalThis : this, function createPolicy() {
  "use strict";

  const TENANT_ID_PATTERN = /^[A-Za-z0-9](?:[A-Za-z0-9_-]{0,62}[A-Za-z0-9])?$/;
  const SOURCE_ID_PATTERN = /^[A-Za-z0-9_-]{43}$/;

  function normalizeTenantId(value) {
    return typeof value === "string" ? value.trim() : "";
  }

  function normalizeApiBaseUrl(value) {
    const raw = String(value || "").trim();
    try {
      const parsed = new URL(raw);
      if (parsed.protocol !== "http:" && parsed.protocol !== "https:") return "";
      parsed.hash = "";
      parsed.search = "";
      parsed.pathname = parsed.pathname.replace(/\/+$/, "") || "/";
      return parsed.href.replace(/\/+$/, "");
    } catch (_error) {
      return "";
    }
  }

  function normalizeSourceOpenUrl(value, sourceId, effectiveTenant, apiBaseUrl) {
    const raw = typeof value === "string" ? value.trim() : "";
    const tenant = normalizeTenantId(effectiveTenant);
    const baseValue = normalizeApiBaseUrl(apiBaseUrl);
    if (
      !raw ||
      !SOURCE_ID_PATTERN.test(sourceId || "") ||
      !TENANT_ID_PATTERN.test(tenant) ||
      !baseValue
    ) return "";
    try {
      const base = new URL(baseValue);
      const candidate = new URL(raw, base);
      const tenantValues = candidate.searchParams.getAll("tenant");
      if (
        candidate.origin !== base.origin ||
        candidate.pathname !== `/api/sources/${sourceId}` ||
        tenantValues.length !== 1 ||
        tenantValues[0] !== tenant ||
        [...candidate.searchParams.keys()].some((key) => key !== "tenant") ||
        (candidate.hash && !/^#page=[1-9]\d{0,6}$/.test(candidate.hash))
      ) return "";
      return candidate.href;
    } catch (_error) {
      return "";
    }
  }

  function usesHebrew(text) {
    const value = typeof text === "string" ? text : "";
    const hebrewCharacters = value.match(/[\u0590-\u05ff]/g)?.length || 0;
    const latinCharacters = value.match(/[A-Za-z]/g)?.length || 0;
    return hebrewCharacters > latinCharacters;
  }

  function normalizeTextDirection(value) {
    const direction = String(value || "").toLowerCase();
    return ["rtl", "ltr", "auto"].includes(direction) ? direction : "auto";
  }

  function numericCitationId(citation) {
    const value = citation && citation.id != null ? String(citation.id) : "";
    return /^[1-9]\d{0,2}$/.test(value) ? value : "";
  }

  function safeCitationDisplayName(value) {
    const normalized = String(value || "")
      .replace(/\\/g, "/")
      .split("/")
      .pop()
      .replace(/[\u0000-\u001f\u007f\u200e\u200f\u202a-\u202e\u2066-\u2069]/g, "")
      .trim();
    return Array.from(normalized || "Unknown source").slice(0, 240).join("");
  }

  function safeCitationRelativePath(value) {
    const relative = String(value || "");
    if (!relative || relative.length > 4096 || relative.startsWith("/") ||
        relative.includes("\\") || /^[A-Za-z]:/.test(relative) ||
        /[\u0000-\u001f\u007f\u200e\u200f\u202a-\u202e\u2066-\u2069]/.test(relative)) return "";
    if (relative.split("/").some((part) => !part || part === "." || part === "..")) return "";
    return relative;
  }

  function isCitationBoundaryWord(character) {
    return Boolean(character && /[\p{L}\p{N}\p{M}_]/u.test(character));
  }

  function citationMarkers(value, citations) {
    const available = new Set(
      (Array.isArray(citations) ? citations : [])
        .map(numericCitationId)
        .filter(Boolean),
    );
    const linked = [];
    const pattern = /\[(\d{1,3})\]/g;
    const text = String(value || "");
    let match;
    while ((match = pattern.exec(text)) !== null) {
      const before = match.index > 0 ? text[match.index - 1] : "";
      const after = pattern.lastIndex < text.length ? text[pattern.lastIndex] : "";
      if (
        available.has(match[1]) &&
        before !== "[" &&
        after !== "]" &&
        !isCitationBoundaryWord(before) &&
        !isCitationBoundaryWord(after)
      ) linked.push(match[1]);
    }
    return linked;
  }

  function splitAnswerTableRow(value) {
    let row = String(value || "").trim();
    if (row.startsWith("|")) row = row.slice(1);
    if (row.endsWith("|")) row = row.slice(0, -1);
    return row.split("|").map((cell) => cell.trim());
  }

  function isAnswerTableSeparator(value) {
    const cells = splitAnswerTableRow(value);
    return cells.length >= 2 && cells.every((cell) => /^:?-{3,}:?$/.test(cell));
  }

  function answerListMatch(value) {
    const unordered = String(value || "").match(/^\s*[-*•]\s+(.+)$/);
    if (unordered) return { ordered: false, content: unordered[1] };
    const ordered = String(value || "").match(/^\s*\d+[.)]\s+(.+)$/);
    return ordered ? { ordered: true, content: ordered[1] } : null;
  }

  return Object.freeze({
    answerListMatch,
    citationMarkers,
    isAnswerTableSeparator,
    isCitationBoundaryWord,
    normalizeSourceOpenUrl,
    normalizeTenantId,
    normalizeTextDirection,
    numericCitationId,
    safeCitationDisplayName,
    safeCitationRelativePath,
    splitAnswerTableRow,
    usesHebrew,
  });
});
