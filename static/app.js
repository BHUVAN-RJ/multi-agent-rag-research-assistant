// Chat UI for the multi-agent RAG backend. Vanilla JS, no framework, no build step.
//
// POST /api/chat answers with text/event-stream. EventSource only supports GET, so the
// stream is read with fetch() + ReadableStream and parsed as SSE frames here. Frame
// names: token, trace, citation, error, done.

"use strict";

const SNIPPET_MAX_CHARS = 260;
const AUTO_SCROLL_SLACK_PX = 160;
const CITATION_FLASH_MS = 1400;

function lookupPageElements() {
  return {
    turns: document.getElementById("turns"),
    emptyState: document.getElementById("empty-state"),
    form: document.getElementById("chat-form"),
    input: document.getElementById("query-input"),
    submitButton: document.getElementById("submit-button"),
    resetButton: document.getElementById("reset-button"),
    sessionLabel: document.getElementById("session-id"),
    traceLog: document.getElementById("trace-log"),
    traceCount: document.getElementById("trace-count"),
    clearTraceButton: document.getElementById("clear-trace-button"),
  };
}

// The test suite loads this file under node, where there is no DOM; only the pure
// helpers (markdown, SSE parser, routing summary) are used there.
const hasDocument = typeof document !== "undefined";
const elements = hasDocument ? lookupPageElements() : {};

const state = {
  sessionId: "",
  isStreaming: false,
  abortController: null,
  turnCounter: 0,
  traceEntries: 0,
};

/* ------------------------------------------------------------------ session */

function createFallbackUuid() {
  // crypto.randomUUID only exists in secure contexts (https or localhost). Build an
  // RFC 4122 v4 id from getRandomValues when the page is served over plain http.
  const bytes = new Uint8Array(16);
  crypto.getRandomValues(bytes);
  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;
  const hexParts = [];
  for (const byte of bytes) {
    const hexByte = byte.toString(16);
    hexParts.push(hexByte.padStart(2, "0"));
  }
  const hex = hexParts.join("");
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

function createSessionId() {
  if (typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return createFallbackUuid();
}

function startNewSession() {
  state.sessionId = createSessionId();
  elements.sessionLabel.textContent = state.sessionId.slice(0, 8);
  elements.sessionLabel.title = state.sessionId;
}

/* --------------------------------------------------------------- SSE parser */

// Incremental SSE parser. Network chunks can split a frame (or even a line) anywhere,
// so text is buffered until a blank line ends the frame. Handles \n, \r\n and \r line
// endings, comment lines (": ping"), and multi-line data fields.
function createSseParser(handleFrame) {
  let buffer = "";
  let carriedCarriageReturn = "";

  function normalizeLineEndings(text) {
    const combined = carriedCarriageReturn + text;
    // A trailing \r may be the first half of a \r\n split across chunks; hold it back.
    const endsWithCarriageReturn = combined.endsWith("\r");
    const body = endsWithCarriageReturn ? combined.slice(0, -1) : combined;
    carriedCarriageReturn = endsWithCarriageReturn ? "\r" : "";
    return body.replace(/\r\n?/g, "\n");
  }

  function parseFrameBlock(block) {
    let eventName = "message";
    const dataLines = [];
    const lines = block.split("\n");
    for (const line of lines) {
      if (line === "" || line.startsWith(":")) {
        continue;
      }
      const colonIndex = line.indexOf(":");
      const fieldName = colonIndex === -1 ? line : line.slice(0, colonIndex);
      let fieldValue = colonIndex === -1 ? "" : line.slice(colonIndex + 1);
      if (fieldValue.startsWith(" ")) {
        fieldValue = fieldValue.slice(1);
      }
      if (fieldName === "event") {
        eventName = fieldValue;
      } else if (fieldName === "data") {
        dataLines.push(fieldValue);
      }
    }
    // A frame made only of comments (heartbeat) carries no data and is not dispatched.
    if (dataLines.length === 0) {
      return;
    }
    handleFrame({ event: eventName, data: dataLines.join("\n") });
  }

  function drainCompleteFrames() {
    let boundaryIndex = buffer.indexOf("\n\n");
    while (boundaryIndex !== -1) {
      const block = buffer.slice(0, boundaryIndex);
      buffer = buffer.slice(boundaryIndex + 2);
      parseFrameBlock(block);
      boundaryIndex = buffer.indexOf("\n\n");
    }
  }

  function push(text) {
    buffer += normalizeLineEndings(text);
    drainCompleteFrames();
  }

  function flush() {
    buffer += normalizeLineEndings("");
    if (carriedCarriageReturn) {
      buffer += "\n";
      carriedCarriageReturn = "";
    }
    drainCompleteFrames();
    // The spec drops an unterminated final frame; be lenient and dispatch it anyway.
    const trimmedRemainder = buffer.trim();
    if (trimmedRemainder) {
      parseFrameBlock(buffer);
    }
    buffer = "";
  }

  return { push, flush };
}

/* ---------------------------------------------------------- safe markdown */

function escapeHtml(text) {
  let escaped = String(text);
  escaped = escaped.replace(/&/g, "&amp;");
  escaped = escaped.replace(/</g, "&lt;");
  escaped = escaped.replace(/>/g, "&gt;");
  escaped = escaped.replace(/"/g, "&quot;");
  escaped = escaped.replace(/'/g, "&#39;");
  return escaped;
}

function citationAnchorId(turnIndex, citationNumber) {
  return `cite-${turnIndex}-${citationNumber}`;
}

function buildCitationLinks(turnIndex, numberList) {
  const numbers = numberList.split(",");
  const links = [];
  for (const rawNumber of numbers) {
    const citationNumber = rawNumber.trim();
    const anchorId = citationAnchorId(turnIndex, citationNumber);
    links.push(`<a class="cite-ref" href="#${anchorId}" data-target="${anchorId}">[${citationNumber}]</a>`);
  }
  return links.join("");
}

// Inline markdown on text that is ALREADY HTML-escaped, so every tag below is ours.
// Code spans and links are swapped for placeholders first so later rules (bold,
// citations) can't rewrite their contents.
function renderInline(escapedText, turnIndex) {
  const stash = [];

  function stashHtml(html) {
    stash.push(html);
    return `\u0000${stash.length - 1}\u0000`;
  }

  function replaceCodeSpan(match, codeText) {
    return stashHtml(`<code>${codeText}</code>`);
  }

  function replaceLink(match, linkText, linkUrl) {
    return stashHtml(`<a href="${linkUrl}" target="_blank" rel="noopener noreferrer">${linkText}</a>`);
  }

  function replaceCitation(match, numberList) {
    return stashHtml(buildCitationLinks(turnIndex, numberList));
  }

  function restorePlaceholder(match, stashIndex) {
    return stash[Number(stashIndex)];
  }

  // A stashed entry can itself contain placeholders (a link whose text holds a code
  // span), and String.replace does not rescan what it inserted. Restore repeatedly
  // until none remain; each pass unwraps one nesting level, so stash.length passes
  // is an upper bound.
  function restoreAllPlaceholders(htmlWithPlaceholders) {
    const placeholderPattern = /\u0000(\d+)\u0000/g;
    let restored = htmlWithPlaceholders;
    let remainingPasses = stash.length + 1;
    while (restored.includes("\u0000") && remainingPasses > 0) {
      restored = restored.replace(placeholderPattern, restorePlaceholder);
      remainingPasses -= 1;
    }
    return restored;
  }

  let html = escapedText;
  html = html.replace(/`([^`\n]+)`/g, replaceCodeSpan);
  // Only http(s) targets become links; javascript: and friends stay plain text. The URL
  // may not contain a placeholder, so a code span can never end up inside href="".
  html = html.replace(/\[([^\]\n]+)\]\((https?:\/\/[^\s)\u0000]+)\)/g, replaceLink);
  // [1], [12], [1, 2] -> links to this turn's citation entries.
  html = html.replace(/\[(\d{1,3}(?:\s*,\s*\d{1,3})*)\]/g, replaceCitation);
  html = html.replace(/\*\*([^*\n]+?)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/__([^_\n]+?)__/g, "<strong>$1</strong>");
  html = html.replace(/(^|[\s(])\*([^*\s][^*\n]*?)\*(?=[\s).,;:!?]|$)/g, "$1<em>$2</em>");
  return restoreAllPlaceholders(html);
}

/* ------------------------------------------------------------- GFM tables */

// Split one escaped table row into trimmed cell strings. Leading/trailing pipes are
// optional in GFM, and "\|" is a literal pipe inside a cell.
function splitTableRow(line) {
  let trimmed = line.trim();
  if (trimmed.startsWith("|")) {
    trimmed = trimmed.slice(1);
  }
  if (trimmed.endsWith("|") && !trimmed.endsWith("\\|")) {
    trimmed = trimmed.slice(0, -1);
  }
  const rawCells = trimmed.split(/(?<!\\)\|/);
  const cells = [];
  for (const rawCell of rawCells) {
    const unescapedCell = rawCell.replace(/\\\|/g, "|");
    cells.push(unescapedCell.trim());
  }
  return cells;
}

// Delimiter row: cells made only of dashes with optional colons, e.g. |:---|---:|.
// Returns the per-column alignment list, or null when the line is not a delimiter row.
function parseTableDelimiter(line) {
  if (!line.includes("|") && !line.includes("-")) {
    return null;
  }
  const cells = splitTableRow(line);
  const alignments = [];
  for (const cell of cells) {
    if (!/^:?-+:?$/.test(cell)) {
      return null;
    }
    const alignLeft = cell.startsWith(":");
    const alignRight = cell.endsWith(":");
    if (alignLeft && alignRight) {
      alignments.push("center");
    } else if (alignRight) {
      alignments.push("right");
    } else if (alignLeft) {
      alignments.push("left");
    } else {
      alignments.push("");
    }
  }
  return alignments;
}

// A table starts at a row containing a pipe followed by a delimiter row with the same
// number of columns. A lone "---" is a horizontal rule, so the delimiter must have a pipe.
function tableAlignmentsAt(lines, lineIndex) {
  const headerLine = lines[lineIndex];
  const delimiterLine = lines[lineIndex + 1];
  if (delimiterLine === undefined || !headerLine.includes("|") || !delimiterLine.includes("|")) {
    return null;
  }
  const alignments = parseTableDelimiter(delimiterLine);
  if (alignments === null) {
    return null;
  }
  const headerCells = splitTableRow(headerLine);
  return headerCells.length === alignments.length ? alignments : null;
}

function renderTableRow(cells, alignments, cellTag, turnIndex) {
  const cellHtml = [];
  for (let columnIndex = 0; columnIndex < alignments.length; columnIndex += 1) {
    const alignment = alignments[columnIndex];
    // Alignment comes from the fixed set above, never from answer text.
    const alignAttribute = alignment ? ` class="align-${alignment}"` : "";
    const cellText = columnIndex < cells.length ? cells[columnIndex] : "";
    cellHtml.push(`<${cellTag}${alignAttribute}>${renderInline(cellText, turnIndex)}</${cellTag}>`);
  }
  return `<tr>${cellHtml.join("")}</tr>`;
}

function renderTable(headerLine, alignments, bodyLines, turnIndex) {
  const headerCells = splitTableRow(headerLine);
  const headerRow = renderTableRow(headerCells, alignments, "th", turnIndex);
  const bodyRows = [];
  for (const bodyLine of bodyLines) {
    const bodyCells = splitTableRow(bodyLine);
    bodyRows.push(renderTableRow(bodyCells, alignments, "td", turnIndex));
  }
  const bodyHtml = bodyRows.length > 0 ? `<tbody>${bodyRows.join("")}</tbody>` : "";
  return `<div class="table-wrap"><table><thead>${headerRow}</thead>${bodyHtml}</table></div>`;
}

// Block-level markdown subset: headings, bullet and numbered lists, fenced code,
// GFM tables, horizontal rules, paragraphs. Unclosed fences (mid-stream) render as code so far.
function renderMarkdown(rawText, turnIndex) {
  const cleanText = String(rawText).replace(/\u0000/g, "");
  const escapedText = escapeHtml(cleanText);
  const lines = escapedText.split("\n");
  const output = [];
  let paragraphLines = [];
  let openListTag = "";
  let inCodeFence = false;
  let codeLines = [];

  function closeParagraph() {
    if (paragraphLines.length > 0) {
      const joined = paragraphLines.join(" ");
      output.push(`<p>${renderInline(joined, turnIndex)}</p>`);
      paragraphLines = [];
    }
  }

  function closeList() {
    if (openListTag) {
      output.push(`</${openListTag}>`);
      openListTag = "";
    }
  }

  function openList(listTag) {
    if (openListTag !== listTag) {
      closeList();
      output.push(`<${listTag}>`);
      openListTag = listTag;
    }
  }

  function closeCodeFence() {
    output.push(`<pre><code>${codeLines.join("\n")}</code></pre>`);
    codeLines = [];
    inCodeFence = false;
  }

  let lineIndex = 0;
  while (lineIndex < lines.length) {
    const line = lines[lineIndex];
    lineIndex += 1;
    if (/^\s*```/.test(line)) {
      if (inCodeFence) {
        closeCodeFence();
      } else {
        closeParagraph();
        closeList();
        inCodeFence = true;
      }
      continue;
    }
    if (inCodeFence) {
      codeLines.push(line);
      continue;
    }

    // lineIndex already points past this line, so the table check looks at line - 1.
    const tableAlignments = tableAlignmentsAt(lines, lineIndex - 1);
    if (tableAlignments !== null) {
      closeParagraph();
      closeList();
      const bodyLines = [];
      // Skip the delimiter row, then take every following non-blank row with a pipe.
      lineIndex += 1;
      while (lineIndex < lines.length && lines[lineIndex].includes("|") && lines[lineIndex].trim() !== "") {
        bodyLines.push(lines[lineIndex]);
        lineIndex += 1;
      }
      output.push(renderTable(line, tableAlignments, bodyLines, turnIndex));
      continue;
    }

    const headingMatch = line.match(/^\s*(#{1,6})\s+(.*)$/);
    const bulletMatch = line.match(/^\s*[-*+]\s+(.*)$/);
    const numberedMatch = line.match(/^\s*\d+[.)]\s+(.*)$/);

    if (line.trim() === "") {
      closeParagraph();
      closeList();
    } else if (/^\s*([-*_])\s*\1\s*\1[\s\-*_]*$/.test(line)) {
      closeParagraph();
      closeList();
      output.push("<hr>");
    } else if (headingMatch) {
      closeParagraph();
      closeList();
      // Answer headings sit inside a card, so "#" maps to h3 rather than h1.
      const level = Math.min(headingMatch[1].length + 2, 6);
      output.push(`<h${level}>${renderInline(headingMatch[2], turnIndex)}</h${level}>`);
    } else if (bulletMatch) {
      closeParagraph();
      openList("ul");
      output.push(`<li>${renderInline(bulletMatch[1], turnIndex)}</li>`);
    } else if (numberedMatch) {
      closeParagraph();
      openList("ol");
      output.push(`<li>${renderInline(numberedMatch[1], turnIndex)}</li>`);
    } else {
      closeList();
      paragraphLines.push(line.trim());
    }
  }

  if (inCodeFence) {
    closeCodeFence();
  }
  closeParagraph();
  closeList();
  return output.join("");
}

/* ------------------------------------------------------------ DOM helpers */

function createElement(tagName, className, text) {
  const element = document.createElement(tagName);
  if (className) {
    element.className = className;
  }
  if (text !== undefined && text !== null) {
    element.textContent = String(text);
  }
  return element;
}

function safeHttpUrl(candidateUrl) {
  if (typeof candidateUrl !== "string" || candidateUrl === "") {
    return "";
  }
  try {
    const parsedUrl = new URL(candidateUrl, window.location.href);
    const isHttp = parsedUrl.protocol === "http:" || parsedUrl.protocol === "https:";
    return isHttp ? parsedUrl.href : "";
  } catch (parseError) {
    return "";
  }
}

function hostnameOf(urlText) {
  try {
    const parsedUrl = new URL(urlText);
    return parsedUrl.hostname;
  } catch (parseError) {
    return "";
  }
}

function truncateText(text, maxChars) {
  const value = String(text || "");
  if (value.length <= maxChars) {
    return value;
  }
  return `${value.slice(0, maxChars - 1)}…`;
}

function formatDuration(milliseconds) {
  if (typeof milliseconds !== "number" || !Number.isFinite(milliseconds)) {
    return "n/a";
  }
  if (milliseconds < 1000) {
    return `${Math.round(milliseconds)} ms`;
  }
  const seconds = milliseconds / 1000;
  return `${seconds.toFixed(2)} s`;
}

function formatClock(epochSeconds) {
  const hasTimestamp = typeof epochSeconds === "number" && Number.isFinite(epochSeconds);
  const date = hasTimestamp ? new Date(epochSeconds * 1000) : new Date();
  const hours = String(date.getHours()).padStart(2, "0");
  const minutes = String(date.getMinutes()).padStart(2, "0");
  const seconds = String(date.getSeconds()).padStart(2, "0");
  const millis = String(date.getMilliseconds()).padStart(3, "0");
  return `${hours}:${minutes}:${seconds}.${millis}`;
}

function isNearPageBottom() {
  const viewportBottom = window.innerHeight + window.scrollY;
  return viewportBottom >= document.body.scrollHeight - AUTO_SCROLL_SLACK_PX;
}

function scrollPageToBottom() {
  window.scrollTo({ top: document.body.scrollHeight });
}

function createBadge(kind) {
  const safeKind = String(kind || "trace").replace(/[^a-z_]/gi, "");
  return createElement("span", `badge kind-${safeKind}`, safeKind || "trace");
}

/* ------------------------------------------------------------------ turns */

function buildTurnElements(turnIndex, query) {
  const root = createElement("article", "turn");
  root.dataset.turn = String(turnIndex);

  const userMessage = createElement("div", "user-message", query);
  const card = createElement("div", "assistant-card");

  const answer = createElement("div", "answer streaming");
  const placeholder = createElement("span", "placeholder", "Routing your question…");
  answer.appendChild(placeholder);

  const errorBox = createElement("div", "turn-error");
  errorBox.hidden = true;

  const routingCard = createElement("details", "routing-card");
  routingCard.open = true;
  const routingSummary = createElement("summary");
  const routingSummaryLabel = createElement("span", "", "Routing");
  const routingSummaryPath = createElement("span", "node-tag", "");
  routingSummary.appendChild(routingSummaryLabel);
  routingSummary.appendChild(routingSummaryPath);
  routingCard.appendChild(routingSummary);

  const routingBody = createElement("div", "routing-body");
  const agentSection = createElement("div");
  agentSection.appendChild(createElement("div", "routing-section-title", "Agents, in order"));
  const agentPath = createElement("div", "agent-path");
  agentPath.appendChild(createElement("span", "routing-empty", "waiting for the supervisor…"));
  agentSection.appendChild(agentPath);

  const llmSection = createElement("div");
  llmSection.appendChild(createElement("div", "routing-section-title", "LLM calls"));
  const llmCalls = createElement("div", "llm-calls");
  llmCalls.appendChild(createElement("span", "routing-empty", "none yet"));
  llmSection.appendChild(llmCalls);

  const activitySection = createElement("div");
  activitySection.appendChild(createElement("div", "routing-section-title", "Timeline"));
  const activity = createElement("ol", "activity-list");
  activitySection.appendChild(activity);

  routingBody.appendChild(agentSection);
  routingBody.appendChild(llmSection);
  routingBody.appendChild(activitySection);
  routingCard.appendChild(routingBody);

  const citations = createElement("section", "citations");
  citations.hidden = true;
  citations.appendChild(createElement("h3", "citations-title", "Sources"));
  const citationList = createElement("ol", "citation-list");
  citations.appendChild(citationList);

  const meta = createElement("div", "turn-meta");
  meta.appendChild(createElement("span", "", "streaming…"));

  card.appendChild(answer);
  card.appendChild(errorBox);
  card.appendChild(citations);
  card.appendChild(routingCard);
  card.appendChild(meta);
  root.appendChild(userMessage);
  root.appendChild(card);

  return {
    root, answer, errorBox, routingSummaryPath, agentPath, llmCalls, activity,
    citations, citationList, meta,
  };
}

function createTurn(query) {
  state.turnCounter += 1;
  const turnIndex = state.turnCounter;
  const turnElements = buildTurnElements(turnIndex, query);
  elements.emptyState.hidden = true;
  elements.turns.appendChild(turnElements.root);
  appendTraceDivider(`turn ${turnIndex}: ${truncateText(query, 80)}`);
  scrollPageToBottom();
  return {
    index: turnIndex,
    query,
    elements: turnElements,
    rawAnswer: "",
    answerSegments: [],
    restartCount: 0,
    routing: createRoutingState(),
    citationIds: new Set(),
    startedAt: performance.now(),
    firstTokenAt: null,
    finished: false,
    sawDone: false,
    renderPending: false,
  };
}

function renderAnswerNow(turn) {
  turn.renderPending = false;
  const shouldFollow = isNearPageBottom();
  if (turn.rawAnswer === "") {
    return;
  }
  // Read again at render time: a restart may have cleared the buffer after scheduling.
  turn.elements.answer.innerHTML = renderMarkdown(turn.rawAnswer, turn.index);
  if (shouldFollow) {
    scrollPageToBottom();
  }
}

// Re-rendering the whole answer per token is fine at answer sizes, but batch it to one
// render per animation frame so a burst of tokens doesn't thrash layout.
function scheduleAnswerRender(turn) {
  if (turn.renderPending) {
    return;
  }
  turn.renderPending = true;
  function renderScheduledAnswer() {
    renderAnswerNow(turn);
  }
  window.requestAnimationFrame(renderScheduledAnswer);
}

function showTurnError(turn, message) {
  const errorBox = turn.elements.errorBox;
  errorBox.hidden = false;
  const existingText = errorBox.textContent;
  errorBox.textContent = existingText ? `${existingText}\n${message}` : message;
}

/* ----------------------------------------------------------- routing card */

function agentNameFromRoute(traceData) {
  const detail = traceData.detail || {};
  const candidate = detail.next_agent || detail.next || detail.agent || detail.to || detail.worker;
  if (candidate) {
    return String(candidate);
  }
  return String(traceData.summary || "?");
}

/* --------------------------------------------- routing summary (pure, no DOM) */

// Per-turn routing state derived only from trace events, so it can be unit tested:
// agentSteps is the supervisor's hops, llmCalls one record per llm_route event (resolved
// by the matching llm_result), and agentResults the provider/model each worker reported
// (agent_result) by node name.
function createRoutingState() {
  return { agentSteps: [], llmCalls: [], agentResults: {}, finalAgent: "", fallbackCount: 0 };
}

// The supervisor's "-> finish" route and the finish node's own route trace both end the
// run; neither is an agent hop.
function isAgentHop(traceData) {
  const detail = traceData.detail || {};
  return traceData.node !== "finish" && detail.next_agent !== "finish";
}

// "openrouter/free -> some/model:free" means the openrouter/free router picked a model;
// the part before the arrow is the id that was requested.
function requestedModelId(modelText) {
  const model = String(modelText || "");
  const arrowIndex = model.indexOf(" -> ");
  return arrowIndex === -1 ? model : model.slice(0, arrowIndex);
}

function latestCallForNode(routing, nodeName) {
  for (let callIndex = routing.llmCalls.length - 1; callIndex >= 0; callIndex -= 1) {
    const call = routing.llmCalls[callIndex];
    if (call.node === nodeName) {
      return call;
    }
  }
  return null;
}

// The router tags llm_route, llm_fallback and llm_result with the same call_id.
function callById(routing, callId) {
  if (!callId) {
    return null;
  }
  for (const call of routing.llmCalls) {
    if (call.callId === callId) {
      return call;
    }
  }
  return null;
}

// Older traces have no call_id: pair an llm_result with the node's oldest call that has
// no llm_result yet, because a node's calls finish in the order they were routed.
function oldestUnresolvedCallForNode(routing, nodeName) {
  for (const call of routing.llmCalls) {
    if (call.node === nodeName && !call.resolvedByLlmResult) {
      return call;
    }
  }
  return null;
}

function createLlmCallRecord(traceData) {
  const detail = traceData.detail || {};
  return {
    callId: String(detail.call_id || ""),
    node: String(traceData.node || detail.role || ""),
    role: String(detail.role || traceData.node || ""),
    provider: String(detail.provider || "?"),
    model: String(detail.model || ""),
    dimension: String(detail.dimension || ""),
    reason: String(detail.reason || traceData.summary || ""),
    failures: [],
    answered: null,
    resolvedByLlmResult: false,
  };
}

function recordFallback(routing, traceData) {
  const detail = traceData.detail || {};
  const nodeName = String(traceData.node || "");
  let call = callById(routing, detail.call_id);
  if (call === null) {
    call = latestCallForNode(routing, nodeName);
  }
  if (call === null) {
    // A fallback without a preceding llm_route: start a record from the failed candidate.
    const orphanDetail = { provider: detail.provider, model: detail.model, call_id: detail.call_id };
    call = createLlmCallRecord({ node: nodeName, detail: orphanDetail });
    routing.llmCalls.push(call);
  }
  const nextCandidate = Array.isArray(detail.next) && detail.next.length === 2 ? detail.next : null;
  call.failures.push({
    provider: String(detail.provider || "?"),
    model: String(detail.model || ""),
    error: String(detail.error || traceData.summary || ""),
    statusCode: typeof detail.status_code === "number" ? detail.status_code : null,
    next: nextCandidate,
    streamRestarted: detail.stream_restarted === true,
    skipped: detail.skipped === true,
  });
  routing.fallbackCount += 1;
}

// llm_result closes one LLM call: who really answered (or that nobody did).
function recordLlmResult(routing, traceData) {
  const detail = traceData.detail || {};
  const nodeName = String(traceData.node || "");
  let call = callById(routing, detail.call_id);
  if (call === null) {
    call = oldestUnresolvedCallForNode(routing, nodeName);
  }
  if (call === null) {
    return;
  }
  call.answered = {
    ok: detail.ok === true,
    provider: String(detail.provider || ""),
    model: String(detail.model || ""),
    error: detail.error ? String(detail.error) : "",
  };
  call.resolvedByLlmResult = true;
}

// A node can plan several LLM calls (data: SQL generation, then answer). Attach the
// result to the call that planned exactly that provider/model when there is one,
// otherwise to the node's latest call (the one that fell back).
function pickCallForResult(routing, nodeName, answeredProvider, answeredModel) {
  for (const call of routing.llmCalls) {
    const sameProvider = call.provider === answeredProvider;
    const sameModel = call.model === requestedModelId(answeredModel);
    if (call.node === nodeName && call.answered === null && sameProvider && sameModel) {
      return call;
    }
  }
  return latestCallForNode(routing, nodeName);
}

function recordAgentResult(routing, traceData) {
  const detail = traceData.detail || {};
  const nodeName = String(traceData.node || "");
  const result = {
    ok: detail.ok !== false,
    provider: String(detail.provider || ""),
    model: String(detail.model || ""),
    error: detail.error ? String(detail.error) : "",
  };
  routing.agentResults[nodeName] = result;
  if (!result.provider && !result.model) {
    return;
  }
  const call = pickCallForResult(routing, nodeName, result.provider, result.model);
  // An llm_result is the exact outcome of that call; agent_result only fills calls the
  // router never reported on (traces from before llm_result existed).
  if (call !== null && !call.resolvedByLlmResult) {
    call.answered = result;
  }
}

// Fold one trace event into the routing state. Returns true when the state changed.
function applyTraceToRouting(routing, traceData) {
  const kind = traceData.kind;
  const detail = traceData.detail || {};
  if (kind === "agent_route") {
    if (traceData.node === "finish" && detail.final_agent) {
      routing.finalAgent = String(detail.final_agent);
    }
    if (!isAgentHop(traceData)) {
      return traceData.node === "finish";
    }
    routing.agentSteps.push({
      name: agentNameFromRoute(traceData),
      reason: String(detail.reason || traceData.summary || ""),
    });
    return true;
  }
  if (kind === "llm_route") {
    routing.llmCalls.push(createLlmCallRecord(traceData));
    return true;
  }
  if (kind === "llm_fallback") {
    recordFallback(routing, traceData);
    return true;
  }
  if (kind === "llm_result") {
    recordLlmResult(routing, traceData);
    return true;
  }
  if (kind === "agent_result") {
    recordAgentResult(routing, traceData);
    return true;
  }
  return false;
}

function shortError(errorText) {
  // Errors look like "provider/model: HTTP 429: {...json...}"; keep the HTTP part.
  const text = String(errorText || "");
  const httpIndex = text.indexOf("HTTP ");
  const fromHttp = httpIndex === -1 ? text : text.slice(httpIndex);
  const singleLine = fromHttp.replace(/\s+/g, " ");
  return truncateText(singleLine, 90);
}

// Text for one LLM call box: what was planned, which candidates failed, and who
// actually answered. status is one of pending, failing, as-planned, replaced, failed,
// all-failed (llm_result says no provider answered).
function describeLlmCall(call) {
  const plannedLabel = call.model ? `${call.provider} / ${call.model}` : call.provider;
  const failureLines = [];
  for (const failure of call.failures) {
    const failedLabel = failure.model ? `${failure.provider} / ${failure.model}` : failure.provider;
    const nextLabel = failure.next ? ` \u2192 next ${failure.next[0]} / ${failure.next[1]}` : "";
    const restartLabel = failure.streamRestarted ? " (answer restarted)" : "";
    // Skipped entries (no key, cooling down) were never called, so they did not "fail".
    const verb = failure.skipped ? "skipped" : "failed";
    failureLines.push(`${verb}: ${failedLabel}: ${shortError(failure.error)}${nextLabel}${restartLabel}`);
  }

  const answered = call.answered;
  if (answered === null) {
    const status = call.failures.length > 0 ? "failing" : "pending";
    return { plannedLabel, failureLines, status, answeredLabel: "" };
  }
  const answeredLabel = answered.model ? `${answered.provider} / ${answered.model}` : answered.provider;
  if (!answered.ok && !answered.provider) {
    return { plannedLabel, failureLines, status: "all-failed", answeredLabel: "" };
  }
  if (!answered.ok) {
    return { plannedLabel, failureLines, status: "failed", answeredLabel };
  }
  const sameProvider = answered.provider === call.provider;
  const sameModel = requestedModelId(answered.model) === call.model;
  const status = sameProvider && sameModel ? "as-planned" : "replaced";
  return { plannedLabel, failureLines, status, answeredLabel };
}

// Provider/model that produced the turn's final answer: the finish node names the agent,
// and that agent's agent_result names the model.
function answeringModelOfTurn(routing) {
  const result = routing.agentResults[routing.finalAgent];
  if (!result || (!result.provider && !result.model)) {
    return "";
  }
  return result.model ? `${result.provider} / ${result.model}` : result.provider;
}

/* ----------------------------------------------------- routing card (DOM) */

function renderAgentPath(turn) {
  const agentSteps = turn.routing.agentSteps;
  const agentPath = turn.elements.agentPath;
  agentPath.replaceChildren();
  const supervisorStep = createElement("span", "agent-step", "supervisor");
  agentPath.appendChild(supervisorStep);
  for (const step of agentSteps) {
    agentPath.appendChild(createElement("span", "agent-arrow", "\u2192"));
    const stepChip = createElement("span", "agent-step", step.name);
    stepChip.title = step.reason || "";
    agentPath.appendChild(stepChip);
  }
  const names = [];
  for (const step of agentSteps) {
    names.push(step.name);
  }
  turn.elements.routingSummaryPath.textContent = names.length ? names.join(" \u2192 ") : "";
}

const LLM_CALL_STATUS_TEXT = {
  "as-planned": "answered by planned model",
  replaced: "answered by fallback",
  failed: "agent failed on",
  "all-failed": "no provider answered",
};

function buildLlmCallBox(call) {
  const description = describeLlmCall(call);
  const callBox = createElement("div", `llm-call status-${description.status}`);
  const head = createElement("div", "llm-call-head");
  head.appendChild(createBadge("llm_route"));
  head.appendChild(createElement("span", "node-tag", call.role || call.node));
  head.appendChild(createElement("span", "llm-call-label", "planned"));
  const plannedBox = createElement("span", "llm-call-planned");
  plannedBox.appendChild(createElement("strong", "", call.provider));
  plannedBox.appendChild(createElement("span", "llm-call-model", call.model));
  head.appendChild(plannedBox);
  if (call.dimension) {
    head.appendChild(createElement("span", "dimension-tag", call.dimension));
  }
  callBox.appendChild(head);
  if (call.reason) {
    callBox.appendChild(createElement("div", "llm-call-reason", `why: ${call.reason}`));
  }
  for (const failureLine of description.failureLines) {
    callBox.appendChild(createElement("div", "llm-call-failure", failureLine));
  }
  const statusText = LLM_CALL_STATUS_TEXT[description.status];
  if (statusText) {
    const answerLine = createElement("div", "llm-call-answer");
    answerLine.appendChild(createElement("span", "llm-call-label", statusText));
    answerLine.appendChild(createElement("strong", "llm-call-model", description.answeredLabel));
    callBox.appendChild(answerLine);
  }
  const isFailure = description.status === "failed" || description.status === "all-failed";
  if (isFailure && call.answered.error) {
    callBox.appendChild(createElement("div", "llm-call-failure", shortError(call.answered.error)));
  }
  return callBox;
}

function renderLlmCalls(turn) {
  const llmCalls = turn.elements.llmCalls;
  const callRecords = turn.routing.llmCalls;
  if (callRecords.length === 0) {
    return;
  }
  llmCalls.replaceChildren();
  for (const call of callRecords) {
    llmCalls.appendChild(buildLlmCallBox(call));
  }
}

function addTimelineEntry(turn, traceData) {
  const item = createElement("li");
  item.appendChild(createBadge(traceData.kind));
  const nodeName = traceData.node ? `${traceData.node}` : "";
  if (nodeName) {
    item.appendChild(createElement("span", "node-tag", nodeName));
  }
  item.appendChild(createElement("span", "activity-summary", traceData.summary || ""));
  turn.elements.activity.appendChild(item);
}

function handleTraceEvent(turn, traceData) {
  const kind = traceData.kind;
  const routingChanged = applyTraceToRouting(turn.routing, traceData);
  if (routingChanged && kind === "agent_route") {
    renderAgentPath(turn);
  } else if (routingChanged) {
    renderLlmCalls(turn);
  }
  if (kind === "llm_fallback") {
    const fallbackDetail = traceData.detail || {};
    if (fallbackDetail.stream_restarted === true) {
      handleStreamRestart(turn, traceData);
    }
  }
  addTimelineEntry(turn, traceData);
  appendTraceLogEntry(kind, traceData.node, traceData.summary, traceData.detail, traceData.ts);
}

/* -------------------------------------------------------------- citations */

function citationNumberFromId(citationId) {
  const idText = String(citationId || "");
  const digitMatch = idText.match(/\d+/);
  if (digitMatch) {
    return digitMatch[0];
  }
  return idText.replace(/[^a-z0-9_-]/gi, "") || "x";
}

function buildCitationTitle(citation, targetUrl) {
  const titleBox = createElement("div", "citation-title");
  const titleText = citation.title || citation.url || citation.source_id || "Untitled source";
  if (targetUrl) {
    const link = createElement("a", "", titleText);
    link.href = targetUrl;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    titleBox.appendChild(link);
  } else {
    titleBox.textContent = titleText;
  }
  return titleBox;
}

// Pure description of where a citation came from: a tag ("web", "corpus", "SQL"), the
// location parts shown after it, and whether the snippet is SQL text to show as code.
function describeCitationSource(citation, targetUrl) {
  const sourceType = String(citation.source_type || "corpus");
  if (sourceType === "web") {
    const host = hostnameOf(targetUrl);
    return { kind: "web", tag: "web", parts: host ? [host] : [], snippetIsCode: false };
  }
  if (sourceType === "database") {
    // Data-agent results: the "source" is the database file and the snippet is the SQL.
    const databaseParts = citation.source_id ? [String(citation.source_id)] : [];
    return { kind: "database", tag: "SQL", parts: databaseParts, snippetIsCode: true };
  }
  const corpusParts = [];
  if (citation.source_id) {
    corpusParts.push(`arXiv:${citation.source_id}`);
  }
  if (citation.page !== undefined && citation.page !== null && citation.page !== "") {
    corpusParts.push(`p. ${citation.page}`);
  }
  if (citation.section) {
    corpusParts.push(`\u00a7 ${citation.section}`);
  }
  return { kind: "corpus", tag: "corpus", parts: corpusParts, snippetIsCode: false };
}

function buildCitationLocation(sourceDescription) {
  const where = createElement("div", "citation-where");
  where.appendChild(createElement("span", "source-tag", sourceDescription.tag));
  for (const locationPart of sourceDescription.parts) {
    where.appendChild(createElement("span", "", locationPart));
  }
  return where;
}

function buildCitationSnippet(snippetText, snippetIsCode) {
  if (!snippetIsCode) {
    return createElement("div", "citation-snippet", truncateText(snippetText, SNIPPET_MAX_CHARS));
  }
  // SQL is shown whole and verbatim; textContent keeps it inert.
  const snippetBlock = createElement("pre", "citation-snippet citation-sql");
  snippetBlock.appendChild(createElement("code", "", snippetText));
  return snippetBlock;
}

function handleCitationEvent(turn, citation) {
  const citationNumber = citationNumberFromId(citation.id);
  const anchorId = citationAnchorId(turn.index, citationNumber);
  if (turn.citationIds.has(anchorId)) {
    return;
  }
  turn.citationIds.add(anchorId);

  const isDatabase = citation.source_type === "database";
  // Database results have no page to open, so they never get a link.
  const targetUrl = isDatabase ? "" : safeHttpUrl(citation.url);
  const sourceDescription = describeCitationSource(citation, targetUrl);
  const item = createElement("li", `citation source-${sourceDescription.kind}`);
  item.id = anchorId;
  item.appendChild(createElement("span", "citation-id", `[${citationNumber}]`));
  item.appendChild(buildCitationTitle(citation, targetUrl));
  item.appendChild(buildCitationLocation(sourceDescription));
  if (citation.snippet) {
    item.appendChild(buildCitationSnippet(String(citation.snippet), sourceDescription.snippetIsCode));
  }
  turn.elements.citationList.appendChild(item);
  turn.elements.citations.hidden = false;

  const locationParts = [citation.source_type || "corpus", citation.title || citation.url || ""];
  appendTraceLogEntry("citation", "", `${citation.id || ""} ${locationParts.join(": ")}`, citation, null);
}

function flashCitation(targetElement) {
  targetElement.classList.add("flash");
  function removeFlash() {
    targetElement.classList.remove("flash");
  }
  window.setTimeout(removeFlash, CITATION_FLASH_MS);
}

function handleTurnsClick(event) {
  const citationLink = event.target.closest("a.cite-ref");
  if (!citationLink) {
    return;
  }
  event.preventDefault();
  const targetId = citationLink.dataset.target;
  const targetElement = document.getElementById(targetId);
  if (!targetElement) {
    return;
  }
  targetElement.scrollIntoView({ behavior: "smooth", block: "center" });
  flashCitation(targetElement);
}

/* -------------------------------------------------------------- trace log */

function updateTraceCount() {
  elements.traceCount.textContent = String(state.traceEntries);
}

function appendTraceDivider(label) {
  const divider = createElement("li", "trace-divider", label);
  elements.traceLog.appendChild(divider);
}

function appendTraceLogEntry(kind, node, summary, detail, epochSeconds) {
  const shouldFollow = elements.traceLog.scrollTop + elements.traceLog.clientHeight >= elements.traceLog.scrollHeight - 40;
  const item = createElement("li");
  const head = createElement("div", "trace-line-head");
  head.appendChild(createElement("span", "trace-time", formatClock(epochSeconds)));
  head.appendChild(createBadge(kind));
  if (node) {
    head.appendChild(createElement("span", "node-tag", node));
  }
  item.appendChild(head);
  if (summary) {
    item.appendChild(createElement("div", "trace-summary", summary));
  }
  const hasDetail = detail && typeof detail === "object" && Object.keys(detail).length > 0;
  if (hasDetail) {
    const detailToggle = createElement("details");
    detailToggle.appendChild(createElement("summary", "", "detail"));
    detailToggle.appendChild(createElement("pre", "", JSON.stringify(detail, null, 2)));
    item.appendChild(detailToggle);
  }
  elements.traceLog.appendChild(item);
  state.traceEntries += 1;
  updateTraceCount();
  if (shouldFollow) {
    elements.traceLog.scrollTop = elements.traceLog.scrollHeight;
  }
}

function clearTraceLog() {
  elements.traceLog.replaceChildren();
  state.traceEntries = 0;
  updateTraceCount();
}

/* ------------------------------------------------------- stream handling */

function joinAnswerSegments(turn) {
  return joinSegmentTexts(turn.answerSegments);
}

// The backend streams a "separator" pseudo-node token ("\n\n---\n\n") between two
// workers' answers. It belongs to no worker, so a worker's restart never removes it.
const SEPARATOR_NODE = "separator";

function appendTokenSegment(segments, nodeName, tokenText) {
  const lastSegment = segments.length > 0 ? segments[segments.length - 1] : null;
  if (lastSegment && lastSegment.node === nodeName) {
    lastSegment.text += tokenText;
  } else {
    segments.push({ node: nodeName, text: tokenText });
  }
  return segments;
}

// Segments that survive a stream restart of nodeName: everything from other nodes,
// including separators.
function segmentsWithoutNode(segments, nodeName) {
  const keptSegments = [];
  for (const segment of segments) {
    const isSeparator = segment.node === SEPARATOR_NODE;
    if (isSeparator || segment.node !== nodeName) {
      keptSegments.push(segment);
    }
  }
  return keptSegments;
}

function joinSegmentTexts(segments) {
  const segmentTexts = [];
  for (const segment of segments) {
    segmentTexts.push(segment.text);
  }
  return segmentTexts.join("");
}

// Tokens are kept per node, in arrival order, so a provider restart for one node can
// drop only that node's partial text and leave other nodes' streamed text in place.
function handleTokenEvent(turn, tokenData) {
  if (turn.firstTokenAt === null) {
    turn.firstTokenAt = performance.now();
  }
  const nodeName = String(tokenData.node || "");
  const tokenText = String(tokenData.text || "");
  appendTokenSegment(turn.answerSegments, nodeName, tokenText);
  turn.rawAnswer = joinAnswerSegments(turn);
  scheduleAnswerRender(turn);
}

function showRestartNote(turn, nodeName, traceData) {
  turn.restartCount += 1;
  const detail = traceData.detail || {};
  // Router sends detail.next = [provider, model] for the candidate it moves to.
  const nextCandidate = Array.isArray(detail.next) ? detail.next : [];
  const nextProvider = nextCandidate.length === 2 ? `${nextCandidate[0]}/${nextCandidate[1]}` : "";
  const target = nextProvider ? ` (now ${nextProvider})` : "";
  const nodeLabel = nodeName ? `${nodeName}: ` : "";
  const noteText = `${nodeLabel}switched provider, restarting answer${target}`;
  let noteBox = turn.elements.restartNote;
  if (!noteBox) {
    noteBox = createElement("div", "restart-note");
    turn.elements.answer.before(noteBox);
    turn.elements.restartNote = noteBox;
  }
  noteBox.textContent = turn.restartCount > 1 ? `${noteText} [x${turn.restartCount}]` : noteText;
}

// The router emits llm_fallback with detail.stream_restarted === true when a provider
// fails mid-stream and the answer restarts on the next provider: discard that node's
// partial tokens so the restarted answer doesn't get appended to the broken one.
function handleStreamRestart(turn, traceData) {
  const nodeName = String(traceData.node || "");
  turn.answerSegments = segmentsWithoutNode(turn.answerSegments, nodeName);
  turn.rawAnswer = joinAnswerSegments(turn);
  if (turn.rawAnswer === "") {
    const placeholder = createElement("span", "placeholder", "Restarting answer\u2026");
    turn.elements.answer.replaceChildren(placeholder);
  } else {
    scheduleAnswerRender(turn);
  }
  showRestartNote(turn, nodeName, traceData);
}

function handleErrorEvent(turn, errorData) {
  const message = errorData.message || errorData.summary || errorData.error || "Unknown error";
  showTurnError(turn, message);
  appendTraceLogEntry("error", errorData.node || "", message, errorData, errorData.ts);
}

function renderTurnMeta(turn, doneData) {
  const meta = turn.elements.meta;
  meta.replaceChildren();
  const clientTtft = turn.firstTokenAt === null ? null : turn.firstTokenAt - turn.startedAt;
  const serverTtft = typeof doneData.ttft_ms === "number" ? doneData.ttft_ms : null;
  const latencyMs = typeof doneData.latency_ms === "number" ? doneData.latency_ms : performance.now() - turn.startedAt;

  function addMetric(label, value) {
    const metric = createElement("span", "", `${label} `);
    metric.appendChild(createElement("strong", "", value));
    meta.appendChild(metric);
  }

  addMetric("latency", formatDuration(latencyMs));
  addMetric("TTFT", formatDuration(serverTtft !== null ? serverTtft : clientTtft));
  if (doneData.final_agent) {
    addMetric("answered by", String(doneData.final_agent));
  }
  const answeringModel = answeringModelOfTurn(turn.routing);
  if (answeringModel) {
    addMetric("model", answeringModel);
  }
  addMetric("agents", String(turn.routing.agentSteps.length));
  addMetric("LLM calls", String(turn.routing.llmCalls.length));
  if (turn.routing.fallbackCount > 0) {
    addMetric("fallbacks", String(turn.routing.fallbackCount));
  }
  addMetric("sources", String(turn.citationIds.size));
}

function handleDoneEvent(turn, doneData) {
  turn.sawDone = true;
  // Prefer the streamed text (it already reflects restarts); fall back to done.answer
  // only when nothing was streamed, e.g. a node that answers without token streaming.
  const streamedText = turn.rawAnswer.trim();
  const hasDoneAnswer = typeof doneData.answer === "string" && doneData.answer.trim() !== "";
  if (streamedText === "" && hasDoneAnswer) {
    turn.rawAnswer = doneData.answer;
  }
  renderAnswerNow(turn);
  renderTurnMeta(turn, doneData);
  const summary = `latency ${formatDuration(doneData.latency_ms)}, ttft ${formatDuration(doneData.ttft_ms)}`;
  appendTraceLogEntry("done", "", summary, null, null);
}

function parseFrameData(frame) {
  try {
    return JSON.parse(frame.data);
  } catch (parseError) {
    return { text: frame.data };
  }
}

function handleStreamFrame(turn, frame) {
  const frameData = parseFrameData(frame) || {};
  if (frame.event === "token") {
    handleTokenEvent(turn, frameData);
  } else if (frame.event === "trace") {
    handleTraceEvent(turn, frameData);
  } else if (frame.event === "citation") {
    handleCitationEvent(turn, frameData);
  } else if (frame.event === "error") {
    handleErrorEvent(turn, frameData);
  } else if (frame.event === "done") {
    handleDoneEvent(turn, frameData);
  }
}

async function consumeEventStream(responseBody, turn) {
  const reader = responseBody.getReader();
  const decoder = new TextDecoder("utf-8");

  function dispatchFrameToTurn(frame) {
    handleStreamFrame(turn, frame);
  }

  const parser = createSseParser(dispatchFrameToTurn);
  while (true) {
    const readResult = await reader.read();
    if (readResult.done) {
      break;
    }
    // stream: true keeps multi-byte UTF-8 characters intact across chunk boundaries.
    const decodedText = decoder.decode(readResult.value, { stream: true });
    parser.push(decodedText);
  }
  const trailingText = decoder.decode();
  parser.push(trailingText);
  parser.flush();
}

// One validation problem as text. main.py sends {field, message}; FastAPI's default
// 422 body uses {loc, msg}, so both are read before falling back to raw JSON.
function describeValidationIssue(issue) {
  if (issue === null || typeof issue !== "object") {
    return String(issue);
  }
  const message = issue.message || issue.msg;
  if (!message) {
    return JSON.stringify(issue);
  }
  const fieldName = issue.field || (Array.isArray(issue.loc) ? issue.loc.join(".") : "");
  return fieldName ? `${fieldName}: ${message}` : String(message);
}

// Text for an error response body (already parsed JSON), shown above the turn.
function formatErrorDetail(body, statusCode) {
  const detail = body ? body.detail : undefined;
  if (Array.isArray(detail)) {
    const messages = [];
    for (const issue of detail) {
      messages.push(describeValidationIssue(issue));
    }
    return messages.join("; ");
  }
  return detail ? String(detail) : `HTTP ${statusCode}`;
}

async function readErrorDetail(response) {
  try {
    const body = await response.json();
    return formatErrorDetail(body, response.status);
  } catch (parseError) {
    return `HTTP ${response.status}`;
  }
}

function finalizeTurn(turn) {
  turn.finished = true;
  turn.elements.answer.classList.remove("streaming");
  if (turn.rawAnswer === "") {
    const placeholder = createElement("span", "placeholder", "No answer was produced.");
    turn.elements.answer.replaceChildren(placeholder);
  }
  if (!turn.sawDone) {
    renderTurnMeta(turn, {});
  }
}

function setStreaming(isStreaming) {
  state.isStreaming = isStreaming;
  elements.submitButton.disabled = isStreaming;
  elements.submitButton.textContent = isStreaming ? "Streaming…" : "Send";
  const exampleButtons = document.querySelectorAll("[data-query]");
  for (const button of exampleButtons) {
    button.disabled = isStreaming;
  }
}

async function submitQuery(queryText) {
  const query = String(queryText || "").trim();
  if (state.isStreaming || query === "") {
    return;
  }
  setStreaming(true);
  elements.input.value = "";
  resizeInput();

  const turn = createTurn(query);
  const abortController = new AbortController();
  state.abortController = abortController;

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify({ query, session_id: state.sessionId }),
      signal: abortController.signal,
    });
    if (!response.ok || !response.body) {
      const detail = await readErrorDetail(response);
      showTurnError(turn, `Request rejected: ${detail}`);
      return;
    }
    await consumeEventStream(response.body, turn);
    if (!turn.sawDone) {
      showTurnError(turn, "The stream ended before the server sent done.");
    }
  } catch (requestError) {
    if (requestError.name === "AbortError") {
      showTurnError(turn, "Stopped.");
    } else {
      showTurnError(turn, `Network error: ${requestError.message}`);
    }
  } finally {
    if (state.abortController === abortController) {
      state.abortController = null;
    }
    finalizeTurn(turn);
    setStreaming(false);
    elements.input.focus();
  }
}

/* ---------------------------------------------------------------- wiring */

function resizeInput() {
  elements.input.style.height = "auto";
  elements.input.style.height = `${elements.input.scrollHeight}px`;
}

function handleFormSubmit(event) {
  event.preventDefault();
  submitQuery(elements.input.value);
}

function handleInputKeydown(event) {
  // Enter sends, Shift+Enter inserts a newline; ignore Enter while an IME is composing.
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    if (!state.isStreaming) {
      submitQuery(elements.input.value);
    }
  }
}

function handleExampleClick(event) {
  const exampleButton = event.target.closest("[data-query]");
  if (!exampleButton || state.isStreaming) {
    return;
  }
  submitQuery(exampleButton.dataset.query);
}

function handleReset() {
  if (state.abortController) {
    state.abortController.abort();
  }
  const turnNodes = elements.turns.querySelectorAll(".turn");
  for (const turnNode of turnNodes) {
    turnNode.remove();
  }
  elements.emptyState.hidden = false;
  clearTraceLog();
  startNewSession();
  elements.input.value = "";
  resizeInput();
  elements.input.focus();
}

function initialize() {
  startNewSession();
  updateTraceCount();
  elements.form.addEventListener("submit", handleFormSubmit);
  elements.input.addEventListener("keydown", handleInputKeydown);
  elements.input.addEventListener("input", resizeInput);
  elements.resetButton.addEventListener("click", handleReset);
  elements.clearTraceButton.addEventListener("click", clearTraceLog);
  elements.turns.addEventListener("click", handleTurnsClick);
  document.addEventListener("click", handleExampleClick);
}

if (hasDocument) {
  initialize();
}

// Expose the pure helpers to the node test harness (tests/test_frontend.py). Browsers
// have no `module`, so this block is inert on the page.
if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    createSseParser,
    escapeHtml,
    renderInline,
    renderMarkdown,
    createRoutingState,
    isAgentHop,
    applyTraceToRouting,
    describeLlmCall,
    answeringModelOfTurn,
    describeCitationSource,
    formatErrorDetail,
    appendTokenSegment,
    segmentsWithoutNode,
    joinSegmentTexts,
  };
}
