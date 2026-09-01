const state = {
  challenges: [],
  bankId: window.DEFAULT_BANK_ID,
  bank: window.BANK_SUMMARIES[window.DEFAULT_BANK_ID],
  unified: window.UNIFIED_SUMMARY,
  presets: [],
  history: [],
  activeApiPresetName: "",
};
let batchRunning = false;
let resultVisible = false;
let activeApiScan = null;
let activeBatchScan = null;
let selectedHistoryIds = new Set();
let openedHistoryId = null;

class ScanCancelledError extends Error {
  constructor() {
    super("检测已停止");
    this.name = "ScanCancelledError";
  }
}

function ensureScanActive(signal) {
  if (signal && signal.aborted) throw new ScanCancelledError();
}

function isScanCancelled(error) {
  return error instanceof ScanCancelledError || error?.name === "AbortError";
}

const byId = (id) => document.getElementById(id);

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  })[character]);
}

function percent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function optionalNumber(id) {
  const value = byId(id).value.trim();
  return value === "" ? null : Number(value);
}

function setMessage(element, text, type = "error") {
  element.textContent = text;
  element.className = `message ${type}`;
  element.hidden = !text;
}

async function runPresetScan(configuration, view, options = {}) {
  /* 单套预设检测核心：串行探测最多 6 次以取得 3 份有效回答，再归因。
     进度与结果都通过 view 渲染，供单个表单与批量并发共用。 */
  const { signal } = options;
  ensureScanActive(signal);
  view.setMessage && view.setMessage("", "working");
  const challengeResponse = await fetch("/api/challenges", { signal });
  const firstBatch = (await challengeResponse.json()).challenges;
  ensureScanActive(signal);
  const retryResponse = await fetch("/api/challenges", { signal });
  const challenges = firstBatch.concat((await retryResponse.json()).challenges);
  ensureScanActive(signal);
  const states = challenges.map(() => "pending");
  const outputs = [];
  const errors = [];
  const target = 3;
  view.showProgress && view.showProgress();
  view.renderProgress && view.renderProgress(states, "已生成独立挑战，准备调用模型");

  for (let index = 0; index < challenges.length && outputs.length < target; index += 1) {
    ensureScanActive(signal);
    states[index] = "working";
    view.renderProgress && view.renderProgress(states, `正在进行第 ${index + 1} 次尝试，等待模型完整输出……`);
    try {
      const response = await fetch("/api/test/probe", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...configuration,
          prompt: challenges[index].prompt,
          expected_count: challenges[index].expected_count,
        }),
        signal,
      });
      const payload = await response.json();
      ensureScanActive(signal);
      if (!response.ok) throw new Error(payload.error || "接口请求失败");
      if (payload.accepted) {
        outputs.push({ text: payload.text, expected_count: challenges[index].expected_count });
        states[index] = "done";
      } else {
        errors.push(`尝试 ${index + 1}: 有效数字 ${payload.parsed_numbers}/${payload.minimum_numbers}`);
        states[index] = "invalid";
      }
    } catch (error) {
      if (isScanCancelled(error) || signal?.aborted) {
        states[index] = "cancelled";
        view.renderProgress && view.renderProgress(states, "检测已停止");
        throw new ScanCancelledError();
      }
      errors.push(`尝试 ${index + 1}: ${error.message}`);
      states[index] = "error";
    }
    view.renderProgress && view.renderProgress(states, `当前已有 ${outputs.length}/${target} 份有效回答`);
  }

  if (outputs.length === target) {
    states.forEach((state, index) => { if (state === "pending") states[index] = "skipped"; });
  }
  ensureScanActive(signal);

  if (!outputs.length) {
    view.renderProgress && view.renderProgress(states, "六次尝试后仍没有可用回答");
    view.setMessage && view.setMessage(`没有获得可分析输出。${errors[0] || ""}`, "error");
    return;
  }

  view.renderProgress && view.renderProgress(states, "模型回答已收齐，正在计算归因概率……");
  const analysisResponse = await fetch("/api/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ outputs }),
    signal,
  });
  const result = await analysisResponse.json();
  ensureScanActive(signal);
  if (analysisResponse.ok) {
    const attempted = states.filter((state) => ["done", "invalid", "error"].includes(state)).length;
    result.api_test = { requested: target, attempted, max_attempts: challenges.length, received: outputs.length, errors };
    view.renderProgress && view.renderProgress(states, `测试完成：${outputs.length}/${target} 份有效回答进入归因`);
    if (view.renderResult) await view.renderResult(result);
  } else {
    view.setMessage && view.setMessage(result.error || "API 自动测试失败。", "error");
  }
}

async function testViaApi(event) {
  event.preventDefault();
  if (activeApiScan) return;
  const button = event.currentTarget.querySelector("button[type=submit]");
  const stopButton = byId("api-test-stop");
  const scan = { controller: new AbortController() };
  activeApiScan = scan;
  button.disabled = true;
  stopButton.hidden = false;
  stopButton.disabled = false;
  byId("result").hidden = true;
  resultVisible = false;
  setMessage(byId("test-message"), "");
  const configuration = {
    base_url: byId("test-api-base").value,
    api_key: byId("test-api-key").value,
    api_model: byId("test-api-model").value,
    temperature: optionalNumber("test-temperature"),
  };
  const view = {
    showProgress: () => { byId("api-test-progress").hidden = false; },
    renderProgress: renderApiProgress,
    renderResult: (result) => renderResult(result, {
      testType: "api",
      sourceName: state.activeApiPresetName,
      apiModel: configuration.api_model,
    }),
    setMessage: (text, type) => setMessage(byId("test-message"), text, type),
  };
  try {
    await runPresetScan(configuration, view, { signal: scan.controller.signal });
  } catch (error) {
    if (isScanCancelled(error)) {
      byId("api-progress-status").textContent = "检测已停止";
      setMessage(byId("test-message"), "检测已停止。", "working");
    } else {
      setMessage(byId("test-message"), error.message || "检测失败。", "error");
    }
  } finally {
    if (activeApiScan === scan) activeApiScan = null;
    button.disabled = false;
    stopButton.hidden = true;
    stopButton.disabled = false;
  }
}

function stopApiScan() {
  if (!activeApiScan) return;
  activeApiScan.controller.abort();
  byId("api-test-progress").hidden = false;
  byId("api-progress-status").textContent = "正在停止检测……";
  byId("api-test-stop").disabled = true;
  setMessage(byId("test-message"), "正在停止检测……", "working");
}

function configForPreset(preset) {
  return {
    base_url: preset.base_url || "",
    api_key: preset.api_key || "",
    api_model: preset.model || "",
    temperature: preset.temperature === "" || preset.temperature === undefined || preset.temperature === null
      ? null
      : Number(preset.temperature),
  };
}

function activateWorkspace(name) {
  document.querySelectorAll(".workspace").forEach((item) => item.classList.toggle("active", item.id === `workspace-${name}`));
  document.querySelectorAll("[data-workspace]").forEach((item) => item.classList.toggle("active", item.dataset.workspace === name));
  if (name === "history") loadHistory();
}

function activateMode(group, name) {
  document.querySelectorAll(`[data-${group}-mode]`).forEach((item) => item.classList.toggle("active", item.dataset[`${group}Mode`] === name));
  document.querySelectorAll(`#workspace-${group} .mode-panel`).forEach((item) => {
    const modes = (item.dataset.showFor || "").split(/\s+/).filter(Boolean);
    const active = modes.length ? modes.includes(name) : item.id === `${group}-${name}`;
    item.classList.toggle("active", active);
  });
  if (group === "test") byId("result").hidden = name === "batch" || !resultVisible;
}

async function loadChallenges() {
  byId("regenerate").disabled = true;
  byId("result").hidden = true;
  resultVisible = false;
  setMessage(byId("test-message"), "");
  const response = await fetch("/api/challenges");
  state.challenges = (await response.json()).challenges;
  renderChallenges();
  byId("regenerate").disabled = false;
}

function renderChallenges() {
  byId("challenge-list").innerHTML = state.challenges.map((challenge, index) => `
    <article class="challenge-item">
      <div class="challenge-header">
        <strong>挑战 ${index + 1}</strong>
        <span>${challenge.expected_count} 个数字</span>
        <button type="button" data-copy="${index}">复制提示词</button>
      </div>
      <div class="challenge-columns">
        <div><label>发送给待测模型</label><pre>${escapeHtml(challenge.prompt)}</pre></div>
        <div><label for="output-${index}">粘贴完整输出</label><textarea id="output-${index}" spellcheck="false" placeholder="保留文字、标点、代码块和完整数字序列"></textarea></div>
      </div>
    </article>
  `).join("");
  document.querySelectorAll("[data-copy]").forEach((button) => {
    button.addEventListener("click", async () => {
      await navigator.clipboard.writeText(state.challenges[Number(button.dataset.copy)].prompt);
      button.textContent = "已复制";
      window.setTimeout(() => { button.textContent = "复制提示词"; }, 1000);
    });
  });
}

function buildResultHtml(payload) {
  const diagnostics = (payload.diagnostics || []).map((item, index) => `
    <span class="diagnostic ${item.accepted ? "accepted" : "rejected"}">挑战 ${index + 1}: ${item.parsed_numbers} 个数字 · ${item.accepted ? "计入" : "忽略"}</span>
  `).join("");
  const rows = (payload.results || []).map((item, index) => `
    <tr class="${index === 0 ? "winner" : ""}">
      <td>${index + 1}</td><td><strong>${escapeHtml(item.display_name)}</strong></td><td>${escapeHtml(item.family_name)}</td>
      <td><div class="probability-cell"><span><i style="width:${item.probability * 100}%"></i></span><strong>${percent(item.probability)}</strong></div></td>
      <td>${percent(item.profile_similarity)}</td>
    </tr>
  `).join("");
  const apiErrors = payload.api_test?.errors || [];
  const apiNote = payload.api_test
    ? `<span>API 获得 ${payload.api_test.received}/${payload.api_test.requested} 份有效回答，实际尝试 ${payload.api_test.attempted}/${payload.api_test.max_attempts}${apiErrors.length ? `，${apiErrors.length} 次未采用` : ""}</span>`
    : "";
  return `
    <div class="result-summary">
      <div><span>最可能模型</span><strong>${escapeHtml(payload.prediction_name)}</strong></div>
      <div><span>统一库概率</span><strong>${percent(payload.probability)}</strong></div>
      <div><span>自动识别家族</span><strong>${escapeHtml(payload.family_prediction_name)} · ${percent(payload.family_probability)}</strong></div>
      <div><span>有效查询</span><strong>${payload.used_outputs}/3</strong></div>
    </div>
    <div class="diagnostics">${diagnostics}</div>
    <div class="table-wrap"><table><thead><tr><th>排序</th><th>候选模型</th><th>家族</th><th>归因概率</th><th>分布相似度</th></tr></thead><tbody>${rows}</tbody></table></div>
    ${apiNote ? `<div class="result-note">${apiNote}</div>` : ""}
  `;
}

function historyMethodLabel(testType) {
  return { manual: "手动测试", api: "API 单次测试", batch: "并发批量检测" }[testType] || "检测";
}

function buildHistoryAutoSaveStatus() {
  return `
    <div class="history-auto-save" aria-live="polite">
      <span class="history-auto-save-status working">正在自动保存检测历史…</span>
    </div>
  `;
}

async function renderResult(payload, context = {}) {
  const resultElement = byId("result");
  resultElement.innerHTML = buildResultHtml(payload) + buildHistoryAutoSaveStatus();
  resultElement.hidden = false;
  resultVisible = true;
  resultElement.scrollIntoView({ behavior: "smooth", block: "start" });
  await persistResultToHistory(payload, context, resultElement);
}

async function persistResultToHistory(result, context, host) {
  const status = host.querySelector(".history-auto-save-status");
  try {
    const response = await fetch("/api/history", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        test_type: context.testType || "manual",
        source_name: context.sourceName || "",
        api_model: context.apiModel || "",
        result,
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "保存检测历史失败。");
    status.textContent = `已自动保存：${payload.record.label}`;
    status.className = "history-auto-save-status success";
    await loadHistory();
  } catch (error) {
    status.textContent = error.message || "自动保存失败";
    status.className = "history-auto-save-status error";
  }
}

function formatHistoryTime(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  return new Intl.DateTimeFormat("zh-CN", {
    year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hour12: false,
  }).format(date);
}

function isSavedToday(value) {
  const date = new Date(value);
  const today = new Date();
  return !Number.isNaN(date.getTime())
    && date.getFullYear() === today.getFullYear()
    && date.getMonth() === today.getMonth()
    && date.getDate() === today.getDate();
}

function filteredHistory() {
  const query = byId("history-search").value.trim().toLocaleLowerCase();
  const method = byId("history-method").value;
  return state.history.filter((record) => {
    if (method && record.test_type !== method) return false;
    if (!query) return true;
    return [record.label, record.source_name, record.api_model, record.prediction_name, record.family_prediction_name]
      .some((value) => String(value || "").toLocaleLowerCase().includes(query));
  });
}

function renderHistory() {
  const records = filteredHistory();
  const currentIds = new Set(state.history.map((record) => record.id));
  selectedHistoryIds = new Set([...selectedHistoryIds].filter((id) => currentIds.has(id)));
  const total = state.history.length;
  const manual = state.history.filter((record) => record.test_type === "manual").length;
  const api = state.history.filter((record) => record.test_type === "api" || record.test_type === "batch").length;
  const passed = state.history.filter((record) => record.status === "passed").length;
  const failed = state.history.filter((record) => record.status === "failed").length;
  byId("history-total").textContent = String(total);
  byId("history-today").textContent = String(state.history.filter((record) => isSavedToday(record.saved_at)).length);
  byId("history-passed").textContent = String(passed);
  byId("history-failed").textContent = String(failed);
  byId("history-manual").textContent = String(manual);
  byId("history-api").textContent = String(api);
  byId("history-count-badge").textContent = `${total} 条记录`;

  byId("history-list").innerHTML = records.length
    ? records.map((record) => {
      const source = [record.source_name, record.api_model].filter(Boolean).join(" · ") || "—";
      return `
        <tr>
          <td class="history-check-column"><input class="history-row-check" type="checkbox" value="${escapeHtml(record.id)}"${selectedHistoryIds.has(record.id) ? " checked" : ""} aria-label="选择 ${escapeHtml(record.label)}"></td>
          <td class="history-record-cell"><strong>${escapeHtml(record.label)}</strong><span>${escapeHtml(source)}</span></td>
          <td><span class="history-method ${escapeHtml(record.test_type)}">${historyMethodLabel(record.test_type)}</span></td>
          <td><span class="history-status ${escapeHtml(record.status || "failed")}" title="${escapeHtml(record.status_reason || "")}">${escapeHtml(record.status_label || "检测未通过")}</span></td>
          <td><strong>${escapeHtml(record.prediction_name || "—")}</strong><span class="history-family">${escapeHtml(record.family_prediction_name || "")}</span></td>
          <td>${percent(Number(record.probability) || 0)}</td>
          <td>${record.used_outputs || 0}/3</td>
          <td>${formatHistoryTime(record.saved_at)}</td>
          <td class="history-row-actions"><button class="history-row-button" data-history-open="${escapeHtml(record.id)}" type="button">查看</button><button class="history-row-button danger" data-history-delete="${escapeHtml(record.id)}" type="button">删除</button></td>
        </tr>
      `;
    }).join("")
    : `<tr><td class="history-empty" colspan="9">暂无符合条件的检测记录</td></tr>`;

  const allVisibleSelected = records.length > 0 && records.every((record) => selectedHistoryIds.has(record.id));
  const someVisibleSelected = records.some((record) => selectedHistoryIds.has(record.id));
  const selectAll = byId("history-select-all");
  selectAll.checked = allVisibleSelected;
  selectAll.indeterminate = !allVisibleSelected && someVisibleSelected;
  selectAll.disabled = records.length === 0;
  byId("history-selected-count").textContent = selectedHistoryIds.size ? `已选择 ${selectedHistoryIds.size} 条记录` : "未选择记录";
  byId("history-delete-selected").disabled = selectedHistoryIds.size === 0;
}

async function loadHistory() {
  try {
    const response = await fetch("/api/history");
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "读取检测历史失败。");
    state.history = payload.records || [];
    if (openedHistoryId && !state.history.some((record) => record.id === openedHistoryId)) {
      openedHistoryId = null;
      byId("history-detail").hidden = true;
    }
    renderHistory();
  } catch (error) {
    state.history = [];
    renderHistory();
    setMessage(byId("history-message"), error.message || "读取检测历史失败。", "error");
  }
}

function toggleHistorySelection(recordId, selected) {
  if (selected) selectedHistoryIds.add(recordId);
  else selectedHistoryIds.delete(recordId);
  renderHistory();
}

function toggleAllHistorySelection() {
  const records = filteredHistory();
  if (byId("history-select-all").checked) records.forEach((record) => selectedHistoryIds.add(record.id));
  else records.forEach((record) => selectedHistoryIds.delete(record.id));
  renderHistory();
}

async function openHistoryRecord(recordId) {
  try {
    const response = await fetch(`/api/history/${encodeURIComponent(recordId)}`);
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "读取检测详情失败。");
    const record = payload.record;
    openedHistoryId = record.id;
    const source = [record.source_name, record.api_model].filter(Boolean).join(" · ");
    byId("history-detail").innerHTML = `
      <div class="history-detail-head"><div><strong>${escapeHtml(record.label)}</strong><span>${historyMethodLabel(record.test_type)}${source ? ` · ${escapeHtml(source)}` : ""} · ${formatHistoryTime(record.saved_at)}</span></div><div class="history-detail-actions"><span class="history-status ${escapeHtml(record.status || "failed")}" title="${escapeHtml(record.status_reason || "")}">${escapeHtml(record.status_label || "检测未通过")}</span><button id="history-detail-close" class="button secondary" type="button">关闭详情</button></div></div>
      <div class="history-detail-result">${buildResultHtml(record.result)}</div>
    `;
    byId("history-detail").hidden = false;
    byId("history-detail").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    setMessage(byId("history-message"), error.message || "读取检测详情失败。", "error");
  }
}

async function deleteHistoryRecords(ids) {
  if (!ids.length) return;
  const prompt = ids.length === 1 ? "确定删除这条检测历史吗？" : `确定删除选中的 ${ids.length} 条检测历史吗？`;
  if (!window.confirm(prompt)) return;
  try {
    const response = await fetch("/api/history", {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ids }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "删除检测历史失败。");
    ids.forEach((id) => selectedHistoryIds.delete(id));
    if (ids.includes(openedHistoryId)) {
      openedHistoryId = null;
      byId("history-detail").hidden = true;
    }
    setMessage(byId("history-message"), `已删除 ${payload.deleted} 条检测历史。`, "success");
    await loadHistory();
  } catch (error) {
    setMessage(byId("history-message"), error.message || "删除检测历史失败。", "error");
  }
}

async function analyzeManual() {
  const button = byId("analyze");
  button.disabled = true;
  setMessage(byId("test-message"), "正在计算……", "working");
  const outputs = state.challenges.map((challenge, index) => ({
    text: byId(`output-${index}`).value,
    expected_count: challenge.expected_count,
  }));
  const response = await fetch("/api/analyze", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ outputs }) });
  const payload = await response.json();
  if (response.ok) {
    setMessage(byId("test-message"), "");
    await renderResult(payload, { testType: "manual" });
  } else {
    setMessage(byId("test-message"), payload.error || "无法完成归因。", "error");
    byId("result").hidden = true;
    resultVisible = false;
  }
  button.disabled = false;
}

function renderApiProgress(states, status) {
  const valid = states.filter((state) => state === "done").length;
  const attempted = states.filter((state) => ["done", "invalid", "error"].includes(state)).length;
  const target = 3;
  byId("api-test-progress").hidden = false;
  byId("api-progress-status").textContent = status;
  byId("api-progress-count").textContent = `有效 ${valid}/${target} · 已尝试 ${attempted}/${states.length}`;
  byId("api-progress-fill").style.width = `${(valid / target) * 100}%`;
  byId("api-progress-steps").innerHTML = states.map((state, index) => {
    const labels = { pending: "等待", working: "请求中", done: "有效", invalid: "数字不足", error: "接口失败", skipped: "无需调用", cancelled: "已停止" };
    return `<span class="progress-step ${state}"><b>${index + 1}</b>挑战 ${index + 1} · ${labels[state]}</span>`;
  }).join("");
}

function updateUnifiedSummary(summary) {
  state.unified = summary;
  byId("topbar-bank-count").textContent = `${summary.model_count} 个候选模型`;
  byId("active-bank-badge").textContent = `${summary.model_count} 个候选模型`;
}

function renderPresets(presets) {
  state.presets = presets;
  const selected = byId("preset-select").value;
  byId("preset-select").innerHTML = ['<option value="">— 请选择或新建 —</option>']
    .concat(presets.map((preset) => `<option value="${escapeHtml(preset.name)}">${escapeHtml(preset.name)}</option>`))
    .join("");
  if (presets.some((preset) => preset.name === selected)) {
    byId("preset-select").value = selected;
  }
  renderPresetBatchList();
}

function batchCheckedNames() {
  return Array.from(document.querySelectorAll("#preset-batch-list input[type=checkbox]:checked"))
    .map((checkbox) => checkbox.value);
}

function updateCheckedCount() {
  const checkboxes = Array.from(document.querySelectorAll("#preset-batch-list input[type=checkbox]"));
  const checked = checkboxes.filter((checkbox) => checkbox.checked).length;
  byId("preset-checked-count").textContent = String(checked);
  byId("preset-select-all").checked = checkboxes.length > 0 && checked === checkboxes.length;
  byId("preset-select-all").disabled = checkboxes.length === 0;
  byId("preset-batch-run").disabled = checkboxes.length === 0 || batchRunning;
  byId("preset-batch-stop").hidden = !batchRunning;
}

function renderPresetBatchList() {
  const checkedNames = new Set(batchCheckedNames());
  const host = byId("preset-batch-list");
  host.innerHTML = state.presets.length
    ? state.presets.map((preset) => `
        <div class="preset-batch-item" draggable="true" data-name="${escapeHtml(preset.name)}" title="${escapeHtml(preset.base_url)}">
          <span class="preset-batch-grip" aria-hidden="true">⋮⋮</span>
          <label class="preset-batch-label">
            <input type="checkbox" value="${escapeHtml(preset.name)}"${checkedNames.has(preset.name) ? " checked" : ""}>
            <span class="preset-batch-name">${escapeHtml(preset.name)}</span>
            <span class="preset-batch-meta">${escapeHtml(preset.model || "未填模型名")}</span>
          </label>
        </div>
      `).join("")
    : `<span class="preset-batch-empty">暂无预设，请先在上方保存。</span>`;
  updateCheckedCount();
}

async function applyPresetOrder(names) {
  try {
    const response = await fetch("/api/presets/reorder", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ names }),
    });
    const payload = await response.json();
    if (!response.ok) {
      setMessage(byId("test-message"), payload.error || "调整预设排序失败。", "error");
      return;
    }
    renderPresets(payload.presets || []);
  } catch (_) {
    setMessage(byId("test-message"), "调整预设排序失败。", "error");
  }
}

function reorderPreset(name, referenceName, insertBefore) {
  const current = state.presets.map((preset) => preset.name);
  const from = current.indexOf(name);
  if (from < 0) return;
  const next = current.filter((item) => item !== name);
  let to = next.indexOf(referenceName);
  if (to < 0) return;
  if (!insertBefore) to += 1;
  if (to === from) return;
  next.splice(to, 0, name);
  applyPresetOrder(next);
}

function toggleAllPresets() {
  const selectAll = byId("preset-select-all").checked;
  document.querySelectorAll("#preset-batch-list input[type=checkbox]").forEach((checkbox) => { checkbox.checked = selectAll; });
  updateCheckedCount();
}

function makeBatchView(preset, host) {
  const card = document.createElement("article");
  card.className = "batch-result-card";
  card.innerHTML = `
    <div class="batch-result-head">
      <strong>${escapeHtml(preset.name)}</strong>
      <span>${escapeHtml(preset.base_url)} · ${escapeHtml(preset.model || "未填模型名")}</span>
    </div>
    <section class="progress-panel batch-progress" hidden>
      <div class="progress-heading"><strong class="batch-status">准备…</strong><span class="batch-count">有效 0/3 · 已尝试 0/6</span></div>
      <div class="progress-track"><i class="batch-fill" style="width:0%"></i></div>
      <div class="progress-steps batch-steps"></div>
    </section>
    <div class="batch-message message" hidden></div>
    <section class="batch-result-host" hidden></section>
  `;
  host.appendChild(card);
  const query = (selector) => card.querySelector(selector);
  const statusEl = query(".batch-status");
  const countEl = query(".batch-count");
  const fillEl = query(".batch-fill");
  const stepsEl = query(".batch-steps");
  const progressEl = query(".batch-progress");
  const resultEl = query(".batch-result-host");
  return {
    preset,
    showProgress: () => { progressEl.hidden = false; },
    renderProgress: (states, status) => {
      const valid = states.filter((item) => item === "done").length;
      const attempted = states.filter((item) => ["done", "invalid", "error"].includes(item)).length;
      statusEl.textContent = status;
      countEl.textContent = `有效 ${valid}/3 · 已尝试 ${attempted}/${states.length}`;
      fillEl.style.width = `${(valid / 3) * 100}%`;
      stepsEl.innerHTML = states.map((item, index) => {
        const labels = { pending: "等待", working: "请求中", done: "有效", invalid: "数字不足", error: "接口失败", skipped: "无需调用", cancelled: "已停止" };
        return `<span class="progress-step ${item}"><b>${index + 1}</b>挑战 ${index + 1} · ${labels[item]}</span>`;
      }).join("");
    },
    renderResult: (result) => {
      resultEl.innerHTML = buildResultHtml(result) + buildHistoryAutoSaveStatus();
      resultEl.hidden = false;
      return persistResultToHistory(result, {
        testType: "batch",
        sourceName: preset.name,
        apiModel: preset.model,
      }, resultEl);
    },
    markCancelled: () => {
      progressEl.hidden = false;
      statusEl.textContent = "检测已停止";
    },
    setMessage: (text, type) => setMessage(query(".batch-message"), text, type),
  };
}

async function startBatchScan() {
  if (batchRunning) return;
  const names = batchCheckedNames();
  if (!names.length) {
    setMessage(byId("test-message"), "请先勾选至少一个预设，再并发检测。", "error");
    return;
  }
  const presets = names.map((name) => state.presets.find((preset) => preset.name === name)).filter(Boolean);
  const runButton = byId("preset-batch-run");
  const stopButton = byId("preset-batch-stop");
  const resultsHost = byId("batch-test-results");
  const scan = { controller: new AbortController() };
  activeBatchScan = scan;
  batchRunning = true;
  runButton.disabled = true;
  stopButton.hidden = false;
  stopButton.disabled = false;
  runButton.textContent = "检测中……";
  byId("result").hidden = true;
  resultVisible = false;
  resultsHost.innerHTML = "";
  setMessage(byId("test-message"), `已并发启动 ${presets.length} 个预设的检测。`, "working");

  const views = presets.map((preset) => makeBatchView(preset, resultsHost));
  const outcomes = await Promise.all(
    views.map(async (view) => {
      try {
        await runPresetScan(configForPreset(view.preset), view, { signal: scan.controller.signal });
        return { name: view.preset.name, ok: true };
      } catch (error) {
        if (isScanCancelled(error)) {
          view.markCancelled && view.markCancelled();
          view.setMessage("检测已停止。", "working");
          return { name: view.preset.name, ok: false, cancelled: true };
        }
        view.setMessage(error.message || "检测失败。", "error");
        return { name: view.preset.name, ok: false };
      }
    })
  );
  const cancelled = scan.controller.signal.aborted;
  batchRunning = false;
  if (activeBatchScan === scan) activeBatchScan = null;
  updateCheckedCount();
  runButton.textContent = "并发检测勾选预设";
  stopButton.hidden = true;
  stopButton.disabled = false;
  if (cancelled) {
    setMessage(byId("test-message"), "检测已停止。已完成的结果仍保留在下方。", "working");
    resultsHost.scrollIntoView({ behavior: "smooth", block: "start" });
    return;
  }
  const failed = outcomes.filter((item) => !item.ok).map((item) => item.name);
  setMessage(
    byId("test-message"),
    failed.length ? `检测完成，${failed.length} 个预设执行失败：${failed.join("、")}` : `检测完成：${outcomes.length} 个预设全部执行完毕。`,
    failed.length ? "error" : "success"
  );
  resultsHost.scrollIntoView({ behavior: "smooth", block: "start" });
}

function stopBatchScan() {
  if (!activeBatchScan) return;
  activeBatchScan.controller.abort();
  byId("preset-batch-stop").disabled = true;
  setMessage(byId("test-message"), "正在停止检测……", "working");
}

async function reloadPresets() {
  try {
    const response = await fetch("/api/presets");
    const payload = await response.json();
    renderPresets(payload.presets || []);
  } catch (_) {
    renderPresets([]);
  }
}

async function loadSelectedPreset() {
  const name = byId("preset-select").value;
  const preset = state.presets.find((item) => item.name === name);
  if (!preset) return;
  byId("test-api-base").value = preset.base_url || "";
  byId("test-api-model").value = preset.model || "";
  byId("test-api-key").value = preset.api_key || "";
  byId("test-temperature").value = preset.temperature === "" ? "" : preset.temperature;
  byId("preset-name").value = "";
  state.activeApiPresetName = preset.name;
  activateMode("test", "api");
  setMessage(byId("test-message"), `已载入预设「${preset.name}」`, "success");
}

async function saveCurrentPreset() {
  const name = byId("preset-name").value.trim();
  if (!name) {
    setMessage(byId("test-message"), "请先填写“新预设名称”再保存。", "error");
    byId("preset-name").focus();
    return;
  }
  const response = await fetch("/api/presets", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      name,
      base_url: byId("test-api-base").value,
      api_key: byId("test-api-key").value,
      model: byId("test-api-model").value,
      temperature: optionalNumber("test-temperature"),
    }),
  });
  const payload = await response.json();
  if (!response.ok) {
    setMessage(byId("test-message"), payload.error || "保存预设失败。", "error");
    return;
  }
  renderPresets(payload.presets || []);
  byId("preset-select").value = name;
  state.activeApiPresetName = name;
  setMessage(byId("test-message"), `已保存预设「${name}」，下次启动可直接载入。`, "success");
}

async function deleteSelectedPreset() {
  const name = byId("preset-select").value;
  if (!name) return;
  if (!window.confirm(`确定删除预设「${name}」吗？`)) return;
  const response = await fetch("/api/presets", {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  const payload = await response.json();
  if (!response.ok) {
    setMessage(byId("test-message"), payload.error || "删除预设失败。", "error");
    return;
  }
  renderPresets(payload.presets || []);
  if (state.activeApiPresetName === name) state.activeApiPresetName = "";
  setMessage(byId("test-message"), `已删除预设「${name}」。`, "success");
}

async function renameSelectedPreset() {
  const name = byId("preset-select").value;
  if (!name) {
    setMessage(byId("test-message"), "请先在“预设”中选择要重命名的项。", "error");
    return;
  }
  const input = window.prompt(`修改预设「${name}」的名称：`, name);
  if (input === null) return;
  const newName = input.trim();
  if (!newName) {
    setMessage(byId("test-message"), "预设名称不能为空。", "error");
    return;
  }
  if (newName === name) return;
  const response = await fetch("/api/presets/rename", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, new_name: newName }),
  });
  const payload = await response.json();
  if (!response.ok) {
    setMessage(byId("test-message"), payload.error || "重命名失败。", "error");
    return;
  }
  const remapped = new Set(batchCheckedNames().map((item) => (item === name ? newName : item)));
  renderPresets(payload.presets || []);
  document.querySelectorAll("#preset-batch-list input[type=checkbox]").forEach((checkbox) => {
    checkbox.checked = remapped.has(checkbox.value);
  });
  byId("preset-select").value = newName;
  if (state.activeApiPresetName === name) state.activeApiPresetName = newName;
  updateCheckedCount();
  setMessage(byId("test-message"), `已将预设「${name}」重命名为「${newName}」。`, "success");
}

function renderInventory() {
  byId("selected-bank-name").textContent = state.bank.label;
  byId("model-options").innerHTML = state.bank.models.map((model) => `<option value="${escapeHtml(model.id)}"></option>`).join("");
  byId("bank-inventory").innerHTML = state.bank.models.length
    ? state.bank.models.map((model) => `<span class="fingerprint-item">${escapeHtml(model.display_name)}</span>`).join("")
    : `<span class="empty-inventory">暂无指纹</span>`;
}

async function refreshBank() {
  const response = await fetch(`/api/bank?bank_id=${encodeURIComponent(state.bankId)}`);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "无法读取指纹库");
  state.bank = payload;
  renderInventory();
}

async function selectBank(bankId) {
  state.bankId = bankId;
  await refreshBank();
}

async function enrollAutomatically(event) {
  event.preventDefault();
  const button = event.currentTarget.querySelector("button[type=submit]");
  button.disabled = true;
  const requested = Number(byId("sample-count").value);
  const started = Date.now();
  const progressTimer = window.setInterval(() => {
    const seconds = Math.floor((Date.now() - started) / 1000);
    setMessage(byId("enrollment-message"), `正在自动识别协议并采集 ${requested} 份回答 · 已等待 ${seconds} 秒`, "working");
  }, 1000);
  setMessage(byId("enrollment-message"), `正在自动识别协议并采集 ${requested} 份回答`, "working");
  let response;
  try {
    response = await fetch("/api/enroll/auto", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        base_url: byId("api-base").value,
        api_key: byId("api-key").value,
        api_model: byId("api-model").value,
        bank_id: state.bankId,
        model_label: byId("auto-model").value,
        sample_count: requested,
        temperature: optionalNumber("temperature"),
      }),
    });
  } catch (error) {
    window.clearInterval(progressTimer);
    setMessage(byId("enrollment-message"), error.message, "error");
    button.disabled = false;
    return;
  }
  window.clearInterval(progressTimer);
  const payload = await response.json();
  if (response.ok) {
    state.bank = payload.bank;
    updateUnifiedSummary(payload.unified);
    renderInventory();
    setMessage(byId("enrollment-message"), `采集完成：收到 ${payload.received}/${payload.requested} 份，${payload.accepted} 份进入指纹库，${payload.rejected} 份无效，${payload.errors.length} 次接口错误。`, "success");
  } else {
    setMessage(byId("enrollment-message"), payload.error || "自动采集失败。", "error");
  }
  button.disabled = false;
}

function renderBankOptions(summaries, selected) {
  byId("bank-select").innerHTML = Object.entries(summaries)
    .map(([bankId, bank]) => `<option value="${escapeHtml(bankId)}"${bankId === selected ? " selected" : ""}>${escapeHtml(bank.label)}</option>`)
    .join("");
}

async function createBank(event) {
  event.preventDefault();
  const button = event.currentTarget.querySelector("button[type=submit]");
  button.disabled = true;
  const response = await fetch("/api/banks", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ label: byId("new-bank-name").value }),
  });
  const payload = await response.json();
  if (response.ok) {
    window.BANK_SUMMARIES = payload.banks;
    state.bankId = payload.bank.id;
    state.bank = payload.bank;
    updateUnifiedSummary(payload.unified);
    renderBankOptions(payload.banks, state.bankId);
    renderInventory();
    byId("new-bank-name").value = "";
    byId("create-bank-form").hidden = true;
    setMessage(byId("enrollment-message"), `已创建 ${payload.bank.label}`, "success");
  } else {
    setMessage(byId("enrollment-message"), payload.error || "创建失败。", "error");
  }
  button.disabled = false;
}

document.querySelectorAll("[data-workspace]").forEach((button) => button.addEventListener("click", () => activateWorkspace(button.dataset.workspace)));
document.querySelectorAll("[data-test-mode]").forEach((button) => button.addEventListener("click", () => activateMode("test", button.dataset.testMode)));
byId("bank-select").addEventListener("change", (event) => selectBank(event.target.value));
byId("regenerate").addEventListener("click", loadChallenges);
byId("analyze").addEventListener("click", analyzeManual);
byId("api-test-form").addEventListener("submit", testViaApi);
byId("api-test-stop").addEventListener("click", stopApiScan);
["test-api-base", "test-api-model", "test-api-key", "test-temperature"].forEach((id) => {
  byId(id).addEventListener("input", () => { state.activeApiPresetName = ""; });
});
byId("preset-load").addEventListener("click", loadSelectedPreset);
byId("preset-rename").addEventListener("click", renameSelectedPreset);
byId("preset-save").addEventListener("click", saveCurrentPreset);
byId("preset-delete").addEventListener("click", deleteSelectedPreset);
byId("preset-select-all").addEventListener("change", toggleAllPresets);
byId("preset-batch-run").addEventListener("click", startBatchScan);
byId("preset-batch-stop").addEventListener("click", stopBatchScan);
byId("preset-batch-list").addEventListener("change", (event) => {
  if (event.target.matches("input[type=checkbox]")) updateCheckedCount();
});
let dragPresetName = null;
let dropInsertBefore = false;

byId("preset-batch-list").addEventListener("dragstart", (event) => {
  const item = event.target.closest(".preset-batch-item");
  if (!item) {
    event.preventDefault();
    return;
  }
  dragPresetName = item.dataset.name;
  dropInsertBefore = false;
  item.classList.add("dragging");
  event.dataTransfer.effectAllowed = "move";
  event.dataTransfer.setData("text/plain", dragPresetName);
});

byId("preset-batch-list").addEventListener("dragover", (event) => {
  if (!dragPresetName) return;
  event.preventDefault();
  event.dataTransfer.dropEffect = "move";
  const host = byId("preset-batch-list");
  host.querySelectorAll(".drop-before, .drop-after").forEach((node) => node.classList.remove("drop-before", "drop-after"));
  const item = event.target.closest(".preset-batch-item");
  if (!item || item.dataset.name === dragPresetName) return;
  const rect = item.getBoundingClientRect();
  dropInsertBefore = event.clientY < rect.top + rect.height / 2;
  item.classList.add(dropInsertBefore ? "drop-before" : "drop-after");
});

byId("preset-batch-list").addEventListener("drop", (event) => {
  if (!dragPresetName) return;
  event.preventDefault();
  const item = event.target.closest(".preset-batch-item");
  const name = dragPresetName;
  dragPresetName = null;
  if (!item || item.dataset.name === name) return;
  reorderPreset(name, item.dataset.name, dropInsertBefore);
});

byId("preset-batch-list").addEventListener("dragend", () => {
  dragPresetName = null;
  byId("preset-batch-list").querySelectorAll(".dragging, .drop-before, .drop-after")
    .forEach((item) => item.classList.remove("dragging", "drop-before", "drop-after"));
});
byId("auto-enrollment").addEventListener("submit", enrollAutomatically);
byId("show-create-bank").addEventListener("click", () => { byId("create-bank-form").hidden = !byId("create-bank-form").hidden; });
byId("create-bank-form").addEventListener("submit", createBank);
byId("history-refresh").addEventListener("click", () => loadHistory());
byId("history-search").addEventListener("input", renderHistory);
byId("history-method").addEventListener("change", renderHistory);
byId("history-select-all").addEventListener("change", toggleAllHistorySelection);
byId("history-delete-selected").addEventListener("click", () => deleteHistoryRecords([...selectedHistoryIds]));
byId("history-list").addEventListener("change", (event) => {
  if (event.target.matches(".history-row-check")) toggleHistorySelection(event.target.value, event.target.checked);
});
byId("history-list").addEventListener("click", (event) => {
  const openButton = event.target.closest("[data-history-open]");
  if (openButton) openHistoryRecord(openButton.dataset.historyOpen);
  const deleteButton = event.target.closest("[data-history-delete]");
  if (deleteButton) deleteHistoryRecords([deleteButton.dataset.historyDelete]);
});
byId("history-detail").addEventListener("click", (event) => {
  if (event.target.id === "history-detail-close") {
    openedHistoryId = null;
    byId("history-detail").hidden = true;
  }
});

renderInventory();
loadChallenges();
reloadPresets();
loadHistory();
