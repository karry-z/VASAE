const defaultFilters = {
  view: "curated",
  evidence: "all",
  model: "all",
  type: "all",
  group: "all",
};

const state = {
  items: [],
  filters: { ...defaultFilters },
};

const languageStorageKey = "vasae.lang";
const initialText = {};
let currentLang = "en";

const translations = {
  en: {
    documentTitle: "VASAE: Vocabulary-Aligned Sparse Autoencoders",
  },
  zh: {
    documentTitle: "VASAE：词表对齐稀疏自编码器",
    skip: "跳到主要内容",
    siteMark: "项目",
    "hero.meta": "预印本版本 · SAE 特征命名 · 词表对齐字典",
    "hero.title": "VASAE：词表对齐稀疏自编码器",
    "hero.authors": "Kairui Zhang, Ziwen Yu, Zahraa S. Abdallah, Martha Lewis",
    "hero.status": "预印本版本。正式版本发布后会更新引用元数据。",
    "hero.lead": "当你讨论通过词表对齐字典方向进行训练时 SAE 特征命名时，可以引用 VASAE。",
    "hero.caveat": "Intrinsic token name 是几何词表锚点，不是因果解释，也不是完整语义解释。",
    "cta.pending": "待发布",
    "cta.gallery": "检查证据",
    "cta.figures": "查看图表",
    "cta.cite": "引用状态",
    "why.eyebrow": "问题 / 为什么是 VASAE",
    "why.title": "SAE 特征名通常在训练之后才分配。",
    "why.lead": "VASAE 研究能否把一部分特征命名移入训练过程，同时不破坏重构质量。",
    "why.problem.kicker": "问题",
    "why.problem.title": "Post-hoc 命名有用，但比较间接。",
    "why.problem.body": "标准 SAE 先学习稀疏字典方向。研究者随后检查高激活上下文，或使用自动解释工具，给这些方向命名。",
    "why.method.kicker": "方法",
    "why.method.title": "把词表作为几何锚点集合。",
    "why.method.body": "VASAE 保持字典方向可学习，同时加入软目标，把它们拉向固定的 token embedding。",
    "why.scope.kicker": "范围",
    "why.scope.title": "名称是锚点，不是解释。",
    "why.scope.body": "Token name 是已学习方向最近的 embedding 锚点。它是用于检查证据的紧凑标签，不是完整特征解释。",
    "method.eyebrow": "方法快照",
    "method.title": "标准 SAE 的命名是 post-hoc；VASAE 在训练时提供 nearest-token names。",
    "method.standard": "标准 SAE",
    "flow.residual": "Residual stream",
    "flow.code": "稀疏编码",
    "flow.direction": "已学习字典方向",
    "flow.posthoc": "训练后检查",
    "flow.manual": "人工 / LLM 特征名",
    "flow.anchor": "软锚定",
    "flow.nearest": "最近 token embedding",
    "flow.name": "Intrinsic token name",
    "method.callout": "decoder 仍然可学习；token embedding 是固定锚点，不是被冻结的 decoder features。",
    "results.eyebrow": "关键结果",
    "results.title": "有用的声明是：保持重构质量的对齐，同时清楚展示边界。",
    "results.lead": "下面每个数字都给出模型、层范围、指标和 caveat，便于检查证据。",
    "results.recon.kicker": "保持重构",
    "results.recon.body": "GPT-2 VASAE-Soft 的解释方差为 0.965 ± 0.054，与 Plain SAE 的 0.965 ± 0.053 在层聚合结果中匹配。",
    "results.gpt2.kicker": "GPT-2 对齐强",
    "results.gpt2.body": "GPT-2-small L0-L10 的特征超过强对齐阈值 s_i >= 0.8；L11 下降到 68.5%。",
    "results.llama.kicker": "Llama 对齐依赖层",
    "results.llama.body": "Llama-3.1-8B 在 lambda=5e-3 时浅层对齐强，末层是边界案例。",
    "table.model": "模型",
    "table.layers": "层",
    "table.metric": "指标",
    "table.result": "结果",
    "table.caveat": "Caveat",
    "table.gpt2recon.layers": "层聚合",
    "table.gpt2recon.metric": "解释方差",
    "table.gpt2recon.caveat": "这支持 soft anchoring，不支持 hard tying。",
    "table.gpt2align.metric": "特征对齐，s_i >= 0.8",
    "table.gpt2align.caveat": "Token coverage 更低，因为多个 feature 可以共享同一个 token 锚点。",
    "table.gpt2l11.metric": "特征对齐，s_i >= 0.8",
    "table.gpt2l11.caveat": "最终层对齐弱于 L0-L10。",
    "table.llama.metric": "lambda=5e-3 下的特征对齐",
    "table.llama.caveat": "这是依赖层位置的证据；L31 是边界条件。",
    "gallery.eyebrow": "特征证据库",
    "gallery.title": "检查 feature-level 证据和上下文 heatmap。",
    "gallery.lead": "Feature card 在现有 artifact 提供 feature id、aligned token、相似度、相关性和 activation 时显示这些字段。Heatmap card 会明确标注，因为当前 case-study manifest 不保存 selected feature id。",
    "filter.view": "视图",
    "filter.representative": "代表证据",
    "filter.allCases": "全部案例",
    "filter.evidence": "证据类型",
    "filter.allEvidence": "全部证据",
    "filter.featureCards": "特征证据卡",
    "filter.heatmaps": "上下文 heatmap",
    "filter.model": "模型",
    "filter.allModels": "全部模型",
    "filter.caseType": "案例类型",
    "filter.clear": "清晰",
    "filter.ambiguous": "模糊",
    "filter.boundary": "边界",
    "filter.topic": "主题",
    "filter.allTopics": "全部主题",
    "filter.reset": "重置",
    "figures.eyebrow": "图表锚点",
    "figures.title": "三张图说明主证据和边界。",
    "figure.align.title": "GPT-2 对齐分布",
    "figure.takeaway": "Takeaway",
    "figure.align.takeaway": "VASAE-Soft 把许多字典方向推近 token embedding，而 Plain SAE 的对齐较弱。",
    "figure.how": "如何读图",
    "figure.align.how": "s_i 是字典方向与任意 token embedding 的最大 cosine similarity。虚线标记 s_i >= 0.8。",
    "figure.why": "为什么重要",
    "figure.align.why": "这支持在 SAE 训练中引入词表对齐，同时不冻结 decoder。",
    "figure.layer.title": "Llama 层边界",
    "figure.layer.takeaway": "Llama 对齐依赖层位置：浅层强，中层部分，末层弱。",
    "figure.layer.how": "每个箱线图总结一层 feature 的 nearest-token alignment scores。虚线标记 s_i = 0.8。",
    "figure.layer.why": "这是边界条件，不只是成功结果。",
    "figure.examples.title": "特征例子 / heatmap",
    "figure.examples.takeaway": "上下文 heatmap 让读者检查输入 token 周围的 nearest-token names。",
    "figure.examples.how": "行是层，列是 token 位置，单元格文字是经过句内 sparse-code centering 后选出的 aligned token name。",
    "figure.examples.why": "可读的局部聚类支持命名接口，同时也展示噪声和边界案例。",
    "figure.view": "查看",
    "figure.explore": "浏览证据库",
    "boundary.eyebrow": "声明边界",
    "boundary.title": "把 token names 读作锚点，不要读作解释。",
    "boundary.means.kicker": "含义",
    "boundary.means.title": "已学习 SAE 字典方向最近的 token embedding。",
    "boundary.means.body": "Intrinsic token name 是词表层面的几何标签。它帮助定位和比较 feature directions，但没有完成解释。",
    "boundary.not.kicker": "不意味着",
    "boundary.not.1": "该 feature 已有完整语义解释。",
    "boundary.not.2": "模型因果使用该 token 或概念。",
    "boundary.not.3": "干预该 feature 会控制被命名的 token。",
    "boundary.not.4": "每个 feature 都有唯一的一对一 token name。",
    "citation.eyebrow": "引用工具",
    "citation.title": "引用元数据待更新。",
    "citation.lead": "正式版本发布后会添加 Preprint/BibTeX。",
    "citation.status.kicker": "状态",
    "citation.status.title": "预印本版本。",
    "citation.status.body": "正式版本发布后会更新引用元数据。这个页面暂不放 BibTeX 区块。",
    "citation.short.kicker": "可复用句子",
    "citation.short.title": "一句话描述",
    "citation.short.body": "VASAE 用软词表锚定目标训练 SAE 字典方向，在保持重构质量的同时，为许多已学习特征生成 nearest-token names。",
    "citation.when.kicker": "何时引用",
    "citation.when.title": "在讨论训练时特征命名时引用 VASAE。",
    "citation.when.body": "当讨论 SAE 特征命名、词表对齐字典学习、residual-stream 特征与 token embedding 的几何接口，或 post-hoc 解释的训练时替代方案时，可以引用 VASAE。",
    "footer.tagline": "面向 SAE 字典方向的、保持重构质量的几何 token 对齐。",
    "footer.status": "Code/data status：项目 artifact 正在为正式版本整理。",
  },
};

const uiLabels = {
  en: {
    missing: "Not available in current artifact",
    shown: "shown",
    representative: "representative evidence item",
    evidence: "evidence item",
    source: "Source",
    context: "Context",
    why: "Why this is evidence",
    boundary: "Boundary",
    meta: {
      intrinsicToken: "Intrinsic token name",
      model: "Model",
      layer: "Layer",
      featureId: "Feature ID",
      alignmentScore: "Alignment score s_i",
      rho: "rho_in / rho_out",
      selectedActivation: "Selected activation z",
      caseType: "Case type",
      topic: "Topic",
      evidenceType: "Evidence type",
    },
    evidenceType: {
      "feature-card": "feature evidence card",
      "heatmap-case": "case-study heatmap",
    },
    caseType: {
      clear: "Clear",
      ambiguous: "Ambiguous",
      boundary: "Boundary",
    },
  },
  zh: {
    missing: "当前 artifact 不可用",
    shown: "已显示",
    representative: "代表证据",
    evidence: "证据",
    source: "来源",
    context: "上下文",
    why: "为什么这是证据",
    boundary: "边界",
    meta: {
      intrinsicToken: "Intrinsic token name",
      model: "模型",
      layer: "层",
      featureId: "Feature ID",
      alignmentScore: "对齐分数 s_i",
      rho: "rho_in / rho_out",
      selectedActivation: "Selected activation z",
      caseType: "案例类型",
      topic: "主题",
      evidenceType: "证据类型",
    },
    evidenceType: {
      "feature-card": "特征证据卡",
      "heatmap-case": "case-study heatmap",
    },
    caseType: {
      clear: "清晰",
      ambiguous: "模糊",
      boundary: "边界",
    },
  },
};

const curatedIds = new Set([
  "llama-l0-townsend",
  "llama-l0-fey",
  "llama-l0-nicole",
  "gpt2-l6-professional",
  "gpt2-l6-english",
  "gpt2-l6-response",
  "gpt2-l10-comes",
  "gpt2-l11-prog",
  "llama-l15-int",
  "llama-l31-no-aligned",
  "gpt2-place-street-heatmap",
  "gpt2-morph-ible-heatmap",
  "gpt2-factual-einstein-heatmap",
  "gpt2-ioi-simple-heatmap",
]);

const elements = {
  grid: document.querySelector("#gallery-grid"),
  count: document.querySelector("#gallery-count"),
  view: document.querySelector("#view-filter"),
  evidence: document.querySelector("#evidence-filter"),
  model: document.querySelector("#model-filter"),
  type: document.querySelector("#type-filter"),
  group: document.querySelector("#group-filter"),
  reset: document.querySelector("#reset-gallery"),
  langButtons: document.querySelectorAll("[data-lang-switch]"),
};

function labels() {
  return uiLabels[currentLang] || uiLabels.en;
}

function safeGetStorage(key) {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

function safeSetStorage(key, value) {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    return false;
  }
  return true;
}

function collectInitialText() {
  document.querySelectorAll("[data-i18n]").forEach((node) => {
    const key = node.getAttribute("data-i18n");
    initialText[key] = node.textContent;
  });
}

function textFor(key, lang = currentLang) {
  return (translations[lang] && translations[lang][key]) || initialText[key] || "";
}

function applyLanguage(lang) {
  currentLang = lang === "zh" ? "zh" : "en";
  document.documentElement.lang = currentLang === "zh" ? "zh-CN" : "en";
  document.title = textFor("documentTitle", currentLang);
  safeSetStorage(languageStorageKey, currentLang);

  document.querySelectorAll("[data-i18n]").forEach((node) => {
    const key = node.getAttribute("data-i18n");
    const text = textFor(key, currentLang);
    if (text) {
      node.textContent = text;
    }
  });

  elements.langButtons.forEach((button) => {
    const active = button.getAttribute("data-lang-switch") === currentLang;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-pressed", active ? "true" : "false");
  });

  if (state.items.length) {
    renderGroups();
    renderGallery();
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function localized(item, field) {
  if (currentLang === "zh" && item[`${field}Zh`]) {
    return item[`${field}Zh`];
  }
  return item[field];
}

function hasValue(value) {
  return value !== null && value !== undefined && value !== "";
}

function displayValue(value) {
  return hasValue(value) ? value : labels().missing;
}

function formatNumber(value) {
  if (!hasValue(value)) {
    return labels().missing;
  }
  if (typeof value !== "number") {
    return value;
  }
  return Number.isInteger(value) ? String(value) : value.toFixed(4);
}

function formatLayer(item) {
  if (item.layerLabel) {
    return item.layerLabel;
  }
  if (Array.isArray(item.layers) && item.layers.length) {
    return item.layers.map((layer) => `L${layer}`).join(", ");
  }
  return labels().missing;
}

function caseTypeLabel(value) {
  return labels().caseType[value] || value;
}

function evidenceTypeLabel(value) {
  return labels().evidenceType[value] || value;
}

function itemId(item) {
  return item.id || `${item.model}/${item.slug}`;
}

function filteredItems() {
  return state.items.filter((item) => {
    const viewMatch = state.filters.view === "all" || item.curated || curatedIds.has(itemId(item));
    const evidenceMatch = state.filters.evidence === "all" || item.evidenceType === state.filters.evidence;
    const modelMatch = state.filters.model === "all" || item.model === state.filters.model;
    const typeMatch = state.filters.type === "all" || item.caseType === state.filters.type;
    const groupMatch = state.filters.group === "all" || item.group === state.filters.group;
    return viewMatch && evidenceMatch && modelMatch && typeMatch && groupMatch;
  });
}

function renderGroups() {
  const currentValue = elements.group.value || "all";
  while (elements.group.options.length > 1) {
    elements.group.remove(1);
  }
  const groups = [...new Set(state.items.map((item) => item.group).filter(Boolean))].sort();
  elements.group.insertAdjacentHTML(
    "beforeend",
    groups.map((group) => `<option value="${escapeHtml(group)}">${escapeHtml(group)}</option>`).join("")
  );
  elements.group.value = groups.includes(currentValue) ? currentValue : "all";
  state.filters.group = elements.group.value;
}

function renderMetaRows(item) {
  const rowData = [
    [labels().meta.intrinsicToken, displayValue(item.intrinsicToken)],
    [labels().meta.model, item.modelLabel],
    [labels().meta.layer, formatLayer(item)],
    [labels().meta.featureId, displayValue(item.featureId)],
    [labels().meta.alignmentScore, formatNumber(item.alignmentScore)],
    [labels().meta.rho, hasValue(item.rhoIn) || hasValue(item.rhoOut) ? `${formatNumber(item.rhoIn)} / ${formatNumber(item.rhoOut)}` : labels().missing],
    [labels().meta.selectedActivation, formatNumber(item.selectedActivation)],
    [labels().meta.caseType, caseTypeLabel(item.caseType)],
    [labels().meta.topic, displayValue(item.group)],
    [labels().meta.evidenceType, evidenceTypeLabel(item.evidenceType)],
  ];

  return `
    <dl class="metadata-grid">
      ${rowData
        .map(([term, value]) => `
          <div>
            <dt>${escapeHtml(term)}</dt>
            <dd>${escapeHtml(value)}</dd>
          </div>
        `)
        .join("")}
    </dl>
  `;
}

function renderGallery() {
  const items = filteredItems();
  const noun = state.filters.view === "curated" ? labels().representative : labels().evidence;
  elements.count.textContent = currentLang === "zh"
    ? `${labels().shown} ${items.length} 个${noun}`
    : `${items.length} ${noun}${items.length === 1 ? "" : "s"} ${labels().shown}`;

  elements.grid.innerHTML = items
    .map((item) => {
      const title = localized(item, "title") || item.slug;
      const context = localized(item, "context") || item.text || labels().missing;
      const whyEvidence = localized(item, "whyEvidence") || labels().missing;
      const boundary = localized(item, "boundary") || labels().missing;
      const altText = `${item.modelLabel} ${evidenceTypeLabel(item.evidenceType)}: ${title}`;
      const imageMarkup = item.image
        ? `<img src="${escapeHtml(item.image)}" alt="${escapeHtml(altText)}">`
        : `<div class="image-placeholder">${escapeHtml(labels().missing)}</div>`;

      return `
        <article class="gallery-card ${item.evidenceType === "feature-card" ? "feature-evidence" : "heatmap-evidence"}">
          ${imageMarkup}
          <div class="gallery-card-body">
            <div class="card-kicker">
              <span class="pill">${escapeHtml(evidenceTypeLabel(item.evidenceType))}</span>
              <span class="pill secondary">${escapeHtml(item.modelLabel)}</span>
              <span class="pill secondary">${escapeHtml(caseTypeLabel(item.caseType))}</span>
            </div>
            <h3>${escapeHtml(title)}</h3>
            ${renderMetaRows(item)}
            <div class="case-details">
              <div>
                <dt>${escapeHtml(labels().context)}</dt>
                <dd>${escapeHtml(context)}</dd>
              </div>
              <div>
                <dt>${escapeHtml(labels().why)}</dt>
                <dd>${escapeHtml(whyEvidence)}</dd>
              </div>
              <div>
                <dt>${escapeHtml(labels().boundary)}</dt>
                <dd>${escapeHtml(boundary)}</dd>
              </div>
              <div>
                <dt>${escapeHtml(labels().source)}</dt>
                <dd>${escapeHtml(displayValue(item.source))}</dd>
              </div>
            </div>
          </div>
        </article>
      `;
    })
    .join("");
}

function bindLanguageSwitch() {
  elements.langButtons.forEach((button) => {
    button.addEventListener("click", () => {
      applyLanguage(button.getAttribute("data-lang-switch"));
    });
  });
}

function resetFilterControls() {
  state.filters = { ...defaultFilters };
  elements.view.value = state.filters.view;
  elements.evidence.value = state.filters.evidence;
  elements.model.value = state.filters.model;
  elements.type.value = state.filters.type;
  elements.group.value = state.filters.group;
}

function promoteToAllView() {
  state.filters.view = "all";
  elements.view.value = "all";
}

function bindControls() {
  elements.view.addEventListener("change", (event) => {
    state.filters.view = event.target.value;
    renderGallery();
  });

  elements.evidence.addEventListener("change", (event) => {
    promoteToAllView();
    state.filters.evidence = event.target.value;
    renderGallery();
  });

  elements.model.addEventListener("change", (event) => {
    promoteToAllView();
    state.filters.model = event.target.value;
    renderGallery();
  });

  elements.type.addEventListener("change", (event) => {
    promoteToAllView();
    state.filters.type = event.target.value;
    renderGallery();
  });

  elements.group.addEventListener("change", (event) => {
    promoteToAllView();
    state.filters.group = event.target.value;
    renderGallery();
  });

  elements.reset.addEventListener("click", () => {
    resetFilterControls();
    renderGallery();
  });
}

async function init() {
  collectInitialText();
  bindControls();
  bindLanguageSwitch();
  const data = window.VASAE_GALLERY || (await fetch("assets/gallery.json").then((response) => response.json()));
  state.items = data.items || [];
  renderGroups();
  resetFilterControls();
  applyLanguage(safeGetStorage(languageStorageKey) || "en");
  renderGallery();
}

init().catch(() => {
  elements.count.textContent = currentLang === "zh" ? "图库数据加载失败。" : "Gallery data failed to load.";
});
