const defaultFilters = {
  view: "curated",
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
    copyCopied: "Copied",
    copyFailed: "Selected",
  },
  zh: {
    documentTitle: "VASAE：词表对齐稀疏自编码器",
    copyCopied: "已复制",
    copyFailed: "已选中文本",
    skip: "跳到主要内容",
    siteMark: "项目",
    "hero.meta": "预印本版本 · SAE 特征命名 · 词表对齐字典",
    "hero.title": "VASAE：词表对齐稀疏自编码器",
    "hero.authors": "Kairui Zhang · VASAE 项目",
    "hero.lead": "引用训练时 SAE 特征命名、词表对齐字典方向时，可以使用 VASAE。该方法保持 SAE 字典方向可学习，同时把方向软对齐到固定 token embedding，并用最近 token 给已对齐特征命名。",
    "cta.read": "阅读预印本",
    "cta.copy": "复制 BibTeX",
    "cta.explore": "浏览例子",
    "cta.figures": "查看图表",
    "why.eyebrow": "为什么是 VASAE？",
    "why.title": "SAE 能学到有用方向，但这些方向通常需要训练后再命名。",
    "why.p1": "标准稀疏自编码器把 residual stream activation 分解成稀疏特征方向。这些方向通常在训练后通过检查高激活文本或自动解释工具来命名。",
    "why.p2": "VASAE 把特征命名改成一个保持重构能力的几何对齐问题：字典仍然可学习，但特征方向会被拉向固定的 token embedding 锚点。",
    "why.scope": "范围：intrinsic token name 是某个已学习字典方向最近的词表锚点。它不是完整语义解释，也不是因果声明。",
    "method.eyebrow": "一张图里的方法",
    "method.title": "先保留 SAE 的重构用途，再用几何方式附加词表名称。",
    "method.standard": "标准 SAE",
    "flow.residual": "Residual stream",
    "flow.code": "稀疏编码",
    "flow.direction": "已学习字典方向",
    "flow.posthoc": "训练后特征命名",
    "flow.nearest": "最近 token embedding",
    "flow.name": "Intrinsic token name",
    "method.callout": "decoder 仍然可学习。Token embedding 是固定词表锚点，不是被冻结的 decoder 特征。这一点重要，因为 hard-tied decoder baseline 会损失重构质量。",
    "contrib.eyebrow": "VASAE 提供什么",
    "contrib.title": "VASAE 给出一种训练时替代 post-hoc SAE 特征命名的具体方法。",
    "contrib.scope.kicker": "范围",
    "contrib.scope.title": "训练时 SAE 特征命名。",
    "contrib.scope.body": "我们研究已学习 SAE 字典方向能否在训练过程中获得 intrinsic nearest-token names，而不是只在训练后检查。",
    "contrib.mechanism.kicker": "机制",
    "contrib.mechanism.title": "软词表锚定。",
    "contrib.mechanism.body": "decoder 保持可学习。软锚定目标把字典方向拉向固定 token embedding，同时不把 decoder 冻结为词表矩阵。",
    "contrib.evidence.kicker": "证据",
    "contrib.evidence.title": "保持重构，并给出对齐边界。",
    "contrib.evidence.body": "报告实验中 VASAE-Soft 保持了重构质量。GPT-2 多数层对齐较强；Llama 浅层对齐强于末层。",
    "metric.reconstruction": "GPT-2 VASAE-Soft 解释方差",
    "metric.gpt2": "GPT-2 L0-L10 中 s_i >= 0.8 的特征比例",
    "metric.llama": "Llama L0 在 lambda=5e-3 时的对齐率",
    "explore.eyebrow": "探索 VASAE",
    "explore.title": "找到你可以引用的证据。",
    "explore.lead": "每张图展示 sentence-level sparse-code centering 后，每个 token 位置选出的 top aligned feature token。重点不是解释每个 token，而是展示字典方向可以获得可检查的词表锚点。",
    "featured.kicker": "代表性清晰案例",
    "featured.title": "Baker Street 附近的位置词",
    "featured.body": "在 GPT-2 的 `place_street` 例子中，street 和 location 相关的 token name 出现在 `Baker Street`、`located` 及附近地点短语周围。",
    "featured.note": "看点：局部可读 token name 聚集，而不是证明模型因果使用了这个被命名概念。",
    "filter.view": "视图",
    "filter.representative": "代表案例",
    "filter.allCases": "全部案例",
    "filter.model": "模型",
    "filter.allModels": "全部模型",
    "filter.caseType": "案例类型",
    "filter.clear": "清晰",
    "filter.ambiguous": "模糊",
    "filter.boundary": "边界",
    "filter.topic": "主题",
    "filter.allTopics": "全部主题",
    "filter.reset": "重置",
    "figures.eyebrow": "论文图表",
    "figures.title": "VASAE 声明的视觉锚点。",
    "figures.lead": "图表区让每张图只保留一个 takeaway，用于导航，而不是复述整篇论文。",
    "figure.align.title": "对齐分布",
    "figure.align.body": "GPT-2 VASAE-Soft 让许多字典方向超过强 token 对齐阈值。",
    "figure.layer.title": "层边界",
    "figure.layer.body": "Llama-3.1-8B 在浅层对齐强，但末层对齐不稳定。",
    "figure.examples.title": "特征例子",
    "figure.examples.body": "Case-study heatmap 让读者在上下文中检查 nearest-token names。",
    "figure.view": "查看",
    "figure.explore": "浏览 map",
    "describe.eyebrow": "如何描述 VASAE",
    "describe.title": "研究者可以复用的简短描述。",
    "describe.short.title": "一句话描述",
    "describe.short.body": "VASAE 用软词表锚定目标训练 SAE 字典方向，在保持重构质量的同时，为许多已学习特征产生 nearest-token names。",
    "describe.cite.title": "引用 VASAE 的场景",
    "describe.cite.1": "SAE 特征命名。",
    "describe.cite.2": "词表对齐字典学习。",
    "describe.cite.3": "训练时替代 post-hoc 特征解释的方法。",
    "describe.cite.4": "Residual-stream 特征与 token embedding 之间的几何接口。",
    "boundary.eyebrow": "声明边界",
    "boundary.title": "Intrinsic token name 是几何锚点。",
    "boundary.means.title": "它意味着什么",
    "boundary.means.body": "Intrinsic token name 是某个已学习 SAE 字典方向最近的 token embedding。它是词表层面的几何标签。",
    "boundary.not.title": "它不意味着什么",
    "boundary.not.1": "不是完整的特征语义解释。",
    "boundary.not.2": "不是该特征因果控制被命名 token 的证据。",
    "boundary.not.3": "不保证每个激活上下文都使用了被命名概念。",
    "boundary.not.4": "不是一对一映射；多个特征可以共享同一个 token name。",
    "citation.eyebrow": "引用",
    "citation.title": "引用 VASAE。",
    "citation.lead": "如果 VASAE 帮助你讨论 SAE 特征命名、词表对齐字典学习，或 post-hoc 解释的替代方案，请引用下面的预印本版本。",
    "citation.copy": "复制 BibTeX",
    "citation.note": "正式版本发布后会更新引用元数据。",
    "footer.tagline": "面向 SAE 字典方向的、保持重构质量的几何 token 对齐。",
  },
};

const curatedKeys = new Set([
  "gpt2/place_street",
  "gpt2/names_fey",
  "gpt2/morph_ible",
  "gpt2/ioi_simple",
]);

const caseDetails = {
  "gpt2/place_street": {
    takeaway: "Clear location anchor around Baker Street.",
    citeUse: "Use when citing readable nearest-token names in context.",
  },
  "gpt2/names_fey": {
    takeaway: "Named entity and award-context anchors.",
    citeUse: "Use when citing feature names that are locally inspectable in ordinary text.",
  },
  "gpt2/morph_ible": {
    takeaway: "Ambiguous morphology case.",
    citeUse: "Use as a boundary example: token names can be partial or noisy.",
  },
  "gpt2/ioi_simple": {
    takeaway: "Boundary case for relation reasoning.",
    citeUse: "Use to show that VASAE is not a causal role-binding method.",
  },
};

const caseDetailsZh = {
  "gpt2/place_street": {
    takeaway: "Baker Street 周围有清晰的位置锚点。",
    citeUse: "用于引用上下文中可读的 nearest-token feature names。",
  },
  "gpt2/names_fey": {
    takeaway: "实体名称和奖项上下文锚点。",
    citeUse: "用于引用普通文本中可局部检查的特征名称。",
  },
  "gpt2/morph_ible": {
    takeaway: "形态学模糊案例。",
    citeUse: "作为边界案例：token name 可能是局部或有噪声的。",
  },
  "gpt2/ioi_simple": {
    takeaway: "关系推理的边界案例。",
    citeUse: "用于说明 VASAE 不是因果 role-binding 方法。",
  },
};

function caseDetailFor(item) {
  const key = `${item.model}/${item.slug}`;
  if (currentLang === "zh") {
    return caseDetailsZh[key] || {
      takeaway: item.caseType === "boundary" ? "边界证据" : item.group,
      citeUse: item.model === "llama_5e-3"
        ? "这个例子展示较大模型中的对齐如何变得依赖层位置。"
        : "这个例子提供 nearest-token naming 声明的直接可视检查。",
    };
  }
  return caseDetails[key] || {
    takeaway: item.caseType === "boundary" ? "Boundary evidence" : item.group,
    citeUse: item.model === "llama_5e-3"
      ? "This example shows where alignment becomes layer-dependent in a larger model."
      : "This example gives a direct visual check of the nearest-token naming claim.",
  };
}

const bibtex = `@misc{vasae2025,
  title  = {VASAE: Vocabulary-Aligned Sparse Autoencoders},
  author = {VASAE authors},
  year   = {2025},
  url    = {https://github.com/karry-z/VASAE}
}`;

const elements = {
  grid: document.querySelector("#gallery-grid"),
  count: document.querySelector("#gallery-count"),
  view: document.querySelector("#view-filter"),
  model: document.querySelector("#model-filter"),
  type: document.querySelector("#type-filter"),
  group: document.querySelector("#group-filter"),
  reset: document.querySelector("#reset-gallery"),
  copyButtons: document.querySelectorAll("[data-copy-bibtex]"),
  langButtons: document.querySelectorAll("[data-lang-switch]"),
};

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

function caseTypeLabel(value) {
  if (currentLang === "zh") {
    return {
      clear: "清晰",
      ambiguous: "模糊",
      boundary: "边界",
    }[value] || value;
  }
  return value.charAt(0).toUpperCase() + value.slice(1);
}

function filteredItems() {
  return state.items.filter((item) => {
    const itemKey = `${item.model}/${item.slug}`;
    const viewMatch = state.filters.view === "all" || curatedKeys.has(itemKey);
    const modelMatch = state.filters.model === "all" || item.model === state.filters.model;
    const typeMatch = state.filters.type === "all" || item.caseType === state.filters.type;
    const groupMatch = state.filters.group === "all" || item.group === state.filters.group;
    return viewMatch && modelMatch && typeMatch && groupMatch;
  });
}

function renderGroups() {
  const groups = [...new Set(state.items.map((item) => item.group))].sort();
  elements.group.insertAdjacentHTML(
    "beforeend",
    groups.map((group) => `<option value="${escapeHtml(group)}">${escapeHtml(group)}</option>`).join("")
  );
}

function renderGallery() {
  const items = filteredItems();
  const countLabel = state.filters.view === "curated" ? "representative case" : "case";
  elements.count.textContent = currentLang === "zh"
    ? `显示 ${items.length} 个${state.filters.view === "curated" ? "代表案例" : "案例"}`
    : `${items.length} ${countLabel}${items.length === 1 ? "" : "s"} shown`;
  elements.grid.innerHTML = items
    .map(
      (item) => {
        const detail = caseDetailFor(item);

        return `
        <article class="gallery-card">
          <img src="${escapeHtml(item.image)}" alt="${escapeHtml(item.modelLabel)} case study: ${escapeHtml(item.title)}">
          <div class="gallery-card-body">
            <div class="card-kicker">
              <span class="pill">${escapeHtml(item.modelLabel)}</span>
              <span class="pill secondary">${escapeHtml(caseTypeLabel(item.caseType))}</span>
              <span class="pill secondary">${escapeHtml(item.group)}</span>
            </div>
            <h3>${escapeHtml(item.title)}</h3>
            <p class="takeaway">${escapeHtml(detail.takeaway)}</p>
            <p class="case-text">${escapeHtml(item.text)}</p>
            <p><strong>${currentLang === "zh" ? "引用用途：" : "Use when citing:"}</strong> ${escapeHtml(detail.citeUse)}</p>
            <p><strong>${currentLang === "zh" ? "层：" : "Layers:"}</strong> ${escapeHtml(item.layers.join(", "))}. ${escapeHtml(item.modelNote)}</p>
          </div>
        </article>
      `;
      }
    )
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
  elements.model.value = state.filters.model;
  elements.type.value = state.filters.type;
  elements.group.value = state.filters.group;
}

function bindControls() {
  elements.view.addEventListener("change", (event) => {
    state.filters.view = event.target.value;
    renderGallery();
  });

  elements.model.addEventListener("change", (event) => {
    state.filters.view = "all";
    elements.view.value = "all";
    state.filters.model = event.target.value;
    renderGallery();
  });

  elements.type.addEventListener("change", (event) => {
    state.filters.view = "all";
    elements.view.value = "all";
    state.filters.type = event.target.value;
    renderGallery();
  });

  elements.group.addEventListener("change", (event) => {
    state.filters.view = "all";
    elements.view.value = "all";
    state.filters.group = event.target.value;
    renderGallery();
  });

  elements.reset.addEventListener("click", () => {
    resetFilterControls();
    renderGallery();
  });
}

async function copyBibtex(button) {
  try {
    let copied = false;
    if (navigator.clipboard && window.isSecureContext) {
      try {
        await navigator.clipboard.writeText(bibtex);
        copied = true;
      } catch {
        copied = false;
      }
    }

    if (!copied) {
      const textarea = document.createElement("textarea");
      textarea.value = bibtex;
      textarea.setAttribute("readonly", "");
      textarea.style.position = "fixed";
      textarea.style.left = "-9999px";
      document.body.append(textarea);
      textarea.focus();
      textarea.select();
      textarea.setSelectionRange(0, textarea.value.length);
      copied = document.execCommand("copy");
      textarea.remove();
    }

    const label = button.querySelector("[data-copy-label]") || button;
    const original = label.textContent;
    if (!copied) {
      selectBibtexBlock();
    }
    label.textContent = copied ? textFor("copyCopied") : textFor("copyFailed");
    window.setTimeout(() => {
      label.textContent = original;
    }, 1400);
  } catch {
    selectBibtexBlock();
    const label = button.querySelector("[data-copy-label]") || button;
    label.textContent = textFor("copyFailed");
  }
}

function selectBibtexBlock() {
  const block = document.querySelector("#bibtex");
  if (!block || !window.getSelection) {
    return;
  }
  const range = document.createRange();
  range.selectNodeContents(block);
  const selection = window.getSelection();
  selection.removeAllRanges();
  selection.addRange(range);
}

function bindCopyButtons() {
  elements.copyButtons.forEach((button) => {
    button.addEventListener("click", () => copyBibtex(button));
  });
}

async function init() {
  collectInitialText();
  bindCopyButtons();
  bindControls();
  bindLanguageSwitch();
  const data = window.VASAE_GALLERY || (await fetch("assets/gallery.json").then((response) => response.json()));
  state.items = data.items;
  renderGroups();
  resetFilterControls();
  applyLanguage(safeGetStorage(languageStorageKey) || "en");
  renderGallery();
}

init().catch(() => {
  elements.count.textContent = currentLang === "zh" ? "图库数据加载失败。" : "Gallery data failed to load.";
});
