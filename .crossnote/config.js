({
  katexConfig: {
  "macros": {}
},
  
  mathjaxConfig: {
  "tex": {
    macros: {
      // notations
      a:"{\\mathbf{a}}", // 一层feature 与 feature token 的相似度
      A: "{\\mathbf{A}}", // feature 与 feature token 的相似度矩阵
      da: "{d_{\\text{aligned}}}", // aligned feature dim
      df: "{\\mathbf{d}}", // feature direction vector
      dmodel: "{d_{\\mathrm{model}}}", // model dim
      dsparse: "{d_{\\mathrm{sparse}}}", // sparse code dim
      dvocab: "{|\\mathcal{V}|}", // vocab size
      dvec: "{d}", // model dim
      E: "{\\mathbf{E}}", // aligned feature token embedding
      H: "{\\mathbf{H}}", // hidden state / embedding 表示
      layeri: "{\\ell}", // layer index
      loss: "{\\mathcal{L}}",
      lossrecon: "{\\mathcal{L}_{\\mathrm{rec}}}", // reconst loss
      losssparse: "{\\mathcal{L}_{\\mathrm{sparse}}}", // sparse loss
      layerL: "{L}", // number of layers
      R: "{\\mathbb{R}}", // 实数域
      t: "{\\mathbf{t}}", // 对齐 token id 向量
      timeT: "{T}", // sequence length
      U: "{\\mathbf{U}}", // feature token 与输入 token 相似度矩阵
      vocab: "{\\mathcal{V}}", // vocab space
      WD: "{\\mathbf{W_\\mathcal{D}}}", // decoder 字典
      WDnorm: "{{\\mathbf{\\hat{W}_\\mathcal{D}}}}", // 归一化 decoder 矩阵
      WE: "{\\mathbf{W_E}}", // token embedding 矩阵
      WEnorm: "{{\\mathbf{\\hat{W}_E}}}", // 归一化 token embedding 矩阵
      X: "{\\mathbf{X}}", // 输入 batch token id 矩阵
      Z: "{\\mathbf{Z}}", // sparse code


      // Common operators
      argmax: "{\\operatorname*{arg\\,max}}",
      argmin: "{\\operatorname*{arg\\,min}}",
      softmax: "{\\operatorname{softmax}}",
      sigmoid: "{\\operatorname{sigmoid}}",
      relu: "{\\operatorname{ReLU}}",
      gelu: "{\\operatorname{GELU}}",
      silu: "{\\operatorname{SiLU}}",
      LN: "{\\operatorname{LN}}",
      MLP: "{\\operatorname{MLP}}",
      Attn: "{\\operatorname{Attn}}",
      FFN: "{\\operatorname{FFN}}",
      KL: "{\\operatorname{KL}}",
      CE: "{\\operatorname{CE}}",
      Var: "{\\operatorname{Var}}",
      Cov: "{\\operatorname{Cov}}",

      // Common helpers
      norm: ["{\\left\\lVert #1 \\right\\rVert}", 1],
      abs: ["{\\left\\lvert #1 \\right\\rvert}", 1],
      inner: ["{\\left\\langle #1, #2 \\right\\rangle}", 2],

      // Mechanistic interpretability
      clean: "{\\mathrm{clean}}",
      corr: "{\\mathrm{corr}}",
      patch: "{\\mathrm{patch}}",
      ablate: "{\\mathrm{ablate}}",
    }
  },
  "options": {},
  "loader": {}
},
  
  mermaidConfig: {
  "startOnLoad": false
},
})