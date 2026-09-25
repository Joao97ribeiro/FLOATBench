// FLOATBench project page: dataset explorer, live Hugging Face rows,
// leaderboard and findings. Data comes from static/data/*.json, written by
// docs/scripts/build_data.py.

const HF_API = "https://datasets-server.huggingface.co";
const HF_DATASET = "DeCoDELab/FLOATBench";
const TOWERS = ["ref", "opt1", "opt2"];
const TOWER_LABEL = { ref: "Reference", opt1: "Opt1", opt2: "Opt2" };
const GROUP_NAME = { IT: "In-train", IP: "Interpolate", EX: "Extrapolate" };
const FAMILIES = [
  "Ensemble", "NeuralNetFastAI", "NeuralNetTorch", "TabM", "CatBoost",
  "LightGBM", "XGBoost", "RandomForest", "ExtraTrees",
];

const state = {
  data: null,
  lb: null,
  damage: {},         // tower -> Int16Array (n_sims * n_sections), log10 x 1000
  sel: 0,             // selected simulation index
  lbSel: null,        // selected leaderboard model key
  families: new Set(FAMILIES),
  hfOffset: 0,
};

// ------------------------------------------------------------------ theme --
function css(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function theme() {
  const dark = css("color-scheme") === "dark";
  return {
    dark,
    ink: css("--ink"),
    ink2: css("--ink-2"),
    ink3: css("--ink-3"),
    line: css("--line"),
    surface: css("--surface"),
    it: css("--it"),
    ip: css("--ip"),
    ex: css("--ex"),
    navy: css("--navy"),
    red: css("--red"),
    teal: css("--teal"),
    tower: { ref: "#888888", opt1: "#1f77b4", opt2: "#d62728" },
    family: {
      Ensemble: css("--navy"),
      NeuralNetFastAI: css("--red"),
      NeuralNetTorch: dark ? "#e8a19c" : "#d06662",
      TabM: dark ? "#c9867f" : "#6f1b17",
      CatBoost: css("--teal"),
      LightGBM: dark ? "#6fb3a0" : "#2f7d6d",
      XGBoost: dark ? "#c7b37a" : "#9a7b2a",
      RandomForest: dark ? "#b09f93" : "#8f7a6e",
      ExtraTrees: dark ? "#9aa3ab" : "#6d7780",
    },
    damageScale: dark
      ? [[0, "#1e2b3d"], [0.45, "#4f86a8"], [0.75, "#d9a45b"], [1, "#f06a5e"]]
      : [[0, "#dfe7f1"], [0.45, "#5c93b3"], [0.75, "#d08a3a"], [1, "#b02c27"]],
  };
}

function baseLayout(t, extra = {}) {
  return Object.assign({
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "Helvetica, Arial, sans-serif", size: 13, color: t.ink2 },
    margin: { l: 58, r: 16, t: 36, b: 48 },
    hoverlabel: { font: { family: "Helvetica, Arial, sans-serif", size: 12 } },
    showlegend: false,
  }, extra);
}

function axis(t, title, extra = {}) {
  return Object.assign({
    title: { text: title, font: { size: 12, color: t.ink } },
    gridcolor: t.line,
    zerolinecolor: t.line,
    linecolor: t.line,
    tickfont: { color: t.ink3 },
  }, extra);
}

const PLOT_CFG = { displaylogo: false, responsive: true, modeBarButtonsToRemove: ["lasso2d", "select2d"] };

function render(id, traces, layout) {
  const el = document.getElementById(id);
  if (!el || !window.Plotly) return;
  Plotly.react(el, traces, layout, PLOT_CFG);
}

// ------------------------------------------------------------------- data --
function decodeDamage(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Int16Array(bytes.buffer);
}

function logDamage(tower, sim, section) {
  const n = state.data.n_sections;
  return state.damage[tower][sim * n + (section - 1)] / 1000;
}

function fmtSci(x, digits = 2) {
  if (!isFinite(x)) return "";
  return x.toExponential(digits).replace("e", "e");
}

// --------------------------------------------------------------- explorer --
function simCategory(i, view) {
  const s = state.data.sims;
  if (s.train[i]) return "train";
  if (view === "wind") return s.wind[i];
  if (view === "wave") return s.wave[i];
  const w = s.wind[i], v = s.wave[i];
  if (w === "EX" && v === "EX") return "EXEX";
  if (w === "EX" || v === "EX") return "EX1";
  if (w === "IP" || v === "IP") return "IP";
  return "IT";
}

function categories(t, view) {
  if (view === "3d") {
    return [
      ["train", "Train", t.navy],
      ["IT", "Test, in-train on both", t.teal],
      ["IP", "Test, interpolate", t.ip],
      ["EX1", "Test, extrapolate on one axis", t.dark ? "#b86560" : "#dd9c98"],
      ["EXEX", "Test, EX_EX", t.ex],
    ];
  }
  const axisName = view === "wind" ? "wind" : "wave";
  return [
    ["train", "Train", t.navy],
    ["IT", `Test, ${axisName} in-train`, t.teal],
    ["IP", `Test, ${axisName} interpolate`, t.ip],
    ["EX", `Test, ${axisName} extrapolate`, t.ex],
  ];
}

function explorerIndices() {
  const ws = document.getElementById("ex-ws").value;
  const s = state.data.sims;
  const idx = [];
  for (let i = 0; i < s.sim_id.length; i++) {
    if (ws === "all" || s.ws[i] === Number(ws)) idx.push(i);
  }
  return idx;
}

function coords(view, i) {
  const s = state.data.sims;
  if (view === "wind") return [s.mean_ws[i], s.std_ws[i]];
  if (view === "wave") return [s.tp[i], s.hs[i]];
  return [s.mean_ws[i], s.tp[i], s.hs[i]];
}

function hoverText(i, tower, section) {
  const s = state.data.sims;
  const d = Math.pow(10, logDamage(tower, i, section));
  return `sim ${s.sim_id[i]} · seed ${s.seed[i]}<br>U = ${s.ws[i]} m/s (mean ${s.mean_ws[i].toFixed(2)}, std ${s.std_ws[i].toFixed(2)})`
    + `<br>Hs = ${s.hs[i].toFixed(2)} m, Tp = ${s.tp[i].toFixed(2)} s`
    + `<br>wind ${GROUP_NAME[s.wind[i]]}, wave ${GROUP_NAME[s.wave[i]]}${s.train[i] ? " (train)" : ""}`
    + `<br>damage @ section ${section}: ${fmtSci(d)}`;
}

function drawEnvelope() {
  const t = theme();
  const view = document.getElementById("ex-view").value;
  const color = document.getElementById("ex-color").value;
  const tower = document.getElementById("ex-tower").value;
  const section = Number(document.getElementById("ex-sec").value);
  const idx = explorerIndices();
  const is3d = view === "3d";
  const size = is3d ? 3 : (idx.length > 2000 ? 5 : 8);
  const traces = [];
  const legend = document.getElementById("ex-legend");

  function trace(ids, extra) {
    const xs = [], ys = [], zs = [], text = [], cd = [];
    for (const i of ids) {
      const c = coords(view, i);
      xs.push(c[0]); ys.push(c[1]); if (is3d) zs.push(c[2]);
      text.push(hoverText(i, tower, section));
      cd.push(i);
    }
    const tr = Object.assign({
      type: is3d ? "scatter3d" : "scattergl",
      mode: "markers",
      x: xs, y: ys, text, customdata: cd,
      hovertemplate: "%{text}<extra></extra>",
    }, extra);
    if (is3d) tr.z = zs;
    return tr;
  }

  if (color === "damage") {
    const vals = idx.map((i) => logDamage(tower, i, section));
    const tr = trace(idx, {
      marker: {
        size, color: vals, colorscale: t.damageScale, opacity: 0.9,
        colorbar: {
          title: { text: "log₁₀ D", font: { color: t.ink2, size: 11 } },
          tickfont: { color: t.ink3, size: 10 }, thickness: 10, len: 0.8, outlinewidth: 0,
        },
      },
    });
    traces.push(tr);
    legend.innerHTML = `<span>Colour: 600 s fatigue damage of ${TOWER_LABEL[tower]} at section ${section} (log scale)</span>`;
  } else if (color === "split") {
    const tr = idx.filter((i) => state.data.sims.train[i]);
    const te = idx.filter((i) => !state.data.sims.train[i]);
    traces.push(trace(te, { marker: { size, color: t.ip, opacity: 0.55 } }));
    traces.push(trace(tr, { marker: { size, color: t.navy, opacity: 0.95 } }));
    legend.innerHTML = `<span><i style="background:${t.navy}"></i>Train (${tr.length})</span><span><i style="background:${t.ip}"></i>Test (${te.length})</span>`;
  } else {
    const cats = categories(t, view);
    const html = [];
    for (const [key, label, col] of cats) {
      const ids = idx.filter((i) => simCategory(i, view) === key);
      if (!ids.length) continue;
      traces.push(trace(ids, { marker: { size, color: col, opacity: key === "train" ? 0.95 : 0.8 } }));
      html.push(`<span><i style="background:${col}"></i>${label} (${ids.length})</span>`);
    }
    legend.innerHTML = html.join("");
  }

  // Selected simulation marker.
  if (idx.includes(state.sel)) {
    const c = coords(view, state.sel);
    const sel = {
      type: is3d ? "scatter3d" : "scatter",
      mode: "markers",
      x: [c[0]], y: [c[1]],
      hoverinfo: "skip",
      marker: { size: is3d ? 8 : 16, color: "rgba(0,0,0,0)", line: { color: t.ink, width: 2.5 } },
    };
    if (is3d) sel.z = [c[2]];
    traces.push(sel);
  }

  const titles = {
    wind: ["Mean wind speed [m/s]", "Std of wind speed [m/s]"],
    wave: ["Peak period Tp [s]", "Significant wave height Hs [m]"],
    "3d": ["Mean wind speed [m/s]", "Tp [s]", "Hs [m]"],
  }[view];
  let layout;
  if (is3d) {
    const ax = (title) => ({
      title: { text: title, font: { size: 11, color: t.ink } },
      gridcolor: t.line, zerolinecolor: t.line, backgroundcolor: "rgba(0,0,0,0)",
      tickfont: { size: 10, color: t.ink3 }, showbackground: false,
    });
    layout = baseLayout(t, {
      margin: { l: 0, r: 0, t: 10, b: 0 },
      scene: {
        xaxis: ax(titles[0]), yaxis: ax(titles[1]), zaxis: ax(titles[2]),
        camera: { eye: { x: 1.6, y: -1.5, z: 0.9 } },
        aspectmode: "manual", aspectratio: { x: 1.4, y: 1, z: 0.8 },
      },
      uirevision: "envelope3d",
    });
  } else {
    layout = baseLayout(t, {
      xaxis: axis(t, titles[0]),
      yaxis: axis(t, titles[1]),
      margin: { l: 58, r: 16, t: 14, b: 48 },
      uirevision: view,
    });
  }
  render("plot-envelope", traces, layout);
  const el = document.getElementById("plot-envelope");
  if (!el._clickBound) {
    el.on("plotly_click", (ev) => {
      const p = ev.points && ev.points[0];
      if (p && p.customdata !== undefined) selectSim(p.customdata);
    });
    el._clickBound = true;
  }
}

function drawProfile() {
  const t = theme();
  const i = state.sel;
  const section = Number(document.getElementById("ex-sec").value);
  const n = state.data.n_sections;
  const traces = TOWERS.map((tw) => {
    const h = state.data.towers[tw].height;
    const d = [];
    for (let k = 1; k <= n; k++) d.push(Math.pow(10, logDamage(tw, i, k)));
    return {
      x: d, y: h, mode: "lines+markers", name: TOWER_LABEL[tw],
      line: { color: t.tower[tw], width: 2.2 },
      marker: { size: 4 },
      hovertemplate: `${TOWER_LABEL[tw]}<br>%{y:.1f} m: %{x:.3e}<extra></extra>`,
    };
  });
  const hSel = state.data.towers.ref.height[section - 1];
  const layout = baseLayout(t, {
    title: { text: `Simulation ${state.data.sims.sim_id[i]}: damage along the tower`, font: { size: 13, color: t.ink }, x: 0.02, xanchor: "left" },
    xaxis: axis(t, "600 s fatigue damage [-] (log)", { type: "log", exponentformat: "power" }),
    yaxis: axis(t, "Section height [m]"),
    showlegend: true,
    legend: { orientation: "h", x: 0, y: -0.2, font: { color: t.ink2 } },
    margin: { l: 58, r: 16, t: 36, b: 78 },
    shapes: [{
      type: "line", xref: "paper", x0: 0, x1: 1, y0: hSel, y1: hSel,
      line: { color: t.ink3, width: 1, dash: "dot" },
    }],
  });
  render("plot-profile", traces, layout);

  const s = state.data.sims;
  const cells = [
    ["sim_id", s.sim_id[i]],
    ["seed", s.seed[i]],
    ["U [m/s]", s.ws[i]],
    ["split", s.train[i] ? "train" : "test"],
    ["Hs [m]", s.hs[i].toFixed(2)],
    ["Tp [s]", s.tp[i].toFixed(2)],
    ["wind", GROUP_NAME[s.wind[i]]],
    ["wave", GROUP_NAME[s.wave[i]]],
  ];
  document.getElementById("sim-card").innerHTML = cells
    .map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join("");
}

function drawLifetime() {
  const t = theme();
  const traces = TOWERS.map((tw) => ({
    x: state.data.towers[tw].lifetime,
    y: state.data.towers[tw].height,
    mode: "lines+markers",
    name: TOWER_LABEL[tw],
    line: { color: t.tower[tw], width: 2.4 },
    marker: { size: 4 },
    hovertemplate: `${TOWER_LABEL[tw]}<br>%{y:.1f} m: D = %{x:.3f}<extra></extra>`,
  }));
  const layout = baseLayout(t, {
    xaxis: axis(t, "25-year fatigue damage D [-] (log)", { type: "log", exponentformat: "power" }),
    yaxis: axis(t, "Section height [m]"),
    showlegend: true,
    legend: { orientation: "h", x: 0, y: -0.22, font: { color: t.ink2 } },
    margin: { l: 58, r: 16, t: 24, b: 80 },
    shapes: [{ type: "line", yref: "paper", x0: 1, x1: 1, y0: 0, y1: 1, line: { color: t.ex, width: 1.2, dash: "dash" } }],
    annotations: [{ x: 0, xref: "x", y: 1, yref: "paper", text: "D = 1", showarrow: false, xanchor: "left", yanchor: "bottom", font: { color: t.ex, size: 11 } }],
  });
  render("plot-lifetime", traces, layout);
}

function selectSim(i) {
  state.sel = i;
  drawEnvelope();
  drawProfile();
  document.getElementById("sim-hf-status").innerHTML = "";
}

function updateSectionLabel() {
  const sec = Number(document.getElementById("ex-sec").value);
  document.getElementById("ex-sec-out").textContent = sec;
  const h = state.data.towers.ref.height[sec - 1];
  const where = sec === 1 ? "tower base" : sec === state.data.n_sections ? "tower top" : "";
  document.getElementById("ex-sec-h").textContent = `(${h.toFixed(1)} m${where ? ", " + where : ""})`;
}

function initExplorer() {
  const wsSel = document.getElementById("ex-ws");
  [...new Set(state.data.sims.ws)].sort((a, b) => a - b).forEach((w) => {
    const o = document.createElement("option");
    o.value = w; o.textContent = `${w} m/s`;
    wsSel.appendChild(o);
  });
  // Start on a deep-extrapolation simulation near rated wind.
  const s = state.data.sims;
  let start = s.sim_id.findIndex((_, i) => s.wind[i] === "EX" && s.wave[i] === "EX" && s.ws[i] > 20);
  state.sel = start < 0 ? 0 : start;
  document.getElementById("ex-color").value = "regime";
  ["ex-view", "ex-color", "ex-tower", "ex-ws"].forEach((id) =>
    document.getElementById(id).addEventListener("change", () => { drawEnvelope(); drawProfile(); }));
  document.getElementById("ex-sec").addEventListener("input", () => {
    updateSectionLabel();
    const color = document.getElementById("ex-color");
    if (color.value !== "damage") color.value = "damage";
    drawEnvelope(); drawProfile();
  });
  document.getElementById("sim-hf").addEventListener("click", loadSimFromHF);
  updateSectionLabel();
  drawEnvelope();
  drawProfile();
  drawLifetime();
}

// ------------------------------------------------------------ hugging face --
async function hfFetch(path) {
  const res = await fetch(`${HF_API}/${path}`);
  let body = null;
  try { body = await res.json(); } catch (e) { /* non-JSON error page */ }
  if (!res.ok || (body && body.error)) {
    const msg = (body && body.error) || `HTTP ${res.status}`;
    throw new Error(msg);
  }
  return body;
}

function rowsTable(features, rows) {
  const cols = features.map((f) => f.name);
  const head = `<thead><tr>${cols.map((c) => `<th class="${c.includes("group") ? "l" : ""}">${c}</th>`).join("")}</tr></thead>`;
  const body = rows.map((r) => `<tr>${cols.map((c) => {
    const v = r.row[c];
    if (typeof v === "number") {
      const txt = c === "damage" ? fmtSci(v, 3) : Number.isInteger(v) ? v : v.toFixed(4);
      return `<td>${txt}</td>`;
    }
    if (c.includes("group")) {
      const cls = v === "Extrapolate" ? "tag-ex" : v === "Interpolate" ? "tag-ip" : "tag-it";
      return `<td class="l"><span class="tag ${cls}">${v}</span></td>`;
    }
    return `<td>${v}</td>`;
  }).join("")}</tr>`).join("");
  return head + `<tbody>${body}</tbody>`;
}

const HF_PAGE = 20;

async function loadHFRows() {
  const config = document.getElementById("hf-config").value;
  const split = document.getElementById("hf-split").value;
  const where = document.getElementById("hf-filter").value;
  const status = document.getElementById("hf-status");
  const table = document.getElementById("hf-table");
  status.textContent = "Loading rows from Hugging Face…";
  const q = `dataset=${encodeURIComponent(HF_DATASET)}&config=${config}&split=${split}&offset=${state.hfOffset}&length=${HF_PAGE}`;
  try {
    const body = where
      ? await hfFetch(`filter?${q}&where=${encodeURIComponent(where)}`)
      : await hfFetch(`rows?${q}`);
    table.innerHTML = rowsTable(body.features, body.rows);
    const total = body.num_rows_total;
    const a = state.hfOffset + 1, b = state.hfOffset + body.rows.length;
    document.getElementById("hf-page").textContent = `rows ${a.toLocaleString()}–${b.toLocaleString()} of ${total.toLocaleString()}`;
    document.getElementById("hf-prev").disabled = state.hfOffset === 0;
    document.getElementById("hf-next").disabled = b >= total;
    status.innerHTML = `<code>${config}</code> / <code>${split}</code>${where ? " filtered" : ""}, live from <a href="https://huggingface.co/datasets/${HF_DATASET}/viewer/${config}/${split}">the Hugging Face viewer</a>.`;
  } catch (err) {
    const hint = /loading|index/i.test(err.message)
      ? "The Hugging Face index for this query is warming up. Try again in a minute."
      : "The Hugging Face API could not be reached from this page.";
    status.innerHTML = `${hint} <a href="https://huggingface.co/datasets/${HF_DATASET}/viewer/${config}/${split}">Open the rows on Hugging Face</a>. <span class="muted">(${err.message})</span>`;
  }
}

function initHF() {
  ["hf-config", "hf-split", "hf-filter"].forEach((id) =>
    document.getElementById(id).addEventListener("change", () => { state.hfOffset = 0; loadHFRows(); }));
  document.getElementById("hf-prev").addEventListener("click", () => {
    state.hfOffset = Math.max(0, state.hfOffset - HF_PAGE); loadHFRows();
  });
  document.getElementById("hf-next").addEventListener("click", () => {
    state.hfOffset += HF_PAGE; loadHFRows();
  });
  loadHFRows();
}

async function loadSimFromHF() {
  const i = state.sel;
  const s = state.data.sims;
  const tower = document.getElementById("ex-tower").value;
  const split = s.train[i] ? "train" : "test";
  const status = document.getElementById("sim-hf-status");
  status.textContent = `Querying ${HF_DATASET} (${tower}/${split}) for sim_id ${s.sim_id[i]}…`;
  const q = `dataset=${encodeURIComponent(HF_DATASET)}&config=${tower}&split=${split}&offset=0&length=30`
    + `&where=${encodeURIComponent(`"sim_id"=${s.sim_id[i]}`)}&orderby=${encodeURIComponent('"section_id"')}`;
  try {
    const body = await hfFetch(`filter?${q}`);
    status.innerHTML = `<div class="table-container" style="max-height:260px;"><table class="table is-narrow is-fullwidth data">${rowsTable(body.features, body.rows)}</table></div>`;
  } catch (err) {
    const hint = /loading|index/i.test(err.message)
      ? "The Hugging Face filter index is warming up. Try again in a minute."
      : "The Hugging Face API could not be reached from this page.";
    status.innerHTML = `${hint} <span class="muted">(${err.message})</span>`;
  }
}

// ------------------------------------------------------------ regime grid --
function initRegimeGrid() {
  const s = state.data.sims;
  const counts = {};
  for (let i = 0; i < s.sim_id.length; i++) {
    if (s.train[i]) continue;
    const k = `${s.wind[i]}_${s.wave[i]}`;
    counts[k] = (counts[k] || 0) + 1;
  }
  const order = ["IT", "IP", "EX"];
  let html = `<div class="h"></div>` + order.map((w) => `<div class="h">wave ${GROUP_NAME[w]}</div>`).join("");
  for (const wi of order) {
    html += `<div class="h">wind ${GROUP_NAME[wi]}</div>`;
    for (const wa of order) {
      const k = `${wi}_${wa}`;
      html += `<div class="c${k === "EX_EX" ? " deep" : ""}"><b>${(counts[k] || 0).toLocaleString()}</b><span>${k}</span></div>`;
    }
  }
  document.getElementById("regime-grid").innerHTML = html;
}

// ------------------------------------------------------------ leaderboard --
function rankBy(rows, key) {
  const vals = rows.map((r) => r.rel_l2[key]).slice().sort((a, b) => a - b);
  const out = new Map();
  rows.forEach((r) => out.set(r, 1 + vals.indexOf(r.rel_l2[key])));
  return out;
}

function modelKey(r) { return `${r.model}|${r.preset}`; }

function shortName(m) { return m.replace(/_BAG_L1$/, "").replace(/_L2$/, ""); }

function lbRows() {
  const tower = document.getElementById("lb-tower").value;
  const regime = document.getElementById("lb-regime").value;
  const rows = state.lb[tower];
  const gRank = rankBy(rows, "Global");
  const rRank = rankBy(rows, regime);
  return { tower, regime, rows, gRank, rRank };
}

function drawLeaderboard() {
  const t = theme();
  const { regime, rows, gRank, rRank } = lbRows();
  const q = document.getElementById("lb-search").value.trim().toLowerCase();
  const n = Number(document.getElementById("lb-n").value);
  const shown = rows
    .filter((r) => state.families.has(r.family))
    .filter((r) => !q || r.model.toLowerCase().includes(q))
    .sort((a, b) => rRank.get(a) - rRank.get(b))
    .slice(0, n);
  if (!state.lbSel || !rows.some((r) => modelKey(r) === state.lbSel)) {
    state.lbSel = modelKey(rows.find((r) => rRank.get(r) === 1));
  }
  const regimeLabel = regime === "Global" ? "Global" : regime;
  const head = `<thead><tr><th>#</th><th class="l">Model</th><th class="l">Preset</th><th>Rel L² ${regimeLabel}</th>`
    + (regime === "Global" ? "" : `<th>Global #</th><th>Shift</th>`)
    + `<th>R² DEL</th></tr></thead>`;
  const body = shown.map((r) => {
    const g = gRank.get(r), k = rRank.get(r), d = g - k;
    const shift = d > 0 ? `<span class="up">▲ ${d}</span>` : d < 0 ? `<span class="down">▼ ${-d}</span>` : "·";
    return `<tr data-key="${modelKey(r)}" class="${modelKey(r) === state.lbSel ? "sel" : ""}">`
      + `<td>${k}</td>`
      + `<td class="l mono"><span class="fam-dot" style="background:${t.family[r.family]}"></span>${shortName(r.model)}</td>`
      + `<td class="l">${r.preset}</td>`
      + `<td><b>${r.rel_l2[regime].toFixed(4)}</b></td>`
      + (regime === "Global" ? "" : `<td>${g}</td><td>${shift}</td>`)
      + `<td>${r.r2_del.toFixed(4)}</td></tr>`;
  }).join("");
  const table = document.getElementById("lb-table");
  table.innerHTML = head + `<tbody>${body || `<tr><td colspan="7" class="l muted">No model matches the filters.</td></tr>`}</tbody>`;
  table.querySelectorAll("tbody tr[data-key]").forEach((tr) =>
    tr.addEventListener("click", () => { state.lbSel = tr.dataset.key; drawLeaderboard(); }));
  drawCrossover();
  drawHeat();
}

function drawCrossover() {
  const t = theme();
  const { tower, regime, rows, gRank, rRank } = lbRows();
  const traces = [];
  for (const fam of FAMILIES) {
    const rs = rows.filter((r) => r.family === fam && state.families.has(fam));
    if (!rs.length) continue;
    traces.push({
      type: "scatter", mode: "markers", name: fam,
      x: rs.map((r) => gRank.get(r)), y: rs.map((r) => rRank.get(r)),
      customdata: rs.map(modelKey),
      text: rs.map((r) => `${shortName(r.model)} (${r.preset})<br>Global #${gRank.get(r)} · ${regime} #${rRank.get(r)}`),
      hovertemplate: "%{text}<extra></extra>",
      marker: { size: 7, color: t.family[fam], opacity: 0.85, line: { width: 0 } },
    });
  }
  const sel = rows.find((r) => modelKey(r) === state.lbSel);
  if (sel) {
    traces.push({
      type: "scatter", mode: "markers", hoverinfo: "skip",
      x: [gRank.get(sel)], y: [rRank.get(sel)],
      marker: { size: 16, color: "rgba(0,0,0,0)", line: { color: t.ink, width: 2 } },
    });
  }
  const N = rows.length;
  const layout = baseLayout(t, {
    title: { text: `${TOWER_LABEL[tower]}: Global rank vs ${regime} rank`, font: { size: 13, color: t.ink }, x: 0.02, xanchor: "left" },
    xaxis: axis(t, "Global rank", { range: [0, N + 3] }),
    yaxis: axis(t, `${regime} rank`, { range: [0, N + 3] }),
    shapes: [{ type: "line", x0: 1, y0: 1, x1: N, y1: N, line: { color: t.line, width: 1.5, dash: "dot" } }],
    margin: { l: 52, r: 12, t: 36, b: 44 },
  });
  render("plot-crossover", traces, layout);
  const el = document.getElementById("plot-crossover");
  if (!el._clickBound) {
    el.on("plotly_click", (ev) => {
      const p = ev.points && ev.points[0];
      if (p && p.customdata) { state.lbSel = p.customdata; drawLeaderboard(); }
    });
    el._clickBound = true;
  }
}

function drawHeat() {
  const t = theme();
  const { tower, rows } = lbRows();
  const r = rows.find((x) => modelKey(x) === state.lbSel);
  if (!r) return;
  const order = ["IT", "IP", "EX"];
  const z = order.map((wi) => order.map((wa) => r.rel_l2[`${wi}_${wa}`]));
  const all = rows.flatMap((x) => order.flatMap((wi) => order.map((wa) => x.rel_l2[`${wi}_${wa}`])));
  const lo = Math.min(...all), hi = Math.min(Math.max(...all), 0.35);
  const trace = {
    type: "heatmap", z, zmin: lo, zmax: hi,
    x: order.map((o) => `wave ${o}`), y: order.map((o) => `wind ${o}`),
    colorscale: t.damageScale, showscale: false,
    text: z.map((row) => row.map((v) => v.toFixed(3))),
    texttemplate: "%{text}", textfont: { family: "Helvetica, Arial, sans-serif", size: 12 },
    hovertemplate: "%{y} × %{x}<br>Rel L² DEL %{z:.4f}<extra></extra>",
    xgap: 3, ygap: 3,
  };
  const layout = baseLayout(t, {
    title: { text: `${shortName(r.model)} (${r.preset}) on ${TOWER_LABEL[tower]}: Rel L² DEL per cell`, font: { size: 12, color: t.ink }, x: 0.02, xanchor: "left" },
    xaxis: axis(t, "", { side: "bottom", showgrid: false }),
    yaxis: axis(t, "", { showgrid: false }),
    margin: { l: 80, r: 12, t: 36, b: 36 },
  });
  render("plot-heat", [trace], layout);
}

function initLeaderboard() {
  const t = theme();
  const chips = document.getElementById("lb-families");
  chips.innerHTML = FAMILIES.map((f) =>
    `<button type="button" class="chip" aria-pressed="true" data-fam="${f}"><span class="dot" style="background:${t.family[f]}"></span>${f}</button>`).join("");
  chips.querySelectorAll(".chip").forEach((c) => c.addEventListener("click", () => {
    const f = c.dataset.fam;
    if (state.families.has(f)) state.families.delete(f); else state.families.add(f);
    c.setAttribute("aria-pressed", state.families.has(f));
    drawLeaderboard();
  }));
  ["lb-tower", "lb-regime", "lb-n"].forEach((id) =>
    document.getElementById(id).addEventListener("change", drawLeaderboard));
  document.getElementById("lb-search").addEventListener("input", drawLeaderboard);
  drawLeaderboard();
}

function recolorChips() {
  const t = theme();
  document.querySelectorAll("#lb-families .chip").forEach((c) => {
    c.querySelector(".dot").style.background = t.family[c.dataset.fam];
  });
}

// --------------------------------------------------------------- findings --
function drawFindings() {
  const t = theme();
  const models = [
    ["WeightedEnsemble_L2", "Ensemble (global rank-1)", t.navy],
    ["NeuralNetFastAI_r102_BAG_L1", "NeuralNetFastAI_r102 (EX_EX rank-1)", t.red],
  ];
  const traces = [];
  for (const [m, label, col] of models) {
    const xs = [], ys = [], text = [];
    TOWERS.forEach((tw) => {
      const rows = state.lb[tw];
      const g = rankBy(rows, "Global"), e = rankBy(rows, "EX_EX");
      const r = rows.filter((x) => x.model === m).sort((a, b) => g.get(a) - g.get(b))[0];
      xs.push(g.get(r), e.get(r), null);
      ys.push(TOWER_LABEL[tw], TOWER_LABEL[tw], null);
      text.push(`Global #${g.get(r)}`, `EX_EX #${e.get(r)}`, "");
    });
    traces.push({
      type: "scatter", mode: "lines+markers", name: label, x: xs, y: ys, text,
      hovertemplate: `${label}<br>%{y}: %{text}<extra></extra>`,
      line: { color: col, width: 2 },
      marker: { size: 10, color: col, symbol: xs.map((_, k) => (k % 3 === 0 ? "circle-open" : "circle")) },
      connectgaps: false,
    });
  }
  render("plot-findings-e2", traces, baseLayout(t, {
    xaxis: axis(t, "Rank in the pool (open = Global, filled = EX_EX)", { range: [0, 100] }),
    yaxis: axis(t, "", { autorange: "reversed" }),
    showlegend: true,
    legend: { orientation: "h", x: 0, y: 1.22, font: { color: t.ink2 } },
    margin: { l: 52, r: 16, t: 40, b: 48 },
  }));

  const folds = ["Ref+Opt1 → Opt2", "Ref+Opt2 → Opt1", "Opt1+Opt2 → Ref"];
  const vals = [0.067, 0.098, 0.423];
  render("plot-findings-e3", [{
    type: "bar", orientation: "h", y: folds, x: vals,
    marker: { color: [t.navy, t.navy, t.red] },
    text: vals.map((v) => v.toFixed(3)), textposition: "outside",
    textfont: { family: "Helvetica, Arial, sans-serif", color: t.ink },
    hovertemplate: "%{y}<br>rank-1 Rel L² DEL %{x:.3f}<extra></extra>",
    cliponaxis: false,
  }], baseLayout(t, {
    xaxis: axis(t, "Rank-1 Rel L² DEL on the held-out tower", { range: [0, 0.5] }),
    yaxis: axis(t, "", { autorange: "reversed" }),
    margin: { l: 130, r: 30, t: 20, b: 48 },
    bargap: 0.45,
  }));
}

// ------------------------------------------------------------------ misc --
function initCopy() {
  const btn = document.getElementById("copy-bibtex");
  btn.addEventListener("click", async () => {
    const text = document.getElementById("bibtex").textContent;
    try {
      await navigator.clipboard.writeText(text);
      btn.innerHTML = '<span class="icon"><i class="fas fa-check"></i></span><span>Copied</span>';
      btn.classList.add("is-success");
    } catch (e) {
      const range = document.createRange();
      range.selectNodeContents(document.getElementById("bibtex"));
      const sel = window.getSelection();
      sel.removeAllRanges(); sel.addRange(range);
      btn.innerHTML = "<span>Selected, press Ctrl+C</span>";
    }
    setTimeout(() => {
      btn.innerHTML = '<span class="icon"><i class="fas fa-copy"></i></span><span>Copy BibTeX</span>';
      btn.classList.remove("is-success");
    }, 1800);
  });
}

function initNav() {
  const links = [...document.querySelectorAll(".nav-inner a")];
  const ids = links.map((a) => a.getAttribute("href").slice(1));
  const obs = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (!e.isIntersecting) return;
      links.forEach((a) => a.classList.toggle("active", a.getAttribute("href") === `#${e.target.id}`));
    });
  }, { rootMargin: "-45% 0px -50% 0px" });
  ids.forEach((id) => { const el = document.getElementById(id); if (el) obs.observe(el); });
}

function redrawAll() {
  if (!state.data || !state.lb) return;
  recolorChips();
  drawEnvelope(); drawProfile(); drawLifetime();
  drawLeaderboard(); drawFindings();
}

function watchTheme() {
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", redrawAll);
  new MutationObserver(redrawAll).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
}

async function main() {
  initCopy();
  initNav();
  initHF();
  const [data, lb] = await Promise.all([
    fetch("static/data/dataset.json").then((r) => r.json()),
    fetch("static/data/leaderboard.json").then((r) => r.json()),
  ]);
  state.data = data;
  state.lb = lb;
  TOWERS.forEach((tw) => { state.damage[tw] = decodeDamage(data.towers[tw].log_damage_i16); });
  initRegimeGrid();
  initExplorer();
  initLeaderboard();
  drawFindings();
  watchTheme();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", main);
} else {
  main();
}
