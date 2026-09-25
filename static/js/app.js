// FLOATBench project page: dataset explorer, live Hugging Face rows,
// leaderboard and findings. Data comes from static/data/*.json, written by
// docs/scripts/build_data.py.

const HF_API = "https://datasets-server.huggingface.co";
const HF_DATASET = "DeCoDELab/FLOATBench";
const TOWERS = ["ref", "opt1", "opt2"];
const TOWER_LABEL = { ref: "REF", opt1: "OPT1", opt2: "OPT2" };
const GROUP_NAME = { IT: "In-train", IP: "Interpolate", EX: "Extrapolate" };
// FAMILY_ORDER of scripts/figures/heatmap_bars.
const FAMILIES = [
  "NeuralNet", "RandomForest", "ExtraTrees", "CatBoost", "LightGBM",
  "XGBoost", "Ensemble", "TabM",
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

// Paper palette: floatbench/colors.py and floatbench/plots/paper_style.py.
const PAPER = {
  red: "#b02c27", lightRed: "#d69b99", middleRed: "#d06662", darkRed: "#932421",
  dark2Red: "#802421", blue: "#294366", lightBlue: "#ade1f4", middleBlue: "#7cc0cd",
  darkBlue: "#1c2b4a", grey: "#b8b8b8", greyDark: "#8f8f8f", lightGrey: "#f2f2f2",
  darkGrey: "#555555", lightBrown: "#8f7a6e", brown: "#66574e", deepRed: "#6f1b17",
};

function theme() {
  return {
    ink: "#000000",          // axis names and panel titles
    ink2: PAPER.darkGrey,     // legends and annotations
    ink3: PAPER.darkGrey,     // tick labels
    line: PAPER.lightGrey,    // grid and spines
    it: PAPER.blue,
    ip: PAPER.grey,
    ipText: PAPER.greyDark,
    ex: PAPER.red,
    navy: PAPER.blue,
    red: PAPER.red,
    teal: PAPER.middleBlue,
    train: PAPER.darkBlue,
    tower: { ref: PAPER.grey, opt1: PAPER.blue, opt2: PAPER.red },
    // FAMILY_COLORS in floatbench/plots/benchmark.py.
    family: {
      NeuralNet: PAPER.blue,
      RandomForest: PAPER.brown,
      ExtraTrees: PAPER.middleBlue,
      CatBoost: PAPER.lightRed,
      LightGBM: PAPER.lightBlue,
      XGBoost: PAPER.lightBrown,
      Ensemble: PAPER.red,
      TabM: PAPER.grey,
    },
    // CUSTOM_MAP_RED_SEQ: light grey (low) to dark red (high).
    damageScale: [
      [0, PAPER.lightGrey], [0.2, PAPER.lightRed], [0.4, PAPER.middleRed],
      [0.6, PAPER.red], [0.8, PAPER.darkRed], [1, PAPER.dark2Red],
    ],
    // Regime heatmap of scripts/figures/heatmap_bars: white, red, deep red.
    heatScale: [[0, "#ffffff"], [0.5, PAPER.red], [1, PAPER.deepRed]],
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
// log10 damage per (simulation, section), row-major, as Float32Array.
function decodeDamage(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const packed = new Int16Array(bytes.buffer);
  const out = new Float32Array(packed.length);
  for (let i = 0; i < packed.length; i++) out[i] = packed[i] / 1000;
  return out;
}

function logDamage(tower, sim, section) {
  const n = state.data.n_sections;
  return state.damage[tower][sim * n + (section - 1)];
}

// ------------------------------------------------ dataset from hugging face --
const HYPARQUET = "https://cdn.jsdelivr.net/npm/hyparquet@1.31.1/+esm";
const PARQUET_URL = (tower, split) =>
  `https://huggingface.co/api/datasets/${HF_DATASET}/parquet/${tower}/${split}/0.parquet`;
const GROUP_CODE = { "In-train": "IT", Interpolate: "IP", Extrapolate: "EX" };
const INPUT_COLS = [
  "sim_id", "wind_speed", "mean_wind_speed", "std_wind_speed", "wave_hs",
  "wave_tp", "wind_seed_id", "wind_group", "wave_group", "damage_weight",
];
const TOWER_COLS = [
  "sim_id", "section_id", "section_height_m", "section_radius_m",
  "section_thickness_m", "damage",
];

async function readParquet(hp, tower, split, columns) {
  const res = await fetch(PARQUET_URL(tower, split));
  if (!res.ok) throw new Error(`${tower}/${split}: HTTP ${res.status}`);
  const file = await res.arrayBuffer();
  state.hfBytes += file.byteLength;
  return hp.parquetReadObjects({ file, columns });
}

// Rebuild the explorer data (same layout as static/data/dataset.json) from
// the train and test parquet files of the three towers on Hugging Face.
async function datasetFromHF() {
  const hp = await import(HYPARQUET);
  state.hfBytes = 0;
  const jobs = [];
  for (const tower of TOWERS) {
    for (const split of ["train", "test"]) {
      const cols = tower === "ref" ? [...new Set([...INPUT_COLS, ...TOWER_COLS])] : TOWER_COLS;
      jobs.push(readParquet(hp, tower, split, cols).then((rows) => ({ tower, split, rows })));
    }
  }
  const parts = await Promise.all(jobs);

  // Simulations (inputs, labels, split) from the reference tower.
  const simRows = new Map();
  for (const { tower, split, rows } of parts) {
    if (tower !== "ref") continue;
    for (const r of rows) {
      if (!simRows.has(Number(r.sim_id))) simRows.set(Number(r.sim_id), { r, train: split === "train" ? 1 : 0 });
    }
  }
  const ids = [...simRows.keys()].sort((a, b) => a - b);
  const index = new Map(ids.map((id, i) => [id, i]));
  const sims = {
    sim_id: [], ws: [], mean_ws: [], std_ws: [], hs: [], tp: [], seed: [],
    wind: [], wave: [], train: [], weight: [],
  };
  for (const id of ids) {
    const { r, train } = simRows.get(id);
    sims.sim_id.push(id);
    sims.ws.push(Number(r.wind_speed));
    sims.mean_ws.push(Number(r.mean_wind_speed));
    sims.std_ws.push(Number(r.std_wind_speed));
    sims.hs.push(Number(r.wave_hs));
    sims.tp.push(Number(r.wave_tp));
    sims.seed.push(Number(r.wind_seed_id));
    sims.wind.push(GROUP_CODE[r.wind_group]);
    sims.wave.push(GROUP_CODE[r.wave_group]);
    sims.train.push(train);
    sims.weight.push(Number(r.damage_weight));
  }

  let nSections = 0;
  for (const { rows } of parts) for (const r of rows) nSections = Math.max(nSections, Number(r.section_id));
  const towers = {};
  const damage = {};
  for (const tower of TOWERS) {
    const logd = new Float32Array(ids.length * nSections).fill(NaN);
    const geom = { height: [], radius: [], thickness: [] };
    for (const { tower: tw, rows } of parts) {
      if (tw !== tower) continue;
      for (const r of rows) {
        const i = index.get(Number(r.sim_id));
        const k = Number(r.section_id) - 1;
        if (i === undefined) continue;
        logd[i * nSections + k] = Math.log10(Math.max(Number(r.damage), 1e-30));
        if (geom.height[k] === undefined) {
          geom.height[k] = Number(r.section_height_m);
          geom.radius[k] = Number(r.section_radius_m);
          geom.thickness[k] = Number(r.section_thickness_m) * 1000;
        }
      }
    }
    if (logd.some(Number.isNaN)) throw new Error(`${tower}: incomplete damage table`);
    const lifetime = new Array(nSections).fill(0);
    for (let i = 0; i < ids.length; i++) {
      for (let k = 0; k < nSections; k++) {
        lifetime[k] += Math.pow(10, logd[i * nSections + k]) * sims.weight[i];
      }
    }
    towers[tower] = Object.assign(geom, { lifetime });
    damage[tower] = logd;
  }
  return { data: { sims, n_sections: nSections, towers }, damage };
}

async function bundledDataset() {
  const data = await fetch("static/data/dataset.json").then((r) => r.json());
  const damage = {};
  TOWERS.forEach((tw) => { damage[tw] = decodeDamage(data.towers[tw].log_damage_i16); });
  return { data, damage };
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
      ["train", "Train", t.train],
      ["IT", "Test, in-train on both", t.it],
      ["IP", "Test, interpolate", t.ip],
      ["EX1", "Test, extrapolate on one axis", PAPER.lightRed],
      ["EXEX", "Test, EX_EX", t.ex],
    ];
  }
  const axisName = view === "wind" ? "wind" : "wave";
  return [
    ["train", "Train", t.train],
    ["IT", `Test, ${axisName} in-train`, t.it],
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
    traces.push(trace(te, { marker: { size, color: t.ip, opacity: 0.7 } }));
    traces.push(trace(tr, { marker: { size, color: t.train, opacity: 0.95, symbol: is3d ? "diamond" : "x" } }));
    legend.innerHTML = `<span><i style="background:${t.train}"></i>Train (${tr.length})</span><span><i style="background:${t.ip}"></i>Test (${te.length})</span>`;
  } else {
    const cats = categories(t, view);
    const html = [];
    for (const [key, label, col] of cats) {
      const ids = idx.filter((i) => simCategory(i, view) === key);
      if (!ids.length) continue;
      const symbol = key === "train" ? (is3d ? "diamond" : "x") : "circle";
      traces.push(trace(ids, { marker: { size, color: col, symbol, opacity: key === "train" ? 0.95 : 0.8 } }));
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
    colorscale: t.heatScale, showscale: false,
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
  if (state.data) { drawEnvelope(); drawProfile(); }
  if (state.lb) { recolorChips(); drawLeaderboard(); }
}

function watchTheme() {
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", redrawAll);
  new MutationObserver(redrawAll).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
}

async function loadDataset() {
  const note = document.getElementById("ex-source");
  const hfLink = `<a href="https://huggingface.co/datasets/${HF_DATASET}">${HF_DATASET}</a>`;
  let loaded;
  try {
    loaded = await datasetFromHF();
    note.innerHTML = `<span class="icon"><i class="fas fa-circle-check"></i></span> Loaded live from ${hfLink} `
      + `(6 parquet files, ${(state.hfBytes / 1e6).toFixed(1)} MB).`;
  } catch (err) {
    loaded = await bundledDataset();
    note.innerHTML = `Hugging Face could not be reached (${err.message}), so the explorer shows the copy of ${hfLink} bundled with this page.`;
  }
  state.data = loaded.data;
  state.damage = loaded.damage;
  initRegimeGrid();
  initExplorer();
}

async function main() {
  initCopy();
  initNav();
  initHF();
  watchTheme();
  const lbReady = fetch("static/data/leaderboard.json").then((r) => r.json()).then((lb) => {
    state.lb = lb;
    initLeaderboard();
  });
  await Promise.all([lbReady, loadDataset()]);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", main);
} else {
  main();
}
