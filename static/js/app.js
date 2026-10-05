const $ = (s) => document.querySelector(s);
const state = { model: "word2vec_skipgram", view: "single", models: [], ready: false };
const examples = ["shipping update", "forgot my password", "refund for damaged item", "track my order", "change delivery address"];

// ---------- theme ----------
const savedTheme = localStorage.getItem("theme");

if (savedTheme) {
  document.documentElement.dataset.theme = savedTheme;
} else {
  document.documentElement.dataset.theme = "light";
}

$("#themeBtn").onclick = () => {
  const t =
    document.documentElement.dataset.theme === "light"
      ? "dark"
      : "light";

  document.documentElement.dataset.theme = t;
  localStorage.setItem("theme", t);
};

// ---------- helpers ----------
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
function banner(msg, type = "info") {
  const b = $("#banner"); b.className = "banner show " + type; b.textContent = msg;
}
function hideBanner() { $("#banner").className = "banner"; }

// ---------- boot ----------
async function boot() {
  const r = await fetch("/stats").then((r) => r.json()).catch(() => null);
  if (!r) { banner("Cannot reach the server.", "err"); return; }
  if (!r.ready) {
    banner(r.message + " …", "info");
    setTimeout(boot, 2000);
    return;
  }
  hideBanner();
  state.ready = true; state.models = r.models;
  $("#stats").innerHTML = `
    <div class="stat"><b>${r.documents}</b><span>Documents</span></div>
    <div class="stat"><b>${r.categories.length}</b><span>Categories</span></div>
    <div class="stat"><b>${r.vector_size}</b><span>Vector dimensions</span></div>
    <div class="stat"><b>${r.models.length}</b><span>Embedding models</span></div>`;
  $("#category").innerHTML = `<option>All</option>` + r.categories.map((c) => `<option value="${esc(c.name)}">${esc(c.name)} (${c.count})</option>`).join("");
  $("#modelSeg").innerHTML = r.models.map((m) => `<button data-m="${m.id}" class="${m.id === state.model ? "on" : ""}" title="Vocabulary: ${m.vocab}">${m.label}</button>`).join("");
  $("#modelSeg").querySelectorAll("button").forEach((b) => b.onclick = () => {
    state.model = b.dataset.m;
    $("#modelSeg").querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
    if ($("#query").value.trim()) search();
  });
}

$("#chips").innerHTML = examples.map((e) => `<button class="chip">${e}</button>`).join("");
$("#chips").querySelectorAll(".chip").forEach((c) => c.onclick = () => { $("#query").value = c.textContent; search(); });
$("#topk").oninput = (e) => $("#kVal").textContent = e.target.value;
$("#viewSeg").querySelectorAll("button").forEach((b) => b.onclick = () => {
  state.view = b.dataset.v;
  $("#viewSeg").querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
  $("#modelSeg").style.opacity = state.view === "compare" ? .4 : 1;
  if ($("#query").value.trim()) search();
});

// ---------- search ----------
function resultCard(r, delay = 0) {
  const pct = Math.max(0, Math.min(100, r.similarity_score * 100));
  return `<div class="result" style="animation-delay:${delay}ms">
    <div class="rank">${r.rank}</div>
    <div>
      <div class="rtitle">${esc(r.title)}</div>
      <div class="badges"><span class="badge">${esc(r.category)}</span><span class="badge id">${esc(r.document_id)}</span></div>
      <div class="content">${esc(r.content)}</div>
      ${r.content.length > 180 ? `<button class="more" onclick="this.closest('.result').classList.toggle('open'); this.textContent = this.closest('.result').classList.contains('open') ? 'Show less' : 'Read more'">Read more</button>` : ""}
      ${r.keywords ? `<div class="kw">🏷 ${esc(r.keywords)}</div>` : ""}
    </div>
    <div class="score"><b>${pct.toFixed(1)}%</b><div class="bar"><i style="width:${pct}%"></i></div><small>similarity</small></div>
  </div>`;
}

function metaBlock(d) {
  const toks = d.known_tokens.map((t) => `<span class="tok ok">${esc(t)}</span>`).join("") +
               d.unknown_tokens.map((t) => `<span class="tok oov" title="Not in vocabulary">${esc(t)}</span>`).join("");
  return `<div class="meta"><div>Processed query: ${toks || "<i>(nothing left after cleaning)</i>"}</div>
          <div>${state.view === "single" ? esc(d.model_label) : "All models"} · ${d.results.length} result(s)</div></div>`;
}

async function search() {
  const q = $("#query").value.trim();

  if (!q) {
    banner("Please enter a search query.", "err");
    $("#query").focus();
    return;
  }
  if (!state.ready) { banner("Index is still building, please wait…", "info"); return; }
  $("#searchBtn").disabled = true;
  $("#results").innerHTML = `<div class="skeleton"></div><div class="skeleton"></div><div class="skeleton"></div>`;
  const params = new URLSearchParams({
  query: q,
  model: state.model,
  top_k: $("#topk").value,
  category: $("#category").value
});

try {
  const endpoint = state.view === "compare"
    ? "/compare"
    : "/search";

  const res = await fetch(`${endpoint}?${params.toString()}`);
    const data = await res.json();
    if (!data.processed_query && state.view === "single") {
      $("#meta").innerHTML = "";
      $("#results").innerHTML = `
        <div class="empty">
          <div class="big">🤷</div>
          <p>No searchable words remained after cleaning the query.</p>
        </div>
      `;
      banner("Your query became empty after preprocessing. Try different words.", "err");
      return;
    }
    if (!res.ok) throw new Error(data.detail || "Search failed");
    hideBanner();
    if (state.view === "compare") {
      const first = Object.values(data)[0];
      $("#meta").innerHTML = metaBlock(first);
      $("#results").innerHTML = `<div class="compare">` + Object.values(data).map((d) =>
        `<div class="col"><h3>${esc(d.model_label)}</h3>${d.results.map((r, i) => resultCard(r, i * 50)).join("") || '<div class="empty">No results</div>'}</div>`).join("") + `</div>`;
    } else {
      $("#meta").innerHTML = metaBlock(data);
      $("#results").innerHTML = data.results.map((r, i) => resultCard(r, i * 60)).join("") ||
        `<div class="empty"><div class="big">🤷</div><p>No matching documents.</p></div>`;
    }
  } catch (e) {
    $("#results").innerHTML = ""; banner(e.message, "err");
  } finally { $("#searchBtn").disabled = false; }
}
$("#searchBtn").onclick = search;
$("#query").addEventListener("keydown", (e) => { if (e.key === "Enter") search(); });
$("#category").onchange = () => { if ($("#query").value.trim()) search(); };
$("#topk").onchange = () => { if ($("#query").value.trim()) search(); };

// ---------- upload ----------
$("#uploadBtn").disabled = true;
$("#uploadBtn").title = "CSV upload is currently disabled";
// $("#fileInput").onchange = async (e) => {
//   const f = e.target.files[0]; if (!f) return;
//   const fd = new FormData(); fd.append("file", f);
//   banner("Uploading and rebuilding the index (this trains 3 models, may take a moment)…");
//   state.ready = false;
//   try {
//     const res = await fetch("/api/upload", { method: "POST", body: fd });
//     const d = await res.json();
//     if (!res.ok) throw new Error(d.detail || "Upload failed");
//     setTimeout(boot, 1500);
//   } catch (err) { banner(err.message, "err"); state.ready = true; }
//   e.target.value = "";
// };

boot();
