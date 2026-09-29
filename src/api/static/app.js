"use strict";

const $ = (id) => document.getElementById(id);
const fileInput = $("file"), drop = $("drop"), solveBtn = $("solve"), statusBox = $("status");
let currentFile = null, lastResult = null;

function showImage(frameId, src, alt) {
  const frame = $(frameId);
  frame.replaceChildren();
  if (!src) {
    const span = document.createElement("span");
    span.className = "empty";
    span.textContent = "—";
    frame.append(span);
    return;
  }
  const img = document.createElement("img");
  img.src = src;
  img.alt = alt;
  img.addEventListener("click", () => window.open(src, "_blank"));
  frame.append(img);
}

function setStatus(text, kind = "info") {
  statusBox.hidden = false;
  statusBox.className = "panel status" + (kind === "error" ? " error" : "");
  statusBox.replaceChildren();
  if (kind === "loading") {
    const s = document.createElement("span");
    s.className = "spinner";
    statusBox.append(s);
  }
  statusBox.append(document.createTextNode(text));
}

function selectFile(file) {
  if (!file) return;
  currentFile = file;
  lastResult = null;
  $("details").hidden = true;
  showImage("f-original", URL.createObjectURL(file), "Imagen original");
  showImage("f-overlay", null);
  showImage("f-clean", null);
  solveBtn.disabled = false;
  setStatus(`Imagen lista: ${file.name}. Pulsa «Resolver».`);
}

fileInput.addEventListener("change", () => selectFile(fileInput.files[0]));
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => {
  e.preventDefault();
  drop.classList.add("over");
}));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => {
  e.preventDefault();
  drop.classList.remove("over");
}));
drop.addEventListener("drop", (e) => selectFile(e.dataTransfer.files[0]));

async function runExample(name) {
  const blob = await (await fetch(`/api/examples/${encodeURIComponent(name)}`)).blob();
  selectFile(new File([blob], name, { type: blob.type }));
  solve();
}

async function loadExamples() {
  try {
    const names = await (await fetch("/api/examples")).json();
    for (const name of names) {
      const b = document.createElement("button");
      b.className = "chip";
      b.textContent = name.replace(/^ejemplo_/, "").replace(/\.jpg$/, "").replaceAll("_", " ");
      b.addEventListener("click", () => runExample(name));
      $("examples").append(b);
    }
    // ?ejemplo=negro_foto resuelve ese ejemplo al abrir la página (útil para demos)
    const wanted = new URLSearchParams(location.search).get("ejemplo");
    const match = wanted && names.find((n) => n === `ejemplo_${wanted}.jpg` || n === wanted);
    if (match) runExample(match);
  } catch {
    /* sin ejemplos: no es crítico */
  }
}

function row(label, value, cls = "") {
  const div = document.createElement("div");
  const dt = document.createElement("dt");
  const dd = document.createElement("dd");
  dt.textContent = label;
  dd.textContent = value;
  if (cls) dd.className = cls;
  div.append(dt, dd);
  return div;
}

function renderResult(r) {
  showImage("f-overlay", r.images.overlay, "Solución superpuesta sobre la foto");
  showImage("f-clean", r.images.clean, "Grilla limpia con la solución");
  const uniq = r.unique === true ? "Sí" : r.unique === false ? "No" : "—";
  $("summary").replaceChildren(
    row("Grilla", `${r.rows} × ${r.cols}`),
    row("Estado", r.solved ? "Resuelto" : r.status, r.solved ? "ok" : "no"),
    row("Solución única", uniq, r.unique === false ? "no" : ""),
    row("Modelo", r.model),
    row("Pistas corregidas", String(r.corrected_clues.length), r.corrected_clues.length ? "no" : ""),
    row("Tiempo visión", `${(r.timings.vision_s * 1000).toFixed(0)} ms`),
    row("Tiempo solver", `${(r.timings.solver_s * 1000).toFixed(0)} ms`),
  );
  const notes = $("notes");
  notes.replaceChildren();
  const add = (text, cls) => {
    const li = document.createElement("li");
    li.className = cls;
    li.textContent = text;
    notes.append(li);
  };
  const dirName = { right: "horizontal", down: "vertical" };
  r.corrected_clues.forEach((c) => add(
    `Pista ${dirName[c.dir]} en (${c.row}, ${c.col}) corregida por el solver: ${c.from} → ${c.to} (resaltada en naranja).`, "warn"));
  r.conflicts.forEach((c) => add(
    `Tramo en conflicto: pista ${dirName[c.dir]} en (${c.row}, ${c.col}) = ${c.total}.`, "err"));
  r.errors.forEach((e) => add(e, "err"));
  r.warnings.forEach((w) => add(w, "warn"));
  if (!r.solved) add("No se encontró una solución: revisa si la grilla o alguna pista se leyó mal.", "err");
  $("details").hidden = false;
}

async function solve() {
  if (!currentFile) return;
  solveBtn.disabled = true;
  setStatus("Procesando: visión computacional y solver CP…", "loading");
  const form = new FormData();
  form.append("file", currentFile);
  form.append("model", $("model").value);
  try {
    const res = await fetch("/api/solve", { method: "POST", body: form });
    const body = await res.json();
    if (!res.ok) throw new Error(body.detail || `Error ${res.status}`);
    lastResult = body;
    renderResult(body);
    setStatus(body.solved ? "Listo: puzzle resuelto." : `El solver terminó con estado ${body.status}.`,
      body.solved ? "info" : "error");
  } catch (err) {
    setStatus(`No se pudo resolver: ${err.message}`, "error");
  } finally {
    solveBtn.disabled = false;
  }
}

function download(obj, name) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(obj, null, 1)], { type: "application/json" }));
  const a = Object.assign(document.createElement("a"), { href: url, download: name });
  a.click();
  URL.revokeObjectURL(url);
}

solveBtn.addEventListener("click", solve);
$("dl-puzzle").addEventListener("click", () => lastResult && download(lastResult.puzzle, "puzzle.json"));
$("dl-solution").addEventListener("click", () => lastResult && download(lastResult.solution, "solution.json"));
loadExamples();
