/* Traffic Watch SPA. Vanilla JS, no build step, no external libs. */
"use strict";

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));

function el(tag, props = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "class") n.className = v;
    else if (k === "html") n.innerHTML = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else if (v != null) n.setAttribute(k, v);
  }
  for (const kid of kids.flat()) if (kid != null) n.append(kid);
  return n;
}

async function api(path, opts) {
  const r = await fetch(path, { credentials: "same-origin", ...opts });
  // The gate 404s once the cookie is gone (e.g. after sign-out or expiry).
  if (r.status === 404) { showLocked(); throw new Error("locked"); }
  if (!r.ok) throw new Error("HTTP " + r.status);
  return r.json();
}

// -- formatting --------------------------------------------------------------
const ARROW = { left_to_right: "→", right_to_left: "←" };
// COCO's "truck" class bundles pickups/SUVs/vans in with real trucks; on a residential
// street it's almost all the former, so relabel it honestly for DISPLAY only. The stored
// vehicle_type (DB, API, speedtest labels) stays the raw detector class.
const VEH_LABEL = { truck: "truck/SUV" };
const vlabel = (v) => VEH_LABEL[v] || v;
const TYPE_LABEL = { relative_speeding: "Speeding", center_lane_pass: "Center-lane",
  loud_engine: "Loud" };

function sevClass(kmh) { return kmh >= 80 ? "r" : kmh >= 70 ? "a" : "g"; }

function fmtTime(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  return d.toLocaleString([], { month: "short", day: "numeric",
    hour: "numeric", minute: "2-digit" });
}

// -- clip card ---------------------------------------------------------------
function clipCard(ev, rank) {
  const hasVideo = ev.has_video;
  const thumb = el("div", { class: "thumb",
    style: ev.thumb_url ? `background-image:url(${ev.thumb_url})` : "" });
  if (ev.speed_kmh != null)
    thumb.append(el("div", { class: "badge " + sevClass(ev.speed_kmh) },
      `${Math.round(ev.speed_kmh)} km/h`));
  if (rank != null)
    thumb.append(el("div", { class: "rank" },
      rank === 1 ? "🥇" : rank === 2 ? "🥈" : rank === 3 ? "🥉" : `#${rank}`));
  thumb.append(el("div", { class: "tag" },
    `${TYPE_LABEL[ev.event_type] || ev.event_type}${ev.annotated ? " • annotated" : ""}`));
  if (hasVideo) thumb.append(el("div", { class: "play", html: "▶" }));

  const meta = [vlabel(ev.vehicle_type), ARROW[ev.direction]].filter(Boolean).join(" ");
  const card = el("div", { class: "card" + (hasVideo ? "" : " novideo") },
    thumb,
    el("div", { class: "card-body" },
      el("span", {}, meta || (hasVideo ? "clip" : "no video")),
      el("span", { class: "when" }, fmtTime(ev.iso))));
  if (hasVideo) card.addEventListener("click", () => openModal(ev));
  return card;
}

function statCard(num, lbl) {
  return el("div", { class: "stat" },
    el("div", { class: "num" }, num == null ? "—" : String(num)),
    el("div", { class: "lbl" }, lbl));
}

// -- views -------------------------------------------------------------------
async function loadNow() {
  const data = await api("/api/now");
  const s = data.summary;
  $("#now-thresh").textContent = `(${Math.round(data.fast_threshold)} km/h+)`;
  const v = data.volume || {};
  const sg = $("#now-stats"); sg.innerHTML = "";
  sg.append(
    statCard(v.total != null ? v.total : "—", "Cars today"),
    statCard(v.over_limit_pct != null ? v.over_limit_pct + "%" : "—",
             `Over ${Math.round(v.over_kmh || 55)}`),
    statCard(s.over_fast, `Over ${Math.round(data.fast_threshold)}`),
    statCard(s.max_kmh != null ? s.max_kmh : "—", "Top speed"));
  const reel = $("#now-reel"); reel.innerHTML = "";
  if (!data.events.length)
    reel.append(el("p", { class: "muted" }, "No fast clips yet today."));
  else data.events.forEach((ev) => reel.append(clipCard(ev)));
}

let browseOffset = 0;
const BROWSE_PAGE = 200;
// append=false -> fresh load (filter change); append=true -> "Load more" next page.
async function loadBrowse(append = false) {
  if (!append) browseOffset = 0;
  const speed = $("#f-speed").value, type = $("#f-type").value, days = $("#f-days").value;
  const date = $("#f-date").value;
  const q = new URLSearchParams({ limit: String(BROWSE_PAGE), offset: String(browseOffset) });
  if (date) {
    // A specific day (local midnight..+24h) overrides the rolling Range so you can
    // jump straight to an older day instead of paging back through busy recent days.
    const start = Math.floor(new Date(date + "T00:00:00").getTime() / 1000);
    q.set("since", String(start));
    q.set("until", String(start + 86400));
  } else if (+days > 0) {
    q.set("days", days);
  }
  if (+speed > 0) q.set("min_speed", speed);
  if (type) q.set("type", type);
  const data = await api("/api/events?" + q);
  const grid = $("#browse-grid");
  if (!append) grid.innerHTML = "";
  data.events.forEach((ev) => grid.append(clipCard(ev)));
  browseOffset += data.count;
  $("#browse-empty").classList.toggle("hidden", grid.children.length > 0);
  $("#browse-more").classList.toggle("hidden", data.count < BROWSE_PAGE);
}

async function loadHall() {
  const data = await api("/api/hall");
  $("#hall-thresh").textContent = `(${Math.round(data.threshold)} km/h+)`;
  const grid = $("#hall-grid"); grid.innerHTML = "";
  $("#hall-empty").classList.toggle("hidden", data.events.length > 0);
  data.events.forEach((ev, i) => grid.append(clipCard(ev, i + 1)));

  // Manually hidden entries: collapsed by default, restorable from the modal.
  const hidden = data.hidden || [];
  const wrap = $("#hall-hidden-wrap"), hgrid = $("#hall-hidden-grid");
  wrap.classList.toggle("hidden", hidden.length === 0);
  hgrid.innerHTML = "";
  hgrid.classList.add("hidden");
  $("#hall-hidden-toggle").textContent = `Show hidden (${hidden.length})`;
  hidden.forEach((ev) => hgrid.append(clipCard(ev)));
}

async function loadStats() {
  const days = $("#s-days").value;
  const d = await api("/api/stats?days=" + days);
  const s = d.summary, v = d.volume || {};
  const note = $("#stats-note");
  note.classList.toggle("hidden", !d.clamped);
  if (d.clamped) note.textContent =
    `Cars + violations shown together since ${d.coverage_since} (when car-counting began).`;
  const sg = $("#stats-summary"); sg.innerHTML = "";
  sg.append(
    statCard(v.total != null ? v.total : "—", "Total cars"),
    statCard(v.over_limit_pct != null ? v.over_limit_pct + "%" : "—",
             `Over ${Math.round(v.over_kmh || 55)}`),
    statCard(s.count, "Violations (55+)"),
    statCard(s.max_kmh != null ? s.max_kmh : "—", "Top speed"));

  barChart($("#chart-daily"), d.daily.map((x) => ({
    v: x.count, label: x.date.slice(5),
    cls: x.over_fast > 0 ? "amber" : "",
    title: `${x.date}: ${x.count} (${x.over_fast} fast)` })), 8);

  barChart($("#chart-hourly"), d.hourly.map((x) => ({
    v: x.count, label: String(x.hour).padStart(2, "0"),
    title: `${x.hour}:00 — ${x.count}` })), 8);

  barChart($("#chart-dist"), d.distribution.map((x) => ({
    v: x.count, label: x.bucket,
    cls: x.min_kmh >= 80 ? "red" : x.min_kmh >= 70 ? "amber" : "",
    title: `${x.bucket} km/h: ${x.count}` })), 12);

  hbars($("#chart-vehicles"), d.vehicles.map((x) => ({
    v: x.count, label: vlabel(x.vehicle_type) })));
}

// -- tiny charts (pure DOM) --------------------------------------------------
function barChart(container, data, maxLabels) {
  container.innerHTML = "";
  const max = Math.max(1, ...data.map((d) => d.v));
  const bars = el("div", { class: "bars" });
  data.forEach((d) => bars.append(el("div", {
    class: "bar " + (d.cls || ""), title: d.title || `${d.label}: ${d.v}`,
    style: `height:${(d.v / max) * 100}%` })));
  const every = Math.ceil(data.length / maxLabels);
  const axis = el("div", { class: "axis" });
  data.forEach((d, i) => axis.append(
    el("span", {}, i % every === 0 ? d.label : "")));
  container.append(bars, axis);
}

function hbars(container, data) {
  container.innerHTML = "";
  const max = Math.max(1, ...data.map((d) => d.v));
  const list = el("div", { class: "hlist" });
  if (!data.length) { container.append(el("p", { class: "muted" }, "No data.")); return; }
  data.forEach((d) => list.append(el("div", { class: "hrow" },
    el("span", {}, d.label),
    el("div", { class: "track" }, el("div", { class: "fill", style: `width:${(d.v / max) * 100}%` })),
    el("span", { class: "muted" }, String(d.v)))));
  container.append(list);
}

// -- video modal -------------------------------------------------------------
let modalEvent = null;
function openModal(ev) {
  modalEvent = ev;
  const toggle = $("#annot-toggle"), wrap = $("#annot-toggle-wrap");
  wrap.classList.toggle("hidden", !ev.annotated);
  toggle.checked = ev.annotated;
  $("#modal-title").textContent =
    [ev.speed_kmh != null ? Math.round(ev.speed_kmh) + " km/h" : null,
     vlabel(ev.vehicle_type), TYPE_LABEL[ev.event_type] || ev.event_type, fmtTime(ev.iso)]
      .filter(Boolean).join(" · ");
  // The hide/restore control only makes sense on the Top Speeds page.
  const hallBtn = $("#hall-btn");
  hallBtn.classList.toggle("hidden", current !== "hall");
  hallBtn.textContent = ev.excluded ? "↩ Restore to Hall" : "🚫 Hide from Hall";
  setPlayerSrc();
  $("#modal").classList.remove("hidden");
}

async function toggleHall() {
  if (!modalEvent) return;
  const path = modalEvent.excluded ? "/api/hall/restore" : "/api/hall/exclude";
  try {
    await api(path, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ event_id: modalEvent.id }),
    });
  } catch (e) { console.error(e); return; }
  closeModal();
  loadHall().catch((e) => console.error(e));
}
function setPlayerSrc() {
  const p = $("#player"), annot = $("#annot-toggle").checked;
  p.src = modalEvent.clip_url + (annot ? "" : "?annotated=false");
  p.load(); p.play().catch(() => {});
  // Download link mirrors the annotated/clean choice and gets a descriptive name.
  const dl = $("#dl-btn");
  dl.href = modalEvent.clip_url + (annot ? "?download=1" : "?download=1&annotated=false");
  const suffix = annot && modalEvent.annotated ? "_annotated" : "";
  dl.setAttribute("download", `${modalEvent.stem}${suffix}.mp4`);
}
function closeModal() {
  const p = $("#player"); p.pause(); p.removeAttribute("src"); p.load();
  $("#modal").classList.add("hidden");
}

// -- speed test --------------------------------------------------------------
// The PC clock so a drive-by can be matched to the system's timestamps. We sync
// an offset to the server once, then tick locally (re-sync periodically).
let clockOffset = null; // serverEpoch - clientEpoch (seconds)
async function syncClock() {
  try { const t = await api("/api/time"); clockOffset = t.ts - Date.now() / 1000; }
  catch (e) { /* locked/offline: leave last good offset */ }
}
function hms(epoch) { return new Date(epoch * 1000).toLocaleTimeString("en-GB"); }
function tickClock() {
  const c = $("#pc-clock");
  if (!c || clockOffset == null) return;
  c.textContent = hms(Date.now() / 1000 + clockOffset);
}

function spPassRow(p) {
  const meta = [vlabel(p.vehicle_type), ARROW[p.direction]].filter(Boolean).join(" ");
  return el("div", { class: "sp-row" },
    el("span", { class: "sp-t" }, hms(p.ts)),
    el("span", { class: "sp-spd" }, `${p.measured}`),
    el("span", { class: "sp-meta muted" }, meta),
    el("button", { class: "ghost sp-pick", onclick: () => labelPass(p) }, "This was me"));
}

async function labelPass(p) {
  const v = window.prompt(
    `True GPS speed for the ${hms(p.ts)} pass (system measured ${p.measured} km/h)?`, "");
  if (v == null) return;
  const t = parseFloat(v);
  if (!(t > 0)) { window.alert("Enter a positive speed in km/h."); return; }
  await api("/api/speedtest/label", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ key: p.key, ts: p.ts, measured: p.measured,
      true_speed: t, direction: p.direction, vehicle_type: p.vehicle_type }) });
  loadSpeed();
}

function spLogRow(r) {
  const errCls = r.error_pct == null ? "" : (Math.abs(r.error_pct) <= 5 ? "ok" : "bad");
  const errTxt = r.error_pct == null ? "—"
    : `${r.error_pct > 0 ? "+" : ""}${r.error_pct}%`;
  return el("div", { class: "sp-row sp-logrow" },
    el("span", { class: "sp-t" }, hms(r.ts)),
    el("span", { class: "sp-spd" }, `${r.measured}→${r.true_speed}`),
    el("span", { class: "sp-err " + errCls }, errTxt),
    el("span", { class: "sp-meta muted" }, ARROW[r.direction] || ""),
    el("button", { class: "ghost icon", title: "Remove",
      onclick: () => delTest(r.id) }, "✕"));
}

async function delTest(key) {
  await api("/api/speedtest/unlabel", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ key }) });
  loadSpeed();
}

async function loadSpeed() {
  syncClock();
  const mins = ($("#sp-mins") && $("#sp-mins").value) || 10;
  const [pd, ld] = await Promise.all([
    api("/api/speedtest/passes?minutes=" + mins),
    api("/api/speedtest/log")]);
  const pg = $("#sp-passes"); pg.replaceChildren(...pd.passes.map(spPassRow));
  $("#sp-passes-empty").classList.toggle("hidden", pd.passes.length > 0);
  const lg = $("#sp-log"); lg.replaceChildren(...ld.tests.map(spLogRow));
  $("#sp-log-empty").classList.toggle("hidden", ld.tests.length > 0);
}

// -- routing + auth ----------------------------------------------------------
const LOADERS = { now: loadNow, browse: loadBrowse, hall: loadHall, stats: loadStats,
  speed: loadSpeed };
let current = "now";
function switchView(view) {
  current = view;
  // Deep-linkable tabs: /#stats opens the Stats view directly (replaceState so
  // tab-hopping doesn't pollute browser history).
  history.replaceState(null, "", "#" + view);
  $$(".tab").forEach((t) => t.classList.toggle("active", t.dataset.view === view));
  $$(".view").forEach((v) => v.classList.add("hidden"));
  $("#view-" + view).classList.remove("hidden");
  LOADERS[view]().catch((e) => console.error(e));
}

function showLocked() {
  $("#app").classList.add("hidden");
  $("#locked").classList.remove("hidden");
}

// -- refresh -----------------------------------------------------------------
// Installed PWAs (Add to Home Screen) have no browser chrome to reload, so the
// data would otherwise only refresh on a full app restart. Reload the current
// view on demand (button) and automatically whenever the app regains focus.
let refreshing = false;
async function refresh() {
  if (refreshing) return;
  refreshing = true;
  const btn = $("#refresh");
  if (btn) btn.classList.add("spinning");
  try { await LOADERS[current](); }
  catch (e) { console.error(e); }
  finally { refreshing = false; if (btn) btn.classList.remove("spinning"); }
}

let lastAuto = 0;
function autoRefresh() {
  // Skip while the locked overlay is up; debounce so visibility+focus don't double-fire.
  if (!$("#locked").classList.contains("hidden")) return;
  const t = Date.now();
  if (t - lastAuto < 1500) return;
  lastAuto = t;
  refresh();
}

function wire() {
  $$(".tab").forEach((t) => t.addEventListener("click", () => switchView(t.dataset.view)));
  ["#f-speed", "#f-type", "#f-days", "#f-date"].forEach((s) =>
    $(s).addEventListener("change", () => loadBrowse()));
  $("#browse-more").addEventListener("click", () => loadBrowse(true));
  $("#s-days").addEventListener("change", () => loadStats());
  $("#sp-mins").addEventListener("change", () => loadSpeed());
  $("#logout").addEventListener("click", async () => {
    try { await fetch("/api/logout", { method: "POST", credentials: "same-origin" }); }
    finally { showLocked(); }
  });
  $("#modal-close").addEventListener("click", closeModal);
  $(".modal-bg").addEventListener("click", closeModal);
  $("#annot-toggle").addEventListener("change", setPlayerSrc);
  $("#hall-btn").addEventListener("click", toggleHall);
  $("#hall-hidden-toggle").addEventListener("click", () => {
    const g = $("#hall-hidden-grid"), shown = !g.classList.contains("hidden");
    g.classList.toggle("hidden", shown);
    $("#hall-hidden-toggle").textContent =
      `${shown ? "Show" : "Hide"} hidden (${g.children.length})`;
  });
  $("#refresh").addEventListener("click", refresh);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !$("#modal").classList.contains("hidden")) closeModal();
  });
  // Auto-refresh when the (installed) app is reopened or brought back to the front.
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") autoRefresh();
  });
  window.addEventListener("focus", autoRefresh);
  window.addEventListener("pageshow", (e) => { if (e.persisted) autoRefresh(); });
}

function start() {
  // If this script loaded at all, the gate already let us in (valid cookie).
  wire();
  // Honor a #view deep link (e.g. /#stats); fall back to the Now page.
  const initial = location.hash.slice(1);
  switchView(LOADERS[initial] ? initial : "now");
  // PC clock: sync the offset to the server, then tick locally; re-sync slowly.
  syncClock();
  setInterval(tickClock, 250);
  setInterval(syncClock, 15000);
}
start();
