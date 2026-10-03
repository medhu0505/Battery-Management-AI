/* BMS Copilot dashboard. All dynamic text goes through textContent. */
(function () {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s);
  const S = { fleet: null, deviceId: null, device: null, tab: "fleet", session: null, sort: { key: "soh", dir: 1 },
              filter: { q: "", profile: "", anomaly: "" }, wf: { status: "", kind: "" }, lastRun: null, plan: null };

  // ---------------------------------------------------------------- helpers
  function h(tag, attrs, ...kids) {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v === null || v === undefined || v === false) continue;
      if (k === "class") n.className = v;
      else if (k === "style") n.style.cssText = v;
      else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
      else n.setAttribute(k, v === true ? "" : v);
    }
    for (const c of kids.flat()) if (c !== null && c !== undefined && c !== false)
      n.appendChild(c instanceof Node ? c : document.createTextNode(String(c)));
    return n;
  }
  async function api(path, opts = {}) {
    const r = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
    const body = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(body.detail || r.statusText);
    return body;
  }
  const pct = (v, d = 1) => (v === null || v === undefined) ? "-" : (100 * v).toFixed(d) + " %";
  const num = (v, d = 0) => (v === null || v === undefined) ? "-" : Number(v).toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: d });
  const days = v => v === null || v === undefined ? "-" : `${num(v)} d`;
  const clock = v => { const t = ((v % 24) + 24) % 24; return `${String(Math.floor(t)).padStart(2, "0")}:${String(Math.round((t % 1) * 60) % 60).padStart(2, "0")}`; };
  const LEVEL_STATUS = { normal: ["good", "Normal"], watch: ["warning", "Watch"], warning: ["serious", "Warning"], critical: ["critical", "Critical"] };
  const ICONS = {
    good: '<path d="M4 8.5l2.5 2.5L12 5.5" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>',
    warning: '<path d="M8 4.5v4.2" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><circle cx="8" cy="11.4" r="1.1" fill="currentColor"/>',
    serious: '<path d="M8 4.5v4.2" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><circle cx="8" cy="11.4" r="1.1" fill="currentColor"/>',
    critical: '<path d="M5.5 5.5l5 5M10.5 5.5l-5 5" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>',
  };
  function statusBadge(level) {
    const [role, label] = LEVEL_STATUS[level] || ["good", level];
    const s = h("span", { class: "status " + (level === "normal" ? "normal" : "") });
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("width", "16"); svg.setAttribute("height", "16"); svg.setAttribute("viewBox", "0 0 16 16");
    svg.style.color = `var(--${role})`;
    svg.innerHTML = `<circle cx="8" cy="8" r="7" fill="currentColor" fill-opacity="0.16"/>` + ICONS[role]; // static markup only
    s.append(svg, h("span", {}, label));
    return s;
  }
  function tile(label, value, note, cls) {
    return h("div", { class: "card tile " + (cls || "") }, h("div", { class: "label" }, label), h("div", { class: "value" }, value),
             note ? h("div", { class: "note" }, note) : null);
  }
  function card(title, desc, ...body) {
    return h("div", { class: "card" }, h("h2", {}, title), desc ? h("p", { class: "desc" }, desc) : null, ...body);
  }
  function busy(btn, on, text) {
    if (!btn) return;
    if (on) { btn.dataset.label = btn.textContent; btn.textContent = text || "Working..."; btn.disabled = true; }
    else { btn.textContent = btn.dataset.label || btn.textContent; btn.disabled = false; }
  }
  function toastError(e, where) {
    (where || document.body).prepend(h("div", { class: "callout", style: "border-color:var(--critical);margin-bottom:10px" }, "Error: " + e.message));
  }
  function kvTable(obj) {
    const g = h("div", { class: "kv" });
    for (const [k, v] of Object.entries(obj)) {
      g.append(h("div", { class: "k" }, k.replaceAll("_", " ")), h("div", { class: "v" }, fmtValue(v)));
    }
    return g;
  }
  function fmtValue(v) {
    if (v === null || v === undefined) return "-";
    if (typeof v === "number") return Math.abs(v) < 10 && !Number.isInteger(v) ? v.toFixed(4) : num(v, Number.isInteger(v) ? 0 : 2);
    if (Array.isArray(v)) return v.map(fmtValue).join(", ");
    // Nested records read as "key: value; key: value" rather than raw JSON.
    if (typeof v === "object") return Object.entries(v).map(([k, x]) => `${k.replaceAll("_", " ")}: ${fmtValue(x)}`).join("; ");
    return String(v);
  }

  // ------------------------------------------------------------------ tabs
  const TABS = { laptop: () => renderLaptop(false), fleet: renderFleet, device: renderDevice, charging: renderCharging,
                 workforce: renderWorkforce, passport: renderPassport, models: renderModels };

  function switchTab(name) {
    if (!TABS[name]) name = "laptop";
    S.tab = name;
    document.querySelectorAll("#tabs button").forEach(b => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
    document.querySelectorAll('section[role="tabpanel"]').forEach(s => s.classList.toggle("hidden", s.id !== "tab-" + name));
    Charts.reset();
    // Shareable deep link, e.g. #tab=device&id=DEV-0120
    const link = "#tab=" + name + (["device", "charging", "passport"].includes(name) && S.deviceId ? "&id=" + S.deviceId : "");
    history.replaceState(null, "", link);
    return Promise.resolve(TABS[name]());
  }
  $("#tabs").addEventListener("click", e => { if (e.target.dataset.tab) switchTab(e.target.dataset.tab); });
  function openDevice(id) { S.deviceId = id; switchTab("device"); }

  function deviceSelect(onChange) {
    const sel = h("select", { "aria-label": "Device", onchange: e => {
      S.deviceId = e.target.value;
      history.replaceState(null, "", `#tab=${S.tab}&id=${S.deviceId}`);
      onChange();
    } });
    (S.fleet?.devices || []).forEach(d => sel.appendChild(h("option", { value: d.device_id, selected: d.device_id === S.deviceId },
      `${d.device_id} - ${d.profile_label} - SoH ${pct(d.soh, 0)}`)));
    return sel;
  }


  // ------------------------------------------------------- this laptop (live)
  let liveTimer = null;
  function recBadge(level) {
    const map = { good: ["good", "Good"], info: ["warning", "Note"], warning: ["serious", "Attention"], critical: ["critical", "Critical"] };
    const [role, label] = map[level] || ["warning", level];
    const s = h("span", { class: "status" });
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("width", "16"); svg.setAttribute("height", "16"); svg.setAttribute("viewBox", "0 0 16 16");
    svg.style.color = `var(--${role})`;
    svg.innerHTML = `<circle cx="8" cy="8" r="7" fill="currentColor" fill-opacity="0.16"/>` + ICONS[role]; // static markup only
    s.append(svg, h("span", {}, label));
    return s;
  }
  const monthLabel = (startIso, day) => { const d = new Date(new Date(startIso).getTime() + day * 86400000); return d.toLocaleDateString(undefined, { month: "short", year: "2-digit" }); };

  async function renderLaptop(refresh) {
    const root = $("#tab-laptop");
    if (liveTimer) { clearInterval(liveTimer); liveTimer = null; }
    root.classList.add("loading");
    if (!root.firstChild) root.appendChild(h("div", { class: "empty" }, "Reading this computer's battery (Windows battery report + live sensors)..."));
    let b;
    try { b = await api("/api/mybattery" + (refresh ? "?refresh=true" : "")); } catch (e) { root.classList.remove("loading"); toastError(e, root); return; }
    root.classList.remove("loading");
    if (!b.available) {
      root.replaceChildren(card("No battery available", null, h("p", {}, b.reason),
        h("p", { class: "secondary" }, "On another Windows laptop run  powercfg /batteryreport /xml /output report.xml  and start the API with BMS_BATTERY_REPORT=report.xml.")));
      return;
    }
    const bat = b.battery, fc = b.forecast, prof = b.profile, live = b.live || {}, proj = b.projection;
    const c60 = fc.crossings?.["60"] || {}, c70 = fc.crossings?.["70"] || {};
    const monthsTxt = c => { if (c.status === "already_below") return "already below"; if (c.median_days == null) return "beyond 3 years";
      const mo = Math.max(1, Math.round(c.median_days / 30.4)); return mo === 1 ? "about 1 month" : `${mo} months`; };
    const planBox = h("div", { id: "lp-plan" }, h("div", { class: "empty" }, "Plan tonight's charge from the live state of charge and your learned routine."));
    root.replaceChildren(
      h("div", { class: "callout", style: "margin-bottom:12px" },
        `Real data from ${b.source}: ${bat.manufacturer} ${bat.chemistry} battery, ${bat.history_entries} capacity-history points, live sensor polling every ${b.poller.interval_s} s (${b.poller.samples_logged} samples logged). Nothing here is simulated.`),
      h("div", { class: "grid cols-4" },
        h("div", { class: "card tile hero" }, h("div", { class: "label" }, "Battery health (full charge vs design)"),
          h("div", { class: "value" }, pct(bat.soh_reported)), h("div", { class: "note" }, `${num(bat.full_charge_mwh / 1000, 1)} of ${num(bat.design_mwh / 1000, 1)} Wh after ${num(bat.cycles)} cycles`)),
        h("div", { class: "card tile", id: "lp-live" }),
        tile("Runtime on a full charge", `${num(bat.runtime_now_h, 1)} h`, `${num(bat.runtime_new_h, 1)} h when new (${num((bat.runtime_new_h - bat.runtime_now_h) * 60)} min lost)`),
        tile("Reaches 60 % of design (replace)", monthsTxt(c60), c60.status === "forecast" && c60.median_days != null ? `~${c60.median_date}; 95 %: ${num(c60.lo_days / 30.4)}-${c60.hi_days ? num(c60.hi_days / 30.4) : ">36"} months` : "GP forecast")),
      h("div", { class: "grid cols-2", style: "margin-top:14px" },
        card("Capacity history and forecast", `Windows-reported full-charge capacity (% of design) with the GP trend (RBF + linear) and 95 % band. Fade ${num(fc.fade_pct_per_100d, 1)} % per 100 days; 70 % in ${monthsTxt(c70)}.`, h("div", { id: "lp-cap" })),
        card("Live telemetry", "State of charge from the battery's fuel gauge, polled from WMI while the dashboard runs.", h("div", { id: "lp-trace" }))),
      h("div", { class: "grid cols-2", style: "margin-top:14px" },
        card("What your charging habits are worth", proj.available ? `Fleet ageing model calibrated to this battery (${proj.calibration}). Months until 60 % of design.` : "Projection unavailable.", h("div", { id: "lp-whatif" })),
        card("Recommendations", "Explainable rules over the forecast, usage profile and projection.",
          h("div", {}, b.recommendations.map(r => h("div", { class: "action" }, h("div", { class: "row" }, recBadge(r.level), h("span", { class: "title" }, r.title)),
            h("div", { class: "secondary", style: "font-size:13px;margin-top:4px" }, r.detail)))))),
      h("div", { class: "grid cols-2", style: "margin-top:14px" },
        card("Tonight's adaptive charge", null,
          h("div", { class: "row", style: "margin-bottom:10px" },
            h("button", { class: "btn primary", onclick: e => planLaptop(e.target, false) }, "Plan tonight"),
            h("button", { class: "btn", onclick: e => planLaptop(e.target, true) }, "Plan with a trip tomorrow 07:00")),
          planBox),
        card("Usage profile (learned)", `From ${prof.events} Windows usage events over ${prof.window_days} days and ${bat.history_entries} history periods${prof.invalid_history_entries ? ` (${prof.invalid_history_entries} corrupt entry excluded)` : ""}.`,
          kvTable({ "time on AC (45 days)": pct(prof.ac_share_45d, 0), "on battery per day": `${num(prof.battery_hours_per_day, 1)} h`,
                    "energy from battery per day": `${num(prof.battery_energy_wh_per_day, 1)} Wh (${pct(prof.daily_need_frac, 0)} of a full charge)`,
                    "time-weighted state of charge": pct(prof.mean_soc, 0), "time above 95 %": pct(prof.frac_high_soc, 0),
                    "charge limit detected": prof.charge_limit_detected ? `yes, ~${prof.charge_limit_pct} %` : "no",
                    "typical first unplug": prof.typical_unplug_hour != null ? clock(prof.typical_unplug_hour) : "-",
                    "cycles per day": num(fc.cycles_per_day, 2) }))),
      h("div", { style: "margin-top:14px" },
        card("Real-hardware dQ/dV (experimental)", b.ica_experimental.available ? `${b.ica_experimental.samples} charging samples, ${b.ica_experimental.series_cells} cells in series. ${b.ica_experimental.note}` : b.ica_experimental.reason,
          b.ica_experimental.available ? h("div", { id: "lp-ica" }) : null)),
      h("div", { class: "row", style: "margin-top:14px" }, h("button", { class: "btn", onclick: () => renderLaptop(true) }, "Regenerate battery report"),
        h("span", { class: "muted" }, `Report ${new Date(bat.report_time).toLocaleString()} - GP model: ${fc.model}`)));
    drawLive(live, b.live_trace);
    const toPct = v => v * 100;
    Charts.line($("#lp-cap"), {
      series: [
        { name: "Windows-reported capacity", color: "var(--series-2)", points: fc.points.map(p => [p.day, toPct(p.soh)]) },
        { name: "GP trend and forecast", color: "var(--series-1)",
          points: fc.fit_past.day.map((d, i) => [d, toPct(fc.fit_past.mean[i])]).concat(fc.forecast.day.map((d, i) => [d, toPct(fc.forecast.mean[i])])),
          band: fc.fit_past.day.map((d, i) => [d, toPct(fc.fit_past.lo[i]), toPct(fc.fit_past.hi[i])]).concat(fc.forecast.day.map((d, i) => [d, toPct(fc.forecast.lo[i]), toPct(fc.forecast.hi[i])])) }],
      x: { fmt: d => monthLabel(fc.start_date, d) }, y: { fmt: v => num(v) + " %", tipFmt: v => v.toFixed(1) + " %", min: 40, max: 90 },
      refLines: [{ y: 80, label: "80 % service threshold" }, { y: 60, label: "60 % replace" }], refX: [{ x: fc.now_day, label: "today" }], height: 260 });
    if (proj.available) {
      const names = { current_habits: ["Your current habits", "var(--series-1)"], ai_adaptive: ["AI adaptive charging", "var(--series-3)"], always_100pct: ["Always at 100 %", "var(--series-2)"] };
      Charts.line($("#lp-whatif"), {
        series: Object.entries(proj.scenarios).map(([k, s]) => ({ name: names[k][0], color: names[k][1], points: s.days.map((d, i) => [d / 30.4, s.soh[i] * 100]),
          endLabel: s.days_to_60pct ? `60 % in ${num(s.days_to_60pct / 30.4)} mo` : "> 36 mo" })),
        x: { label: "months from now", fmt: v => num(v) }, y: { fmt: v => num(v) + " %", tipFmt: v => v.toFixed(1) + " %" },
        refLines: [{ y: 60, label: "replace" }], endLabels: true, height: 250 });
    }
    if (b.ica_experimental.available) {
      const ica = b.ica_experimental;
      Charts.line($("#lp-ica"), { series: [{ name: "dQ/dV", color: "var(--series-1)", points: ica.v_cell.map((v, i) => [v, ica.dqdv_wh_per_v[i]]) }],
        x: { label: "cell voltage (V)", fmt: v => v.toFixed(2) }, y: { label: "dQ/dV (Wh/V)", fmt: v => v.toFixed(1) }, height: 220 });
    }
    liveTimer = setInterval(async () => {
      if (S.tab !== "laptop") { clearInterval(liveTimer); liveTimer = null; return; }
      try { const l = await api("/api/mybattery/live"); drawLive(l.latest || {}, l.trace); } catch (e) { /* keep last frame */ }
    }, 20000);
  }

  function drawLive(live, trace) {
    const box = $("#lp-live");
    if (box) box.replaceChildren(h("div", { class: "label" }, "Now"),
      h("div", { class: "value" }, live.percent != null ? `${live.percent} %` : "-"),
      h("div", { class: "note" }, `${live.power_online ? "on AC" : "on battery"}${live.charging ? ", charging" : ""} - ${live.voltage_mv ? (live.voltage_mv / 1000).toFixed(2) + " V" : "- V"}`),
      h("div", { class: "note" }, live.charge_rate_mw ? `+${num(live.charge_rate_mw / 1000, 1)} W` : live.discharge_rate_mw ? `-${num(live.discharge_rate_mw / 1000, 1)} W` : "0 W (idle / held by charge limit)"));
    const tr = $("#lp-trace");
    if (!tr) return;
    const pts = (trace?.time || []).map((t, i) => [new Date(t).getTime(), trace.percent[i]]).filter(p => p[1] != null);
    if (pts.length < 2) { tr.replaceChildren(h("div", { class: "empty" }, "Collecting samples - the trace fills in while the dashboard is open.")); return; }
    const shortSpan = pts[pts.length - 1][0] - pts[0][0] < 20 * 60000;
    const tfmt = v => new Date(v).toLocaleTimeString([], shortSpan ? { hour: "2-digit", minute: "2-digit", second: "2-digit" } : { hour: "2-digit", minute: "2-digit" });
    Charts.line(tr, { series: [{ name: "State of charge", color: "var(--series-1)", points: pts }],
      x: { fmt: tfmt }, y: { fmt: v => num(v) + " %", min: 0, max: 100 }, height: 250 });
  }

  async function planLaptop(btn, trip) {
    busy(btn, true, "Optimising...");
    const dep = new Date(); dep.setDate(dep.getDate() + 1); dep.setHours(7, 0, 0, 0);
    const pad = n => String(n).padStart(2, "0");
    const body = trip ? { calendar_events: [{ title: "Trip", start: `${dep.getFullYear()}-${pad(dep.getMonth() + 1)}-${pad(dep.getDate())}T07:00:00`, kind: "flight" }] } : {};
    try {
      const p = await api("/api/mybattery/charge-plan", { method: "POST", body: JSON.stringify(body) });
      const box = h("div", { id: "lp-plan-chart" });
      $("#lp-plan").replaceChildren(
        h("div", { class: "callout", style: "margin-bottom:10px" }, `${p.reason}. Now ${pct(p.soc_now, 0)}, target ${pct(p.target_soc, 0)} by ${new Date(p.unplug_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}.`),
        h("div", { class: "grid cols-2" },
          tile("Wear avoided vs charge-to-100 %", `${Math.max(0, p.damage_reduction_pct).toFixed(0)} %`, `target met: ${p.ai.met_target ? "yes" : "no"}`),
          tile("Capacity lost per year if every night were like this", `${(p.ai.damage_pct_capacity * 365).toFixed(2)} %`, `legacy: ${(p.legacy.damage_pct_capacity * 365).toFixed(2)} %`)),
        box, h("p", { class: "desc", style: "margin-top:8px" }, p.note));
      Charts.line(box, { series: [
        { name: "Charge to 100 % now", color: "var(--series-2)", points: p.legacy.trace.clock.map((x, i) => [x, p.legacy.trace.soc[i] * 100]) },
        { name: "AI schedule", color: "var(--series-1)", points: p.ai.trace.clock.map((x, i) => [x, p.ai.trace.soc[i] * 100]) }],
        x: { fmt: clock }, y: { fmt: v => num(v) + " %", min: 0, max: 102 }, refLines: [{ y: p.target_soc * 100, label: "need" }], height: 220 });
    } catch (e) { toastError(e, $("#lp-plan")); } finally { busy(btn, false); }
  }

  // ----------------------------------------------------------------- fleet
  async function loadFleet(force) {
    if (!S.fleet || force) S.fleet = await api("/api/fleet");
    if (!S.deviceId) S.deviceId = S.fleet.devices[0].device_id;
    $("#level-pill").textContent = `Autonomy L${S.fleet.workforce.level}`;
    $("#clock-pill").textContent = "Fleet: simulated, " + new Date(S.fleet.now).toUTCString().slice(5, 16);
    return S.fleet;
  }

  async function renderFleet() {
    const root = $("#tab-fleet");
    root.classList.add("loading");
    const f = await loadFleet();
    root.classList.remove("loading");
    const k = f.kpis;
    root.replaceChildren(
      h("div", { class: "grid cols-4" },
        tile("Mean fleet state of health", pct(k.mean_soh), `${k.devices} devices`, "hero"),
        tile("At or below end of first life (80 %)", num(k.below_eol), `${num(k.below_85pct)} below 85 %`),
        tile("Median RUL under 180 days", num(k.rul_under_180d), `${num(k.flagged_for_triage)} flagged for agentic triage`),
        tile("Precursor alerts", `${k.anomalies.critical} critical`, `${k.anomalies.warning} warning, ${k.anomalies.watch} watch`)),
      h("div", { class: "grid cols-3", style: "margin-top:14px" },
        tile("SoH error: AI vs legacy gauge", `${pct(k.soh_mae_ai, 2)} vs ${pct(k.soh_mae_legacy_gauge, 1)}`, "mean absolute error, simulated fleet"),
        tile("Adaptive charging active", pct(k.adaptive_share, 0), "pilot cohort + agent-enabled"),
        tile("Open approvals", num(f.workforce.open_actions), f.workforce.level_description)),
      h("div", { class: "grid cols-2", style: "margin-top:14px" },
        card("State-of-health distribution", "Devices per 2.5 % SoH band (AI estimate from on-device ICA + cloud GPR)", h("div", { id: "soh-hist" })),
        card("How to use", null, h("div", { class: "secondary", style: "font-size:13px" },
          h("p", {}, "Click a device to see its health forecast, dQ/dV fingerprint, precursor signals and what-if projections."),
          h("p", {}, "Adaptive charging compares legacy charge-to-100 % against the MILP schedule and the INT8 RL policy that runs on the NPU."),
          h("p", {}, "The agentic workforce runs triage and service at the autonomy level you choose (L1-L5); approvals stay with you."),
          h("p", {}, "The copilot on the right answers questions using the same tools.")))),
      h("div", { class: "section-title" }, "Devices"),
      fleetFilters(),
      card("Installed base", "Sorted worst-first by default. Intervals are 95 %.", h("div", { class: "table-wrap", id: "fleet-table" })));
    Charts.bars($("#soh-hist"), {
      bars: f.soh_histogram.map((c, i) => ({ label: `${Math.round(f.soh_bins[i] * 100)}`, value: c, color: "var(--series-1)",
                                             series: `${Math.round(f.soh_bins[i] * 100)}-${Math.round(f.soh_bins[i + 1] * 100)} % SoH` })),
      height: 210, valueName: "devices", y: { fmt: v => num(v) }, aria: "SoH histogram" });
    drawFleetTable();
  }

  function fleetFilters() {
    const profiles = [...new Set(S.fleet.devices.map(d => d.profile))];
    return h("div", { class: "row", style: "margin-bottom:10px" },
      h("input", { type: "search", placeholder: "Search device or lot", value: S.filter.q, oninput: e => { S.filter.q = e.target.value; drawFleetTable(); } }),
      h("select", { onchange: e => { S.filter.profile = e.target.value; drawFleetTable(); }, "aria-label": "Profile" },
        h("option", { value: "" }, "All profiles"), ...profiles.map(p => h("option", { value: p, selected: S.filter.profile === p }, p))),
      h("select", { onchange: e => { S.filter.anomaly = e.target.value; drawFleetTable(); }, "aria-label": "Precursor level" },
        h("option", { value: "" }, "Any precursor level"), ...["critical", "warning", "watch", "normal"].map(l => h("option", { value: l, selected: S.filter.anomaly === l }, l))));
  }

  function drawFleetTable() {
    const box = $("#fleet-table"); if (!box) return;
    const q = S.filter.q.toLowerCase();
    const ord = { critical: 0, warning: 1, watch: 2, normal: 3 };
    let rows = S.fleet.devices.filter(d => (!q || d.device_id.toLowerCase().includes(q) || d.lot.toLowerCase().includes(q))
      && (!S.filter.profile || d.profile === S.filter.profile) && (!S.filter.anomaly || d.anomaly === S.filter.anomaly));
    const keyf = { soh: d => d.soh, age: d => d.age_days, rul: d => d.rul.median_days, anomaly: d => ord[d.anomaly], id: d => d.device_id, gauge: d => d.gauge_soh_legacy, warranty: d => d.warranty_days_left }[S.sort.key];
    rows = rows.slice().sort((a, b) => (keyf(a) > keyf(b) ? 1 : keyf(a) < keyf(b) ? -1 : 0) * S.sort.dir);
    const th = (label, key, cls) => h("th", { class: "sortable " + (cls || ""), onclick: () => { S.sort = { key, dir: S.sort.key === key ? -S.sort.dir : 1 }; drawFleetTable(); } },
      label + (S.sort.key === key ? (S.sort.dir > 0 ? " ↑" : " ↓") : ""));
    const table = h("table", {},
      h("thead", {}, h("tr", {}, th("Device", "id"), h("th", {}, "Profile"), th("Age", "age", "num"), th("SoH (AI)", "soh", "num"),
        th("Legacy gauge", "gauge", "num"), th("RUL median [95 %]", "rul", "num"), th("Precursors", "anomaly"), h("th", {}, "Charging"),
        th("Warranty left", "warranty", "num"), h("th", {}, "Status"))),
      h("tbody", {}, rows.map(d => h("tr", { class: "clickable", onclick: () => openDevice(d.device_id) },
        h("td", {}, h("b", {}, d.device_id), h("div", { class: "muted", style: "font-size:11px" }, `${d.lot} - ${d.climate}`)),
        h("td", {}, d.profile_label),
        h("td", { class: "num" }, days(d.age_days)),
        h("td", { class: "num" }, pct(d.soh), h("div", { class: "muted", style: "font-size:11px" }, `± ${pct(1.96 * d.soh_std, 1)}`)),
        h("td", { class: "num secondary" }, pct(d.gauge_soh_legacy)),
        h("td", { class: "num" }, d.rul.status === "past_end_of_first_life" ? "past EOL" : days(d.rul.median_days),
          d.rul.status === "past_end_of_first_life" ? null : h("div", { class: "muted", style: "font-size:11px" }, `${num(d.rul.lo_days)}-${num(d.rul.hi_days)} d`)),
        h("td", {}, statusBadge(d.anomaly)),
        h("td", {}, d.adaptive ? h("span", { class: "badge accent" }, "adaptive") : h("span", { class: "badge" }, "legacy"),
          d.derated ? h("span", { class: "badge", style: "margin-left:4px" }, "derated") : null),
        h("td", { class: "num" }, d.warranty_days_left > 0 ? days(d.warranty_days_left) : h("span", { class: "muted" }, "expired")),
        h("td", { class: "secondary" }, d.status)))));
    box.replaceChildren(table);
  }

  // ---------------------------------------------------------------- device
  async function renderDevice() {
    const root = $("#tab-device");
    await loadFleet();
    root.classList.add("loading");
    let d;
    try { d = await api(`/api/devices/${S.deviceId}`); } catch (e) { root.classList.remove("loading"); toastError(e, root); return; }
    S.device = d;
    root.classList.remove("loading");
    const rul = d.rul;
    const rulText = rul.status === "past_end_of_first_life" ? "Past end of first life" : days(rul.median_days);
    const out = h("div", { id: "device-extra" });
    root.replaceChildren(
      h("div", { class: "row", style: "margin-bottom:12px" }, deviceSelect(renderDevice),
        h("span", { class: "secondary" }, `${d.profile_label} - ${d.climate} climate - lot ${d.lot} - ${d.age_days} days in service`),
        h("span", { class: "spacer" }),
        h("button", { class: "btn", onclick: e => runDiagnose(e.target) }, "Run diagnostic capture"),
        h("button", { class: "btn", onclick: e => runOta(e.target) }, "Push signed OTA recalibration"),
        h("button", { class: "btn", onclick: e => runSocBench(e.target) }, "SoC estimator benchmark"),
        h("button", { class: "btn primary", onclick: () => switchTab("charging") }, "Plan tonight's charge")),
      h("div", { class: "grid cols-4" },
        tile("State of health (AI)", pct(d.soh), `95 %: ${pct(d.soh_interval[0])} - ${pct(d.soh_interval[1])}; legacy gauge ${pct(d.gauge_soh_legacy)}`),
        tile("Remaining useful life", rulText, rul.status === "past_end_of_first_life" ? "SoH at or below 80 %" : `95 % interval ${num(rul.lo_days)}-${num(rul.hi_days)} days`),
        tile("vs expectation for age & usage", `${d.soh_z >= 0 ? "+" : ""}${d.soh_z.toFixed(1)} σ`, `expected ${pct(d.expected_soh)}; anode-LAM excess ${d.lam_ne_excess_z.toFixed(1)} σ`),
        h("div", { class: "card tile" }, h("div", { class: "label" }, "Precursor detector"), h("div", { style: "margin:6px 0 2px" }, statusBadge(d.anomaly.level)),
          h("div", { class: "note" }, d.anomaly.mechanism || "no precursor signature"),
          h("div", { class: "note" }, d.warranty_days_left > 0 ? `Warranty: ${d.warranty_days_left} days left` : "Warranty expired"))),
      h("div", { class: "grid cols-2", style: "margin-top:14px" },
        card("State-of-health history", "AI estimate with 95 % band vs the legacy cycle-counter gauge. Ground truth is shown only because this is a simulation.", h("div", { id: "c-soh" })),
        card("Digital-twin what-if", "Projected SoH from today with this device's usage: legacy charging vs adaptive charging.", h("div", { id: "c-whatif" }))),
      h("div", { class: "grid cols-2", style: "margin-top:14px" },
        card("Incremental capacity (dQ/dV) fingerprint", "On-device ICA of a slow overnight charge. Peak shifts and shrinkage reveal lithium-inventory and active-material loss.", h("div", { id: "c-ica" })),
        card("Precursor score", "Mahalanobis distance of divergence, self-discharge, thermal residual, strain and DCIR growth vs the healthy fleet.", h("div", { id: "c-anom" }))),
      h("div", { class: "grid cols-2", style: "margin-top:14px" },
        card("Degradation modes", "Routine estimate from ICA features (GPR) and, after a diagnostic capture, the full degradation-mode fit.", modesTable(d)),
        card("Agent actions for this device", null, actionList(d.actions, true))),
      out);
    const eol = [{ y: 0.8, label: "End of first life 80 %" }];
    Charts.line($("#c-soh"), {
      series: [
        { name: "AI estimate (ICA + GPR)", color: "var(--series-1)", points: d.series.day.map((x, i) => [x, d.series.soh_est[i]]),
          band: d.series.day.map((x, i) => [x, d.series.soh_est[i] - 1.96 * d.series.soh_std[i], d.series.soh_est[i] + 1.96 * d.series.soh_std[i]]) },
        { name: "Legacy gauge (cycle counter)", color: "var(--series-2)", points: d.series.day.map((x, i) => [x, d.series.gauge_soh_legacy[i]]) },
        { name: "Simulator ground truth", color: "var(--text-muted)", width: 1.5, points: d.series.day.map((x, i) => [x, d.series.soh_truth_sim_only[i]]) }],
      x: { label: "days in service", fmt: v => num(v) }, y: { fmt: v => pct(v, 0), tipFmt: v => pct(v, 1) }, refLines: eol, height: 250 });
    const w = d.whatif;
    Charts.line($("#c-whatif"), {
      series: [{ name: "Legacy charging", color: "var(--series-2)", points: w.legacy.days.map((x, i) => [x, w.legacy.soh[i]]), endLabel: w.legacy.eol_in_days ? `EOL in ${w.legacy.eol_in_days} d` : "Legacy" },
               { name: "Adaptive charging", color: "var(--series-1)", points: w.adaptive.days.map((x, i) => [x, w.adaptive.soh[i]]), endLabel: w.adaptive.eol_in_days ? `EOL in ${w.adaptive.eol_in_days} d` : "Adaptive" }],
      x: { label: "days from today", fmt: v => num(v) }, y: { fmt: v => pct(v, 0), tipFmt: v => pct(v, 1) }, refLines: eol, endLabels: true, height: 250 });
    const icaColors = ["var(--seq-250)", "var(--seq-450)", "var(--seq-650)"];
    Charts.line($("#c-ica"), {
      series: d.ica.map((c, i) => ({ name: `day ${c.day}`, color: icaColors[i % 3], points: c.v.map((v, j) => [v, c.dqdv[j]]) })),
      x: { label: "cell voltage (V)", fmt: v => v.toFixed(2) }, y: { label: "dQ/dV (Ah/V)", fmt: v => v.toFixed(0), tipFmt: v => v.toFixed(2) }, height: 250 });
    Charts.line($("#c-anom"), {
      series: [{ name: "Precursor score (D²)", color: "var(--series-1)", points: d.series.day.map((x, i) => [x, Math.min(d.series.anomaly_d2[i], 200)]) }],
      x: { label: "days in service", fmt: v => num(v) }, y: { fmt: v => num(v), tipFmt: v => v >= 200 ? "≥ 200" : v.toFixed(1), min: 0 },
      refLines: [{ y: 15.1, label: "watch" }, { y: 25.7, label: "warning" }], height: 250 });
  }

  function modesTable(d) {
    const rows = [["Lithium inventory loss (LLI)", "lli"], ["Cathode active-material loss", "lam_pe"], ["Anode active-material loss", "lam_ne"]];
    const dma = d.diagnosis;
    return h("div", {},
      h("table", {}, h("thead", {}, h("tr", {}, h("th", {}, "Mode"), h("th", { class: "num" }, "Routine ICA estimate"), h("th", { class: "num" }, "Diagnostic fit"))),
        h("tbody", {}, rows.map(([label, k]) => h("tr", {}, h("td", {}, label),
          h("td", { class: "num" }, `${pct(d.modes_est[k], 1)} ± ${pct(1.96 * d.modes_est_std[k], 1)}`),
          h("td", { class: "num" }, dma ? pct(dma[k], 1) : "-"))))),
      dma ? h("p", { class: "desc", style: "margin-top:8px" }, `Dominant mechanism: ${dma.dominant}. Fit RMSE ${dma.rmse_mah} mAh, ohmic offset ${dma.offset_mv} mV.`)
          : h("p", { class: "desc", style: "margin-top:8px" }, "Run a diagnostic capture to fit the modes from a full slow charge."));
  }

  async function runDiagnose(btn) {
    busy(btn, true, "Capturing...");
    try { await api(`/api/devices/${S.deviceId}/diagnose`, { method: "POST" }); await renderDevice(); }
    catch (e) { toastError(e, $("#tab-device")); } finally { busy(btn, false); }
  }
  async function runOta(btn) {
    busy(btn, true, "Signing...");
    try {
      const r = await api(`/api/devices/${S.deviceId}/ota`, { method: "POST" });
      const sec = r.security_checks;
      $("#device-extra").replaceChildren(card("Signed OTA recalibration package", "Ed25519-signed; the device verifies signature, digest, binding, expiry and version before installing.",
        h("div", { class: "grid cols-2" },
          kvTable({ kind: r.header.kind, version: r.header.version, device: r.header.device_id, key_id: r.header.key_id, payload_sha256: r.header.payload_sha256.slice(0, 24) + "...",
                    expires: r.header.expires_at.slice(0, 10), installed: r.device_verification.installed, reason: r.device_verification.reason }),
          kvTable({ "new capacity (Ah)": r.payload_summary.capacity_ah, "OCV table points": r.payload_summary.ocv_points, "ICA bands (V)": r.payload_summary.ica_bands_v,
                    "ICA peak drift (mV)": r.payload_summary.ica_peak_drift_mv, "tampered payload": `${sec.tampered_payload.accepted ? "ACCEPTED" : "rejected"} (${sec.tampered_payload.reason})`,
                    "replayed version": `${sec.replayed_same_version.accepted ? "ACCEPTED" : "rejected"} (${sec.replayed_same_version.reason})`,
                    "other device": `${sec.wrong_device.accepted ? "ACCEPTED" : "rejected"} (${sec.wrong_device.reason})` }))));
    } catch (e) { toastError(e, $("#tab-device")); } finally { busy(btn, false); }
  }
  async function runSocBench(btn) {
    busy(btn, true, "Simulating...");
    try {
      const r = await api(`/api/devices/${S.deviceId}/soc-benchmark`);
      const box = h("div", { id: "c-socb" });
      $("#device-extra").replaceChildren(card("State-of-charge estimation on this aged cell",
        `Drive-cycle discharge (${r.duration_h} h). Mean absolute error: legacy coulomb counting ${r.legacy_coulomb_counting.mae_pct.toFixed(2)} %, EKF with beginning-of-life parameters ${r.ekf_bol_parameters.mae_pct.toFixed(2)} %, EKF recalibrated from the twin ${r.ekf_recalibrated.mae_pct.toFixed(2)} %.`, box));
      const t = r.trace;
      Charts.line(box, { series: [
        { name: "Legacy coulomb counting", color: "var(--series-2)", points: t.t_h.map((x, i) => [x, t.legacy_coulomb_counting[i]]) },
        { name: "EKF, recalibrated", color: "var(--series-1)", points: t.t_h.map((x, i) => [x, t.ekf_recalibrated[i]]) },
        { name: "True SoC (simulator)", color: "var(--text-muted)", width: 1.5, points: t.t_h.map((x, i) => [x, t.truth[i]]) }],
        x: { label: "hours", fmt: v => v.toFixed(1) }, y: { fmt: v => num(v) + " %", tipFmt: v => v.toFixed(1) + " %" }, height: 240 });
    } catch (e) { toastError(e, $("#tab-device")); } finally { busy(btn, false); }
  }

  // --------------------------------------------------------------- charging
  const SCENARIOS = {
    routine: { label: "Learned routine (tonight)", body: {} },
    flight: { label: "Flight tomorrow 07:00 (calendar)", body: { calendar_events: [{ title: "Flight to Boston", start: "2026-09-27T07:00:00", kind: "flight", duration_h: 8 }] } },
    light: { label: "Light day tomorrow (calendar)", body: { calendar_events: [{ title: "Workshop day", start: "2026-09-27T09:00:00", kind: "light_day" }] } },
    hot: { label: "Hot room (34 C ambient)", body: { ambient_c: 34 } },
    asap: { label: "Ready as soon as possible", body: { mode: "ready_asap" } },
  };
  async function renderCharging() {
    const root = $("#tab-charging");
    await loadFleet();
    const scen = h("select", { id: "scen", "aria-label": "Scenario" }, ...Object.entries(SCENARIOS).map(([k, v]) => h("option", { value: k }, v.label)));
    const soc = h("input", { type: "range", min: 5, max: 90, value: 25, id: "socnow", "aria-label": "Plug-in state of charge" });
    const socLabel = h("span", { class: "num" }, "25 %");
    soc.addEventListener("input", () => socLabel.textContent = soc.value + " %");
    const go = h("button", { class: "btn primary", onclick: e => planCharge(e.target) }, "Plan charge");
    root.replaceChildren(
      h("div", { class: "row", style: "margin-bottom:12px" }, deviceSelect(() => {}), scen,
        h("label", { class: "row secondary" }, "Plug-in SoC", soc, socLabel), go),
      h("div", { id: "plan-out" }, h("div", { class: "empty" }, "Choose a scenario and press Plan charge. The optimiser solves a MILP (HiGHS), verifies it on the safety-gated twin, and compares it with legacy charging and the INT8 RL policy.")));
  }
  async function planCharge(btn) {
    busy(btn, true, "Optimising...");
    const body = { ...SCENARIOS[$("#scen").value].body, soc_now: Number($("#socnow").value) / 100 };
    try {
      const p = await api(`/api/devices/${S.deviceId}/charge-plan`, { method: "POST", body: JSON.stringify(body) });
      S.plan = p;
      const c = p.context, L = p.legacy, M = p.milp, R = p.edge_rl_int8;
      const pol = [["Legacy charge-to-100 %", L, "var(--series-2)"], ["MILP schedule", M, "var(--series-1)"], ["INT8 RL on NPU", R, "var(--series-3)"]];
      const polTile = ([name, r]) => h("div", { class: "card tile" }, h("div", { class: "label" }, name),
        h("div", { class: "value" }, `${(r.damage_pct_capacity * 365).toFixed(2)} % / yr`),
        h("div", { class: "note" }, `capacity lost per year if every charge were like this; final ${pct(r.final_soc, 0)}; ${r.met_target ? "target met" : "target missed"}`),
        h("div", { class: "note" }, `${r.hours_above_95.toFixed(1)} h above 95 %; peak ${r.peak_temp_c.toFixed(1)} C; ready after ${r.ready_after_h ?? "-"} h`));
      const out = $("#plan-out");
      out.replaceChildren(
        h("div", { class: "callout", style: "margin-bottom:12px" }, `${c.reason}. Target ${pct(c.target_soc, 0)} by ${c.unplug_at.slice(11, 16)} (${c.hours_to_unplug} h), mode ${c.mode}, ambient ${c.ambient_c} C.`),
        h("div", { class: "grid cols-4" },
          h("div", { class: "card tile hero" }, h("div", { class: "label" }, "Battery wear avoided vs legacy (MILP)"),
            h("div", { class: "value delta-good" }, `${Math.max(0, p.comparison.milp_damage_reduction_pct).toFixed(0)} %`),
            h("div", { class: "note" }, `edge RL policy: ${Math.max(0, p.comparison.edge_rl_damage_reduction_pct).toFixed(0)} %`)),
          ...pol.map(polTile)),
        h("div", { class: "grid cols-2", style: "margin-top:14px" },
          card("State of charge", "All three policies run through the same safety-gated physics.", h("div", { id: "c-plan-soc" })),
          card("Charge rate", "Requested rates are clamped by the deterministic supervisor (JEITA, CV headroom, ceilings).", h("div", { id: "c-plan-c" }))),
        h("div", { class: "grid cols-2", style: "margin-top:14px" },
          card("Safety envelope in force", "AI may only tighten these limits.", kvTable(c.effective_limits)),
          card("MILP solver", "HiGHS branch-and-bound; plan verified on the twin and re-planned if short.",
            kvTable({ status: M.status, variables: M.solver.n_variables, binaries: M.solver.n_binary, constraints: M.solver.n_constraints,
                      "objective (EUR)": M.solver.objective_eur, replans: M.replans, "energy cost (EUR)": M.energy_eur, "grid CO2 (g)": M.carbon_g }))));
      const traces = [["Legacy", L, "var(--series-2)"], ["MILP", M, "var(--series-1)"], ["INT8 RL", R, "var(--series-3)"]];
      Charts.line($("#c-plan-soc"), {
        series: traces.map(([n, r, col]) => ({ name: n, color: col, points: r.trace.clock.map((x, i) => [x, r.trace.soc[i]]) })),
        x: { fmt: clock }, y: { fmt: v => pct(v, 0), min: 0, max: 1.02, tipFmt: v => pct(v, 1) },
        refLines: [{ y: c.target_soc, label: `need ${pct(c.target_soc, 0)}` }], height: 240 });
      Charts.line($("#c-plan-c"), {
        series: traces.map(([n, r, col]) => ({ name: n, color: col, points: r.trace.c.map((v, i) => [r.trace.clock[i], v]) })),
        x: { fmt: clock }, y: { fmt: v => v.toFixed(1) + "C", min: 0, tipFmt: v => v.toFixed(2) + "C" }, height: 240 });
    } catch (e) { toastError(e, $("#plan-out")); } finally { busy(btn, false); }
  }

  // -------------------------------------------------------------- workforce
  const KIND_LABEL = { notify_user: "User notification", enable_adaptive_charging: "Adaptive charging", safety_derate: "Safety derate",
    diagnostic_capture: "Diagnostic capture", ota_recalibration: "OTA recalibration", warranty_claim: "Warranty claim",
    proactive_replacement_offer: "Replacement offer", replacement_order: "Replacement order", logistics_shipment: "Logistics",
    asset_recovery: "Asset recovery", design_insight: "Design insight" };
  const STATUS_LABEL = { executed: "executed", pending_approval: "needs approval", awaiting_user: "awaiting user consent", recommended: "recommended", rejected: "rejected" };

  function actionList(actions, compact) {
    if (!actions.length) return h("div", { class: "empty" }, "No actions yet.");
    return h("div", {}, actions.slice(0, compact ? 8 : 400).map(a => {
      const decide = async (approve, btn) => {
        busy(btn, true, "...");
        try { await api(`/api/workforce/actions/${a.id}/decision`, { method: "POST", body: JSON.stringify({ approve }) }); S.fleet = null; S.tab === "workforce" ? renderWorkforce() : renderDevice(); }
        catch (e) { toastError(e); busy(btn, false); }
      };
      const needs = a.status === "pending_approval" || a.status === "awaiting_user";
      return h("div", { class: "action" },
        h("div", { class: "row" }, h("span", { class: "badge " + (a.status === "executed" ? "accent" : "") }, STATUS_LABEL[a.status] || a.status),
          h("span", { class: "badge" }, KIND_LABEL[a.kind] || a.kind), h("span", { class: "meta" }, `${a.id} - ${a.agent} - needs L${a.required_level}`),
          h("span", { class: "spacer" }), a.device_id && !compact ? h("button", { class: "btn small", onclick: () => openDevice(a.device_id) }, a.device_id) : null),
        h("div", { class: "title", style: "margin-top:4px" }, a.title),
        h("div", { class: "secondary", style: "font-size:13px" }, a.rationale),
        needs ? h("div", { class: "row", style: "margin-top:8px" },
          h("button", { class: "btn small primary", onclick: e => decide(true, e.target) }, a.status === "awaiting_user" ? "User confirms" : "Approve"),
          h("button", { class: "btn small danger", onclick: e => decide(false, e.target) }, "Reject")) : null,
        Object.keys(a.result || {}).length ? h("details", {}, h("summary", {}, "Result & evidence"),
          h("pre", { class: "json" }, JSON.stringify({ result: a.result, evidence: a.evidence }, null, 2))) : null);
    }));
  }

  async function renderWorkforce() {
    const root = $("#tab-workforce");
    root.classList.add("loading");
    const wf = await api("/api/workforce");
    root.classList.remove("loading");
    $("#level-pill").textContent = `Autonomy L${wf.level}`;
    const ladder = h("div", { class: "ladder" }, Object.entries(wf.levels).map(([lv, desc]) => h("button", {
      class: "rung", "aria-pressed": String(Number(lv) === wf.level), onclick: async () => { await api("/api/workforce/level", { method: "PUT", body: JSON.stringify({ level: Number(lv) }) }); S.fleet = null; renderWorkforce(); } },
      h("div", { class: "lv" }, `L${lv}`), h("div", { class: "d" }, desc.split(" - ")[1] || desc))));
    const insights = wf.actions.filter(a => a.kind === "design_insight");
    let acts = wf.actions.filter(a => a.kind !== "design_insight");
    const counts = wf.counts;
    if (S.wf.status) acts = acts.filter(a => a.status === S.wf.status);
    if (S.wf.kind) acts = acts.filter(a => a.kind === S.wf.kind);
    const run = h("button", { class: "btn primary", onclick: async e => {
      busy(e.target, true, "Agents working...");
      try { S.lastRun = await api("/api/workforce/run", { method: "POST" }); S.fleet = null; renderWorkforce(); }
      catch (err) { toastError(err, root); busy(e.target, false); } } }, "Run agentic workforce");
    root.replaceChildren(
      card("Autonomy level", "The same agents act differently per level: execute, ask an engineer to approve, or only recommend. Replacement orders always need user consent (pre-approved only at L5).", ladder),
      h("div", { class: "row", style: "margin:14px 0" }, run,
        h("button", { class: "btn", onclick: async () => { if (confirm("Clear all agent actions, passports and device state?")) { await api("/api/workforce/reset", { method: "POST" }); S.fleet = null; S.lastRun = null; renderWorkforce(); } } }, "Reset simulation state"),
        S.lastRun ? h("span", { class: "secondary" }, `Last run at L${S.lastRun.level}: ${S.lastRun.devices_flagged} devices flagged, ${S.lastRun.new_actions} new actions`) : null,
        h("span", { class: "spacer" }),
        h("span", { class: "secondary" }, Object.entries(counts).map(([k, v]) => `${STATUS_LABEL[k] || k}: ${v}`).join("  |  "))),
      h("div", { class: "grid cols-2" },
        h("div", {},
          h("div", { class: "row", style: "margin-bottom:10px" },
            h("select", { "aria-label": "Status filter", onchange: e => { S.wf.status = e.target.value; renderWorkforce(); } },
              h("option", { value: "" }, "All statuses"), ...Object.keys(STATUS_LABEL).map(s => h("option", { value: s, selected: S.wf.status === s }, STATUS_LABEL[s]))),
            h("select", { "aria-label": "Kind filter", onchange: e => { S.wf.kind = e.target.value; renderWorkforce(); } },
              h("option", { value: "" }, "All action types"), ...Object.keys(KIND_LABEL).filter(k => k !== "design_insight").map(k => h("option", { value: k, selected: S.wf.kind === k }, KIND_LABEL[k])))),
          card(`Actions (${acts.length})`, "Newest first. Prescriptive service with agentic triage: predictive -> triage -> service -> logistics -> asset recovery.", actionList(acts, false))),
        h("div", {},
          card("Design feedback (fleet loop, L5)", "Fleet-level findings for supplier quality and the next-generation pack.",
            insights.length ? actionList(insights, false) : h("div", { class: "empty" }, "Run the workforce to generate fleet insights.")))));
  }

  // --------------------------------------------------------------- passport
  async function renderPassport() {
    const root = $("#tab-passport");
    await loadFleet();
    const role = h("select", { id: "role", "aria-label": "Viewer role", onchange: () => loadPassport() },
      h("option", { value: "public" }, "Public"), h("option", { value: "legitimate_interest" }, "Legitimate interest (repairer, recycler)"),
      h("option", { value: "authority" }, "Authority / notified body"));
    root.replaceChildren(h("div", { class: "row", style: "margin-bottom:12px" }, deviceSelect(() => loadPassport()), role), h("div", { id: "pp-out" }));
    loadPassport();
  }
  async function loadPassport() {
    const out = $("#pp-out");
    out.classList.add("loading");
    try {
      const [p, g] = await Promise.all([api(`/api/devices/${S.deviceId}/passport?role=${$("#role").value}`), api(`/api/devices/${S.deviceId}/second-life`)]);
      out.classList.remove("loading");
      const section = (title, obj) => obj ? card(title, null, kvTable(Object.fromEntries(Object.entries(obj).map(([k, v]) =>
        [k, v && typeof v === "object" && "value" in v ? (v.value === null ? "-" : v.value) : v])))) : null;
      const qr = h("div", { class: "qr" });
      if (p._qr_svg) { const tpl = document.createElement("template"); tpl.innerHTML = p._qr_svg; qr.appendChild(tpl.content); } // server-generated SVG
      out.replaceChildren(
        h("div", { class: "callout", style: "margin-bottom:12px" }, p.applicability),
        h("div", { class: "grid cols-3" },
          h("div", { class: "card" }, h("h2", {}, `Passport ${p.passport_id}`), h("p", { class: "desc" }, `Viewed as: ${p.viewer_role}`),
            h("div", { class: "row" }, qr, kvTable({ version: p._version.version, hash: p._version.hash.slice(0, 20) + "...",
              previous: p._version.prev_hash ? p._version.prev_hash.slice(0, 20) + "..." : "genesis", "hash chain valid": p._version.chain_valid, link: p.link }))),
          h("div", { class: "card tile" }, h("div", { class: "label" }, "Second-life readiness"), h("div", { class: "value" }, `Grade ${g.grade}`),
            h("div", { class: "note" }, `score ${g.score} / 100`), h("div", { class: "secondary", style: "margin-top:6px" }, g.pathway),
            g.second_life_days_to_60pct ? h("div", { class: "note" }, `projected second life to 60 % SoH: ${num(g.second_life_days_to_60pct)} days`) : null),
          card("Grading rationale", "Risk-aware: uses the lower bound of the RUL interval; safety gates override the score.",
            h("ul", { style: "margin:0;padding-left:18px;font-size:13px" }, g.reasons.map(r => h("li", {}, r))))),
        h("div", { class: "grid cols-2", style: "margin-top:14px" },
          section("State of health (dynamic)", p.state_of_health), section("Rated performance & durability", p.rated_performance)),
        h("div", { class: "grid cols-2", style: "margin-top:14px" },
          section("General information", p.general), section("Circularity", p.circularity)),
        h("div", { class: "grid cols-3", style: "margin-top:14px" },
          section("Negative events", p.negative_events), section("Carbon footprint", p.carbon_footprint), section("Compliance", p.compliance)),
        h("div", { style: "margin-top:14px" }, section("Status", p.status)));
    } catch (e) { out.classList.remove("loading"); toastError(e, out); }
  }

  // ----------------------------------------------------------------- models
  async function renderModels() {
    const root = $("#tab-models");
    const r = await api("/api/models");
    const hs = r.health, an = r.anomaly, tw = r.twin, ch = r.charging.policy_comparison, tm = r.tinyml;
    const live = hs.live_fleet_rul;
    const polBox = h("div", { id: "c-pol" });
    root.replaceChildren(
      h("div", { class: "callout", style: "margin-bottom:12px" }, `${r.disclaimer} Built ${r.built_at}.`),
      h("div", { class: "grid cols-4" },
        tile("SoH error (MAE)", pct(hs.soh.mae_ai, 2), `legacy gauge ${pct(hs.soh.mae_gauge_baseline, 2)}; 95 % coverage ${pct(hs.soh.coverage95, 0)}`),
        tile("RUL error, live fleet", days(live.all.mae_days), `truth inside 95 % interval: ${pct(live.all.coverage95, 0)} (n=${live.all.n})`),
        tile("Precursor lead time", days(an.median_warning_lead_before_critical_days), `warning ${num(an.median_days_onset_to_warning)} d after onset; false warnings ${pct(an.false_warning_rate, 2)}`),
        tile("Twin voltage error", `${tw.voltage_rmse_mv_hybrid.toFixed(1)} mV`, `ECM only ${tw.voltage_rmse_mv_ecm_only.toFixed(1)} mV (-${tw.improvement_pct.toFixed(0)} %)`)),
      h("div", { class: "grid cols-2", style: "margin-top:14px" },
        card("Charging policies", `Capacity lost per year if every night charged like the average of ${ch.sessions} random sessions (lower is better).`, polBox,
          kvTable({ "targets met - legacy": pct(ch.legacy.met_target_rate, 1), "targets met - MILP": pct(ch.milp.met_target_rate, 1), "targets met - RL": pct(ch.rl.met_target_rate, 1),
                    "damage reduction - MILP": pct(ch.milp.damage_reduction_vs_legacy_pct / 100, 0), "damage reduction - RL": pct(ch.rl.damage_reduction_vs_legacy_pct / 100, 0) })),
        card("TinyML pipeline", "RL teacher -> distilled MLP -> structured pruning -> QAT INT8 -> ONNX (QDQ).",
          h("table", {}, h("thead", {}, h("tr", {}, h("th", {}, "Stage"), h("th", { class: "num" }, "Bytes"), h("th", { class: "num" }, "Agreement (on-policy)"), h("th", { class: "num" }, "Value regret"))),
            h("tbody", {}, tm.stages.map(s => h("tr", {}, h("td", {}, s.stage), h("td", { class: "num" }, s.bytes ? num(s.bytes) : "-"),
              h("td", { class: "num" }, pct(s.agreement_onpolicy, 1)), h("td", { class: "num" }, s.normalised_value_regret_onpolicy.toFixed(3)))))),
          h("p", { class: "desc", style: "margin-top:8px" }, `Closed loop with deadline guard: INT8 student meets ${pct(tm.closed_loop["int8_student+guard"].met_target_rate, 1)} of targets (teacher ${pct(tm.closed_loop["rl_teacher+guard"].met_target_rate, 1)}). ONNX INT8 ${num(tm.onnx.bytes_int8)} B; ORT argmax agreement ${pct(tm.onnx.int8?.argmax_agreement, 2)}; ${tm.numpy_int8_latency_us.toFixed(0)} us per inference (numpy).`))),
      h("div", { class: "grid cols-2", style: "margin-top:14px" },
        card("RUL by cohort (live fleet)", "The adaptive cohort's intervals widen automatically where training data is thinner.",
          h("table", {}, h("thead", {}, h("tr", {}, h("th", {}, "Cohort"), h("th", { class: "num" }, "n"), h("th", { class: "num" }, "MAE"), h("th", { class: "num" }, "95 % coverage"), h("th", { class: "num" }, "Mean interval width"))),
            h("tbody", {}, Object.entries(live).map(([k, v]) => h("tr", {}, h("td", {}, k.replaceAll("_", " ")), h("td", { class: "num" }, v.n), h("td", { class: "num" }, days(v.mae_days)),
              h("td", { class: "num" }, pct(v.coverage95, 0)), h("td", { class: "num" }, days(v.mean_interval_width_days))))))),
        card("Degradation modes from routine ICA", "GPR on the nine dQ/dV features + DCIR; held-out devices.",
          kvTable(Object.fromEntries(Object.entries(hs.modes_from_routine_ica || {}).map(([k, v]) => [k, `R² ${v.r2.toFixed(3)}, MAE ${pct(v.mae, 2)}`]))),
          h("p", { class: "desc", style: "margin-top:10px" }, "Most informative SoH features (smallest ARD length scale): " +
            Object.entries(hs.soh_length_scales).sort((a, b) => a[1] - b[1]).slice(0, 4).map(([k]) => k).join(", ")))),
      h("div", { style: "margin-top:14px" }, card("Full report", null, h("details", {}, h("summary", {}, "model_report.json"), h("pre", { class: "json" }, JSON.stringify(r, null, 2))))));
    Charts.bars(polBox, { bars: [["Legacy", ch.legacy], ["MILP", ch.milp], ["RL (tabular)", ch.rl]].map(([n, v]) => ({ label: n, value: v.damage_pct_capacity_mean * 365, color: "var(--series-1)" })),
      height: 190, valueLabels: true, allLabels: true, valueName: "% capacity lost per year", y: { fmt: v => v.toFixed(1) + " %", tipFmt: v => v.toFixed(2) + " %" } });
  }

  // ---------------------------------------------------------------- copilot
  const CHIPS = ["How is my battery doing?", "Plan tonight's charge for my laptop", "Give me a fleet overview", "Which devices show swelling or thermal precursors?", "Why is DEV-0114 degrading so fast?",
                 "Plan tonight's charge for DEV-0001 - I have a flight tomorrow", "Is there a supplier quality problem?", "How accurate are the models?"];
  function addMsg(role, text, trace, mode) {
    const m = h("div", { class: "msg " + role }, text);
    if (trace && trace.length) m.appendChild(h("div", { class: "trace" }, "Tools: " + trace.map(t => t.tool).join(", ")));
    if (mode && role === "bot") m.appendChild(h("div", { class: "trace" }, mode));
    $("#chat-log").appendChild(m);
    $("#chat-log").scrollTop = 1e9;
    return m;
  }
  async function send(text) {
    if (!text.trim()) return;
    addMsg("user", text);
    const pending = addMsg("bot", "Thinking...");
    try {
      const r = await api("/api/copilot", { method: "POST", body: JSON.stringify({ message: text, session_id: S.session }) });
      S.session = r.session_id;
      pending.remove();
      addMsg("bot", r.answer, r.trace, r.mode.startsWith("offline") ? r.mode : null);
      $("#chat-mode").textContent = "Mode: " + r.mode;
    } catch (e) { pending.textContent = "Error: " + e.message; }
  }
  $("#chat-form").addEventListener("submit", e => { e.preventDefault(); const t = $("#chat-text"); send(t.value); t.value = ""; });
  $("#chat-text").addEventListener("keydown", e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#chat-form").requestSubmit(); } });
  CHIPS.forEach(c => $("#chips").appendChild(h("button", { class: "chip", type: "button", onclick: () => send(c) }, c)));
  const toggleChat = () => $("#app").classList.toggle("chat-collapsed");
  $("#chat-btn").addEventListener("click", toggleChat);
  $("#chat-close").addEventListener("click", toggleChat);
  if (window.innerWidth < 1100) $("#app").classList.add("chat-collapsed");

  $("#theme-btn").addEventListener("click", () => {
    const cur = document.documentElement.dataset.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    document.documentElement.dataset.theme = cur === "dark" ? "light" : "dark";
    try { localStorage.setItem("bms-theme", document.documentElement.dataset.theme); } catch (e) { /* storage unavailable */ }
  });
  try { const t = localStorage.getItem("bms-theme"); if (t) document.documentElement.dataset.theme = t; } catch (e) { /* storage unavailable */ }

  api("/api/status").then(s => {
    $("#chat-mode").textContent = "Mode: " + s.copilot_mode;
    addMsg("bot", "Hi - I'm the BMS Copilot. Ask me about fleet health, a specific device, tonight's charging plan, battery passports or what the agents are doing.");
  }).catch(() => {});
  // Deep links: #tab=<name>&id=<device>&scenario=<charging scenario>&chat=0&ask=<question for the copilot>
  const params = new URLSearchParams(location.hash.slice(1));
  if (params.get("ask")) setTimeout(() => send(params.get("ask")), 500);
  if (params.get("id")) S.deviceId = params.get("id");
  if (params.get("chat") === "0") $("#app").classList.add("chat-collapsed");
  loadFleet().catch(() => {});
  switchTab(params.get("tab") || "laptop").then(() => {
    const scenario = params.get("scenario");
    if (S.tab === "charging" && scenario && SCENARIOS[scenario]) {
      $("#scen").value = scenario;
      planCharge($("#tab-charging .btn.primary"));
    }
  }).catch(() => {});
})();
