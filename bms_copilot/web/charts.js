/* Minimal dependency-free SVG charts (line with bands/crosshair, bars).
   Colors are CSS custom properties, so light/dark switch without re-render. */
(function () {
  const NS = "http://www.w3.org/2000/svg";
  const registry = new Set();

  function el(tag, attrs, parent) {
    const n = document.createElementNS(NS, tag);
    for (const k in attrs || {}) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  }
  function niceTicks(min, max, count) {
    if (!isFinite(min) || !isFinite(max)) return [0, 1];
    if (min === max) { min -= 1; max += 1; }
    const span = max - min, raw = span / Math.max(count, 1);
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => span / s <= count) || 10 * mag;
    const out = [];
    for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }
  function legend(container, items, kind) {
    if (items.length < 2) return;
    const lg = document.createElement("div");
    lg.className = "legend";
    items.forEach(s => {
      const k = document.createElement("span"); k.className = "key";
      const sw = document.createElement("span");
      sw.className = kind === "rect" ? "swatch-rect" : "swatch-line";
      sw.style.background = s.color;
      const t = document.createElement("span"); t.textContent = s.name;
      k.append(sw, t); lg.appendChild(k);
    });
    container.appendChild(lg);
  }
  function tooltipEl(container) {
    const t = document.createElement("div");
    t.className = "tooltip hidden";
    container.appendChild(t);
    return t;
  }
  function tipRows(tip, head, rows) {
    tip.replaceChildren();
    const h = document.createElement("div"); h.className = "t-head"; h.textContent = head; tip.appendChild(h);
    rows.forEach(r => {
      const row = document.createElement("div"); row.className = "t-row";
      const key = document.createElement("span"); key.className = "t-key"; key.style.background = r.color;
      const b = document.createElement("b"); b.textContent = r.value;
      const n = document.createElement("span"); n.className = "secondary"; n.textContent = r.name;
      row.append(key, b, n); tip.appendChild(row);
    });
  }
  function place(tip, container, px, py) {
    tip.classList.remove("hidden");
    const w = container.clientWidth, tw = tip.offsetWidth;
    let left = px + 14; if (left + tw > w) left = px - tw - 14;
    tip.style.left = Math.max(0, left) + "px";
    tip.style.top = Math.max(0, py - 10) + "px";
  }

  function line(container, opts) {
    const render = () => {
      container.replaceChildren();
      container.classList.add("chart");
      const series = opts.series.filter(s => s.points && s.points.length);
      if (!series.length) { container.textContent = "No data"; return; }
      legend(container, series.filter(s => !s.noLegend), "line");
      const W = Math.max(container.clientWidth, 280), H = opts.height || 240;
      const m = { t: 10, r: opts.endLabels ? 86 : 16, b: 34, l: 52 };
      const xs = [], ys = [];
      series.forEach(s => {
        s.points.forEach(p => { xs.push(p[0]); ys.push(p[1]); });
        (s.band || []).forEach(b => { ys.push(b[1], b[2]); });
      });
      (opts.refLines || []).forEach(r => ys.push(r.y));
      let x0 = opts.x?.min ?? Math.min(...xs), x1 = opts.x?.max ?? Math.max(...xs);
      let y0 = opts.y?.min ?? Math.min(...ys), y1 = opts.y?.max ?? Math.max(...ys);
      if (opts.y?.pad !== false) { const pad = (y1 - y0) * 0.06 || 0.5; if (opts.y?.min === undefined) y0 -= pad; if (opts.y?.max === undefined) y1 += pad; }
      const X = v => m.l + (v - x0) / ((x1 - x0) || 1) * (W - m.l - m.r);
      const Y = v => H - m.b - (v - y0) / ((y1 - y0) || 1) * (H - m.t - m.b);
      const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, height: H, role: "img", "aria-label": opts.aria || "chart" }, container);
      const yT = niceTicks(y0, y1, 5).filter(v => v >= y0 && v <= y1);
      yT.forEach(v => {
        el("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), class: "grid-line" }, svg);
        const t = el("text", { x: m.l - 8, y: Y(v) + 4, "text-anchor": "end", class: "tick" }, svg);
        t.textContent = (opts.y?.fmt || String)(v);
      });
      const xT = niceTicks(x0, x1, Math.max(3, Math.floor((W - m.l - m.r) / 90))).filter(v => v >= x0 && v <= x1);
      el("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, class: "axis-line" }, svg);
      xT.forEach(v => {
        const t = el("text", { x: X(v), y: H - m.b + 16, "text-anchor": "middle", class: "tick" }, svg);
        t.textContent = (opts.x?.fmt || String)(v);
      });
      if (opts.x?.label) { const t = el("text", { x: (m.l + W - m.r) / 2, y: H - 4, "text-anchor": "middle", class: "axis-title" }, svg); t.textContent = opts.x.label; }
      if (opts.y?.label) { const t = el("text", { x: 12, y: m.t + 2, class: "axis-title", transform: `rotate(-90 12 ${m.t + 2})`, "text-anchor": "end" }, svg); t.textContent = opts.y.label; }
      (opts.refLines || []).forEach(r => {
        el("line", { x1: m.l, x2: W - m.r, y1: Y(r.y), y2: Y(r.y), class: "ref-line" }, svg);
        if (r.label) { const t = el("text", { x: W - m.r - 4, y: Y(r.y) - 4, "text-anchor": "end", class: "ref-label" }, svg); t.textContent = r.label; }
      });
      (opts.refX || []).forEach(r => {
        if (r.x < x0 || r.x > x1) return;
        el("line", { x1: X(r.x), x2: X(r.x), y1: m.t, y2: H - m.b, class: "ref-line" }, svg);
        if (r.label) { const t = el("text", { x: X(r.x) + 4, y: m.t + 10, class: "ref-label" }, svg); t.textContent = r.label; }
      });
      const clipId = "clip" + Math.random().toString(36).slice(2, 9);
      const cp = el("clipPath", { id: clipId }, el("defs", {}, svg));
      el("rect", { x: m.l, y: m.t - 6, width: W - m.l - m.r, height: H - m.t - m.b + 6 }, cp);
      const plot = el("g", { "clip-path": `url(#${clipId})` }, svg);
      series.forEach(s => {
        if (s.band && s.band.length) {
          const up = s.band.map(b => `${X(b[0])},${Y(b[2])}`), lo = s.band.slice().reverse().map(b => `${X(b[0])},${Y(b[1])}`);
          el("polygon", { points: up.concat(lo).join(" "), fill: s.color, "fill-opacity": 0.12 }, plot);
        }
      });
      const placed = [];
      series.forEach(s => {
        const d = s.points.map((p, i) => `${i ? "L" : "M"}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join("");
        el("path", { d, fill: "none", stroke: s.color, "stroke-width": s.width || 2, "stroke-linejoin": "round",
                     "stroke-linecap": "round", "stroke-opacity": s.opacity || 1 }, plot);
        if (opts.endLabels && !s.noEndLabel) {
          const p = s.points[s.points.length - 1];
          el("circle", { cx: X(p[0]), cy: Y(p[1]), r: 4, fill: s.color, stroke: "var(--surface-1)", "stroke-width": 2 }, svg);
          // Colliding end labels are not stacked: the legend and tooltip carry that series.
          const y = Y(p[1]) + 4;
          if (!placed.some(py => Math.abs(py - y) < 14)) {
            placed.push(y);
            const t = el("text", { x: X(p[0]) + 8, y, class: "end-label" }, svg);
            t.textContent = s.endLabel || s.name;
          }
        }
      });
      // crosshair + tooltip
      const tip = tooltipEl(container);
      const cross = el("line", { y1: m.t, y2: H - m.b, class: "crosshair hidden" }, svg);
      const hit = el("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent" }, svg);
      const allX = Array.from(new Set(series.flatMap(s => s.points.map(p => p[0])))).sort((a, b) => a - b);
      const nearest = (arr, v) => arr.reduce((b, p) => Math.abs(p[0] - v) < Math.abs(b[0] - v) ? p : b, arr[0]);
      const move = ev => {
        const r = svg.getBoundingClientRect();
        const px = (ev.clientX - r.left) * (W / r.width);
        const xv = x0 + (px - m.l) / (W - m.l - m.r) * (x1 - x0);
        const xs2 = allX.reduce((b, v) => Math.abs(v - xv) < Math.abs(b - xv) ? v : b, allX[0]);
        cross.setAttribute("x1", X(xs2)); cross.setAttribute("x2", X(xs2)); cross.classList.remove("hidden");
        const rows = series.filter(s => !s.noTooltip).map(s => {
          const p = nearest(s.points, xs2);
          return { name: s.name, color: s.color, value: (opts.y?.tipFmt || opts.y?.fmt || String)(p[1]) };
        });
        tipRows(tip, (opts.x?.tipFmt || opts.x?.fmt || String)(xs2), rows);
        place(tip, container, X(xs2) * (r.width / W), (ev.clientY - r.top));
      };
      hit.addEventListener("pointermove", move);
      hit.addEventListener("pointerleave", () => { tip.classList.add("hidden"); cross.classList.add("hidden"); });
    };
    render();
    registry.add(render);
    return render;
  }

  function bars(container, opts) {
    const render = () => {
      container.replaceChildren();
      container.classList.add("chart");
      const data = opts.bars;
      legend(container, opts.legend || [], "rect");
      const W = Math.max(container.clientWidth, 260), H = opts.height || 200;
      const m = { t: 16, r: 10, b: opts.xLabelRotate ? 46 : 30, l: 46 };
      const vmax = (opts.y?.max ?? Math.max(...data.map(d => d.value), 0) * 1.12) || 1;
      const Y = v => H - m.b - v / vmax * (H - m.t - m.b);
      const band = (W - m.l - m.r) / data.length;
      const bw = Math.min(24, band - 2);
      const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, height: H, role: "img", "aria-label": opts.aria || "bar chart" }, container);
      niceTicks(0, vmax, 4).filter(v => v <= vmax).forEach(v => {
        el("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), class: "grid-line" }, svg);
        const t = el("text", { x: m.l - 8, y: Y(v) + 4, "text-anchor": "end", class: "tick" }, svg);
        t.textContent = (opts.y?.fmt || String)(v);
      });
      el("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, class: "axis-line" }, svg);
      const tip = tooltipEl(container);
      const every = Math.ceil(data.length / Math.max(1, Math.floor((W - m.l) / 46)));
      data.forEach((d, i) => {
        const cx = m.l + band * i + band / 2, y = Y(d.value), h = H - m.b - y, x = cx - bw / 2, r = Math.min(4, h);
        const path = h > 0 ? `M${x},${H - m.b}V${y + r}Q${x},${y} ${x + r},${y}H${x + bw - r}Q${x + bw},${y} ${x + bw},${y + r}V${H - m.b}Z` : "";
        const bar = el("path", { d: path, fill: d.color || "var(--series-1)", class: "bar" }, svg);
        const hit = el("rect", { x: m.l + band * i, y: m.t, width: band, height: H - m.t - m.b, fill: "transparent", tabindex: 0 }, svg);
        if (opts.valueLabels) { const t = el("text", { x: cx, y: y - 4, "text-anchor": "middle", class: "end-label" }, svg); t.textContent = (opts.y?.fmt || String)(d.value); }
        if (i % every === 0 || opts.allLabels) {
          const t = el("text", { x: cx, y: H - m.b + 14, "text-anchor": "middle", class: "tick" }, svg); t.textContent = d.label;
        }
        const show = ev => {
          bar.style.filter = "brightness(1.15)";
          tipRows(tip, d.label, [{ name: d.series || opts.valueName || "", color: d.color || "var(--series-1)", value: (opts.y?.tipFmt || opts.y?.fmt || String)(d.value) }]);
          const rct = svg.getBoundingClientRect();
          place(tip, container, cx * rct.width / W, (ev.clientY ? ev.clientY - rct.top : y));
        };
        hit.addEventListener("pointermove", show);
        hit.addEventListener("focus", show);
        const hide = () => { bar.style.filter = ""; tip.classList.add("hidden"); };
        hit.addEventListener("pointerleave", hide); hit.addEventListener("blur", hide);
        if (opts.onClick) hit.addEventListener("click", () => opts.onClick(d));
      });
    };
    render();
    registry.add(render);
    return render;
  }

  let timer;
  window.addEventListener("resize", () => {
    clearTimeout(timer);
    timer = setTimeout(() => registry.forEach(r => { try { r(); } catch (e) { /* container gone */ } }), 150);
  });
  window.Charts = { line, bars, reset: () => registry.clear() };
})();
