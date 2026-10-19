/* Random ribbon diagrams for training FlowTrace (INFO/FLOW-CHART-AI-PLAN.md §5.1).
   Runs inside App/index.html (render.py injects it), so it builds diagrams with the app's own
   helpers: bands → Snap it → cuts and piece styles, exactly as a person would.

   window.makeSample(seed, words) → { doc, parts } where parts are the documents to render:
     image  : the diagram as drawn
     ribbon : ribbons only, white on black (cuts and labels stay black)
     text   : ribbon labels only, white on black
     center : the center line of every ribbon, including the parts hidden under crossings */
(function () {
  function rng(seed) {             // mulberry32
    let a = seed >>> 0;
    return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  }
  const BASE = ['#7B1E2B', '#5BA829', '#1F3F73', '#E0A526', '#8A8F99', '#2B8C8C', '#C2410C', '#6D28D9', '#0EA5E9', '#DB2777', '#15803D', '#B45309', '#475569', '#9333EA', '#DC2626', '#0F766E'];
  const hsl = (h, s, l) => {
    const k = n => (n + h / 30) % 12, a = s * Math.min(l, 1 - l);
    const c = n => Math.round(255 * (l - a * Math.max(-1, Math.min(k(n) - 3, 9 - k(n), 1)))).toString(16).padStart(2, '0');
    return ('#' + c(0) + c(8) + c(4)).toUpperCase();
  };

  window.makeSample = function (seed, words) {
    const R = rng(seed), U = (a, b) => a + (b - a) * R(), I = (a, b) => Math.floor(U(a, b + 1)), pick = a => a[Math.floor(R() * a.length)];
    const color = () => R() < 0.7 ? pick(BASE) : hsl(U(0, 360), U(0.35, 0.85), U(0.28, 0.6));
    const word = () => { const w = pick(words); return R() < 0.15 ? w[0] + w.slice(1).toLowerCase() : w; };
    const W = 1600, H = pick([900, 900, 1000, 1200]);
    const shapes = [];
    const add = (type, props) => { const s = Object.assign({ id: 's' + (shapes.length + 1), type }, clone(DEFAULTS[type]), props); s.z = shapes.length; shapes.push(s); return s; };
    const TH = U(24, 58), ringColors = [];

    // ---- ribbons
    const kind = R() < 0.65 ? 'knot' : R() < 0.6 ? 'serpentine' : 'rings';
    const centers = [];
    if (kind === 'knot' || kind === 'rings') {
      const n = kind === 'knot' ? I(1, 5) : I(1, 3);
      const spread = kind === 'knot' ? U(1.85, 2.6) : U(1.3, 2.4);        // close loops weave like the eternal knot; rings may overlap
      const Rr = Math.min(U(80, 180), (W - 240) / ((n - 1) * spread + 2.4));
      const x0 = U(120, 200) + Rr, step = Rr * spread, yMid = H * U(0.42, 0.58), dy = Rr * U(0.35, 0.9);
      for (let i = 0; i < n; i++) centers.push([x0 + i * step, yMid + (i % 2 ? -dy : dy) * (kind === 'knot' ? 1 : U(-0.5, 0.5))]);
      const on = (c, a, r) => [f(c[0] + r * Math.cos(rad(a))), f(c[1] + r * Math.sin(rad(a)))];
      if (kind === 'knot') {
        for (let i = 0; i + 1 < n; i++) {             // connectors first, so the rings weave over them
          const a = centers[i], b = centers[i + 1], ang = deg(Math.atan2(b[1] - a[1], b[0] - a[0]));
          const s = add('bezier', Object.assign(bezierFrom(on(a, ang - U(15, 35), Rr), null, on(b, ang + 180 + U(15, 35), Rr), null), { thickness: TH * U(0.85, 1), fill: color() }));
          s.flipText = autoFlip(s);
        }
      }
      centers.forEach((c, i) => {
        const toward = centers[i + 1] || centers[i - 1] || [c[0] + 1, c[1]];
        const g = deg(Math.atan2(toward[1] - c[1], toward[0] - c[0]));
        const dir = i % 2 ? -1 : 1, GAP = kind === 'knot' && n > 1 ? U(55, 90) : U(4, 40), segs = I(1, 4);
        const total = 360 - GAP, start = g + dir * GAP / 2, seg = total / segs, r = Rr * (kind === 'rings' ? U(0.8, 1.2) : 1);
        const fill = color();
        ringColors.push(fill);
        for (let k = 0; k < segs; k++) {
          const a0 = start + dir * k * seg, a1 = start + dir * (k + 1) * seg;
          const s = add('arc', { cx: f(c[0]), cy: f(c[1]), r: f(r), startAngle: f(a0), endAngle: f(a1), thickness: f(Math.min(TH, r * 0.8)), fill });
          s.flipText = autoFlip(s);
        }
      });
    } else {
      // a serpentine: Béziers that continue each other smoothly across the page
      let p = [U(120, 300), H * U(0.3, 0.7)], d = [1, U(-0.6, 0.6)];
      const n = I(2, 6), len = (W - 400) / n;
      for (let i = 0; i < n; i++) {
        const q = [Math.min(W - 120, p[0] + len * U(0.8, 1.2)), Math.max(150, Math.min(H - 150, p[1] + U(-0.5, 0.5) * H))];
        const s = add('bezier', Object.assign(bezierFrom(p, norm(d), q, null), { thickness: TH, fill: color() }));
        s.p2 = [f(s.p2[0] + U(-80, 80)), f(s.p2[1] + U(-120, 120))];
        s.flipText = autoFlip(s);
        d = norm([s.p3[0] - s.p2[0], s.p3[1] - s.p2[1]]);
        p = s.p3.slice();
      }
    }

    // ---- things around the ribbons (in the picture, never in the masks)
    const ink = R() < 0.5 ? '#1F2328' : color();
    if (R() < 0.8) add('title', { x: f(U(30, 120)), y: f(U(20, 60)), title: `${word()} ${word()}`.slice(0, 28), subtitle: R() < 0.7 ? `${word()} ${word()} ${word()}`.toLowerCase() : '', titleSize: f(U(22, 38)), subtitleSize: f(U(13, 20)), titleColor: ink });
    centers.forEach((c, i) => {
      const col = ringColors[i] || color();
      if (R() < 0.75) add('item', { x: c[0], y: c[1], icon: R() < 0.85 ? pick(Object.keys(ICONS)) : '', iconSize: f(U(30, 70)), iconColor: col, text: word().slice(0, 8), textColor: col, fontSize: f(U(16, 30)) });
      if (R() < 0.6) add('circle', { cx: f(c[0] + U(-1.6, 1.6) * 90), cy: f(c[1] + U(1.1, 1.6) * 90), r: f(U(12, 22)), fill: '#FFFFFF', stroke: col, strokeWidth: f(U(1.5, 3)), text: String(i + 1), textColor: col, fontSize: f(U(12, 20)) });
    });
    for (let k = I(0, 3); k > 0; k--) add('text', { x: f(U(100, W - 100)), y: f(U(60, H - 60)), text: `${word()} ${word()}`.toLowerCase(), textColor: '#4B5563', fontSize: f(U(11, 18)), bold: R() < 0.3 });
    const bg = R() < 0.8 ? '#FFFFFF' : hsl(U(0, 360), U(0, 0.3), U(0.93, 0.99));

    // ---- build it the way a person would: Snap it, then arrows (cuts) and piece styles
    loadDoc(normalizeDoc({ app: APP_ID, version: FILE_VERSION, canvas: { width: W, height: H, background: bg }, shapes }));
    snapIt();
    const styleAll = pick(['chevron', 'triangle', 'notch-tail', 'gap', 'gap']);
    for (const s of state.shapes.filter(x => x.type === 'path')) {
      const L = pathGeo(s).L, th = s.thickness;
      s.arrowStyle = R() < 0.8 ? styleAll : pick(['chevron', 'triangle', 'notch-tail', 'gap']);
      const hs = headSize(s.arrowStyle, th);
      Object.assign(s, { arrowWidth: f(hs.w * U(0.9, 1.15)), arrowLength: f(hs.l * U(0.8, 1.3)), reverse: R() < 0.3 });
      const nCuts = Math.max(0, Math.min(14, Math.round(L / (th * U(3, 7)))));
      for (let k = 0; k < nCuts; k++) insertCut(s, U(0.02, 0.98));
      if (!s.closed && R() < 0.5) s.headArrow = true;
      const scheme = pick(['one', 'one', 'alternate', 'each']), c1 = s.pieces[0].fill, c2 = color();
      const fs = f(th * U(0.32, 0.48)), tc = R() < 0.88 ? '#FFFFFF' : '#1F2328';
      s.pieces.forEach((pc, i) => {
        pc.fill = scheme === 'one' ? c1 : scheme === 'alternate' ? (i % 2 ? c2 : c1) : color();
        pc.text = R() < 0.75 ? word() : '';
        Object.assign(pc, { fontSize: fs, textColor: tc, bold: R() < 0.85, textAlign: pick(['center', 'center', 'start', 'end']), layer: R() < 0.15 ? 1 : 0 });
      });
    }
    render();
    const doc = serialize();

    return { doc, parts: Object.assign({ image: doc }, labelDocs(doc)) };
  };

  /** The three label documents for a diagram (also used for real training pairs and scoring). */
  window.labelDocs = function (doc) {
    const only = d => { d.shapes = d.shapes.filter(s => s.type === 'path' || s.type === 'arc' || s.type === 'bezier' || s.type === 'arrow'); d.canvas.background = '#000000'; return d; };
    const ribbon = only(clone(doc)), text = only(clone(doc)), center = only(clone(doc));
    for (const s of ribbon.shapes) { if (s.type === 'path') s.pieces.forEach(pc => { pc.fill = '#FFFFFF'; pc.text = ''; }); else { s.fill = '#FFFFFF'; s.text = ''; } }
    for (const s of text.shapes) {
      if (s.type === 'path') s.pieces.forEach(pc => { pc.fill = '#000000'; pc.textColor = '#FFFFFF'; });
      else { s.fill = '#000000'; s.textColor = '#FFFFFF'; }
    }
    for (const s of center.shapes) {
      if (s.type === 'path') { s.pieces.forEach(pc => { pc.fill = '#FFFFFF'; pc.text = ''; pc.layer = 0; }); Object.assign(s, { thickness: 3, arrowStyle: 'triangle', arrowWidth: 0.1, arrowLength: 0.1, headArrow: false, tailArrow: false }); }
      else if (s.type === 'arrow') s.width = s.length = 0.1;
      else { s.fill = '#FFFFFF'; s.text = ''; s.thickness = 3; }
    }
    return { ribbon, text, center };
  };
})();
