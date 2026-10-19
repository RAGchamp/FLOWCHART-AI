# FLOW-CHART-APP — Plan

A single-page, browser-based editor for building **curved, ribbon-style flow diagrams** (like the "MLOps: Eternal Knot" sample in `../Sample-flow-diag.jpg`) that are hard to make in PowerPoint, Excel or Publisher.

Users draw thick **arc** and **Bézier** bands, cap them with **chevron arrowheads**, set each piece's **color** and **text** (text follows the curve), and **save / load** the whole diagram as JSON.

---

## 1. Goals and scope

### In scope (v1)
| # | Feature | Notes |
|---|---------|-------|
| 1 | Canvas workspace | Large drawing surface with pan, zoom, and an optional grid. |
| 2 | **Arc** tool | Draws a thick circular band (annular sector) with adjustable center, radius, start/end angle and thickness. |
| 3 | **Bézier** tool | Draws a thick cubic-Bézier band whose control points can be dragged to make S-curves, loops and swoops. |
| 4 | **Arrow** tool | Click near the end of any band to add a chevron arrowhead that follows the curve's direction at that point. |
| 5 | Selection + right panel | Clicking a shape selects it. The right panel edits its fill color, text, text color, font size and geometry. |
| 6 | Text on shapes | The label runs along the band's center line, like "TRAIN" or "EVALUATE" in the sample. |
| 7 | Save JSON | Downloads the diagram as a `.json` file. |
| 8 | Load JSON | Opens a `.json` file and redraws the full diagram. |

### Nice-to-have (v1.1+)
- Undo/redo, copy/paste, duplicate, delete-key support.
- Free text boxes and circles, for labels like "Data", "ML" and "Ops" and the numbered badges ①②③④.
- Icon/image placement (PNG/SVG).
- Export to PNG / SVG.
- Snap: band ends snap to other band ends so ribbons join seamlessly.
- Z-order controls (bring forward / send back), so ribbons can weave over and under each other as in the knot.
- Multi-select, align and group.

### Out of scope
- Real-time collaboration, accounts or a server. The app is a pure client-side static file.

---

## 2. Technology choices

| Decision | Choice | Why |
|----------|--------|-----|
| Delivery | A single `index.html` (inline CSS + JS), optionally split into `app.js` / `styles.css` | Opens by double-clicking, with no build step and no install. |
| Rendering | **SVG**, not `<canvas>` | Every shape is a DOM element, so click hit-testing, selection outlines and handles come for free. `<textPath>` gives text along a curve natively. Export to SVG is trivial. |
| Framework | Vanilla ES2020 JavaScript | Small scope; no dependencies. |
| Persistence | JSON via `Blob` download + `<input type="file">` / `FileReader` | Works offline in every modern browser. |

> The user-facing word stays **"canvas"**. Internally it is an `<svg>` element sized to the work area.

---

## 3. UI layout

```
┌──────────────────────────────────────────────────────────────────────────┐
│ TOP NAV:  [Select] [Arc] [Bézier] [Arrow] │ [Save JSON] [Load JSON] │ [New] [Zoom −/+] │
├──────────────────────────────────────────────────────┬───────────────────┤
│                                                      │  RIGHT PANEL      │
│                                                      │  (Properties)     │
│                    CANVAS (SVG)                      │                   │
│                                                      │  Type: Arc        │
│        ╭───TRAIN───╮                                 │  Fill  [■ #4CAF50]│
│       ╱             ╲                                │  Text  [TRAIN   ] │
│      │               ▶                               │  Text color [■]   │
│                                                      │  Font size [14]   │
│                                                      │  Thickness [40]   │
│                                                      │  ─ Geometry ─     │
│                                                      │  …tool-specific   │
│                                                      │  [Delete]         │
├──────────────────────────────────────────────────────┴───────────────────┤
│ STATUS BAR: active tool • cursor x,y • hint text ("Click to place center…") │
└──────────────────────────────────────────────────────────────────────────┘
```

- **Top nav bar**: tool buttons work as a radio group, with the active tool highlighted. File actions sit in their own group.
- **Right panel**: shows "Nothing selected — choose a tool or click a shape" when empty. When a shape is selected, it shows that shape's properties.
- **Status bar** (optional): shows step-by-step hints for multi-click tools.

---

## 4. Tools — interaction design

### 4.1 Select (default, `V`)
- Click a shape to select it. A dashed outline appears, the shape's edit handles appear, and the right panel fills in.
- Drag the shape body to move it. Drag a handle to reshape.
- Click empty canvas to deselect.

### 4.2 Arc tool (`A`)
Draws an **annular sector**: a thick ring segment.

1. **Click 1**: set the center.
2. **Click 2**: set the radius and the start angle.
3. **Click 3**: set the end angle. A live preview shows while the mouse moves; the sweep follows the mouse direction.

Default thickness is 40 px. Handles after creation:
- **Center** (move the whole arc)
- **Radius** handle on the mid-angle
- **Start-angle** and **end-angle** handles
- **Thickness** handle on the outer edge

Geometry: the outer path is radius `r + t/2` and the inner path is `r − t/2`, joined by straight end caps. This is drawn as a closed SVG `path` built from two `A` (arc) commands, so the fill color works and the shape is clickable everywhere.

### 4.3 Bézier tool (`B`)
Draws a **thick cubic-Bézier band**, which is how you get S-shapes and swooping connectors.

1. **Click 1**: start point P0.
2. **Click 2**: end point P3. Initial control points P1 and P2 are placed one third and two thirds of the way along, offset sideways to make a gentle S.
3. Drag the **P1 / P2 control handles** (shown with dotted guide lines to P0 / P3) to reshape.

Rendering options, simplest first:
- **v1**: draw the center-line path with `stroke-width = thickness`, `stroke-linecap: butt`, and `fill: none`. This is simple and exact, and `pointer-events: stroke` makes it clickable.
- **v1.1** (if gradients or outlines are needed): compute an offset outline by sampling the curve (about 64 samples) along its normals and build a closed polygon path.

**Multi-segment Bézier (v1.1)**: Shift-click adds more anchor points to form a smooth chain (a poly-Bézier) for long serpentine ribbons.

### 4.4 Arrow tool (`R`)
Adds a **chevron arrowhead** to a band.

1. Hover over any arc or Bézier band. The nearest point on its center line is highlighted, and it snaps to the start or end when it is within 20 px.
2. **Click** to attach a chevron at that point:
   - **Direction** = the curve's tangent at that point. For an arc it is perpendicular to the radius, following the sweep. For a Bézier it is the derivative `B'(t)`.
   - **Size**: width = 1.6 × the band's thickness, and length = 0.8 × the band's thickness. Both are editable.
   - **Color**: inherits the band's fill by default and can be changed.
3. The arrow is stored **attached to its parent** (`parentId` + `t` from 0 to 1). When the band is reshaped, the arrow moves and rotates with it.
4. Clicking on empty canvas places a free-standing chevron whose direction is set by dragging.

Chevron styles, set in the right panel:
- `triangle`: a solid ▶ head.
- `chevron`: a ❯ notched head, like the sample's arrow tips.
- `notch-tail`: a notch cut into the *start* of a band, so ribbons look like they flow into each other.

### 4.5 Keyboard shortcuts
`V` select · `A` arc · `B` Bézier · `R` arrow · `Del` delete · `Esc` cancel current drawing · `Ctrl+S` save · `Ctrl+O` load · `Ctrl+Z`/`Ctrl+Y` undo/redo (v1.1)

---

## 5. Right panel — properties

**Common to every shape**
| Control | Input | Applies to |
|---------|-------|------------|
| Fill color | `<input type="color">` + hex text field + a palette of swatches (sample colors: maroon `#7B1E2B`, green `#5BA829`, navy `#1F3F73`, gold `#E0A526`, grey `#8A8F99`) | all |
| Text | text input | arc, Bézier |
| Text color | color input (default white) | arc, Bézier |
| Font size / bold | number input + toggle | arc, Bézier |
| Text position | slider 0–100% along the path (`startOffset`) | arc, Bézier |
| Flip text | checkbox, which reverses path direction so text is never upside-down | arc, Bézier |
| Opacity | slider | all |
| Delete | button | all |

**Arc-specific**: center X/Y, radius, start angle°, end angle°, thickness.
**Bézier-specific**: P0, P1, P2, P3 (x, y), thickness.
**Arrow-specific**: style, width, length, parent link (read-only), "detach" button.

Every input updates the shape **live** with the `input` event.

---

## 6. Data model (JSON schema)

```json
{
  "app": "FLOW-CHART-APP",
  "version": 1,
  "canvas": { "width": 1600, "height": 900, "background": "#FFFFFF" },
  "shapes": [
    {
      "id": "s1",
      "type": "arc",
      "cx": 300, "cy": 400, "r": 120,
      "startAngle": 200, "endAngle": 520,
      "thickness": 40,
      "fill": "#7B1E2B",
      "text": "CURATE",
      "textColor": "#FFFFFF", "fontSize": 14, "bold": true,
      "textOffset": 50, "flipText": false,
      "opacity": 1,
      "z": 0
    },
    {
      "id": "s2",
      "type": "bezier",
      "p0": [420, 400], "p1": [520, 250], "p2": [620, 550], "p3": [720, 400],
      "thickness": 40,
      "fill": "#5BA829",
      "text": "FORMULATE",
      "textColor": "#FFFFFF", "fontSize": 14, "bold": true,
      "textOffset": 50, "flipText": false,
      "opacity": 1,
      "z": 1
    },
    {
      "id": "s3",
      "type": "arrow",
      "parentId": "s2", "t": 1.0,
      "style": "chevron",
      "width": 64, "length": 32,
      "fill": "#5BA829",
      "opacity": 1,
      "z": 2
    }
  ]
}
```

Rules:
- Angles are in degrees, clockwise from 3 o'clock (SVG convention). `endAngle` may exceed 360 to allow sweeps greater than one turn.
- Arrows with a `parentId` compute their position from the parent. Free arrows store `x`, `y` and `angle` instead.
- `z` sets draw order. The renderer sorts by `z` before appending to the SVG.
- **Load validation**: check `app` and `version`, skip unknown shape types with a console warning, and fill in missing fields from defaults so older files still load.

---

## 7. Code architecture

```
FLOW-CHART-APP/
├── index.html          # layout: top nav, <svg id="canvas">, right panel, status bar
├── styles.css
├── js/
│   ├── state.js        # diagram model: shapes[], selectedId, nextId; add/update/remove; emits 'change'
│   ├── geometry.js     # arcPath(), bezierPoint(t), bezierTangent(t), arcPoint/Tangent, nearestPointOnShape()
│   ├── render.js       # model → SVG elements (path + textPath + handles); full re-render, then diff later if needed
│   ├── tools/
│   │   ├── select.js
│   │   ├── arc.js
│   │   ├── bezier.js
│   │   └── arrow.js    # each tool: onPointerDown/Move/Up, onKey, preview(), cancel()
│   ├── panel.js        # builds the right-panel form for the selected shape and binds inputs to state
│   ├── io.js           # saveJSON() / loadJSON(file) / validate+migrate
│   └── main.js         # wires toolbar, keyboard, tool switching, pan/zoom
└── INFO/
    └── FLOW-CHART-APP.md
```

A single-file `index.html` build is also acceptable for easy sharing. Keep the same module boundaries as sections inside one `<script>`.

**Core patterns**
- **Single source of truth**: `state.shapes`. The UI never stores geometry in the DOM. `render()` rebuilds the SVG from state, which keeps save/load trivial: load = replace state, then render.
- **Tool interface**: `{ name, cursor, onPointerDown, onPointerMove, onPointerUp, onKeyDown, cancel }`. `main.js` forwards pointer events to the active tool.
- **Coordinates**: convert screen to SVG coordinates with `svg.getScreenCTM().inverse()` so pan and zoom don't break drawing.
- **Text on path**: each band gets a hidden center-line `<path id="cl-{id}">` and a `<text><textPath href="#cl-{id}" startOffset="50%" text-anchor="middle" dominant-baseline="middle">`.
- **Hit-testing**: native SVG events. Each shape `<g data-id="…">` catches the click, so there is no manual math. The math is needed only for the Arrow tool's "nearest point on curve" (sample the curve and choose the closest sample).

**Key geometry**
- Arc point: `(cx + r·cos θ, cy + r·sin θ)`. Tangent: `(−sin θ, cos θ)` × sweep sign.
- Cubic Bézier: `B(t) = (1−t)³P0 + 3(1−t)²t·P1 + 3(1−t)t²·P2 + t³P3`.
  Its derivative: `B'(t) = 3(1−t)²(P1−P0) + 6(1−t)t(P2−P1) + 3t²(P3−P2)`.
- Chevron at point P with unit tangent T and normal N (width w, length L):
  tip = `P + T·L`, wings = `P ± N·w/2`, notch = `P + T·(L·0.4)` (for the `chevron` style).
- Arcs with a sweep of 180° or more need the SVG `large-arc-flag = 1`. Sweeps of 360° or more must be split into two arc commands.

---

## 8. Save / Load flow

**Save**
1. Serialize `{ app, version, canvas, shapes }` with `JSON.stringify(data, null, 2)`.
2. Create `new Blob([...], {type: "application/json"})`, then use `URL.createObjectURL` with a temporary `<a download="flowchart.json">` and click it.
3. Optional: a filename prompt, defaulting to `flowchart-YYYYMMDD-HHMM.json`.

**Load**
1. **Load JSON** clicks a hidden `<input type="file" accept=".json,application/json">`.
2. `FileReader.readAsText`, then `JSON.parse` inside try/catch. Invalid files show a friendly error toast.
3. Validate and migrate, replace the state, reset the selection, `render()`, and zoom to fit the content.
4. If the current diagram has unsaved changes, confirm before replacing it.

Optional autosave: mirror the state to `localStorage` on each change, and offer "Restore last session?" on startup.

---

## 9. Implementation milestones

| Milestone | Deliverable | Done when… |
|-----------|-------------|-----------|
| **M1 – Shell** | HTML layout: top nav, SVG canvas, right panel, status bar; tool switching | Buttons toggle the active tool; the panel shows its empty state. |
| **M2 – Model + render** | `state.js`, `render.js`, `geometry.js` | Hard-coded sample shapes render correctly. |
| **M3 – Arc tool** | 3-click creation + handles | You can draw and reshape a ring segment. |
| **M4 – Bézier tool** | 2-click creation + P1/P2 handles | You can draw an S-curve band. |
| **M5 – Selection + panel** | Select, move, live property editing, delete | Clicking any shape lets you change its color and text. |
| **M6 – Text on path** | `textPath` labels, offset and flip | Labels follow the curves and stay readable. |
| **M7 – Arrow tool** | Snap to band, tangent-aligned chevron, attached to parent | An arrow added at a band's end follows it when the band is reshaped. |
| **M8 – Save/Load** | JSON export/import + validation | A saved diagram, reloaded, renders identically. |
| **M9 – Polish** | Pan/zoom, grid, keyboard shortcuts, color swatches, status hints | Usable without instructions. |
| **M10 – Extras** | Undo/redo, PNG/SVG export, text boxes, z-order | Optional. |

---

## 10. Test plan

- **Recreate the sample**: rebuild the "MLOps Eternal Knot" using about 4 arcs (maroon, green, navy, gold loops) plus Bézier crossovers (FORMULATE, PLAN, CODE, RELEASE) and chevrons. This is the acceptance test.
- **Round-trip**: save, reload, then save again. The two JSON files must be identical.
- **Geometry edge cases**: arcs with a sweep of 359° and 360° or more, arcs with a negative sweep, a Bézier with coincident control points, a thickness greater than the radius (clamp it to `r`).
- **Arrow attachment**: reshape and move the parent, and the arrow follows. Delete the parent, and its attached arrows are deleted too (after a confirmation).
- **Text**: long text overflowing the path is truncated or shrunk. Flipped text reads left-to-right.
- **Browsers**: latest Chrome, Edge and Firefox. Check Windows display scaling at 125% and 150%.
- **Bad input**: loading a non-JSON file, malformed JSON, or JSON from another app shows an error, and the current diagram stays intact.

---

## 11. Decisions (resolved 2026-10-17)

| # | Question | Decision |
|---|----------|----------|
| 1 | Multi-colored segments on one ring (CURATE / COLLECT / …) | **Separate arcs + end-snapping.** Each segment is its own arc, and the ends of adjacent arcs snap together within 12 px so they join seamlessly. End-snapping moves from "nice-to-have" into **v1** (milestone M3/M4). |
| 2 | Center icons and labels | **Text boxes + circles in v1.** There is a free **Text** shape (for "Data", "ML", "Ops", "Dev") and a **Circle badge** shape (for numbered markers ①②③④), each with fill, border, text, text color and font size. Image/icon placement stays in v1.1. |
| 3 | Image export | **JSON only in v1.** PNG/SVG export is deferred to v1.1. *Update 2026-10-17:* Export to editable PowerPoint (.pptx), PNG and SVG is now built; see [PPT-EXTENSION-PLAN.md](PPT-EXTENSION-PLAN.md). |
| 4 | Packaging | **A single `index.html`** with inline CSS and JS. The module boundaries in §7 become clearly commented sections inside one `<script>`. |

### Resulting v1 changes
- **Top nav**: `[Select] [Arc] [Bézier] [Arrow] [Text] [Circle] │ [Save JSON] [Load JSON] │ [New] [Zoom −/+]`
- **New shape types** in the JSON schema:
  ```json
  { "id": "s4", "type": "text", "x": 300, "y": 420, "text": "Data",
    "textColor": "#7B1E2B", "fontSize": 28, "bold": true, "z": 3 }
  { "id": "s5", "type": "circle", "cx": 120, "cy": 520, "r": 16,
    "fill": "#FFFFFF", "stroke": "#333333", "strokeWidth": 2,
    "text": "1", "textColor": "#333333", "fontSize": 16, "z": 4 }
  ```
- **Tools**: Text (`T`): click to place, then type the text in the right panel. Circle (`C`): click for the center, drag to set the radius.
- **Snapping**: while an arc or Bézier endpoint is being drawn or dragged, it snaps to any other band's endpoint within 12 px. Hold `Alt` to disable snapping.
- **File layout (§7)** collapses to:
  ```
  FLOW-CHART-APP/
  ├── index.html      # everything: styles, layout, state, geometry, render, tools, panel, io
  └── INFO/FLOW-CHART-APP.md
  ```
- **M10 extras** now include: PNG/SVG export, image icons, undo/redo, z-order.
