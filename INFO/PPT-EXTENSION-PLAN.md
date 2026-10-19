# PPT-EXTENSION — Plan: Export to PowerPoint (.pptx)

Add an **Export → PowerPoint (.pptx)** feature to `App/index.html`, so a finished flow chart can be downloaded as a `.pptx` file. That file opens in **Microsoft PowerPoint** (Windows, Mac, web) and imports into **Google Slides** (upload to Drive and open with Slides, or *File → Import slides*).

The goal is **editable** output. Every ribbon piece, arrowhead, badge and label arrives as a native slide shape whose color and text can be changed in PowerPoint or Slides. A pixel-perfect **picture** mode is offered as a fallback.

Status: **implemented in `App/index.html` (2026-10-17)**, phases P1–P5 except testing in real PowerPoint and Google Slides (see §12). Related: [FLOW-CHART-APP.md](FLOW-CHART-APP.md) (main app plan; this replaces the "Export to PNG/SVG" item in §1 / §11 M10 and extends it).

---

## 1. Goals and scope

### In scope (v1)
| # | Feature | Notes |
|---|---------|-------|
| 1 | **Export button** | New toolbar group: `[Export ▾]` → *PowerPoint (.pptx)*, *PNG image*, *SVG image*. Shortcut `Ctrl+E` opens the export dialog. |
| 2 | **Editable mode** (default) | One slide. Each shape is converted to native DrawingML shapes: ribbons and arrowheads become freeform shapes, circles become ellipses, and text, titles and labels become text boxes. Each figure is a **group**. |
| 3 | **Picture mode** | One slide with the whole diagram as a high-resolution PNG (2× or 3×). Looks identical to the app everywhere, but cannot be edited. |
| 4 | **Both** | Slide 1 is editable and slide 2 is the picture, so the user can keep whichever imports better. |
| 5 | Slide size | 16:9 (default, 13.333 × 7.5 in), 4:3, or *Match canvas* (custom size with the canvas's aspect ratio). |
| 6 | Area | *Whole canvas* or *Fit to diagram* (shape bounding box plus a margin), centered and scaled to fit the slide. |
| 7 | Labels along curves | Rebuilt from the browser's own text layout, so they sit where they do on screen (see §5). |
| 8 | Works offline, no install | Stays a single `index.html` with **no new dependencies** and no server. The file is built in the browser and downloaded via `Blob`, like Save JSON. |
| 9 | PNG / SVG export | Comes almost for free from the picture-mode rasterizer. |

### Nice-to-have (v1.1+)
- Embed the picture as **SVG with a PNG fallback** (PowerPoint 365 shows the sharp vector; Google Slides uses the PNG).
- Put the **Title** shape into the slide's real title placeholder so it shows in PowerPoint's outline view and the accessibility checker.
- Speaker notes generated from the labels ("COLLECT → TRAIN → DEPLOY → MONITOR").
- *Copy as image* to the clipboard for quick pasting into an open deck.
- Multiple diagrams, one per slide, in a single file. This is the bridge to the AI deck builder in the business plan.
- A PowerPoint add-in or Google Slides add-on that inserts the diagram into the open deck directly.

### Out of scope
- Importing `.pptx` back into the app. **JSON stays the editable source format**, and the `.pptx` is an output only.
- Animations and slide transitions.
- Keynote-specific tuning. Keynote opens `.pptx`, but it is not a test target.

---

## 2. What has to be converted

All coordinates in the app are canvas pixels in a `state.canvas.width × height` space (default 1600 × 900). The shape types to convert are (from `DEFAULTS` in `index.html`):

| App shape | How it draws today | PowerPoint / Slides shape |
|-----------|-------------------|---------------------------|
| `arc` | Filled annular sector, `arcBandPath(s)` | Freeform (`a:custGeom`) built from **two `arcTo` segments + two lines**, an exact match with no sampling |
| `bezier` | Thick stroked cubic, `bezLinePath` | Freeform polygon of the band **outline** (left and right offsets of the curve, sampled every ~2 px) |
| `path` (snapped figure) | One stroked polyline **per piece**, drawn **layer by layer**, then the arrowheads of that layer (`renderPath`) | One freeform outline per piece, plus one freeform per arrowhead, in the same layer order. The whole figure becomes one **group**. |
| `arrow` | Polygon from `chevronPoints()`; `notch-tail` and `gap` use the background color | Freeform polygon. `gap` becomes a thin open freeform line in the background color (plus the filled point on figures). |
| `circle` | `<circle>` with fill, stroke and centered text | `prstGeom ellipse` with fill, line and centered text inside the shape |
| `text` | Multi-line centered `<text>` | Text box: no wrap, centered, zero insets, one paragraph per line |
| `title` | Title + subtitle, aligned left/center/right | One text box with two paragraphs (title and subtitle sizes and colors) |
| `item` | Built-in icon (inline SVG) or uploaded image, with a label under it | Picture (icon **rasterized to PNG** at 4× in its color, or the uploaded image) plus a text box, grouped |
| Band / piece labels | `<textPath>` along the center line; `fitTexts()` shrinks long labels | Text box(es) placed from the browser's measured glyph positions (§5) |
| Canvas background | `#paper` rect | Slide background fill |
| Grid, trace image, selection handles | Editor aids | **Never exported** |

Per-shape `opacity` maps to fill/line alpha (`<a:alpha val="…"/>`).

---

## 3. Key design decisions

| Decision | Choice | Why |
|----------|--------|-----|
| PPTX writer | **A small hand-written OOXML writer** inside `index.html` (~350 lines for templates and the zip), instead of PptxGenJS | Keeps the app dependency-free and offline, in line with §2 of the main plan. We need things PptxGenJS doesn't expose well: **group shapes**, `arcTo` in custom geometry, multi-subpath paths and exact z-order. PptxGenJS 3.12 is the fallback if the spike (§8 P0) shows that the hand-written package is fragile. |
| Zip | A minimal zip writer (local headers + central directory + CRC-32). Compression via the browser's built-in `CompressionStream('deflate-raw')`, falling back to **store** (no compression) where it's unavailable. | No library needed; both variants are valid zip files that PowerPoint and Google accept. |
| Ribbons as… | **Filled outline freeforms** (decided 2026-10-17). Thick open freeform lines are kept only as an internal fallback for outlines that fold on very tight bends. | A filled shape shows its color as *Shape Fill* in both apps, which is what users expect to change, and it scales cleanly when resized. |
| Units | 1 canvas px → `EMU_PER_PX = slideWidthEMU / exportAreaWidthPx`. For the default (1600 px → 12,192,000 EMU) that is **7,620 EMU/px = 0.6 pt/px**. | All geometry goes through one `toEmu([x, y])` transform that includes the area offset and the fit scale. Font sizes are `px × 0.6 × fitScale` pt. |
| Font | **Arial** for both the canvas and the export (decided 2026-10-17). The dialog still offers Calibri and Segoe UI as alternatives. | Arial exists in PowerPoint and Google Slides on every platform. Today the app uses `system-ui`, which is Segoe UI on Windows and San Francisco on Mac, so its glyph widths would differ from the slide. |
| Self-crossing / weaving | Emit shapes in the **exact paint order** the app uses: `state.shapes` by `z`; inside a figure, layer by layer, pieces then arrows | PowerPoint and Slides draw in document order, so the over/under weave carries over with no extra work. |
| Closed ring with no cuts | One path with an outer and an inner subpath. If the spike shows a fill-rule problem, split the ring into **two half-rings** in one group. | DrawingML fill-rule behavior with multiple subpaths has to be verified in both apps. |

---

## 4. Geometry conversion details

Reuse the existing geometry helpers instead of re-deriving shapes:

- **Arc**: `cx, cy, r, arcThick(s), startAngle, sweep(s)`. Path: move to the outer start → `arcTo(wR=hR=ro, stAng, swAng)` → line to the inner end → `arcTo(ri, …, −swAng)` → close. DrawingML angles are in 60,000ths of a degree, clockwise from +x, the same direction as SVG's y-down space.
- **Bézier**: generalize `bezOutlinePath(s)` into `bandOutline(centerPts, halfWidth)`, which returns the left and right offset polylines. Center points come from `sampleStrip(s)` (already spaced ~3 px).
- **Snapped figure piece**: `subPts(pathGeo(s), r0, r1)` for each range in `pieceRanges(s)` → `bandOutline()`. Joins: miter, falling back to round when the miter is longer than 2 × half-width, to match `stroke-linejoin: round`. Very tight bends (radius < half-width) fold the outline back on itself. Clean this up by splitting at fold points into separate sub-outlines in the same piece group, and check it with the test diagrams (§9).
- **Arrowheads**: `chevronPoints(geoAt(...), T, style, width, length)` for figures (see `pathArrowList(s)`), and `arrowPoints(a)` for free arrows. Each becomes a closed polygon freeform. The `gap` style becomes an open freeform with line color = canvas background and width `max(2, width × 0.07)`, mirroring `drawHead()`.
- **Bounding box** for each freeform: compute the min/max of its EMU points, write `a:off` / `a:ext` from it, and write the path in local coordinates (`a:path w h`). Groups use `a:chOff` / `a:chExt` equal to `a:off` / `a:ext`, so child coordinates stay in slide space.

---

## 5. Labels along curves

PowerPoint has no text-on-arbitrary-path. Its WordArt "Follow path" only does fixed arches, and **Google Slides drops WordArt warps on import**. So labels are rebuilt from what the browser has already laid out.

1. Call `render()` so `fitTexts()` has applied its shrink and edge offsets.
2. For every `text.band-label`, read the SVG text APIs: `getNumberOfChars()`, `getStartPositionOfChar(i)`, `getEndPositionOfChar(i)`, `getRotationOfChar(i)` and the final `font-size`. Each glyph's center = the midpoint of start and end; rotation = `getRotationOfChar(i)`. (`dominant-baseline: central` puts that point on the center line.)
3. Pick the output per label, using the **Labels** option (default *Auto*):
   - **Straight**: one text box, centered on the label's middle, rotated to the mean angle. It is fully editable but drifts off very curved bands.
   - **Per-letter**: one small rotated text box per character, grouped as `Label: TRAIN`. It follows any curve exactly, but editing the word means editing letters.
   - **Auto**: Straight when the spread of glyph rotations is ≤ 20°, Per-letter otherwise. Most labels on gentle bands stay as a single, easily edited text box.
4. Text boxes: `wrap="none"`, all insets 0, `anchor="ctr"`, autofit off, centered paragraph, bold and color from the piece, and letter spacing (`spc`) from the app's `fontSize × 0.08`. Box width = measured advance + a small pad, so Google Slides doesn't wrap the text.
5. The label is placed **after** its piece inside the same group, so it travels with the ribbon when moved.

---

## 6. Package structure (minimal valid PPTX)

```
[Content_Types].xml
_rels/.rels
docProps/app.xml, docProps/core.xml                 # title, creator "Flow Chart Builder", timestamps
ppt/presentation.xml                                # sldSz from the chosen slide size
ppt/_rels/presentation.xml.rels
ppt/presProps.xml, ppt/viewProps.xml, ppt/tableStyles.xml
ppt/slideMasters/slideMaster1.xml (+ _rels)         # one blank master
ppt/slideLayouts/slideLayout1.xml (+ _rels)         # "Blank" layout
ppt/theme/theme1.xml                                # minimal theme; major/minor font = export font
ppt/slides/slide1.xml (+ _rels)                     # editable slide
ppt/slides/slide2.xml (+ _rels)                     # picture slide ("Both" mode only)
ppt/media/image1.png …                              # icons, uploaded images, picture-mode render
```

The static parts (master, layout, theme, props) are **template strings** checked once against PowerPoint and Google Slides in P0. Only `presentation.xml`, the slides, their rels, `[Content_Types].xml` and the media are generated per export. Every shape gets a unique `p:cNvPr id` and a readable `name` (`Piece 2 – TRAIN`, `Arrow`, `Badge 1`). The top group gets `descr` alt text listing the labels in flow order.

---

## 7. Code architecture (inside `index.html`)

A new commented section, **`PPTX export`**, placed after the existing Save/Load (`saveJSON`) code:

```
/* ===== Export: shared ===== */
exportArea(opts)            -> { x, y, w, h }        whole canvas or diagram bbox + margin
makeTransform(area, slide)  -> { toEmu(pt), scale, ptPerPx }
cleanSvgForExport()         -> SVGElement clone      no grid, trace, overlay; inlines styles
rasterize(svgEl, scale)     -> Promise<Blob PNG>     <img src=blob:svg> → <canvas> → PNG
exportPNG(), exportSVG()

/* ===== Export: zip ===== */
crc32(bytes), zipFiles([{ name, data }]) -> Promise<Blob>

/* ===== Export: OOXML ===== */
xmlEsc(s), STATIC_PARTS{...}, presentationXml(), slideXml(shapesXml, bg), contentTypes(media)
spFreeform({ name, paths, fill, line, alpha })   // custGeom
spEllipse(...), spTextBox(...), spPicture(...), grpSp(name, children, descr)

/* ===== Export: shape converters ===== */
convertShape(s, T)  → switch on s.type: arc | bezier | path | arrow | circle | text | title | item
bandOutline(centerPts, halfWidth) → [left, right] polylines
labelBoxes(textEl, mode, T) → text box XML (straight / per-letter)

/* ===== Export: UI ===== */
openExportDialog(), exportPPTX(opts) → builds parts → zipFiles → download `flowchart-<stamp>.pptx`
```

- Estimated size: **~900–1,200 lines** of JS, plus ~40 lines of CSS and HTML for the dialog.
- Reuse `stamp()`, `toast()`, the `Blob` download code from `saveJSON()` (factor out a `download(blob, name)` helper), and all geometry helpers.
- Export never changes `state`, so undo history and autosave are unaffected.
- **No paywall:** PowerPoint export is **free in every version** (decided 2026-10-17). No `canExport()` gating is built; Pro is sold on the AI features, templates and cloud library instead.

### Export dialog
```
┌ Export to PowerPoint ───────────────────────────────┐
│ Content     (•) Editable shapes  ( ) Picture  ( ) Both │
│ Slide size  [16:9 ▾]     Area  [Fit to diagram ▾]      │
│ Labels      [Auto ▾]     Font  [Arial ▾]               │
│ Picture resolution [3× ▾] (picture / both only)        │
│                                [Cancel] [Download .pptx]│
│ Tip: Google Slides → File › Import slides, or upload   │
│ to Drive and "Open with Google Slides".                │
└────────────────────────────────────────────────────────┘
```
The last choices are remembered in `localStorage` (wrapped in try/catch). Exporting an empty canvas shows a toast instead of producing a blank file.

---

## 8. Implementation milestones

| Phase | Work | Output | Est. |
|-------|------|--------|------|
| **P0 Spike** | Hand-build a test `.pptx` containing: an `arcTo` arc band, a filled outline ribbon (including a tight bend), a ring with two subpaths, a group, a per-letter label and a straight rotated label, alpha, and a PNG. Open it in PowerPoint (Windows desktop + web) and Google Slides. | Ring fill rule and text box settings confirmed; validated static template parts | 1–2 days |
| **P1 Package + picture mode** | Zip writer, OOXML templates, `rasterize()`, picture-mode PPTX, PNG and SVG export, Export button and dialog skeleton | A usable export on day 3: an image-based PPTX that works everywhere | 2 days |
| **P2 Editable shapes** | Converters for arc, bezier, path (pieces, layers, groups), arrow (all four styles), circle, text, title; opacity; z-order | Editable slide with correct geometry and colors | 3–4 days |
| **P3 Curved labels** | Glyph extraction, Straight / Per-letter / Auto modes, font switch to Arial on the canvas | Labels match the screen | 2 days |
| **P4 Icons, images, polish** | Rasterize built-in icons in color; embed uploaded images; alt text; shape names; Fit-to-diagram and Match-canvas sizes; "Both" mode | Feature-complete | 1–2 days |
| **P5 Test + docs** | Test matrix (§9), fixes, update `guide.html` (Export section and Google Slides steps), update `FLOW-CHART-APP.md` §1/§11 | Release | 2 days |
| | | **Total** | **~11–14 days** |

---

## 9. Test plan

**Test diagrams** (saved as JSON fixtures under `INFO/pptx-tests/`):
1. The built-in **Example** / sample MLOps knot (weaving, many pieces, curved labels).
2. Every arrow style (`chevron`, `triangle`, `notch-tail`, `gap`) on free arrows and on snapped figures, with head and tail arrows.
3. A closed ring with **no cuts**, and one with 4 cuts.
4. Tight bends where the radius is smaller than half the thickness (outline folding).
5. Long labels that `fitTexts()` shrinks; `flipText`; start/end-aligned labels.
6. Opacity < 1, white-on-color text, a non-white canvas background.
7. Icon items (every built-in icon), plus an uploaded PNG and an uploaded SVG image.
8. Non-default canvas sizes: portrait (900 × 1600) and very wide.
9. An empty canvas (expect a toast and no download).

**Where to open each export:**

| Target | Checks |
|--------|--------|
| PowerPoint for Windows (Microsoft 365) | Opens with **no "repair" prompt**; geometry, colors and labels match the app; clicking a piece shows *Shape Fill* = piece color; labels editable; group moves as one |
| PowerPoint for the web | Same as above |
| PowerPoint for Mac | Opens cleanly; fonts render |
| Google Slides (upload to Drive → *Open with Google Slides*, and *File → Import slides*) | Shapes import as shapes (not a picture); groups intact; labels positioned within ~2 px of PowerPoint; no lost alpha |
| LibreOffice Impress (bonus) | Opens; used for automated visual checks |

**Automated checks** (a small script in `INFO/pptx-tests/`, run by hand):
- Unzip the export and parse every XML part (well-formed; unique shape ids; every rel target exists).
- `python-pptx` loads the file and counts shapes per slide (smoke test).
- `soffice --headless --convert-to png` renders slide 1. Compare it with the app's PNG export of the same diagram, and flag a pixel difference above a threshold.

**Acceptance criteria:** for all 9 fixtures, the file opens without errors in PowerPoint and Google Slides; each ribbon piece, arrowhead and badge is a separate editable shape; labels sit on their bands (Auto mode); export of the sample takes < 2 s and the file is < 1 MB in editable mode.

---

## 10. Risks and mitigations

| Risk | Mitigation |
|------|-----------|
| PowerPoint shows "PowerPoint found a problem with content… Repair" | Static parts are validated once in P0; generated XML goes through a DOMParser check before download; the fixtures are re-tested on every change to the export code |
| Google Slides renders text boxes slightly differently (font metrics, autofit) | Arial everywhere; `wrap="none"`, zero insets, autofit off; boxes sized from measured advances plus a pad; **Both** mode as a safety net |
| Per-letter labels are awkward to edit | *Auto* uses a single straight text box whenever the curve allows; the user can force *Straight* |
| Outline folding on tight bends | Fold detection splits the outline; for a piece that still folds, fall back to a thick-line freeform for that piece only |
| Large files from images | PNG at 2×–3× only for picture mode; icons rasterized at the size they are drawn × 4; `deflate-raw` compression |
| `CompressionStream` missing in older browsers | Fall back to an uncompressed (store) zip, which is still valid |
| Canvas font change (`system-ui` → Arial) alters existing diagrams slightly | Labels already auto-shrink via `fitTexts()`; mention it in the release notes; possibly add a font setting on the canvas |

---

## 11. Decisions (resolved 2026-10-17)

| # | Question | Decision |
|---|----------|----------|
| 1 | Ribbon style in the exported file | **Filled outline shapes.** Each piece's color appears as *Shape Fill*. A thick-line freeform is used only for a piece whose outline still folds on a very tight bend. |
| 2 | Font | **Arial** for both the canvas and the export, so labels line up the same in PowerPoint and Google Slides. Existing diagrams' labels may shift slightly; `fitTexts()` absorbs most of it. |
| 3 | Paywall | **PowerPoint export is always free**, in the local app and in any hosted version. No `canExport()` gating. |

> Note: this differs from `Biz-plans-Github-website/PLAN-13-Flowchart-builder.html`, where editable PPTX export is the main Pro ($99/yr) feature. That plan's Pricing, Business Model and Net Profit sections need updating to sell Pro on the AI features, templates and cloud library.

---

## 12. Implementation notes (2026-10-17)

**Where it lives:** the `Export: PowerPoint (.pptx), PNG and SVG` section of the script in `App/index.html` (just before the *Example* section), plus the `#exportDlg` dialog markup, its CSS, the **Export** toolbar button and `Ctrl+E`. Entry points: `openExport()` (dialog), `runExport(opts)` (download), `buildPptx(opts)` (returns the `.pptx` Blob).

**Differences from the plan above:**
- The toolbar has one **Export** button that opens a dialog with a *Format* list (PowerPoint / PNG / SVG), instead of a drop-down menu.
- Uploaded pictures of any type (JPG, GIF, WebP, SVG) are redrawn as PNG, so the package only ever contains PNG media.
- The canvas now draws all diagram text in Arial (`#shapesLayer text`), per decision 2.
- No "Made with" credit is added to exports (this local app is free and unbranded).

**How it was checked** (scripts and results were kept outside the repo, in the session scratchpad):
- The **Example** diagram, the online **sample** (a 16-piece snapped knot with *gap* arrows and layered pieces) and a **stress** diagram were exported. The stress diagram covers a ring with no cuts, head and tail chevrons, end-aligned labels, an arc at 60% opacity, a Bézier band, all four free-arrow styles, two-line text, a badge with no border, an uploaded picture, an icon, special characters in the title and a non-white background. The exports used every Content / Slide size / Area / Labels combination.
- Every XML part passes **XSD validation against the official ECMA-376 Transitional schemas** (`pml.xsd`, `dml-main.xsd`, extended properties). A deliberately broken slide fails it, so the check is real. The zip CRCs, relationship targets and unique shape ids are checked, and **python-pptx 1.0.2** opens every file.
- Each slide was redrawn from its DrawingML by an independent renderer and compared with the app's own PNG. Geometry, colors, weaving order, gap cuts, curved labels, icons, badges and title match.
- Browser tests: the fold fallback (a hairpin bend becomes a thick-line freeform), unit scale (1600 px canvas → 0.6 pt/px), XML escaping, the dialog rows per format/content, `Ctrl+E`, Delete ignored while the dialog is open, an empty canvas refused with a toast, and the PNG, SVG and PPTX downloads.

**Not yet done:** opening the exports in **real PowerPoint (Windows/Mac/web) and Google Slides** (P0 / §9 target matrix). Text metrics, `arcTo` and group handling should be confirmed there before release, especially in Google Slides import.
