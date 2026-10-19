# FLOW-CHART-AI — Plan: "Trace it" with a self-trained small model

Add an **AI Trace** feature to `App/index.html`. The user uploads a picture of a ribbon-style flow chart with **Trace image**, as today, and clicks **Trace it**. A small model that we train ourselves, called **FlowTrace**, reads the picture and draws the diagram as normal, editable app shapes: bands, a snapped figure, arrow cuts, colored and labelled pieces, icons, numbered badges and the title. The user then fixes whatever it got wrong with the existing tools.

The acceptance picture is `../Sample-flow-diag.jpg`. It is a phone photo of a slide on a monitor showing the "MLOps: Eternal Knot", so it has perspective, glare, a cut-off right edge, and labels written along tight curves.

Status: **implemented 2026-10-18**: P0–P7 and P9, plus the training kit for P2/P3. Trace it works end to end with classic vision. A trained model is waiting on the overnight training runs. §13 lists what was built and where it differs from this plan. All open decisions are resolved (§12). Related: [FLOW-CHART-APP.md](FLOW-CHART-APP.md) (the app), [PPT-EXTENSION-PLAN.md](PPT-EXTENSION-PLAN.md) (export), `guide.html` §11 (Trace image), §6 (Snap it), §7 (Arrows).

---

## 0. Read this first: what this computer can and cannot train

Hardware checked on 2026-10-18:

| Item | Found | What it means |
|------|-------|---------------|
| CPU | Intel Pentium Silver N5030, 4 cores, 1.1 GHz | A low-power laptop CPU. Training is possible, but slow. |
| GPU | Intel UHD 605 only; **no NVIDIA GPU** (no CUDA) | PyTorch trains on the CPU only. |
| RAM | 32 GB | Enough to hold the datasets and the small models below. |
| Disk | ~1.8 TB free | No limit on synthetic data. |
| Software | Python 3.12, PyTorch 2.9, OpenCV 4.13, onnxruntime 1.20, Playwright 1.61, pytesseract + Tesseract, Node 24, Ollama 0.6 | Everything needed is already installed. |

**Consequences for the design:**

1. **A general vision-language model cannot be trained or fine-tuned here.** That includes LLaVA, Qwen-VL, Florence-2 and Pix2Struct, which have 0.2 to 7 billion parameters. Even running one through Ollama would take minutes per picture on this CPU.
2. **What does fit is a small, very specialised model.** That means about 0.5–2 M parameters, 256–384 px input, trained only on ribbon diagrams. It can train here in about **one night to a few days**. Phase 0 measures the real speed before we commit.
3. **The model should not try to "write" the diagram on its own.** Instead it marks *where* the ribbons, center lines, cuts, arrows and text are, pixel by pixel. Deterministic code then turns that into app shapes. That code reuses the app's own `snapIt()`, `insertCut()` and `pathGeo()`. This split is what makes a small model good enough.
4. **The true "small language model" comes in Phase 2.** It is an image → diagram-program decoder (§6.3) that writes a compact token language, like an SLM writing code. At 15–30 M parameters it needs a GPU to train in reasonable time. The plan uses a free cloud GPU for that step only (Kaggle: about 30 h/week; Colab). Inference still runs on this computer. If cloud is not acceptable, Phase 2 can train locally as a multi-day run at a reduced size.
5. **We do not train OCR from scratch.** Tesseract is already installed and reads the labels once we straighten them (§6.2 step 6). Training our own text reader is a large project in itself and would not do better.

Everything runs **locally and offline**. No picture leaves the computer.

---

## 1. Goals and scope

### In scope (v1)
| # | Feature | Notes |
|---|---------|-------|
| 1 | **Trace it** button | Shown in the Trace image section of the panel and in the top bar once a trace picture is loaded. Shortcut `Ctrl+Shift+T`. |
| 2 | **Corner / crop step** | Four draggable corner handles on the trace picture. They are pre-placed by the AI (the slide edges), and the user can adjust them. This fixes perspective and crops away the monitor, the side text and anything that is not the diagram. |
| 3 | **Ribbons** | Detects every band, including where ribbons cross over and under each other. Outputs editable **Arc and Bézier bands**, then runs **Snap it** automatically. |
| 4 | **Pieces and arrows** | Finds color changes and chevrons along the ribbon, inserts **cuts** there, and sets each piece's **fill**. It also detects the **arrow style** (chevron / triangle / notch / gap) and the **direction of flow**. |
| 5 | **Labels on pieces** | Reads the curved text (TRAIN, EVALUATE, CURATE…) and puts it on the right piece, with its alignment (start / center / end) and text color. |
| 6 | **Over / under** | Where the figure crosses itself, the piece seen on top in the picture is brought to the front (piece `layer`). |
| 7 | **Badges, icons, free text, title** | Numbered circles ①②③④ become **Circle** shapes with their number. Loop centers become **Icon** items: the closest of the 16 built-in icons, plus the label ("Data", "ML", "Dev", "Ops"). The heading becomes a **Title**. |
| 8 | **Review mode** | The result arrives as **one undo step**. Low-confidence items get an orange dashed outline and are listed in the panel ("Piece 7: label unclear"), and clicking an entry selects that shape. |
| 9 | **Runs locally** | A small local service (`AI/service`) started with one double-click. The app keeps working without it: the button then explains how to start it. |
| 10 | **Training kit** | A synthetic-data generator that uses the app itself, plus a **Save training pair** button to turn real traced pictures into labelled examples, plus training and evaluation scripts. |

### Nice-to-have (v1.1+)
- Run fully **in the browser** (onnxruntime-web + model bundled as a `.js` file) so no service has to be started (§7.2).
- **Trace a region**: drag a box and trace only that part, merging with what is already drawn.
- **Re-trace a piece**: right-click a piece and choose "Re-read label" or "Re-detect color".
- **Phase 2 model** (§6.3): the image → diagram-program decoder that writes the bands directly.
- Learn from corrections: when the user fixes a traced diagram and saves it, offer to store it as a training pair.

### Out of scope
- Box-and-arrow flowcharts (rectangles, diamonds, straight connectors). The app has no rectangle or connector tools, and the AI can only output what the app can draw. Adding those tools is a separate plan.
- The side paragraphs in the sample ("Data is a … between Curation…"). That text is not part of the diagram and is cropped away in the corner step.
- Hand-drawn sketches on paper. They may partly work, but they are not a target in v1.
- Reproducing photographic detail such as gradients, shadows or 3-D bevels. The output uses the app's flat style.

---

## 2. How it fits the existing app

The app already has everything the AI needs to **build** a diagram. The AI only has to decide *what* to build.

| Existing piece of the app | How the AI uses it |
|---------------------------|--------------------|
| `state.trace` + `openTrace()` (Trace image) | The picture to trace. It is already fitted inside the canvas (`preserveAspectRatio="xMidYMid meet"`), so a picture pixel maps to a canvas unit by one scale + offset. |
| Arc / Bézier shapes (`DEFAULTS`) | The AI's first output: editable bands with handles. They are easy to fix by hand before snapping. |
| `snapIt()` | Joins the AI's bands into one figure, closes gaps and trims overlaps. The AI does not have to place the joins exactly, just as a person does not. |
| `pathGeo()` / `geoAt()` | Converts an AI-detected point (a color change or a chevron) into a cut fraction 0–1 on the figure. |
| `insertCut()` + `pieces[]` | Splits the figure and sets `fill`, `text`, `textAlign`, `textColor` per piece. |
| `reorder()` / piece `layer` | Sets the over/under order at crossings. |
| `ICONS` (16 built-in icons), Circle, Title, Text | Targets for the icon classifier, badge detector and title reader. |
| `commit()` | Makes the whole trace a single undo step, so `Ctrl+Z` removes it. |
| `normalizeDoc()` / `loadDoc()` | Validate the AI's output exactly as if it were a loaded file. Bad output can never break the app. |
| PNG export rasterizer | Renders synthetic training pictures with the app's real drawing code (§5). |

**The key idea:** the trained model plus the post-processing code produce the same things a person produces by hand when tracing. Those are rough bands, snap, arrow clicks, colors and labels, so the rest of the app works unchanged.

### 2.1 New code in `index.html` (about 250–350 lines)
| Section | What it adds |
|---------|-------------|
| `Automation API` | `window.FlowAPI = { loadDoc, getDoc, snapIt, insertCut, pathGeo, renderToPNG, … }`. Used by the AI result applier, and by the Playwright data generator (§5). It is the same functions the buttons call, only made reachable by name. |
| `AI trace` | `traceCorners` (the 4 handles on the trace picture), `runTrace()` (POST to the service, progress, cancel), `applyTrace(result)` (§4), and the review list in the panel. |
| `Save training pair` | Downloads `pair-YYYYMMDD-HHMM.zip`, containing the original picture, the corner points, and the current diagram JSON, with the picture → canvas transform. |
| Panel | Trace image section gains: **Set corners**, **Trace it**, what to detect (Ribbons ✓ Labels ✓ Icons ✓ Badges ✓ Title ✓), and **Detail** (simplify ↔ precise). |

---

## 3. User flow

```
[Trace image] → picture appears faded
      │
      ▼
[Trace it] ──► corner handles appear on the picture, pre-placed on the slide edges
      │        (drag to fix; Enter = OK, Esc = cancel)
      ▼
progress toast: Straightening… Finding ribbons… Reading labels… Building shapes…   [Cancel]
      │        (target: < 20 s on this computer for a 1600×1200 photo)
      ▼
result drawn on top of the picture, selected as a group, ONE undo step
panel ▸ "AI Trace result": 4 loops, 1 figure, 16 pieces, 14 labels, 4 icons, 4 badges
        Needs a look (3):  • Piece 9 label "PLAN" (low confidence)
                           • Crossing near CODE/FORMULATE: which is on top?
                           • Icon in loop 3: picked "code"
        [Accept]  [Undo trace]  [Trace again with other settings]
      │
      ▼
user fixes with normal tools → Clear image → Save / Export
```

Rules:
- **Trace it** is disabled until a trace picture exists. If the service is not running, the button shows how to start it (`AI\start-flowtrace.bat`), and nothing else changes.
- Tracing never deletes existing shapes. If the canvas is not empty, the user is asked: **Add to drawing** or **Replace drawing**.
- Every traced item stays a normal shape. The orange "needs a look" outline exists only in the overlay. It is not saved and it disappears on Accept.
- Option **Stop after bands**: the user gets the editable arcs and Béziers without snapping or cutting, so they can fine-tune the geometry first and then click Snap it themselves.

---

## 4. Contract between the app and the AI

The app sends the picture and the corners. The AI returns **app shapes plus a figure plan**. The app applies the plan with its own functions, so geometry logic is not duplicated in Python.

**Request** `POST http://127.0.0.1:8765/trace`
```json
{
  "image": "data:image/jpeg;base64,…",
  "corners": [[112,58],[1580,40],[1602,1130],[96,1150]],
  "canvas": { "width": 1600, "height": 1200 },
  "detect": { "ribbons": true, "labels": true, "icons": true, "badges": true, "title": true },
  "detail": 0.5
}
```
`corners` are in picture pixels (top-left, top-right, bottom-right, bottom-left). The returned coordinates are already in **canvas units**.

**Response**
```json
{
  "flowtraceVersion": "1.0.0",
  "model": "seg-v3-2026-11-02",
  "timings": { "straighten": 0.2, "segment": 4.1, "graph": 1.3, "ocr": 6.8, "total": 13.0 },
  "bands": [
    { "type": "arc", "cx": 300, "cy": 520, "r": 140, "startAngle": 200, "endAngle": 470, "thickness": 46, "conf": 0.91 },
    { "type": "bezier", "p0": [430,470], "p1": [520,420], "p2": [600,520], "p3": [690,470], "thickness": 44, "conf": 0.78 }
  ],
  "figures": [
    {
      "closed": true,
      "direction": "forward",
      "arrowStyle": "gap", "arrowWidth": 50, "arrowLength": 16,
      "cuts": [ { "at": [262,402], "conf": 0.88 }, { "at": [350,388], "conf": 0.80 } ],
      "pieces": [
        { "fill": "#7B1E2B", "text": "CURATE", "textColor": "#FFFFFF", "textAlign": "center", "layer": 0, "conf": { "fill": 0.97, "text": 0.83 } }
      ]
    }
  ],
  "items":   [ { "type": "item", "x": 300, "y": 520, "icon": "cloud", "text": "Data", "iconColor": "#7B1E2B", "textColor": "#7B1E2B", "conf": 0.74 } ],
  "circles": [ { "type": "circle", "cx": 95, "cy": 700, "r": 18, "text": "1", "stroke": "#7B1E2B", "conf": 0.95 } ],
  "titles":  [ { "type": "title", "x": 60, "y": 90, "title": "MLOps: Eternal Knot", "subtitle": "Machine Learning Lifecycle", "conf": 0.9 } ],
  "warnings": [ { "kind": "crossing", "at": [560,610], "msg": "Could not tell which ribbon is on top" } ]
}
```

**`applyTrace(result)` in the app**
1. Add `bands` as arc/Bézier shapes. If **Stop after bands** is on, stop here.
2. Call `snapIt()`. Match each new figure to the planned `figures[i]` by overlap.
3. Turn each `cuts[k].at` into a fraction with the nearest point from `pathGeo()`, then call `insertCut()`.
4. Copy each piece's fill, text, colors, alignment and layer. Set `reverse` from `direction`, and the arrow style and size.
5. Add `items`, `circles` and `titles` through the same code paths as their tools, then `normalizeDoc()` the result.
6. Call `commit()` once, select everything new, and fill the review list from the `conf` values (below 0.6 = "needs a look") and `warnings`.

The `conf` values are never saved to the JSON. The JSON schema does not change.

---

## 5. Training data: the app makes its own

A small model is only as good as its data. Labelling ribbon diagrams by hand is slow, but **the app can draw unlimited correct examples**, so most data is synthetic and every label is exact.

### 5.1 Synthetic generator (`AI/datagen/`)
1. **`random_doc.js`** builds a random diagram from templates, with the parameters randomised:
   | Template | Examples |
   |----------|----------|
   | Knot of *n* loops (2–6) | The Eternal Knot, figure-eight, trefoil-like chains |
   | Ring of segments | One ring cut into 3–8 pieces (CURATE/COLLECT/…) |
   | Serpentine / S-chain | Left-to-right swoops, timelines |
   | Spiral / nested arcs | Concentric partial rings |
   | Mixed | Several independent figures plus free arrows |

   It also randomises colors (sample palette plus random palettes), thickness, arrow style and size, the flow direction and which pieces are on top. Labels come from a word list of ~3,000 process words (TRAIN, DEPLOY, REVIEW, PLAN, …, in 1–2 words), with random case, font size, bold and text color. It adds 0–6 icons with labels, 0–8 numbered badges and an optional title and subtitle.
2. **`render.py`** (Playwright, headless Chromium) opens `App/index.html` and, for each doc, calls `FlowAPI.loadDoc()`, then `snapIt()`, `insertCut()` and so on. It saves:
   - `img.png`: the picture, drawn by the app's own renderer.
   - `doc.json`: the true diagram (the bands *before* snapping, and the final figure).
   - `masks.npz`: the training targets (§6.1), made by re-rendering the same doc in special "label styles". For example, every piece is filled with its piece ID color, text is hidden, the center line is drawn alone, each band is drawn alone for the hidden (under) parts, and arrowheads are drawn alone.
3. **`augment.py`** makes the clean renders look like real pictures:
   - **Photo of a screen**: perspective warp (up to ~25°), moiré, glare and gradient lighting, blur, noise, JPEG at quality 30–90, color shift and white balance. The sample photo is exactly this case.
   - **Slide context**: a random slide border, side text and bullet text, a header bar, and a monitor bezel. The figure is pasted onto it, so the corner step and the crop are learned too.
   - **Style variety**: 3-D shading on ribbons, drop shadows, outlines, and gradients. Real slides have these, and the app does not draw them.
   - **Damage**: partial crop at the edges (like the cut-off right side of the sample) and small occluders (a cursor or a tooltip box like "Click the arrow to change the look of the bullet.").

Targets: **20,000 training pictures** plus **1,000 validation pictures** for v1. Rendering runs at about 2–4 per second, so this takes a few hours and can run overnight. The 32 GB of RAM and the free disk space are plenty.

### 5.2 Real pairs (`AI/data/real/`)
- **Save training pair** (§2.1) turns every picture the user traces by hand, or traces with the AI and then corrects, into a labelled example: picture + corners + correct diagram JSON.
- Goal: **60–100 real pairs** from slides, screenshots and photos the user is allowed to use. Use 70 % for fine-tuning and keep **30 % as a frozen test set** that is never trained on.
- `real_to_masks.py` produces the same `masks.npz` targets from a real pair. It renders the corrected JSON and maps it back into the photo through the corner transform.

> The first real pair is `Sample-flow-diag.jpg`. Trace it by hand with today's tool, save the pair, and it becomes the acceptance test (§9).

---

## 6. The model: FlowTrace

### 6.1 Phase 1 — "FlowTrace-Seg": a small dense model plus geometry code (v1)

**Network**: a compact U-Net with a MobileNetV3-style encoder (depthwise-separable convolutions), about **1.2 M parameters**. Input is a 384×384 RGB picture, after straightening. Training uses 256×256 crops at first and finishes at 384. It has one shared encoder and decoder with several output heads, all at ¼ resolution plus one full-resolution mask:

| Head | Output | Used for |
|------|--------|----------|
| `ribbon` | ribbon / background mask (text counts as ribbon) | Ribbon area, thickness (distance transform) |
| `center` | **amodal** center-line heat map: center lines *including the hidden parts under a crossing ribbon* | Following a ribbon through a crossing |
| `tangent` | flow direction as (cos 2θ, sin 2θ) along center lines | Pairing branches at crossings and fitting curves |
| `ontop` | at crossings, which branch is visible | Piece `layer` |
| `cut` | heat map of piece boundaries (color changes, chevrons, gaps) | Cut positions |
| `arrow` | arrow-tip points + pointing direction; arrow-style class | `reverse`, `arrowStyle`, end arrows |
| `text` | mask of text pixels, split into *on-ribbon* and *free* text | Where to read labels; keep text out of the color sampling |
| `objects` | center + size heat maps for badges, icons and the title block | Circle / Item / Title |

Plus a tiny **icon classifier**: a 64×64 crop → 17 classes (16 `ICONS` + "other / uploaded picture"), ~150 k parameters, trained on icon renders in many colors and sizes.

**Loss**: Dice + BCE for the masks, focal loss for the heat maps, and cosine loss for the direction field. The `center` head weights hidden parts ×2, because crossings are the hard part.

**Post-processing** (`AI/service/build.py`, deterministic, no learning):
1. **Straighten**: the 4 corners → homography → 1600-wide straightened picture. The same transform maps results back onto the canvas.
2. **Segment**: run the network. For pictures larger than 384 px, use overlapping tiles and blend them.
3. **Center-line graph**: thin the `center` map to one-pixel lines → a graph of junctions and branches. At each crossing, pair the branches that continue each other in direction (`tangent`). This is the same idea Snap it uses to pair facing ends. A dead end inside a ribbon is a cut or an arrow, not the end of the ribbon.
4. **Chains → bands**: split each chain where its curvature changes. Fit a circle to each part (least squares). If it fits within 2 % of the radius, emit an **Arc**; otherwise fit a **cubic Bézier** (Schneider's algorithm). Thickness comes from 2 × the median distance-transform value. The **Detail** slider sets the fit tolerance (fewer bands ↔ closer fit).
5. **Pieces and colors**: sample the picture along each chain at the center line ± ¼ thickness, skipping `text` pixels. Convert to Lab color and smooth along the chain. Combine `cut` heat map peaks with sudden color changes to get cuts. Each piece's color is its median color, snapped to a palette swatch if within ΔE < 6.
6. **Labels**: for each piece, **unroll the band into a straight strip image**. Sample the picture along the center line, across the thickness, so curved text becomes a straight line. Read it with Tesseract (`--psm 7`, upper case and digits preferred), both ways round. Keep the reading with the higher confidence; that also sets **Flip text**. The position of the text inside the strip sets `textAlign` (start / center / end).
7. **Free text, badges and icons**: find text lines in the `text` (free) mask with OpenCV, read them with Tesseract, and assign them to the nearest icon (label) or to the title (top-left block, largest font = title, next line = subtitle). Badges come from the `objects` heat map, confirmed by a circle fit, and their digit is read by Tesseract. The badge outline color is sampled from the picture.
8. **Assemble** the response (§4) with a `conf` value per item, taken from the head scores and the OCR confidence.

### 6.2 Why this split works on a slow CPU
- The network is small because it only has to answer local questions ("is this pixel a ribbon center? which way does it run?"). The global structure comes from geometry code that is exact and needs no training.
- Exact labels for every head come free from the app's renderer, including the parts hidden under crossings, which no person could label.
- Every step can be inspected. When a trace is wrong, the debug view (§8) shows which stage failed, so we know whether to fix data, the model, or the code.

### 6.3 Phase 2 — "FlowTrace-Seq": the small language model (v1.1+)

Once Phase 1 works and real pairs exist, train a true **image → diagram-program** model. It is an encoder (a small CNN or ViT-tiny on the same 384 px input) plus a **6-layer transformer decoder** (~15–30 M parameters) that writes the diagram in a compact token language:

```
<fig closed dir=fwd style=gap>
  ARC 300 520 140 200 470 T46
  BEZ 430 470 520 420 600 520 690 470 T44
  CUT 262 402 | PIECE #7B1E2B "CURATE" C L0
  CUT 350 388 | PIECE #7B1E2B "COLLECT" S L0
</fig>
<item cloud 300 520 #7B1E2B "Data">
<circ 95 700 18 "1" #7B1E2B>
<title 60 90 "MLOps: Eternal Knot" "Machine Learning Lifecycle">
```

- Coordinates are quantised to a 256-step grid (one token each), colors to a 64-color palette plus a refine step, and text is written in characters.
- It trains on the same synthetic and real pairs, since `doc.json` turns straight into this token language.
- **Training needs a GPU.** Plan a free Kaggle or Colab GPU: roughly 6–12 GPU hours for 200 k synthetic pictures. On this CPU the same run would take weeks. A reduced version (~8 M parameters, 256 px) could train locally in about 4–7 days.
- **Inference runs locally**: an ONNX export with greedy decoding. The target is under 15 s on this CPU.
- **Use**: the decoder suggests the *structure* (how many loops, which bands join, the order of pieces). Phase 1's post-processing then *refines* the geometry and colors against the actual pixels. Both outputs are scored on the test set (§9), and the service keeps whichever wins, per stage.

### 6.4 Not chosen, and why
| Option | Why not (now) |
|--------|---------------|
| Fine-tune a big open VLM (Qwen-VL, LLaVA, Florence-2) | Cannot fine-tune without a GPU. Too slow to run on this CPU. It would also still have to learn the app's JSON. |
| Prompt a local VLM through Ollama | Good for a quick test, but vague about geometry (it can say "four loops" but not exact arcs) and slow here. **It can still be a Phase 0 baseline**: ask it to list the labels and colors and compare with ours. |
| Classical computer vision only, no learning | Works on clean flat slides, but breaks on photos, 3-D shading, crossings and text that sits on the ribbon. The learned heads are exactly what fixes those cases. |
| Our own OCR | Tesseract on straightened strips is good enough; see §0. |

---

## 7. Where it runs

### 7.1 v1: local service (recommended)
- `AI/service/flowtrace_server.py`: a Python HTTP server on `127.0.0.1:8765`, using only the standard library plus the packages that are already installed. It loads the ONNX models once and runs onnxruntime + OpenCV + Tesseract.
- Endpoints: `GET /health` (version, model name), `POST /corners` (suggested slide corners), `POST /trace` (§4), `POST /debug` (all intermediate maps as PNGs, for §8).
- Started with `AI\start-flowtrace.bat`, which runs a minimised window. It listens on localhost only. It sends CORS headers so that the app, opened from a `file://` page, can call it.
- **Why a service first**: OpenCV, Tesseract and onnxruntime are already installed in Python, and the app stays a single file with no new dependencies.

### 7.2 v1.1: fully in the browser (optional)
- `App/vendor/ort.min.js` (onnxruntime-web) and `App/vendor/flowtrace-model.js` are loaded with `<script>` tags, which work from `file://`. The model is stored as base64 inside the `.js` file, about 2 MB for FlowTrace-Seg. The geometry code is ported to JavaScript; a lot of it already exists (`resample`, `pathGeo`, the Bézier helpers).
- Labels need `tesseract.js` (vendored, ~10 MB with the English data). Alternatively, keep only the label step in the service.
- Decide after v1 based on speed: WebAssembly on this CPU may be slower than native onnxruntime.

---

## 8. Folder layout

```
FLOW-CHART-APP/
├── App/index.html                 # + FlowAPI, AI trace UI, Save training pair (§2.1)
├── AI/
│   ├── start-flowtrace.bat        # starts the local service
│   ├── requirements.txt           # pinned versions of what is already installed
│   ├── datagen/
│   │   ├── random_doc.js          # random diagram templates (§5.1)
│   │   ├── words.txt              # label vocabulary
│   │   ├── render.py              # Playwright: doc → img.png, doc.json, masks.npz
│   │   ├── augment.py             # photo / slide / damage effects
│   │   └── real_to_masks.py       # real pairs → same targets
│   ├── train/
│   │   ├── dataset.py
│   │   ├── model_seg.py           # FlowTrace-Seg (§6.1)
│   │   ├── model_icon.py
│   │   ├── train_seg.py           # CPU, resumable, checkpoints every 30 min
│   │   ├── export_onnx.py
│   │   └── seq/                   # Phase 2 decoder + tokenizer + Kaggle/Colab notebook
│   ├── service/
│   │   ├── flowtrace_server.py
│   │   ├── straighten.py  segment.py  graph.py  fit.py  colors.py  labels.py  objects.py
│   │   └── build.py               # assembles the §4 response
│   ├── eval/
│   │   ├── score.py               # metrics in §9 for a folder of pairs
│   │   └── report.html            # side-by-side: picture | trace | truth, per test image
│   ├── models/                    # *.onnx + model-card.md (data, date, scores)
│   └── data/                      # synthetic/, real/train, real/test  (not shared)
└── INFO/FLOW-CHART-AI-PLAN.md
```

**Debug view**: with `?debug` in the app URL, **Trace it** also shows the intermediate layers as switchable overlays on the canvas: straightened picture, ribbon mask, center lines, crossings, cuts, label strips with their OCR text. This is how failures get diagnosed.

---

## 9. How we measure "good"

Scored by `AI/eval/score.py` on the **real test set** (never trained on), plus 500 synthetic test pictures:

| Metric | What it checks | v1 target (real test) |
|--------|----------------|-----------------------|
| Ribbon IoU | Overlap of traced and true ribbon area | ≥ 0.85 |
| Center-line distance | Average distance between traced and true center lines, as a fraction of thickness | ≤ 0.25 |
| Topology | Number of figures and loops correct; each crossing paired correctly | ≥ 90 % of crossings |
| Cuts | Piece boundaries found within ½ thickness (precision / recall) | ≥ 0.85 / 0.85 |
| Piece color | ΔE (Lab) between traced and true fill | median ≤ 8 |
| Labels | Character error rate on piece labels | ≤ 10 % |
| Over / under | Correct piece on top at crossings | ≥ 80 % |
| Badges / icons / title | Found and correct | ≥ 90 % / 70 % / 90 % |
| **Fix-up effort** | Number of edits needed to turn the trace into the truth (count of shapes moved, added or deleted, and fields retyped) | Sample photo: **≤ 15 edits** (by hand it takes ~100+ actions) |
| Speed | Total time on this computer, for a 1600×1200 photo | ≤ 20 s |

**Acceptance test**: tracing `Sample-flow-diag.jpg` gives 4 loops in one figure crossing correctly, about 16 pieces in the right colors, at least 12 of the 16 labels right, the 4 badges, 4 icons with "Data / ML / Dev / Ops", and the title with its subtitle. All of it is editable and saves and exports like a hand-drawn diagram.

---

## 10. Implementation milestones

| Milestone | Deliverable | Done when… |
|-----------|-------------|-----------|
| **P0 – Spike & benchmark** | Measure the CPU's training speed for a 1.2 M-parameter U-Net at 256 and 384 px. Hand-trace the sample photo and save it as the first real pair. Optional Ollama baseline. | We know the hours per epoch and have chosen the final input size and dataset size. |
| **P1 – FlowAPI** | `window.FlowAPI` in `index.html`; nothing in the UI changes | Playwright can load a doc, snap, cut and export a PNG headlessly. |
| **P2 – Data generator** | `random_doc.js`, `render.py`, `augment.py` with all mask targets | 20 k + 1 k pictures generated. Overlays of the masks on the pictures look right on 50 random samples. |
| **P3 – Train FlowTrace-Seg** | `model_seg.py`, `train_seg.py` (resumable), icon classifier, ONNX export | On synthetic validation: ribbon IoU ≥ 0.9, center-line ≤ 0.2 × thickness. |
| **P4 – Geometry builder** | `graph.py`, `fit.py`, `colors.py` → bands + cuts + pieces | Perfect synthetic masks → exact diagram. Model masks → topology correct on ≥ 90 % of synthetic validation. |
| **P5 – Labels & objects** | Strip unrolling + Tesseract, badges, icons, title | Label character error ≤ 10 % on synthetic validation. |
| **P6 – Service** | `flowtrace_server.py`, `start-flowtrace.bat`, `/debug` | `curl` with the sample photo returns a valid §4 response in ≤ 20 s. |
| **P7 – App UI** | Corner handles, Trace it, progress/cancel, `applyTrace()`, review list, debug overlays, Save training pair | Tracing the sample in the app gives an editable diagram as one undo step. |
| **P8 – Real data loop** | 60–100 real pairs, fine-tuning, evaluation report | §9 targets met on the real test set; the acceptance test passes. |
| **P9 – Guide** | New section in `guide.html` ("AI Trace") + a model card | A user can install, start and use it from the guide alone. |
| **P10 – Phase 2 (optional)** | FlowTrace-Seq on a cloud GPU, merged with Phase 1 | Beats Phase 1 on topology or fix-up effort on the real test set. |
| **P11 – In-browser (optional)** | §7.2 | Trace works with the service stopped. |

Rough effort: P0–P2 about 1 week, P3 about 1 week (mostly unattended CPU training), P4–P7 about 2–3 weeks, P8 ongoing.

---

## 11. Risks

| Risk | Mitigation |
|------|-----------|
| CPU training is too slow | P0 measures it first. Fall back to 256 px, fewer heads, or train P3 on a free cloud GPU too (the model is the same, so only the training location changes). |
| Synthetic pictures don't look like real slides | Heavy augmentation (§5.1). Fine-tune on real pairs. Track the real-test score, not only the synthetic one. |
| Crossings and over/under are wrong | Amodal center-line and `ontop` heads with extra loss weight. The review list flags uncertain crossings, and one click on **Bring piece to front** fixes it. |
| Curved labels are misread | Unroll strips before OCR. Prefer words from `words.txt` when confidence is low. Everything stays editable in the panel. |
| Photo is badly skewed or cropped | The corner step is manual-override. Parts of ribbons cut off at the edge become open ends, not invented loops. |
| Output breaks the app | All output goes through `normalizeDoc()`. The whole trace is one undo step. |
| Picture rights for real training data | Use only the user's own slides and pictures the user is allowed to use. `AI/data/` is never shared or committed. |

---

## 12. Decisions (resolved 2026-10-18)

| # | Question | Decision |
|---|----------|----------|
| 1 | Is a free cloud GPU acceptable for **training only** (Phase 2, possibly P3)? | **Yes, for synthetic data only.** Kaggle or Colab trains on generated diagrams. Real pairs never leave this computer, and fine-tuning on them runs locally. Tracing always runs locally. |
| 2 | Service (§7.1) or in-browser (§7.2) for v1? | **Local Python service.** The in-browser version (§7.2) stays an optional v1.1 idea, not a commitment. |
| 3 | Should traced output arrive as raw bands (editable curves) or already snapped and cut? | **Snapped and cut by default**, with a **Stop after bands** checkbox. |
| 4 | Label languages | **English only in v1**: upper/lower case letters and digits, with the English `words.txt` used for corrections. |
| 5 | Should corrected traces be saved as training data automatically? | **No. Only when the user clicks Save training pair.** The app never prompts and never stores pairs on its own. |

---

## 13. Implementation notes (2026-10-18)

### What was built
| Milestone | Where | Result |
|-----------|-------|--------|
| P0 benchmark | `AI/train/train_seg.py --benchmark` | **~0.4 s per training picture** at 256 px on this CPU, after the model was changed to start at half resolution (it was 1.3 s before that). 20 k pictures × 10 epochs ≈ 19–23 h, over 2–3 nights. |
| P1 FlowAPI | `App/index.html`, "Automation API" | `window.FlowAPI`: `loadDoc`, `getDoc`, `snapIt`, `insertCut`, `pathGeo`, `setTrace`, `applyTrace`, `renderPNG(scale, area)`. |
| P2 data generator | `AI/datagen/random_doc.js`, `render.py`, `apppage.py`, `real_to_masks.py` | Knots, rings and serpentines are built with the app's own Snap it and cuts. Masks: ribbon / ribbon labels / **amodal** center lines. The photo look (perspective, light, moiré, blur, noise, JPEG, slide clutter) is added on the fly in `AI/train/dataset.py` instead of a separate `augment.py`. |
| P3 training | `AI/train/model_seg.py`, `train_seg.py`, `export_onnx.py` | Resumable CPU training and ONNX export. Export refuses to publish below a validation score of 0.6. **Not yet run for real** (needs the overnight runs). |
| P4–P5 geometry, labels, objects | `AI/service/*.py` | Classic vision stands in for the network until a model is published. On the sample photo: the whole knot is one closed figure with correct over/under at all 3 crossings, ~21 pieces in the right colors, 12–16 of the 16 labels, 4 icons + names, 4 badges, title + subtitle. |
| P6 service | `AI/service/flowtrace_server.py`, `AI/start-flowtrace.bat` | `/health`, `/corners`, `/trace`. Takes **~8–12 s** for the sample. |
| P7 app UI | `App/index.html`, "AI trace" | Trace it button, corner handles, add/replace dialog, progress + cancel, one undo step, the "Needs a look" list with orange marks, Stop after bands, Detail, Save training pair, `?debug` overlay. |
| P8 real data loop | `AI/eval/score.py` | The tooling works. **Real pairs still need to be collected**, starting with the sample photo, hand-corrected. |
| P9 guide | `guide.html` §11 "Let the AI trace it", shortcuts, limits; `AI/README.md` | Done. |

### Changes from the plan
- **Coordinates (§4):** the service returns coordinates in **pixels of the straightened picture** and includes that picture (`image.src`, `width`, `height`). The app maps them onto the canvas and swaps the faded trace picture for the straightened one, so the result lines up with it. On **Replace drawing**, the canvas height follows the picture's shape.
- **Figures (§4 applyTrace):** the figure is built **directly from the traced center line** (resampled like Snap it does), not by snapping bands. That is exact, and avoids Snap it pairing wrong ends at crossings. The bands are still produced and used for **Stop after bands**, after which the user clicks Snap it.
- **Cuts and pieces** are matched by position: each planned cut point → `nearestOnPath` → `insertCut`. Each planned piece's middle point → the app piece that contains it. One figure's label sizes are kept within ±20 % of their median, to stop a misread from producing a giant label.
- **Shortcut:** `Shift+T`, not `Ctrl+Shift+T`, because browsers reserve the latter (reopen tab).
- **Model:** about 0.25 M parameters (not 1.2 M), starting at half resolution, with 3 heads (ribbon, text, center). The tangent, on-top, cut, arrow and object heads of §6.1 are left for later. Classic vision handles those for now.
- **Safety:** if a published model's ribbon area or thickness disagrees wildly with classic vision on a picture, the service uses classic vision for that picture.
- **Label reading:** each label strip is read 4 ways (both directions × Otsu/soft) in one Tesseract call, and snapped to `words.txt` (exact, close, or the start of a word hidden under a crossing). Low-confidence labels go to the review list.
- **Remembered answers** (added 2026-10-18): `AI/service/memory.py` matches the straightened picture against every saved pair (ORB features + RANSAC homography, at least 40 agreeing features) and returns the corrected diagram moved onto the new picture. The panel option **Use remembered answers** is on by default; `score.py` tests without memory unless `--recall` is given. This is memory, not learning: it helps only the same picture or a re-photo of the same slide.
- The service now refuses to start a second copy on the same port. On Windows both copies could otherwise listen, and the old one would answer.
- Added the Python package **`onnx`** (needed by PyTorch's ONNX export).

### Next steps
1. Correct the traced sample photo (TRAIN, the green EVALUATE piece, BUILD, MONITOR…) and **Save training pair**. That makes the first real test pair and the acceptance test of §9.
2. `python render.py` (synthetic data), then 2–3 nights of `python train_seg.py --hours 8`, then `export_onnx.py`.
3. Collect 60–100 real pairs, fine-tune, and track `score.py`.
4. Phase 2 (FlowTrace-Seq) once the real test set exists.
