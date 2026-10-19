# FlowTrace — the AI behind "Trace it"

FlowTrace reads a picture of a ribbon flow chart and returns a plan that the app builds as normal, editable shapes. It runs on this computer only. Design: [../INFO/FLOW-CHART-AI-PLAN.md](../INFO/FLOW-CHART-AI-PLAN.md). User steps: guide.html §11, "Let the AI trace it".

## 1. Use it

1. Double-click **`start-flowtrace.bat`**, and leave the window open.
2. Open `App\index.html`. Click **Trace image** and choose a picture.
3. Click **Trace it** (or press `Shift+T`), drag the 4 orange corners onto the diagram, then press **Enter**.

It needs Python 3.12 with `requirements.txt`, and **Tesseract OCR** for reading text. Tesseract is found automatically in `C:\Program Files\Tesseract-OCR`.

## 2. What is inside

| Folder | What |
|--------|------|
| `service/` | The local web service (`flowtrace_server.py`, port 8765) and the tracing pipeline. The steps are: `straighten` (corners → flat picture) → `segment` (ribbon / label pixels) → `graph` (center lines through crossings) → `colors` (cuts, piece colors, flow direction, over/under) → `labels` (Tesseract on unrolled label strips, free text, badge numbers) → `objects` (icons, badges, title) → `fit` (Arc/Bézier bands) → `pipeline` (assembles the response). |
| `datagen/` | Synthetic training data made **by the app itself**. `random_doc.js` builds random knots, rings and serpentines with the app's own Snap it and cuts. `render.py` renders them, together with exact masks, in headless Chromium. `real_to_masks.py` turns your saved training pairs into masks. `words.txt` holds the label vocabulary, which is also used to correct misread labels. |
| `train/` | **FlowTrace-Seg**, a 0.25 M-parameter U-Net (`model_seg.py`) with the outputs ribbon / label / center line. Also the training script (`train_seg.py`, resumable, CPU) and `export_onnx.py`. |
| `eval/` | `score.py`, which scores traces on the frozen real test set and writes `report.html`. |
| `models/` | `flowtrace-seg.onnx`, created by export. While it is missing, the service uses **classic vision** (hand-written image processing). |
| `data/` | `synthetic/`, `real/` (your `pair-*.zip`, `train_masks/`, `test/`). Created as you go; keep it private. |

## 3. Train your own model

Measured on this computer (Pentium N5030, CPU only), 2026-10-18: **~0.4 s per training picture** at 256 px. So 20,000 pictures × 10 epochs ≈ **19–23 hours**, run over 2–3 nights. Training resumes where it stopped.

```bat
cd AI\datagen
python -m playwright install chromium        &rem once
python render.py --count 20000 --val 1000      &rem synthetic data, ~2-3 hours

cd ..\train
python train_seg.py --benchmark                &rem optional: re-measure the speed
python train_seg.py --hours 8                  &rem one night; run again to continue
python export_onnx.py                          &rem publishes models\flowtrace-seg.onnx (needs val score >= 0.6)
```

Restart `start-flowtrace.bat` afterwards. The panel then says "trained model". If the model's output looks implausible on a picture (very different ribbon area or thickness from classic vision), the service falls back to classic vision for that picture and prints why.

**Real pictures teach it the most.** Trace a picture (by hand or with Trace it), correct the result, and click **Save training pair**. Put the zips in `AI\data\real\`. Then:

```bat
cd AI\datagen && python real_to_masks.py       &rem ~70 % train, ~30 % frozen test (fixed by file name)
cd ..\train  && python train_seg.py --hours 2  &rem fine-tune; real pairs count 20x
cd ..\eval   && python score.py                &rem scores + report.html
```

**Already have the correct diagram as JSON?** Make answer keys without retyping anything:

```bat
cd AI\datagen
rem a photo + its AI trace (Save JSON) + the correct diagram -> answer key for the photo
python answer_pair.py --picture ..\..\Sample-flow-diag.jpg --traced ..\..\JSON-FILES\flowchart-ai-generated-20261018-2135.json --answer ..\..\JSON-FILES\sample-json.json --out ..\data\real\test\pair-sample-photo.zip
rem a correct diagram alone -> the app draws the picture; the diagram is the answer
python render_pair.py ..\..\JSON-FILES\sample-json.json ..\data\real\test\pair-sample-render.zip
```

`answer_pair.py` keeps the trace's geometry (it fits the photo), takes labels, pieces, colors, icons, badges and title from the answer, and lines the two up using the icon names they share (at least 3). Zips in `data\real\test\` are the frozen test set; put them in `data\real\` instead to let `real_to_masks.py` use them for training.

**Remembered answers.** Every pair zip in `data\real\` (and `data\real\test\`) is also remembered. When you trace the same picture again, even with other corners or a re-photo of the same slide, FlowTrace recognizes it (`service\memory.py`, feature matching) and returns your corrected diagram exactly. The panel's **Use remembered answers** box turns this off. `score.py` tests the AI *without* memory unless you add `--recall`.

**Faster training (optional):** the plan allows a free cloud GPU (Kaggle or Colab) for *synthetic* data only. Copy `AI\train` and `AI\data\synthetic` there, then run `python train_seg.py --threads 2 --batch 32` and bring back `runs\seg-v1\best.pt`. Fine-tune on real pairs here.

## 4. Status (2026-10-18)

- Trace it works end to end on `Sample-flow-diag.jpg` with **classic vision**, in about 8–12 s. It finds 1 closed figure (the whole knot, with correct over/under at its 3 crossings), about 21 pieces in the right colors, about 12–16 labels, 4 icons with names, 4 badges, and the title with its subtitle.
- No trained model yet: training needs the overnight runs above.
- Not built yet: Phase 2 (FlowTrace-Seq decoder), the in-browser version, and a hand-checked real test set.
