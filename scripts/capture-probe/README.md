# capture-probe

Verifies whether gimme's content-protected overlays are truly **invisible** (not a
**black rectangle**) under a browser screen-share — i.e. Google Meet, which uses
Chrome's `getDisplayMedia`. This probe drives that exact API, grabs **one** frame
with no live preview (so there's no infinity-mirror confound), and saves it for
inspection.

## Why this exists

`SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)` (what tao/Tauri's
`content_protected` sets) makes a window **absent** from capture on compliant
capturers — but `hide()`/`show()` on Windows degrades it to `WDA_MONITOR`, which
renders the window as a solid **black box** in the share (tauri#14189). This probe
is the regression test for that: an overlay must stay invisible across hide/show
cycles and after every screenshot.

## Usage

```bash
python3 scripts/capture-probe/server.py      # serves http://localhost:8137
```

1. Open <http://localhost:8137> in the browser under test (Chrome = Meet's path).
   From Windows this reaches a WSL-side server via `localhost` (WSL2 forwarding).
2. Launch gimme, place the overlays (main + pop-out) **over the colored stripes**.
3. Click **📸 Capture screen** → pick **"Entire Screen"** → Share.
4. Read `scripts/capture-probe/shot_latest.png`.

- Stripes show through where an overlay sat → **invisible** ✅
- Black rectangle over the stripes → **black** ❌ (protection degraded)

Use a still, known-black window as a "ruler" to disambiguate *invisible* from
*not-present*. Never judge from Meet's own share preview — sharing the whole
screen with the meeting window visible creates a recursive black rectangle that
is NOT the overlay.

`shot_*.png` are gitignored.
