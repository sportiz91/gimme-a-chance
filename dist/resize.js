// Manual edge-resize for the undecorated overlays (main, answer, manager).
//
// Native undecorated resizing is disabled on purpose (`resizable: false`):
// Tauri implements it with a hidden hit-test child window whose HTLEFT/…
// answers make Windows paint the resize arrows on hover, and the OS modal
// sizing loop pins those arrows for the whole drag. Screen-share viewers see
// the system cursor, so a cursor that mutates over an "empty" patch of screen
// betrays the invisible overlay — same class of leak as the capture black box.
//
// So resizing is reimplemented: invisible strips along the window edges grab
// the pointer and stream ticks to Rust (`begin_resize` / `resize_tick` /
// `end_resize`), which reads the cursor and does all geometry math in
// physical pixels (no devicePixelRatio here) and moves the window itself.
// No OS sizing loop ever runs, so the cursor stays whatever the page CSS
// says — a plain arrow.
(() => {
  'use strict';
  const { invoke } = window.__TAURI__.core;

  const EDGE = 6; // logical px per side strip (native was ~8 physical px)
  const CORNER = 14; // logical px per corner square

  const style = document.createElement('style');
  style.textContent = [
    // Max z-index so content never covers a handle; cursor stays the arrow.
    '.rz{position:fixed;z-index:2147483647;cursor:default;}',
    `.rz-n{top:0;left:${CORNER}px;right:${CORNER}px;height:${EDGE}px;}`,
    `.rz-s{bottom:0;left:${CORNER}px;right:${CORNER}px;height:${EDGE}px;}`,
    `.rz-w{left:0;top:${CORNER}px;bottom:${CORNER}px;width:${EDGE}px;}`,
    `.rz-e{right:0;top:${CORNER}px;bottom:${CORNER}px;width:${EDGE}px;}`,
    `.rz-nw{top:0;left:0;width:${CORNER}px;height:${CORNER}px;}`,
    `.rz-ne{top:0;right:0;width:${CORNER}px;height:${CORNER}px;}`,
    `.rz-sw{bottom:0;left:0;width:${CORNER}px;height:${CORNER}px;}`,
    `.rz-se{bottom:0;right:0;width:${CORNER}px;height:${CORNER}px;}`,
  ].join('\n');

  // One drag at a time (one mouse); self-pace ticks to the IPC round-trip so
  // a high-polling mouse can't flood the channel.
  let inflight = false;

  function makeHandle(edge) {
    const el = document.createElement('div');
    el.className = `rz rz-${edge}`;
    el.addEventListener('pointerdown', (e) => {
      if (e.button !== 0) return;
      e.preventDefault();
      // Capture keeps the drag alive when the cursor outruns the window.
      el.setPointerCapture(e.pointerId);
      invoke('begin_resize', { edge });
    });
    el.addEventListener('pointermove', async (e) => {
      if (!el.hasPointerCapture(e.pointerId) || inflight) return;
      inflight = true;
      try {
        await invoke('resize_tick');
      } finally {
        inflight = false;
      }
    });
    const end = (e) => {
      if (!el.hasPointerCapture(e.pointerId)) return;
      el.releasePointerCapture(e.pointerId);
      invoke('end_resize');
    };
    el.addEventListener('pointerup', end);
    // Covers rug-pulls like Ctrl+Shift+H parking the window mid-drag.
    el.addEventListener('pointercancel', end);
    return el;
  }

  const mount = () => {
    document.head.appendChild(style);
    for (const edge of ['n', 's', 'e', 'w', 'ne', 'nw', 'se', 'sw']) {
      document.body.appendChild(makeHandle(edge));
    }
  };
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', mount);
  } else {
    mount();
  }
})();
