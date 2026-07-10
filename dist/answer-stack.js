// The Answer stack — the list of answers rendered identically by the main
// window's Answer box and by the pop-out overlay (answer.html).
//
// Newest answer on top. A press prepends a card and scrolls to it, so firing a
// new answer never destroys the one you were still reading. Each card is a
// native <details>: collapse/expand costs no JS, is keyboard-accessible, and
// its state lives in the `open` attribute. Collapse state is per-window on
// purpose — the overlay mirrors CONTENT, not the reader's local view.
//
// smd's markdown parser is stateful and append-only, so it cannot be shared
// across cards: a streaming card owns its own parser, created on its first
// delta and thrown away when the authoritative full text arrives.
//
// Loaded as a classic script (not a module): both windows' inline scripts must
// see `window.createAnswerStack` at parse time. `window.smd` / `window.hljs`
// are only touched inside the methods, which run long after those load.

(function () {
  'use strict';

  const PREVIEW_CHARS = 90;

  function fmtTime(ts) {
    return new Date(ts || Date.now()).toTimeString().slice(0, 8);
  }

  // First non-empty line, stripped of the markdown punctuation that reads as
  // garbage on a one-line collapsed preview.
  function previewOf(raw) {
    const line = raw
      .split('\n')
      .map((s) => s.replace(/[#>*`_]/g, '').trim())
      .find(Boolean) || '';
    return line.length > PREVIEW_CHARS ? line.slice(0, PREVIEW_CHARS) + '…' : line;
  }

  // An Ask card shows the question you typed; an agent press has none by
  // design (the backend hands the model the whole transcript, no extraction).
  function labelOf(meta) {
    if (meta && meta.kind === 'ask') return '💬 ' + (meta.label || 'Ask');
    return '🤖 Agent';
  }

  function createAnswerStack(root) {
    root.classList.add('answer-stack');
    const cards = new Map();
    const order = []; // ids, newest first — mirrors DOM order

    function build(id, meta) {
      const el = document.createElement('details');
      el.className = 'answer-card loading';
      el.open = true;

      const head = document.createElement('summary');
      head.className = 'answer-head';

      const chev = document.createElement('span');
      chev.className = 'answer-chev';
      chev.setAttribute('aria-hidden', 'true');

      const time = document.createElement('span');
      time.className = 'answer-time';
      time.textContent = fmtTime(meta && meta.ts);

      const label = document.createElement('span');
      label.className = 'answer-label';
      label.textContent = labelOf(meta);

      const preview = document.createElement('span');
      preview.className = 'answer-preview';

      head.append(chev, time, label, preview);

      const body = document.createElement('div');
      body.className = 'answer-body';

      el.append(head, body);
      return {
        id,
        meta: meta || null,
        el,
        body,
        preview,
        raw: '',
        parser: null,
        state: 'loading',
        previewDone: false,
      };
    }

    // The preview only matters while the card can still be collapsed to one
    // line, so stop recomputing it once it fills or the answer settles.
    function refreshPreview(c) {
      if (c.previewDone) return;
      const p = previewOf(c.raw);
      c.preview.textContent = p;
      if (p.length >= PREVIEW_CHARS || c.state === 'done' || c.state === 'error') {
        c.previewDone = true;
      }
    }

    function renderFull(c, text) {
      c.body.innerHTML = '';
      if (window.smd) {
        const p = window.smd.parser(window.smd.default_renderer(c.body));
        window.smd.parser_write(p, text);
        window.smd.parser_end(p);
      } else {
        c.body.textContent = text;
      }
      if (window.hljs) {
        c.body.querySelectorAll('pre code').forEach((b) => window.hljs.highlightElement(b));
      }
    }

    return {
      // A new answer: a loading card at the top of the stack.
      begin(id, meta) {
        if (cards.has(id)) return false;
        const c = build(id, meta);
        cards.set(id, c);
        order.unshift(id);
        root.insertBefore(c.el, root.firstChild);
        root.scrollTop = 0;
        return true;
      },

      // Route a streamed token to its card. An unknown or already-settled id is
      // a stale delta (a press that finished, or one from before a clear) — drop it.
      delta(id, text) {
        const c = cards.get(id);
        if (!c || c.state === 'done' || c.state === 'error') return false;
        c.state = 'streaming';
        c.el.classList.remove('loading');
        c.raw += text;
        if (window.smd) {
          if (!c.parser) c.parser = window.smd.parser(window.smd.default_renderer(c.body));
          window.smd.parser_write(c.parser, text);
        } else {
          c.body.textContent = c.raw;
        }
        refreshPreview(c);
        return true;
      },

      // The invoke's full answer is authoritative (deltas can be stale or
      // dropped): re-render the card whole, then colorize its code blocks —
      // highlighting mid-stream would fight smd's live text nodes.
      final(id, fullText) {
        const c = cards.get(id);
        if (!c) return false;
        c.state = 'done';
        c.parser = null;
        c.raw = fullText;
        c.previewDone = false;
        c.el.classList.remove('loading');
        renderFull(c, fullText);
        refreshPreview(c);
        return true;
      },

      // Failures keep their card: you want to see which press died and why.
      error(id, message) {
        const c = cards.get(id);
        if (!c) return false;
        c.state = 'error';
        c.parser = null;
        c.raw = message;
        c.previewDone = false;
        c.el.classList.remove('loading');
        c.el.classList.add('error');
        c.body.innerHTML = '';
        c.body.textContent = message;
        refreshPreview(c);
        return true;
      },

      clear() {
        cards.clear();
        order.length = 0;
        root.innerHTML = '';
      },

      // Newest first. Feeds the overlay's catch-up replay.
      snapshot() {
        return order.map((id) => {
          const c = cards.get(id);
          return { id: c.id, meta: c.meta, raw: c.raw, state: c.state };
        });
      },
    };
  }

  window.createAnswerStack = createAnswerStack;
})();
