/* live2 dashboard -- vanilla JS, view-only. */
'use strict';

const $ = (sel) => document.querySelector(sel);

/* ---------------------------------------------------------------- tabs */
document.querySelectorAll('nav button').forEach((b) => {
  b.addEventListener('click', () => {
    document.querySelectorAll('nav button').forEach((x) => x.classList.remove('active'));
    document.querySelectorAll('.tab').forEach((x) => x.classList.remove('active'));
    b.classList.add('active');
    $('#tab-' + b.dataset.tab).classList.add('active');
    if (b.dataset.tab === 'opponents') loadOpponents();
    if (b.dataset.tab === 'hands') loadHands();
    if (b.dataset.tab === 'flags') loadFlags();
  });
});

/* ---------------------------------------------------------------- cards */
const SUIT = { s: '♠', h: '♥', d: '♦', c: '♣' };
function cardEl(c) {
  const el = document.createElement('div');
  if (!c || c.length < 2) { el.className = 'card back'; el.textContent = '?'; return el; }
  const suit = c[c.length - 1].toLowerCase();
  el.className = 'card' + ('hd'.includes(suit) ? ' red' : '');
  el.innerHTML = `<span>${c.slice(0, -1)}</span><span>${SUIT[suit] || suit}</span>`;
  return el;
}
function renderCards(el, cards, min) {
  el.innerHTML = '';
  (cards || []).forEach((c) => el.appendChild(cardEl(c)));
  for (let i = (cards || []).length; i < (min || 0); i++) el.appendChild(cardEl(null));
}

/* ---------------------------------------------------------------- live feed */
let lastMsgTime = 0;

function connectWS() {
  const ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onmessage = (ev) => {
    lastMsgTime = Date.now();
    renderLive(JSON.parse(ev.data));
    loadShadow();
  };
  ws.onclose = () => setTimeout(connectWS, 2000);
}
connectWS();

/* ---------------------------------------------------------------- pilot control */
async function pilotStatus() {
  let st;
  try { st = await (await fetch('/api/pilot/status')).json(); } catch { return; }
  const chip = $('#pilot-chip');
  const running = !!st.running;
  chip.className = 'pilot-chip ' + (running ? (st.mode === 'auto' ? 'auto' : 'rec') : 'off');
  chip.textContent = running
    ? `pilot ${st.mode === 'auto' ? 'AUTO' : 'recommending'} · pid ${st.pid}`
    : 'pilot off';
  $('#pilot-start').classList.toggle('hidden', running);
  $('#pilot-auto').classList.toggle('hidden', running);
  $('#pilot-stop').classList.toggle('hidden', !running);
  const panel = $('#pilot-panel');
  const hasLog = (st.log || []).length > 0;
  panel.classList.toggle('hidden', !running && !hasLog && $('#probe-out').classList.contains('hidden'));
  $('#pilot-mode').textContent = running ? `${st.mode} · started ${st.started}` : 'stopped';
  if (hasLog) {
    const log = $('#pilot-log');
    const atEnd = log.scrollTop + log.clientHeight >= log.scrollHeight - 8;
    log.textContent = st.log.join('');
    if (atEnd) log.scrollTop = log.scrollHeight;
  }
}
async function pilotPost(path, body) {
  try {
    const r = await (await fetch(path, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body || {}),
    })).json();
    if (r.error) alert(r.error);
    return r;
  } catch (e) { alert('pilot control failed: ' + e); }
}
$('#pilot-start').addEventListener('click', async () => {
  await pilotPost('/api/pilot/start', { mode: 'recommend' });
  pilotStatus();
});
$('#pilot-auto').addEventListener('click', async () => {
  if (!confirm('Start pilot in AUTO mode? It will click the real table.\n' +
               'Abort at any time: slam the cursor into a screen corner, or press Stop.')) return;
  await pilotPost('/api/pilot/start', { mode: 'auto' });
  pilotStatus();
});
$('#pilot-stop').addEventListener('click', async () => {
  await pilotPost('/api/pilot/stop');
  pilotStatus();
});
$('#pilot-probe').addEventListener('click', async () => {
  const btn = $('#pilot-probe');
  btn.disabled = true; btn.textContent = 'Probing…';
  const r = await pilotPost('/api/pilot/probe');
  btn.disabled = false; btn.textContent = 'Probe';
  if (!r) return;
  $('#pilot-panel').classList.remove('hidden');
  $('#probe-out').classList.remove('hidden');
  $('#probe-text').textContent = (r.output || []).join('\n');
  const img = $('#probe-img');
  img.classList.toggle('hidden', !r.png);
  if (r.png) img.src = '/api/pilot/probe.png?t=' + Date.now();
});
setInterval(pilotStatus, 3000);
pilotStatus();

/* ---------------------------------------------------------------- flag turn (old F12) */
async function flagTurn() {
  const btn = $('#flag-btn');
  const r = await pilotPost('/api/flag');
  if (!r || !r.ok) return;
  btn.classList.add('flagged');
  btn.innerHTML = `&#9873; Flagged turn ${r.turn}`;
  setTimeout(() => {
    btn.classList.remove('flagged');
    btn.innerHTML = '&#9873; Flag turn';
  }, 2500);
}
$('#flag-btn').addEventListener('click', flagTurn);
document.addEventListener('keydown', (e) => {
  // 'F' anywhere on the Live tab (not while typing) -- browser owns F12, so plain F it is
  if ((e.key === 'f' || e.key === 'F') && !e.ctrlKey && !e.altKey && !e.metaKey
      && !/INPUT|TEXTAREA|SELECT/.test(document.activeElement?.tagName || '')
      && $('#tab-live').classList.contains('active')) {
    e.preventDefault();
    flagTurn();
  }
});

/* assembler shadow (per-turn corrections + provenance) */
const PROV_LABEL = {
  'quarantine': 'quarantine', 'sticky-identity': 'sticky', 'carry-over': 'carry-over',
  'derived': 'derived',
};
async function loadShadow() {
  let data;
  try { data = await (await fetch('/api/shadow')).json(); } catch { return; }
  const status = $('#shadow-status');
  const list = $('#shadow-list');
  if (!data.active) { status.textContent = 'not running'; return; }
  status.textContent = `active · ${data.board_id}`;
  list.classList.remove('dim');
  list.innerHTML = '';
  const turns = (data.turns || []).slice().reverse();
  if (!turns.length) { list.innerHTML = '<span class="dim">no turns yet</span>'; return; }
  turns.forEach((t) => {
    const row = document.createElement('div');
    row.className = 'shadow-turn';
    let html = `<span class="st-turn">turn ${t.turn}</span><span>`;
    const provOf = (text) => {
      for (const [field, src] of Object.entries(t.provenance || {})) {
        if (text.startsWith(field.split('.')[0])) return src;
      }
      return null;
    };
    if ((t.corrections || []).length === 0 && (t.contradictions || []).length === 0) {
      html += '<span class="shadow-clean">clean — vision confirmed</span>';
    } else {
      html += (t.corrections || []).map((c) => {
        const src = provOf(c);
        const badge = src ? `<span class="prov-badge ${src}">${PROV_LABEL[src] || src}</span>` : '';
        return `${badge}${c}`;
      }).join('<br>');
      if ((t.contradictions || []).length) {
        if ((t.corrections || []).length) html += '<br>';
        html += (t.contradictions || []).map((c) =>
          `<span class="shadow-contra">⚠ ${c.field}: vision=${c.vision} vs ${c.derived ?? c.carry_over} (${c.note})</span>`
        ).join('<br>');
      }
    }
    row.innerHTML = html + '</span>';
    list.appendChild(row);
  });
}
setInterval(() => {
  const badge = $('#feed-status');
  const age = (Date.now() - lastMsgTime) / 1000;
  if (!lastMsgTime) { badge.textContent = 'no feed'; badge.classList.add('stale'); return; }
  badge.textContent = age < 15 ? 'live' : `stale ${Math.round(age)}s`;
  badge.classList.toggle('stale', age >= 15);
}, 1000);

function bb(v, big) { return big ? (v / big).toFixed(1) + 'bb' : v; }

function renderLive(snap) {
  const t = snap.turn;
  if (!t) return;
  $('#live-empty').classList.add('hidden');
  $('#live-view').classList.remove('hidden');
  $('#board-id').textContent = `${snap.board_id}  ·  turn ${t.turn}  ·  feed: ${snap.feed}`;

  const obs = t.observation || {};
  const big = obs.big_blind || 1;
  renderCards($('#community-cards'), obs.community_cards, 5);
  $('#pot-line').textContent =
    `${obs.street} · pot ${bb(obs.pot_size, big)} · to call ` +
    (obs.call_amount_known ? bb(obs.call_amount, big) : (obs.call_amount == null ? '—' : bb(obs.call_amount, big) + '?'));

  const seats = $('#seats');
  seats.innerHTML = '';
  const hero = document.createElement('div');
  hero.className = 'seat hero';
  hero.innerHTML =
    `<div class="nm">Hero ${obs.hero_is_small_blind ? '<span class="blind-tag">SB</span>' : ''}` +
    `${obs.hero_is_big_blind ? '<span class="blind-tag">BB</span>' : ''}` +
    `${obs.hero_position === 0 ? '<span class="blind-tag">BU</span>' : ''}</div>` +
    `<div class="sub">stack ${bb(obs.hero_stack, big)} · in ${bb(obs.hero_committed, big)} · pos ${obs.hero_position}</div>`;
  seats.appendChild(hero);
  (obs.seats || []).forEach((s) => {
    if (!s.occupied) return;
    const el = document.createElement('div');
    el.className = 'seat' + (s.is_active ? '' : ' folded');
    el.innerHTML =
      `<div class="nm"><span class="dot ${s.vpip_color || 'none'}" title="VPIP ${s.vpip_color || '?'}"></span>` +
      `<span class="dot ${s.agg_color || 'none'}" title="AGG ${s.agg_color || '?'}"></span>` +
      `${s.name || s.seat_key} ` +
      `${s.is_small_blind ? '<span class="blind-tag">SB</span>' : ''}` +
      `${s.is_big_blind ? '<span class="blind-tag">BB</span>' : ''}` +
      `${s.position === 0 ? '<span class="blind-tag">BU</span>' : ''}</div>` +
      `<div class="sub">${s.state_label} · stack ${bb(s.stack, big)} · in ${bb(s.committed, big)}` +
      `${s.raised_this_street ? ' · raised' : ''}</div>`;
    seats.appendChild(el);
  });

  const ev = t.evaluation || {};
  $('#model-name').textContent = ev.model || '';
  renderCards($('#hero-cards'), obs.hero_cards, 2);
  $('#eval-line').textContent =
    `equity ${(ev.equity ?? 0).toFixed(3)} (${ev.equity_method || '?'}) · ` +
    `strength ${(ev.hand_strength ?? 0).toFixed(3)} · edge ${(ev.equity_edge ?? 0).toFixed(2)}`;

  const bars = $('#policy-bars');
  bars.innerHTML = '';
  const act = t.action || {};
  const pol = ev.actor_policy || {};
  const q = ev.critic_q || {};
  Object.keys(pol).forEach((k) => {
    const row = document.createElement('div');
    row.className = 'pbar' + (k === act.chosen ? ' chosen' : '');
    row.innerHTML =
      `<span>${k}</span>` +
      `<span class="track"><span class="fill" style="width:${(pol[k] * 100).toFixed(1)}%"></span></span>` +
      `<span class="q">${(pol[k] * 100).toFixed(1)}% · Q ${(q[k] ?? 0).toFixed(2)}</span>`;
    bars.appendChild(row);
  });
  $('#action-line').textContent =
    act.chosen ? `→ ${act.chosen}${act.bet_size ? ' ' + bb(act.bet_size, big) : ''}` : '';
  $('#reason-line').textContent = act.reason || '';
}

/* ---------------------------------------------------------------- opponents */
function fmtN(v, d) { return v == null ? '—' : Number(v).toFixed(d ?? 0); }

async function loadOpponents() {
  const data = await (await fetch('/api/opponents?window=100&min_hands=10')).json();
  $('#opp-total').textContent = data.total_hands;
  $('#opp-window').textContent = data.window;
  const tb = $('#opp-table tbody');
  tb.innerHTML = '';
  data.players.forEach((p) => {
    const scopes = [['lifetime', p.lifetime]];
    const w = p['last_' + data.window];
    if (w && w.hands < p.lifetime.hands) scopes.push(['last-' + data.window, w]);
    scopes.forEach(([label, s], i) => {
      const tr = document.createElement('tr');
      if (i === 1) tr.className = 'scope-window';
      tr.innerHTML =
        `<td>${i === 0 ? p.name : ''}</td><td>${label}</td>` +
        `<td class="num">${s.hands}</td><td class="num">${fmtN(s.vpip)}</td>` +
        `<td class="num">${fmtN(s.pfr)}</td><td class="num">${fmtN(s.limp)}</td>` +
        `<td class="num">${fmtN(s.threebet, 1)}</td><td class="num">${fmtN(s.af, 1)}</td>` +
        `<td class="num">${fmtN(s.wtsd)}</td><td class="num">${fmtN(s.wsd)}</td>` +
        `<td class="num">${fmtN(s.think_avg_s, 1)}</td><td class="num">${fmtN(s.pf_raise_avg_bb, 1)}</td>`;
      tb.appendChild(tr);
    });
  });
}

/* ---------------------------------------------------------------- hands */
async function loadHands() {
  const hands = await (await fetch('/api/hands?limit=100')).json();
  const tb = $('#hands-table tbody');
  tb.innerHTML = '';
  hands.forEach((h) => {
    const tr = document.createElement('tr');
    tr.className = 'rowlink' + (h.winner === 'Zwonkie' ? ' winrow' : '');
    tr.innerHTML =
      `<td>${h.hand_id}</td><td>${h.players.join(', ')}</td>` +
      `<td>${h.winner ?? '—'}</td><td class="num">${h.pot ?? '—'}</td>` +
      `<td>${h.last_street ?? '—'}</td><td>${h.source}</td>`;
    tr.addEventListener('click', () => loadHandDetail(h.sessioncode, h.hand_id));
    tb.appendChild(tr);
  });
}

async function loadHandDetail(sessioncode, handId) {
  const h = await (await fetch(`/api/hand/${sessioncode}/${handId}`)).json();
  const el = $('#hand-detail');
  if (h.error) { el.textContent = 'Not found.'; return; }
  el.classList.remove('empty');
  el.innerHTML = `<div><b>${handId}</b> <span class="dim">session ${sessioncode} · ${h.source}` +
    ` · blinds ${(h.blinds || {}).sb ?? '?'}/${(h.blinds || {}).bb ?? '?'}</span></div>`;
  const seats = {};
  (h.players || []).forEach((p) => {
    seats[p.seat] = p.name + (p.hole_cards && p.hole_cards.length ? ` [${p.hole_cards.join(' ')}]` : '');
  });
  const streets = {};
  (h.actions || []).forEach((a) => (streets[a.street] = streets[a.street] || []).push(a));
  const board = h.board || [];
  const BOARD_AT = { flop: board.slice(0, 3), turn: board.slice(3, 4), river: board.slice(4, 5) };
  ['preflop', 'flop', 'turn', 'river'].forEach((st) => {
    if (!streets[st]) return;
    const blk = document.createElement('div');
    blk.className = 'street-block';
    const boardCards = BOARD_AT[st];
    blk.innerHTML = `<div class="sb-title">${st}${boardCards && boardCards.length ? ' ' + boardCards.join(' ') : ''}</div>`;
    streets[st].forEach((a) => {
      const row = document.createElement('div');
      row.className = 'act-row';
      row.innerHTML = `<span class="who">${seats[a.seat] || a.seat}</span>` +
        `<span>${a.action}${a.amount ? ' ' + a.amount : ''}</span>` +
        `<span class="dim">${a.think_time_s != null ? a.think_time_s.toFixed(1) + 's' : ''}</span>`;
      blk.appendChild(row);
    });
    el.appendChild(blk);
  });
  const res = h.result || {};
  const done = document.createElement('div');
  done.innerHTML = `<b>${res.winner_name || res.winner_seat || '?'}</b> wins ${res.pot ?? '?'}`;
  el.appendChild(done);
}

/* ---------------------------------------------------------------- flags */
async function loadFlags() {
  const flags = await (await fetch('/api/flags')).json();
  const el = $('#flags-list');
  el.innerHTML = flags.length ? '' : '<div class="empty">No flagged turns.</div>';
  flags.forEach((f) => {
    const rec = f.record || {};
    const obs = rec.observation || {};
    const act = rec.action || {};
    const ev = rec.evaluation || {};
    const item = document.createElement('div');
    item.className = 'flag-item';
    let detail = f.record
      ? `<div>hero ${(obs.hero_cards || []).join(' ')} · board ${(obs.community_cards || []).join(' ') || '—'}` +
        ` · ${obs.street} · equity ${(ev.equity ?? 0).toFixed(3)}</div>` +
        `<div>→ <b>${act.chosen ?? '?'}</b> <span class="dim">${act.reason ?? ''}</span></div>`
      : `<div class="dim">no turn record (pre-format-2 session)</div>`;
    item.innerHTML =
      `<div class="fi-head"><b>${f.board_id}</b> <span>turn ${f.flag.turn}</span>` +
      `<span>${f.flag.action ?? ''}</span><span class="dim">${f.flag.ts}</span>` +
      `<span class="dim">${f.artifacts.length} artifact(s)</span></div>` + detail;
    el.appendChild(item);
  });
}
