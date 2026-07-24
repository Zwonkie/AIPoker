/* live2 dashboard -- vanilla JS, view-only. */
'use strict';

const $ = (sel) => document.querySelector(sel);

/* ---------------------------------------------------------------- tabs */
function activateTab(name) {
  document.querySelectorAll('nav button').forEach((x) =>
    x.classList.toggle('active', x.dataset.tab === name));
  document.querySelectorAll('.tab').forEach((x) => x.classList.remove('active'));
  $('#tab-' + name).classList.add('active');
  if (name === 'pilot') pilotStatus();
  if (name === 'opponents') loadOpponents();
  if (name === 'hands') loadHands();
  if (name === 'flags') loadFlags();
}
document.querySelectorAll('nav button').forEach((b) => {
  b.addEventListener('click', () => activateTab(b.dataset.tab));
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
    loadTableLog();
  };
  ws.onclose = () => setTimeout(connectWS, 2000);
}
connectWS();

/* ---------------------------------------------------------------- pilot control
   The same actions exist twice (header shortcuts + Pilot tab) -- buttons carry
   data-act and every handler binds/toggles them as a set. */
const actBtns = (act) => document.querySelectorAll(`.pbtn[data-act="${act}"]`);

async function pilotStatus() {
  let st;
  try { st = await (await fetch('/api/pilot/status')).json(); } catch { return; }
  const chip = $('#pilot-chip');
  const running = !!st.running;
  const model = st.model || '';                 // what the RUNNING pilot loaded
  const configured = st.configured_model || ''; // what a fresh start WOULD load (decision.py)
  const short = (s) => ((s.match(/\(([^)]+)\)/) || [null, s])[1] || s);
  const stale = running && model && configured && model !== configured;
  chip.className = 'pilot-chip ' + (running ? (st.mode === 'auto' ? 'auto' : 'rec') : 'off');
  const where = st.table === 'at-table' ? ` · at ${st.table_id || 'table'}`
    : st.table === 'waiting' ? ' · waiting for table' : '';
  chip.textContent = running
    ? `pilot ${st.mode === 'auto' ? 'AUTO' : 'recommending'}${model ? ' · ' + short(model) : ''}${where}`
    : 'pilot off';
  chip.title = running
    ? `pid ${st.pid} · started ${st.started}${model ? ' · model ' + model : ' · model loading…'}`
      + (stale ? ` · STALE: decision.py is set to ${configured} — Stop then Start to load it` : '')
    : (configured ? `next start will load ${configured}` : '');
  // Decision panel header reflects the running pilot's loaded model even between decisions
  // (ev.model only writes on a live decision event; this keeps it populated + flags staleness).
  const nameEl = $('#model-name');
  if (running) {
    nameEl.textContent = model || 'loading model…';
    nameEl.title = stale
      ? `Serving ${model}, but ${configured} is configured — restart the pilot to load it`
      : (model ? `pilot is serving ${model}` : '');
  }
  nameEl.classList.toggle('stale', !!stale);
  // Header stale badge: loud, always-visible cue that the process predates a decision.py change.
  const staleEl = $('#model-stale');
  if (staleEl) {
    staleEl.classList.toggle('hidden', !stale);
    staleEl.textContent = stale ? `⚠ ${short(configured)} configured — restart` : '';
    staleEl.title = stale
      ? `Running ${model}; core/decision.py is set to ${configured}. Stop then Start the pilot to load it.`
      : '';
  }
  actBtns('start').forEach((b) => b.classList.toggle('hidden', running));
  actBtns('auto').forEach((b) => b.classList.toggle('hidden', running));
  actBtns('stop').forEach((b) => b.classList.toggle('hidden', !running));
  const hasLog = (st.log || []).length > 0;
  const whereLong = st.table === 'at-table' ? ` · at table ${st.table_id || ''}`
    : st.table === 'waiting' ? ' · waiting for a table' : '';
  $('#pilot-mode').textContent = running
    ? `${st.mode} · ${model || 'loading model…'}${whereLong} · pid ${st.pid} · started ${st.started}`
    : (configured ? `stopped · next start loads ${configured}` : 'stopped');
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
actBtns('start').forEach((b) => b.addEventListener('click', async () => {
  await pilotPost('/api/pilot/start', { mode: 'recommend' });
  pilotStatus();
}));
actBtns('auto').forEach((b) => b.addEventListener('click', async () => {
  if (!confirm('Start pilot in AUTO mode? It will click the real table.\n' +
               'Abort at any time: slam the cursor into a screen corner, or press Stop.')) return;
  await pilotPost('/api/pilot/start', { mode: 'auto' });
  pilotStatus();
}));
actBtns('stop').forEach((b) => b.addEventListener('click', async () => {
  await pilotPost('/api/pilot/stop');
  pilotStatus();
}));
/* Always-visible hard release: pilot + any orphaned pilot/probe processes -> fresh state. */
actBtns('stopall').forEach((b) => b.addEventListener('click', async () => {
  if (!confirm('Stop ALL pilot activity — the pilot plus any probe/orphan processes — ' +
               'and return to a fresh, nothing-running state?')) return;
  const t = b.textContent;
  b.disabled = true; b.textContent = 'Releasing…';
  const r = await pilotPost('/api/pilot/stop_all');
  b.disabled = false; b.textContent = t;
  if (r && r.ok) {
    const n = (r.orphans_killed || []).length;
    b.textContent = n ? `Released (+${n})` : 'Released';
    setTimeout(() => { b.textContent = t; }, 2000);
  }
  pilotStatus();
}));
actBtns('probe').forEach((b) => b.addEventListener('click', async () => {
  activateTab('pilot');            // result renders in the Pilot tab, wherever it was clicked
  const btns = [...actBtns('probe')];
  btns.forEach((x) => { x.disabled = true; x.textContent = 'Probing…'; });
  const r = await pilotPost('/api/pilot/probe');
  btns.forEach((x) => { x.disabled = false; x.textContent = 'Probe'; });
  if (!r) return;
  $('#probe-out').classList.remove('hidden');
  $('#probe-text').textContent = (r.output || []).join('\n');
  const img = $('#probe-img');
  img.classList.toggle('hidden', !r.png);
  if (r.png) img.src = '/api/pilot/probe.png?t=' + Date.now();
}));
setInterval(pilotStatus, 3000);
pilotStatus();
// Opponent-profile cache for the live seat mini-stats (independent of the Opponents tab).
refreshOppStats();
setInterval(refreshOppStats, 20000);

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

$('#shadow-clear').addEventListener('click', async () => {
  await pilotPost('/api/shadow/clear');
  $('#shadow-list').innerHTML = '<span class="shadow-clean">cleared</span>';
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
        html += (t.contradictions || []).map((c) => {
          // contradictions carry per-rule evidence keys -- render whatever is present
          const vals = Object.entries(c)
            .filter(([k]) => k !== 'field' && k !== 'note')
            .filter(([, v]) => v !== null && v !== undefined)
            .map(([k, v]) => `${k}=${v}`).join(' · ');
          return `<span class="shadow-contra">⚠ ${c.field}: ${vals} (${c.note})</span>`;
        }).join('<br>');
      }
    }
    row.innerHTML = html + '</span>';
    list.appendChild(row);
  });
}

/* ---------------------------------------------------------------- table log
   Per-hand OUTCOME log for the live table, straight from the ground-truth hand store
   (winner/street/pot + hero's action & chip delta) -- NOT derived from the live board
   state, which never sees showdowns. Newest hand first. */
const ACT_LABEL = {
  post_sb: 'posts SB', post_bb: 'posts BB', ante: 'ante', fold: 'folds', check: 'checks',
  call: 'calls', bet: 'bets', raise: 'raises', allin: 'all-in',
};
// "Clear" just clears the Table log VIEW: it hides the hands on screen now; newer hands still
// stream in. In-memory only (a page reload shows all again); resets when the table changes.
let lastTableLog = null;
let tablelogClearSeq = 0;
async function loadTableLog() {
  let data;
  try { data = await (await fetch('/api/table_log?limit=15')).json(); } catch { return; }
  if (!lastTableLog || data.tournament_id !== lastTableLog.tournament_id) tablelogClearSeq = 0;
  lastTableLog = data;
  const rows = (data.rows || []).filter((r) => Number(r.seq) > tablelogClearSeq);
  $('#tablelog-status').textContent = data.tournament_id
    ? `ground truth · table ${data.tournament_id} · ${rows.length} hand(s)`
    : 'ground truth · from hand history';
  $('#tablelog-table').classList.toggle('hidden', rows.length === 0);
  $('#tablelog-empty').classList.toggle('hidden', rows.length > 0);
  const tb = $('#tablelog-table tbody');
  tb.innerHTML = '';
  rows.forEach((r) => {
    const tr = document.createElement('tr');
    tr.className = 'rowlink' + (r.hero_won ? ' winrow' : '');
    const act = r.hero_action ? (ACT_LABEL[r.hero_action] || r.hero_action) : 'sits out';
    const street = r.hero_street ? `<span class="dim"> ${r.hero_street}</span>` : '';
    const net = r.hero_net == null ? ''
      : `<span class="net ${r.hero_net >= 0 ? 'pos' : 'neg'}">` +
        `${r.hero_net > 0 ? '+' : ''}${r.hero_net}</span>`;
    const result = r.winner_name
      ? `<b>${r.winner_name}</b> won<span class="dim"> ${r.winner_street || '—'}</span> · ${r.pot ?? '—'}`
      : '<span class="dim">—</span>';
    tr.innerHTML =
      `<td class="num">${r.seq}</td>` +
      `<td>${act}${street} ${net}</td>` +
      `<td>${result}</td>`;
    if (r.sessioncode && r.hand_id != null) {
      tr.addEventListener('click', () => {
        activateTab('hands');
        loadHandDetail(r.sessioncode, r.hand_id);
      });
    }
    tb.appendChild(tr);
  });
}
$('#tablelog-clear').addEventListener('click', () => {
  const seqs = (lastTableLog?.rows || []).map((r) => Number(r.seq)).filter(Number.isFinite);
  tablelogClearSeq = seqs.length ? Math.max(...seqs) : tablelogClearSeq;
  loadTableLog();
});
loadTableLog();
setInterval(loadTableLog, 8000);

setInterval(() => {
  const badge = $('#feed-status');
  const age = (Date.now() - lastMsgTime) / 1000;
  if (!lastMsgTime) { badge.textContent = 'no feed'; badge.classList.add('stale'); return; }
  badge.textContent = age < 15 ? 'live' : `stale ${Math.round(age)}s`;
  badge.classList.toggle('stale', age >= 15);
}, 1000);

function bb(v, big) { return big ? (v / big).toFixed(1) + 'bb' : v; }

/* Opponent-profile cache (name -> profile), joined into the live seat cards. Refreshed
   independently of the Opponents tab so the mini-stats line is populated on the Live view.
   The 3 stats shown are the most telling ones NOT already encoded by the VPIP/AGG dots:
   PFR (preflop aggression), 3Bet% (re-raise tendency), WTSD% (showdown/station tendency). */
let oppStatsByName = {};
async function refreshOppStats() {
  try {
    const data = await (await fetch('/api/opponents?window=100&min_hands=1')).json();
    const m = {};
    (data.players || []).forEach((p) => { m[p.name] = p; });
    oppStatsByName = m;
  } catch { /* keep last cache on a transient failure */ }
}
function statsLine(name) {
  const p = name ? oppStatsByName[name] : null;
  const s = p && p.lifetime;
  if (!s || !s.hands) return '<div class="stats none">no reads yet</div>';
  const cell = (k, v, d) => {
    const val = fmtN(v, d);
    return `<span class="st"><i>${k}</i>${val === '—' ? val : val + '%'}</span>`;
  };
  return `<div class="stats" title="${s.hands} hands in store">` +
    cell('PFR', s.pfr, 0) + cell('3B', s.threebet, 1) + cell('WTSD', s.wtsd, 0) +
    '</div>';
}

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
  // Fixed 3x3 table mirror: every seat_key keeps its physical cell (CSS .pos-*), so a
  // vacated seat leaves a dashed "empty" placeholder instead of the others reflowing.
  // Hero is pinned to the lower-centre cell. The board/pot occupies the true centre and
  // is shown above, so that cell stays blank.
  const byKey = {};
  (obs.seats || []).forEach((s) => { byKey[s.seat_key] = s; });
  ['seat_1', 'seat_2', 'seat_3', 'seat_4', 'seat_5'].forEach((key) => {
    const s = byKey[key];
    const el = document.createElement('div');
    if (!s || !s.occupied) {
      el.className = `seat empty pos-${key}`;
      el.textContent = 'empty';
      seats.appendChild(el);
      return;
    }
    el.className = `seat pos-${key}` + (s.is_active ? '' : ' folded');
    el.innerHTML =
      `<div class="nm"><span class="dot ${s.vpip_color || 'none'}" title="VPIP ${s.vpip_color || '?'}"></span>` +
      `<span class="dot ${s.agg_color || 'none'}" title="AGG ${s.agg_color || '?'}"></span>` +
      `${s.name || s.seat_key} ` +
      `${s.is_small_blind ? '<span class="blind-tag">SB</span>' : ''}` +
      `${s.is_big_blind ? '<span class="blind-tag">BB</span>' : ''}` +
      `${s.position === 0 ? '<span class="blind-tag">BU</span>' : ''}</div>` +
      `<div class="sub">stack ${bb(s.stack, big)} · in ${bb(s.committed, big)}` +
      `${s.raised_this_street ? ' · raised' : ''}</div>` +
      statsLine(s.name);
    seats.appendChild(el);
  });
  const hero = document.createElement('div');
  hero.className = 'seat hero';
  hero.innerHTML =
    `<div class="nm">Hero ${obs.hero_is_small_blind ? '<span class="blind-tag">SB</span>' : ''}` +
    `${obs.hero_is_big_blind ? '<span class="blind-tag">BB</span>' : ''}` +
    `${obs.hero_position === 0 ? '<span class="blind-tag">BU</span>' : ''}</div>` +
    `<div class="sub">stack ${bb(obs.hero_stack, big)} · in ${bb(obs.hero_committed, big)} · pos ${obs.hero_position}</div>` +
    statsLine(obs.hero_name || 'Zwonkie');
  seats.appendChild(hero);

  const ev = t.evaluation || {};
  // Only write when the decision carries a model; otherwise leave the status-driven running
  // model in place (pilotStatus keeps #model-name populated between decisions).
  if (ev.model) $('#model-name').textContent = ev.model;
  renderCards($('#hero-cards'), obs.hero_cards, 2);
  $('#eval-line').textContent =
    `equity ${(ev.equity ?? 0).toFixed(3)} (${ev.equity_method || '?'}) · ` +
    `strength ${(ev.hand_strength ?? 0).toFixed(3)} · edge ${(ev.equity_edge ?? 0).toFixed(2)}`;

  const bars = $('#policy-bars');
  bars.innerHTML = '';
  const act = t.action || {};
  const pol = ev.actor_policy || {};
  const q = ev.critic_q || {};
  // critic preference as a comparable percentage: softmax over the Q values present
  const qKeys = Object.keys(pol).filter((k) => q[k] != null);
  const qMax = Math.max(...qKeys.map((k) => q[k]), -Infinity);
  const qExp = Object.fromEntries(qKeys.map((k) => [k, Math.exp(q[k] - qMax)]));
  const qSum = qKeys.reduce((s, k) => s + qExp[k], 0) || 1;
  Object.keys(pol).forEach((k) => {
    const pPct = (pol[k] * 100);
    const qPct = q[k] != null ? (qExp[k] / qSum) * 100 : null;
    const row = document.createElement('div');
    row.className = 'pbar' + (k === act.chosen ? ' chosen' : '');
    row.innerHTML =
      `<span>${k}</span>` +
      `<span class="track">` +
      `<span class="fill" style="width:${pPct.toFixed(1)}%"></span>` +
      (qPct != null ? `<span class="qfill" style="width:${qPct.toFixed(1)}%"></span>` : '') +
      `</span>` +
      `<span class="q">${pPct.toFixed(1)}%` +
      (qPct != null ? ` · Q ${qPct.toFixed(0)}% (${(q[k]).toFixed(2)})` : '') + `</span>`;
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
  // players currently seated at the live table first, highlighted
  const players = [...data.players.filter((p) => p.at_table),
                   ...data.players.filter((p) => !p.at_table)];
  players.forEach((p) => {
    const scopes = [['lifetime', p.lifetime]];
    const w = p['last_' + data.window];
    if (w && w.hands < p.lifetime.hands) scopes.push(['last-' + data.window, w]);
    scopes.forEach(([label, s], i) => {
      const tr = document.createElement('tr');
      if (i === 1) tr.className = 'scope-window';
      if (p.at_table) tr.classList.add('at-table');
      tr.innerHTML =
        `<td>${i === 0 ? p.name + (p.at_table ? ' <span class="table-badge">at table</span>' : '') : ''}</td><td>${label}</td>` +
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
