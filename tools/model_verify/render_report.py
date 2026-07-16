"""Renders a model_verify results JSON (from run.py --dump-json) into a self-contained HTML
report with the raw per-scenario data as interactive charts/heatmaps, for visual analysis
beyond the PASS/FAIL table. Reusable across versions -- point it at any results JSON.

Usage:
  .venv/Scripts/python.exe -m tools.model_verify.render_report \
      tools/model_verify/results/v15__expert_main.pth.json \
      --out tools/model_verify/results/v15_report.html
"""
import argparse
import json
import os

TEMPLATE = """<!doctype html>
<title>{title}</title>
<style>
.viz-root {{
  color-scheme: light;
  --surface-1:      #fcfcfb;
  --surface-2:      #f9f9f7;
  --text-primary:   #0b0b0b;
  --text-secondary: #52514e;
  --text-muted:     #898781;
  --grid:           #e1e0d9;
  --baseline:       #c3c2b7;
  --border:         rgba(11,11,11,0.10);
  --good:           #0ca30c;
  --warning:        #fab219;
  --serious:        #ec835a;
  --critical:       #d03b3b;
  --seq-100: #cde2fb; --seq-200: #9ec5f4; --seq-300: #6da7ec; --seq-400: #3987e5;
  --seq-500: #256abf; --seq-600: #184f95; --seq-700: #0d366b;
  --c-fold:      #2a78d6;
  --c-call:      #1baf7a;
  --c-raise_33:  #eda100;
  --c-raise_66:  #008300;
  --c-raise_pot: #4a3aa7;
  --c-allin:     #e34948;
}}
@media (prefers-color-scheme: dark) {{
  :root:where(:not([data-theme="light"])) .viz-root {{
    color-scheme: dark;
    --surface-1: #1a1a19; --surface-2: #0d0d0d;
    --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #898781;
    --grid: #2c2c2a; --baseline: #383835; --border: rgba(255,255,255,0.10);
    --good: #0ca30c; --warning: #fab219; --serious: #ec835a; --critical: #d03b3b;
    --seq-100: #b7d3f6; --seq-200: #9ec5f4; --seq-300: #5598e7; --seq-400: #3987e5;
    --seq-500: #256abf; --seq-600: #184f95; --seq-700: #0d366b;
    --c-fold: #3987e5; --c-call: #199e70; --c-raise_33: #c98500;
    --c-raise_66: #008300; --c-raise_pot: #9085e9; --c-allin: #e66767;
  }}
}}
:root[data-theme="dark"] .viz-root {{
  color-scheme: dark;
  --surface-1: #1a1a19; --surface-2: #0d0d0d;
  --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #898781;
  --grid: #2c2c2a; --baseline: #383835; --border: rgba(255,255,255,0.10);
  --good: #0ca30c; --warning: #fab219; --serious: #ec835a; --critical: #d03b3b;
  --seq-100: #b7d3f6; --seq-200: #9ec5f4; --seq-300: #5598e7; --seq-400: #3987e5;
  --seq-500: #256abf; --seq-600: #184f95; --seq-700: #0d366b;
  --c-fold: #3987e5; --c-call: #199e70; --c-raise_33: #c98500;
  --c-raise_66: #008300; --c-raise_pot: #9085e9; --c-allin: #e66767;
}}

* {{ box-sizing: border-box; }}
body {{
  margin: 0; background: var(--surface-2); color: var(--text-primary);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
}}
.wrap {{ max-width: 1180px; margin: 0 auto; padding: 32px 24px 80px; }}
h1 {{ font-size: 22px; margin: 0 0 4px; }}
.subtitle {{ color: var(--text-secondary); font-size: 14px; margin: 0 0 28px; }}
.subtitle code {{ background: var(--surface-1); border: 1px solid var(--border); border-radius: 4px; padding: 1px 5px; }}

.stat-row {{ display: flex; gap: 12px; margin-bottom: 32px; flex-wrap: wrap; }}
.stat-tile {{
  background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px;
  padding: 14px 18px; min-width: 92px;
}}
.stat-tile .n {{ font-size: 26px; font-weight: 600; font-variant-numeric: tabular-nums; }}
.stat-tile .l {{ font-size: 12px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: .04em; }}
.stat-tile.pass .n {{ color: var(--good); }}
.stat-tile.warn .n {{ color: var(--warning); }}
.stat-tile.fail .n {{ color: var(--critical); }}

.card {{
  background: var(--surface-1); border: 1px solid var(--border); border-radius: 12px;
  padding: 20px 22px 22px; margin-bottom: 22px;
}}
.card h2 {{ font-size: 15px; margin: 0 0 2px; }}
.card .issue {{ font-size: 12px; color: var(--text-muted); margin: 0 0 4px; }}
.card .detail {{ font-size: 13px; color: var(--text-secondary); margin: 0 0 16px; }}
.badge {{ display: inline-block; font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 999px; margin-left: 8px; vertical-align: middle; }}
.badge.PASS {{ background: color-mix(in srgb, var(--good) 18%, transparent); color: var(--good); }}
.badge.WARN {{ background: color-mix(in srgb, var(--warning) 22%, transparent); color: #8a5c00; }}
.badge.FAIL {{ background: color-mix(in srgb, var(--critical) 18%, transparent); color: var(--critical); }}
:root[data-theme="dark"] .badge.WARN, @media (prefers-color-scheme: dark) {{ .badge.WARN {{ color: var(--warning); }} }}

table.heatmap {{ border-collapse: collapse; font-size: 12px; }}
table.heatmap th {{ font-weight: 500; color: var(--text-muted); padding: 4px 8px; text-align: center; }}
table.heatmap td {{
  text-align: center; padding: 8px 10px; font-variant-numeric: tabular-nums;
  border: 2px solid var(--surface-1); border-radius: 4px; min-width: 54px; position: relative;
}}
table.heatmap td .ring {{ position: absolute; inset: 2px; border-radius: 3px; border: 2px solid var(--critical); pointer-events: none; }}
.axis-label {{ font-size: 11px; color: var(--text-muted); }}

.legend {{ display: flex; gap: 14px; flex-wrap: wrap; margin-top: 12px; font-size: 12px; color: var(--text-secondary); }}
.legend .sw {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 5px; vertical-align: -1px; }}

table.kv-table {{ border-collapse: collapse; font-size: 12.5px; width: 100%; }}
table.kv-table th {{
  font-weight: 500; color: var(--text-muted); text-align: left; padding: 7px 14px 7px 0;
  border-bottom: 1px solid var(--border); text-transform: capitalize; white-space: nowrap;
  position: sticky; top: 0; background: var(--surface-1);
}}
table.kv-table td {{
  padding: 7px 14px 7px 0; border-bottom: 1px solid var(--grid); font-variant-numeric: tabular-nums;
  white-space: nowrap; vertical-align: middle;
}}
table.kv-table tr:last-child td {{ border-bottom: none; }}
table.kv-table tr:hover td {{ background: var(--surface-2); }}
table.kv-table td.truncate {{ max-width: 260px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}

.policy-chip {{ display: inline-flex; align-items: center; gap: 7px; }}
.policy-chip-bar {{ display: flex; width: 84px; height: 9px; border-radius: 3px; overflow: hidden; background: var(--surface-2); flex-shrink: 0; }}
.policy-chip-seg {{ height: 100%; }}
.policy-chip-label {{ font-size: 11px; color: var(--text-secondary); }}

.bars {{ display: flex; align-items: flex-end; gap: 18px; height: 140px; padding-top: 10px; }}
.bar-col {{ display: flex; flex-direction: column; align-items: center; gap: 6px; width: 56px; }}
.bar-track {{ width: 32px; height: 100px; background: var(--surface-2); border-radius: 4px 4px 0 0; display: flex; align-items: flex-end; }}
.bar-fill {{ width: 100%; border-radius: 4px 4px 0 0; }}
.bar-val {{ font-size: 11px; color: var(--text-secondary); font-variant-numeric: tabular-nums; }}
.bar-label {{ font-size: 11px; color: var(--text-muted); }}

.two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
@media (max-width: 720px) {{ .two-col {{ grid-template-columns: 1fr; }} }}

svg text {{ fill: var(--text-muted); font-size: 10px; font-family: inherit; }}
.line-path {{ fill: none; stroke-width: 2px; }}
.line-dot {{ cursor: pointer; }}
</style>
<div class="viz-root">
<div class="wrap">
  <h1>model_verify report</h1>
  <p class="subtitle">version <code>{version}</code> &middot; weights <code>{weights}</code> &middot; action space <code>{action_space}</code></p>

  <div class="stat-row">
    <div class="stat-tile pass"><div class="n">{n_pass}</div><div class="l">Pass</div></div>
    <div class="stat-tile warn"><div class="n">{n_warn}</div><div class="l">Warn</div></div>
    <div class="stat-tile fail"><div class="n">{n_fail}</div><div class="l">Fail</div></div>
    <div class="stat-tile"><div class="n">{n_skip}</div><div class="l">Skip</div></div>
  </div>

  <div id="charts"></div>
</div>
</div>

<script>
const RESULTS = {results_json};
const ACTION_COLOR = {{
  fold: 'var(--c-fold)', call: 'var(--c-call)', raise_33: 'var(--c-raise_33)',
  raise_66: 'var(--c-raise_66)', raise_pot: 'var(--c-raise_pot)', allin: 'var(--c-allin)'
}};
const ACTION_LABEL = {{
  fold: 'Fold', call: 'Call', raise_33: 'Raise 33%', raise_66: 'Raise 66%',
  raise_pot: 'Raise Pot', allin: 'All-In'
}};
const SEQ = ['var(--seq-100)','var(--seq-200)','var(--seq-300)','var(--seq-400)','var(--seq-500)','var(--seq-600)','var(--seq-700)'];

function seqColor(v) {{ // v in [0,1] -> sequential ramp step
  const idx = Math.max(0, Math.min(SEQ.length - 1, Math.round(v * (SEQ.length - 1))));
  return SEQ[idx];
}}
function textOn(v) {{ return v > 0.55 ? '#ffffff' : 'var(--text-primary)'; }}

function el(tag, attrs, children) {{
  const e = document.createElementNS(tag === 'svg' || tag === 'g' || tag === 'path' || tag === 'circle' || tag === 'line' || tag === 'text'
    ? 'http://www.w3.org/2000/svg' : null, tag);
  for (const k in (attrs || {{}})) e.setAttribute(k, attrs[k]);
  (children || []).forEach(c => e.appendChild(c));
  return e;
}}
function htmlEl(tag, attrs, html) {{
  const e = document.createElement(tag);
  for (const k in (attrs || {{}})) {{
    if (k === 'class') e.className = attrs[k]; else e.setAttribute(k, attrs[k]);
  }}
  if (html !== undefined) e.innerHTML = html;
  return e;
}}

function card(title, issue, detail, status) {{
  const c = htmlEl('div', {{class: 'card'}});
  c.appendChild(htmlEl('h2', {{}}, `${{title}} <span class="badge ${{status}}">${{status}}</span>`));
  if (issue) c.appendChild(htmlEl('p', {{class: 'issue'}}, `guards: ${{issue}}`));
  c.appendChild(htmlEl('p', {{class: 'detail'}}, detail));
  return c;
}}

function lineChart(data, actionKeys) {{
  const W = 640, H = 220, PAD = 34;
  const xs = data.map(d => d.equity);
  const svg = el('svg', {{viewBox: `0 0 ${{W}} ${{H}}`, width: '100%', height: H}});
  // gridlines
  for (let i = 0; i <= 4; i++) {{
    const y = PAD + (H - 2 * PAD) * (1 - i / 4);
    svg.appendChild(el('line', {{x1: PAD, x2: W - 10, y1: y, y2: y, stroke: 'var(--grid)', 'stroke-width': 1}}));
    const t = el('text', {{x: 4, y: y + 3}}); t.textContent = (i / 4).toFixed(2); svg.appendChild(t);
  }}
  xs.forEach((eq, i) => {{
    const x = PAD + (W - PAD - 10) * (i / (xs.length - 1));
    const t = el('text', {{x: x - 8, y: H - 8}}); t.textContent = eq.toFixed(2); svg.appendChild(t);
  }});
  const xFor = i => PAD + (W - PAD - 10) * (i / (xs.length - 1));
  const yFor = v => PAD + (H - 2 * PAD) * (1 - v);
  actionKeys.forEach(ak => {{
    const pts = data.map((d, i) => [xFor(i), yFor(d.policy[ak] ?? 0)]);
    const d = pts.map((p, i) => (i === 0 ? 'M' : 'L') + p[0].toFixed(1) + ',' + p[1].toFixed(1)).join(' ');
    svg.appendChild(el('path', {{d, class: 'line-path', stroke: ACTION_COLOR[ak]}}));
    pts.forEach((p, i) => {{
      const dot = el('circle', {{cx: p[0], cy: p[1], r: 3, fill: ACTION_COLOR[ak], class: 'line-dot'}});
      const title = el('title'); title.textContent = `${{ACTION_LABEL[ak]}} @ equity ${{data[i].equity}}: ${{(data[i].policy[ak] ?? 0).toFixed(2)}}`;
      dot.appendChild(title);
      svg.appendChild(dot);
    }});
  }});
  const wrap = htmlEl('div', {{}});
  wrap.appendChild(svg);
  const legend = htmlEl('div', {{class: 'legend'}});
  actionKeys.forEach(ak => {{
    const item = htmlEl('span', {{}});
    item.innerHTML = `<span class="sw" style="background:${{ACTION_COLOR[ak]}}"></span>${{ACTION_LABEL[ak]}}`;
    legend.appendChild(item);
  }});
  wrap.appendChild(legend);
  return wrap;
}}

function getPath(obj, path) {{
  // resolves a possibly-dotted field path like 'policy.allin' against a record
  return path.split('.').reduce((o, k) => (o == null ? undefined : o[k]), obj);
}}

function heatmap(data, xField, yField, valueField, opts) {{
  opts = opts || {{}};
  const xs = [...new Set(data.map(d => d[xField]))].sort((a, b) => a - b);
  const ys = [...new Set(data.map(d => d[yField]))].sort((a, b) => a - b);
  const table = htmlEl('table', {{class: 'heatmap'}});
  const head = htmlEl('tr');
  head.appendChild(htmlEl('th', {{}}, `${{opts.yLabel || yField}} \\\\ ${{opts.xLabel || xField}}`));
  xs.forEach(x => head.appendChild(htmlEl('th', {{}}, String(x))));
  table.appendChild(head);
  ys.slice().reverse().forEach(y => {{
    const row = htmlEl('tr');
    row.appendChild(htmlEl('th', {{}}, String(y)));
    xs.forEach(x => {{
      const rec = data.find(d => d[xField] === x && d[yField] === y);
      const v = rec ? getPath(rec, valueField) : null;
      const td = htmlEl('td', {{}});
      if (v !== null && v !== undefined) {{
        td.style.background = seqColor(v);
        td.style.color = textOn(v);
        td.textContent = v.toFixed(2);
        td.title = `${{opts.xLabel || xField}}=${{x}}, ${{opts.yLabel || yField}}=${{y}}: ${{valueField}}=${{v.toFixed(3)}}` + (rec.argmax ? `, argmax=${{rec.argmax}}` : '');
        if (opts.ringField && rec[opts.ringField]) td.appendChild(htmlEl('div', {{class: 'ring'}}));
      }}
      row.appendChild(td);
    }});
    table.appendChild(row);
  }});
  return table;
}}

function categoricalGrid(data, xField, yField, catField) {{
  const xs = [...new Set(data.map(d => d[xField]))].sort((a, b) => a - b);
  const ys = [...new Set(data.map(d => d[yField]))].sort((a, b) => a - b);
  const table = htmlEl('table', {{class: 'heatmap'}});
  const head = htmlEl('tr');
  head.appendChild(htmlEl('th', {{}}, `equity \\\\ stack`));
  xs.forEach(x => head.appendChild(htmlEl('th', {{}}, String(x))));
  table.appendChild(head);
  ys.slice().reverse().forEach(y => {{
    const row = htmlEl('tr');
    row.appendChild(htmlEl('th', {{}}, String(y)));
    xs.forEach(x => {{
      const rec = data.find(d => d[xField] === x && d[yField] === y);
      const td = htmlEl('td', {{}});
      if (rec) {{
        const cat = rec[catField];
        td.style.background = ACTION_COLOR[cat] || '#888';
        td.style.color = '#fff';
        td.textContent = ACTION_LABEL[cat] || cat;
        td.title = `stack=${{x}}, equity=${{y}}: argmax=${{cat}}`;
      }}
      row.appendChild(td);
    }});
    table.appendChild(row);
  }});
  const wrap = htmlEl('div', {{}});
  wrap.appendChild(table);
  const legend = htmlEl('div', {{class: 'legend'}});
  Object.keys(ACTION_COLOR).forEach(ak => {{
    const item = htmlEl('span', {{}});
    item.innerHTML = `<span class="sw" style="background:${{ACTION_COLOR[ak]}}"></span>${{ACTION_LABEL[ak]}}`;
    legend.appendChild(item);
  }});
  wrap.appendChild(legend);
  return wrap;
}}

function barChart(data, xField, valueField, color, label) {{
  const wrap = htmlEl('div', {{class: 'bars'}});
  data.forEach(d => {{
    const v = d[valueField];
    const col = htmlEl('div', {{class: 'bar-col'}});
    const track = htmlEl('div', {{class: 'bar-track'}});
    const fill = htmlEl('div', {{class: 'bar-fill'}});
    fill.style.height = Math.round(v * 100) + '%';
    fill.style.background = color;
    fill.title = `${{xField}}=${{d[xField]}}: ${{label}}=${{v.toFixed(2)}}`;
    track.appendChild(fill);
    col.appendChild(htmlEl('div', {{class: 'bar-val'}}, v.toFixed(2)));
    col.appendChild(track);
    col.appendChild(htmlEl('div', {{class: 'bar-label'}}, String(d[xField]) + 'bb'));
    wrap.appendChild(col);
  }});
  return wrap;
}}

const ACTION_KEY_SET = new Set(Object.keys(ACTION_COLOR));

function isPolicyLike(v) {{
  if (!v || typeof v !== 'object' || Array.isArray(v)) return false;
  const keys = Object.keys(v);
  return keys.length > 0 && keys.every(k => ACTION_KEY_SET.has(k));
}}

// Compact stacked-bar rendering of a {{fold, call, raise_33, ...}} distribution, in place of
// dumping the raw object as JSON text (which was blowing out table columns).
function policyChip(policy) {{
  const wrap = htmlEl('div', {{class: 'policy-chip'}});
  const bar = htmlEl('div', {{class: 'policy-chip-bar'}});
  const parts = [];
  let bestKey = null, bestVal = -Infinity;
  Object.keys(ACTION_COLOR).forEach(ak => {{
    const v = policy[ak];
    if (v === undefined) return;
    const seg = htmlEl('span', {{class: 'policy-chip-seg'}});
    seg.style.width = (Math.max(v, 0) * 100) + '%';
    seg.style.background = ACTION_COLOR[ak];
    bar.appendChild(seg);
    parts.push(`${{ACTION_LABEL[ak]}} ${{(v * 100).toFixed(0)}}%`);
    if (v > bestVal) {{ bestVal = v; bestKey = ak; }}
  }});
  wrap.title = parts.join(' · ');
  wrap.appendChild(bar);
  wrap.appendChild(htmlEl('span', {{class: 'policy-chip-label'}}, `${{ACTION_LABEL[bestKey] || bestKey}} ${{(bestVal * 100).toFixed(0)}}%`));
  return wrap;
}}

function formatScalar(v) {{
  if (typeof v !== 'number' || Number.isInteger(v)) return String(v);
  return v.toFixed(3).replace(/0+$/, '').replace(/\\.$/, '');
}}

// Generic key-value table for checks whose `data` is a flat list of per-scenario records
// without a bespoke chart above (SLOW checks: vpip_adapts_to_style, bb100_vs_standard_fields,
// beats_frozen_predecessor, beats_offformula_stress, no_nan_or_crash) -- every check's raw data
// is visible somewhere, even ones that don't warrant a custom visualization.
function genericTable(data) {{
  if (!data || !data.length) return null;
  const keys = [];
  data.forEach(row => Object.keys(row).forEach(k => {{ if (!keys.includes(k)) keys.push(k); }}));
  const table = htmlEl('table', {{class: 'kv-table'}});
  const head = htmlEl('tr');
  keys.forEach(k => head.appendChild(htmlEl('th', {{}}, k.replace(/_/g, ' '))));
  table.appendChild(head);
  data.forEach(row => {{
    const tr = htmlEl('tr');
    keys.forEach(k => {{
      const v = row[k];
      const td = htmlEl('td', {{}});
      if (v === null || v === undefined) td.textContent = '—';
      else if (typeof v === 'boolean') {{
        td.textContent = v ? '✓' : '✗';
        td.style.color = v ? 'var(--good)' : 'var(--critical)';
      }}
      else if (isPolicyLike(v)) td.appendChild(policyChip(v));
      else if (typeof v === 'object') {{
        td.classList.add('truncate');
        td.textContent = JSON.stringify(v);
        td.title = JSON.stringify(v, null, 2);
      }}
      else if (typeof v === 'number') td.textContent = formatScalar(v);
      else td.textContent = String(v);
      tr.appendChild(td);
    }});
    table.appendChild(tr);
  }});
  const scroller = htmlEl('div', {{}});
  scroller.style.overflowX = 'auto';
  scroller.appendChild(table);
  return scroller;
}}

const root = document.getElementById('charts');
const byId = {{}};
RESULTS.checks.forEach(c => byId[c.id] = c);
const actionKeys = RESULTS.action_space;
const consumed = new Set();

function addCard(id, title, xLabel, yLabel, valueField, ringField, kind) {{
  const c = byId[id];
  if (!c) return;
  consumed.add(id);
  if (!c.data || !c.data.length) {{
    root.appendChild(card(title, id, c.detail, c.status));
    return;
  }}
  const wrap = card(title, id, c.detail, c.status);
  if (kind === 'line') wrap.appendChild(lineChart(c.data, actionKeys));
  else if (kind === 'heatmap') wrap.appendChild(heatmap(c.data, 'stack_bb', 'equity', valueField, {{xLabel: 'stack (bb)', yLabel: 'equity', ringField}}));
  else if (kind === 'categorical') wrap.appendChild(categoricalGrid(c.data, 'stack_bb', 'equity', 'argmax'));
  root.appendChild(wrap);
}}

addCard('equity_ablation_monotonic', 'Equity ablation — full policy vs equity', null, null, null, null, 'line');
addCard('deep_stack_ood_guard', 'Deep-stack OOD guard — P(All-In), ring = All-In is argmax', null, null, 'policy.allin', 'argmax_is_allin', 'heatmap');
addCard('short_stack_polarization', 'Short-stack polarization — P(Call) in shove-or-fold spots', null, null, 'policy.call', null, 'heatmap');
addCard('free_check_low_fold', 'Free-check hygiene — raw P(Fold) when call = 0', null, null, 'policy.fold', null, 'heatmap');
addCard('action_diversity', 'Action diversity — argmax action by equity x stack', null, null, null, null, 'categorical');

// air/nuts as a paired two-col card
(function() {{
  const air = byId['air_folds_mostly'], nuts = byId['nuts_aggressive_mostly'];
  if (!air || !nuts) return;
  consumed.add('air_folds_mostly'); consumed.add('nuts_aggressive_mostly');
  const wrap = htmlEl('div', {{class: 'card'}});
  wrap.appendChild(htmlEl('h2', {{}}, `Air / Nuts spot checks`));
  wrap.appendChild(htmlEl('p', {{class: 'issue'}}, 'guards: V14 spot-test baseline'));
  const two = htmlEl('div', {{class: 'two-col'}});
  const airCol = htmlEl('div');
  airCol.appendChild(htmlEl('p', {{class: 'detail'}}, `Air (~12% eq) — P(Fold) <span class="badge ${{air.status}}">${{air.status}}</span>`));
  airCol.appendChild(barChart(air.data.map(d => ({{stack_bb: d.stack_bb, v: d.policy.fold}})), 'stack_bb', 'v', 'var(--c-fold)', 'P(Fold)'));
  const nutsCol = htmlEl('div');
  nutsCol.appendChild(htmlEl('p', {{class: 'detail'}}, `Nuts (~92% eq) — P(All-In) <span class="badge ${{nuts.status}}">${{nuts.status}}</span>`));
  nutsCol.appendChild(barChart(nuts.data.map(d => ({{stack_bb: d.stack_bb, v: d.policy.allin}})), 'stack_bb', 'v', 'var(--c-allin)', 'P(All-In)'));
  two.appendChild(airCol); two.appendChild(nutsCol);
  wrap.appendChild(two);
  root.appendChild(wrap);
}})();

// Every remaining check (SLOW checks with flat-record data, SKIPs, anything not explicitly
// chart-rendered above) still gets a card -- a generic key/value table when there's data,
// otherwise just the status + detail text. Guarantees no check is silently omitted.
RESULTS.checks.forEach(c => {{
  if (consumed.has(c.id)) return;
  const wrap = card(c.id, null, c.detail, c.status);
  const table = genericTable(c.data);
  if (table) wrap.appendChild(table);
  root.appendChild(wrap);
}});
</script>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results_json')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    with open(args.results_json, 'r') as f:
        results = json.load(f)

    n_pass = sum(1 for c in results['checks'] if c['status'] == 'PASS')
    n_warn = sum(1 for c in results['checks'] if c['status'] == 'WARN')
    n_fail = sum(1 for c in results['checks'] if c['status'] == 'FAIL')
    n_skip = sum(1 for c in results['checks'] if c['status'] == 'SKIP')

    html = TEMPLATE.format(
        title=f"model_verify: {results['version']}",
        version=results['version'],
        weights=results['weights'],
        action_space=', '.join(results['action_space']),
        n_pass=n_pass, n_warn=n_warn, n_fail=n_fail, n_skip=n_skip,
        results_json=json.dumps(results),
    )

    out = args.out or os.path.splitext(args.results_json)[0] + '_report.html'
    with open(out, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"wrote {out}")


if __name__ == '__main__':
    main()
