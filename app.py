"""NeuroVigi - Real-Time EEG Drowsiness Monitor (Streamlit dashboard)."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
# Model code prints emoji; the default Windows console encoding (cp1252) can't encode them.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:
        pass
from backend.inference import BCIPredictor
from backend.simulator import EEGSimulator

st.set_page_config(page_title="NeuroVigi · EEG Monitor", page_icon="🧠",
                   layout="wide", initial_sidebar_state="expanded")

# Live demo: the v2 leak-free model of each fold, run on that fold's held-out session.
MODELS_DIR   = Path("./processed_data/saved_models_v2")
DATA_PATH    = "./processed_data/seed_vig_5ch_processed.npz"
METRICS_PATH = Path("./processed_data/final_metrics.json")
CH_NAMES  = ["FT7", "FT8", "T7", "PZ", "O2"]
CH_COLORS = ["#38bdf8", "#a78bfa", "#34d399", "#fb923c", "#f472b6"]
FS, STEP_S = 200, 1.0          # epochs are 2 s windows taken every 1 s
SMOOTH_S   = 45                # causal smoothing window used by the final system
EEG_VIEW_S = 8
# Person number of each SEED-VIG session (sorted file order); persons 4 and 5 have two sessions.
PERSON = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 1, 20, 21, 2, 3, 4, 4, 5, 5, 6, 7, 8, 9]

GOOD, BAD, ACCENT = "#22c55e", "#ef4444", "#38bdf8"
TXT, SUB, MUTED, LINE, PANEL = "#e6edf7", "#a3b3cc", "#6b7f9e", "#1c2a44", "#0d1528"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
.stApp{{background:#070c1a;}}
html,body,[class*=css]{{font-family:'Inter',sans-serif;}}
section[data-testid=stSidebar]{{background:#0a1122!important;border-right:1px solid {LINE};}}
#MainMenu,footer{{visibility:hidden;}}
header[data-testid=stHeader]{{background:transparent;}}
.block-container{{padding-top:1.4rem!important;max-width:1400px;}}
h1,h2,h3,h4{{color:{TXT}!important;}}
p,li,label,.stMarkdown{{color:{SUB};}}
.mono{{font-family:'JetBrains Mono',monospace;}}

.topbar{{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;
  padding:14px 20px;margin-bottom:18px;border:1px solid {LINE};border-radius:14px;
  background:linear-gradient(120deg,#0d1730,#0a1224);}}
.brand{{font-size:1.45rem;font-weight:700;color:{TXT};letter-spacing:-.02em;}}
.brand span{{color:{ACCENT};}}
.tag{{font-size:.78rem;color:{MUTED};margin-top:2px;}}
.chips{{display:flex;gap:8px;flex-wrap:wrap;align-items:center;}}
.chip{{font-size:.74rem;color:{SUB};border:1px solid {LINE};border-radius:999px;padding:4px 11px;background:#0b142a;}}
.chip b{{color:{TXT};font-weight:600;}}
.live{{color:{GOOD};border-color:rgba(34,197,94,.45);background:rgba(34,197,94,.08);font-weight:700;}}
.idle{{color:{MUTED};font-weight:700;}}
.dot{{width:7px;height:7px;border-radius:50%;display:inline-block;margin-right:6px;vertical-align:1px;}}
.dlive{{background:{GOOD};animation:blink 1s infinite;}} .didle{{background:{MUTED};}}
@keyframes blink{{0%,100%{{opacity:1}}50%{{opacity:.2}}}}

.card{{background:{PANEL};border:1px solid {LINE};border-radius:14px;padding:16px 18px;height:100%;min-height:128px;}}
.ct{{font-size:.68rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:{MUTED};margin-bottom:8px;}}
.cv{{font-family:'JetBrains Mono',monospace;font-size:1.9rem;font-weight:600;color:{TXT};line-height:1.1;}}
.cs{{font-size:.78rem;color:{SUB};margin-top:6px;}}
.state{{border-radius:14px;padding:18px;text-align:center;min-height:128px;display:flex;flex-direction:column;justify-content:center;}}
.s-ok{{background:linear-gradient(145deg,#06200f,#0a2a16);border:1px solid rgba(34,197,94,.45);}}
.s-bad{{background:linear-gradient(145deg,#2a0a0e,#360d13);border:1px solid rgba(239,68,68,.55);animation:pulse 1.4s infinite;}}
.s-idle{{background:{PANEL};border:1px solid {LINE};}}
@keyframes pulse{{0%,100%{{box-shadow:0 0 0 rgba(239,68,68,0)}}50%{{box-shadow:0 0 26px rgba(239,68,68,.35)}}}}
.sv{{font-size:1.75rem;font-weight:700;letter-spacing:.04em;}}
.alertbar{{border-radius:12px;padding:11px 16px;margin:14px 0 2px 0;font-size:.9rem;font-weight:600;
  background:rgba(239,68,68,.12);border:1px solid rgba(239,68,68,.45);color:#fecaca;}}
.infobar{{border-radius:12px;padding:11px 16px;margin:14px 0 2px 0;font-size:.86rem;
  background:rgba(56,189,248,.07);border:1px solid rgba(56,189,248,.3);color:{SUB};}}

.sec{{font-size:.72rem;font-weight:700;letter-spacing:.16em;text-transform:uppercase;color:{SUB};
  border-bottom:1px solid {LINE};padding-bottom:6px;margin:22px 0 8px 0;}}
.sec span{{color:{MUTED};font-weight:500;letter-spacing:.04em;text-transform:none;}}
.row{{display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #121d33;font-size:.82rem;}}
.row .l{{color:{MUTED};}} .row .v{{color:{TXT};font-family:'JetBrains Mono',monospace;font-weight:600;}}
.side-h{{font-size:.7rem;font-weight:700;letter-spacing:.16em;text-transform:uppercase;color:{MUTED};margin:18px 0 6px 0;}}
.meter{{position:relative;height:8px;border-radius:4px;background:#16213a;margin-top:10px;overflow:visible;}}
.meter .fill{{height:100%;border-radius:4px;transition:width .4s ease;}}
.meter .mark{{position:absolute;left:50%;top:-3px;width:2px;height:14px;background:{TXT};opacity:.7;}}
.strip{{display:flex;gap:1px;height:16px;border-radius:5px;overflow:hidden;background:#0b142a;}}
.strip div{{flex:1;min-width:2px;}}
.striplbl{{font-size:.72rem;color:{MUTED};margin:8px 0 4px 0;display:flex;justify-content:space-between;}}

.stButton>button{{background:#0f1a33!important;border:1px solid {LINE}!important;color:{TXT}!important;
  border-radius:10px!important;font-weight:600!important;width:100%!important;}}
.stButton>button:hover{{border-color:{ACCENT}!important;color:{ACCENT}!important;}}
button[data-baseweb=tab]{{font-size:.86rem!important;font-weight:600!important;}}
button[data-baseweb=tab] p{{color:{SUB}!important;}}
button[data-baseweb=tab][aria-selected=true] p{{color:{ACCENT}!important;}}
div[data-baseweb=tab-highlight]{{background:{ACCENT}!important;}}
</style>
"""

DEFAULTS = {"running": False, "pos": 0, "hist": [], "eeg": None, "subject": 8, "loaded": None}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


@st.cache_resource(show_spinner="Loading EEG dataset (1.2 GB)…")
def _sim():
    return EEGSimulator(DATA_PATH)


@st.cache_resource(show_spinner="Starting inference engine…")
def _pred():
    return BCIPredictor()


@st.cache_data
def _metrics():
    try:
        return json.loads(METRICS_PATH.read_text())
    except Exception:
        return None


# ── Plot helpers ─────────────────────────────────────────────────────────
def style(fig, h, **kw):
    axis = dict(gridcolor="#15213a", zerolinecolor="#15213a", linecolor=LINE,
                tickfont=dict(color=SUB, size=11), title_font=dict(color=SUB, size=11))
    layout = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0a1224", height=h,
                  font=dict(color=SUB, family="Inter"), margin=dict(l=56, r=16, t=16, b=40),
                  legend=dict(orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)",
                              font=dict(color=SUB, size=11)),
                  hoverlabel=dict(bgcolor="#0d1528", font=dict(color=TXT)))
    layout.update(kw)   # per-chart settings (margin, legend, …) override the defaults
    fig.update_layout(**layout)
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


def empty(h, msg):
    fig = go.Figure()
    style(fig, h, xaxis=dict(visible=False), yaxis=dict(visible=False))
    fig.add_annotation(text=msg, x=.5, y=.5, xref="paper", yref="paper", showarrow=False,
                       font=dict(color=MUTED, size=13))
    return fig


def eeg_chart(sig):
    if sig is None:
        return empty(270, "Press ▶ Start to stream EEG from the selected session")
    sig = sig[-EEG_VIEW_S * FS:]
    t = np.arange(len(sig)) / FS - len(sig) / FS
    fig, off = go.Figure(), 5.0
    for i, (name, col) in enumerate(zip(CH_NAMES, CH_COLORS)):
        ch = sig[:, i]
        y = (ch - ch.mean()) / (ch.std() + 1e-8) * 1.2 + (4 - i) * off
        fig.add_trace(go.Scatter(x=t, y=y, mode="lines", name=name, hoverinfo="skip",
                                 line=dict(color=col, width=1)))
    style(fig, 270, showlegend=False)
    fig.update_xaxes(title="seconds (0 = now)", range=[t[0], 0])
    fig.update_yaxes(tickmode="array", tickvals=[(4 - i) * off for i in range(5)], ticktext=CH_NAMES,
                     showgrid=False)
    return fig


def prob_chart(hist):
    if not hist:
        return empty(230, "Drowsiness probability will appear here")
    h = hist[-300:]
    t = [x["t"] for x in h]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=[x["true"] for x in h], mode="lines", line=dict(width=0, shape="hv"),
                             fill="tozeroy", fillcolor="rgba(239,68,68,.10)", name="Drowsy (PERCLOS truth)",
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=t, y=[x["p_raw"] for x in h], mode="lines", name="Raw (per 2 s window)",
                             line=dict(color="#64748b", width=1)))
    fig.add_trace(go.Scatter(x=t, y=[x["p_s"] for x in h], mode="lines", name=f"Smoothed ({SMOOTH_S} s)",
                             line=dict(color=ACCENT, width=3)))
    fig.add_hline(y=.5, line=dict(color="#94a3b8", width=1, dash="dot"),
                  annotation_text="threshold", annotation_font=dict(color=MUTED, size=10))
    style(fig, 230)
    fig.update_xaxes(title="session time (s)")
    fig.update_yaxes(range=[0, 1.02], title="P(drowsy)")
    return fig


def meter(p):
    """Card with the smoothed drowsiness probability and a 0-100% bar with the 50% threshold marked."""
    title = f'<div class="ct">Drowsiness · {SMOOTH_S} s average</div>'
    if p is None:
        return (f'<div class="card">{title}<div class="cv" style="color:{MUTED}">—</div>'
                f'<div class="meter"><div class="mark"></div></div><div class="cs">decision threshold 50%</div></div>')
    col = BAD if p >= .5 else GOOD
    return (f'<div class="card">{title}<div class="cv" style="color:{col}">{p * 100:.0f}%</div>'
            f'<div class="meter"><div class="fill" style="width:{p * 100:.1f}%;background:{col}"></div>'
            f'<div class="mark"></div></div><div class="cs">decision threshold 50%</div></div>')


def strip(vals):
    return "".join(f'<div style="background:{BAD if v else GOOD};opacity:{.85 if v else .55}"></div>'
                   for v in vals)


# ── Analysis figures ────────────────────────────────────────────────────
def progress_fig(m):
    steps = [("Always guess 'alert'", 65.3, "#475569"),
             ("Single model (v2), raw", m["single_raw"]["acc"] * 100, "#64748b"),
             (f"+ {SMOOTH_S} s smoothing", 81.7, "#0ea5e9"),
             ("+ per-session norm model (v3)", 82.6, "#38bdf8"),
             ("+ 4 s window model (v4)", m["final"]["acc"] * 100, GOOD)]
    fig = go.Figure(go.Bar(y=[s[0] for s in steps], x=[s[1] for s in steps], orientation="h",
                           marker=dict(color=[s[2] for s in steps]),
                           text=[f"{s[1]:.1f}%" for s in steps], textposition="outside",
                           textfont=dict(color=TXT, size=12), hoverinfo="skip"))
    style(fig, 260, showlegend=False, margin=dict(l=210, r=50, t=10, b=36))
    fig.update_xaxes(range=[50, 90], title="accuracy on held-out sessions (%)")
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(color=TXT, size=12))
    return fig


def per_session_fig(m):
    acc = np.array(m["final"]["per_subject"]) * 100
    cols = [GOOD if a >= 80 else "#f59e0b" if a >= 70 else BAD for a in acc]
    fig = go.Figure(go.Bar(x=[f"S{i + 1:02d}" for i in range(len(acc))], y=acc, marker=dict(color=cols),
                           customdata=PERSON, hovertemplate="%{x} · person %{customdata}<br>%{y:.1f}%<extra></extra>"))
    fig.add_hline(y=acc.mean(), line=dict(color=ACCENT, width=1.5, dash="dash"),
                  annotation_text=f"mean {acc.mean():.1f}%", annotation_position="bottom left",
                  annotation_font=dict(color=ACCENT, size=11), annotation_bgcolor="rgba(10,18,36,.85)")
    fig.add_hline(y=80, line=dict(color="#94a3b8", width=1, dash="dot"))
    style(fig, 280, showlegend=False)
    fig.update_yaxes(range=[50, 100], title="accuracy (%)")
    return fig


def roc_fig(m):
    fig = go.Figure()
    for key, name, col, dash in (("final", "Final system", GOOD, None),
                                 ("single_raw", "Single model, raw", "#94a3b8", "dash")):
        r = m[key]["roc"]
        fig.add_trace(go.Scatter(x=r["fpr"], y=r["tpr"], mode="lines", name=f"{name} · AUC {m[key]['auc']:.3f}",
                                 line=dict(color=col, width=2.5 if key == "final" else 1.8, dash=dash)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", showlegend=False, hoverinfo="skip",
                             line=dict(color="#334155", width=1, dash="dot")))
    style(fig, 280, legend=dict(x=.42, y=.08, bgcolor="rgba(10,18,36,.9)", font=dict(color=TXT, size=11)))
    fig.update_xaxes(title="false positive rate", range=[0, 1])
    fig.update_yaxes(title="true positive rate", range=[0, 1])
    return fig


def cm_fig(m):
    cm = np.array(m["final"]["confusion"])
    pct = cm / cm.sum(axis=1, keepdims=True) * 100
    txt = [[f"{cm[i, j]:,}<br>{pct[i, j]:.1f}%" for j in range(2)] for i in range(2)]
    fig = go.Figure(go.Heatmap(z=pct, x=["Predicted alert", "Predicted drowsy"], y=["Truly alert", "Truly drowsy"],
                               colorscale=[[0, "#0a1224"], [1, "#0369a1"]], showscale=False, zmin=0, zmax=100,
                               text=txt, texttemplate="%{text}", hoverinfo="skip",
                               textfont=dict(size=15, color="white", family="JetBrains Mono")))
    style(fig, 280, margin=dict(l=100, r=16, t=16, b=40))
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(color=TXT))
    fig.update_xaxes(showgrid=False, tickfont=dict(color=TXT))
    return fig


def noise_fig(m):
    n = m["noise"]
    fig = go.Figure()
    for key, name, col in (("v4", "v4 (4 s windows)", GOOD), ("v2", "v2 (2 s windows)", ACCENT)):
        fig.add_trace(go.Scatter(x=n["levels"], y=n[key], mode="lines+markers", name=name,
                                 line=dict(color=col, width=2.5), marker=dict(size=7),
                                 hovertemplate="noise %{x}%<br>%{y:.1f}%<extra></extra>"))
    style(fig, 280)
    fig.update_xaxes(title="added Gaussian noise (% of signal std)  ·  50% ≈ 6 dB SNR")
    fig.update_yaxes(range=[70, 80], title="raw accuracy (%)")
    return fig


# ── Live step ───────────────────────────────────────────────────────────
def reset(subject, start_s):
    st.session_state.update(subject=subject, pos=int(start_s / STEP_S), hist=[], eeg=None,
                            loaded=(subject, start_s))


def step(sim, pred, noise, n):
    subj = st.session_state.subject
    mpath = str(MODELS_DIR / f"fold_{subj + 1}_subject_{subj}.pth")
    for _ in range(n):
        pos = st.session_state.pos
        if pos >= sim.n(subj):
            st.session_state.running = False
            return
        epoch, true = sim.epoch(subj, pos)
        _, _, probs = pred.predict(epoch, mpath, noise)
        hist = st.session_state.hist
        recent = [x["p_raw"] for x in hist[-(SMOOTH_S - 1):]] + [float(probs[1])]
        p_s = float(np.mean(recent))
        hist.append(dict(t=pos * STEP_S, p_raw=float(probs[1]), p_s=p_s, pred=int(p_s >= .5),
                         true=true, warm=len(recent) >= SMOOTH_S))
        eeg = st.session_state.eeg
        new = epoch if eeg is None else epoch[-int(STEP_S * FS):]
        eeg = new if eeg is None else np.concatenate([eeg, new])[-EEG_VIEW_S * FS:]
        st.session_state.eeg = eeg
        st.session_state.pos = pos + 1
        if len(hist) > 900:
            del hist[:len(hist) - 900]


def card(title, value, sub="", color=TXT):
    return (f'<div class="card"><div class="ct">{title}</div>'
            f'<div class="cv" style="color:{color}">{value}</div><div class="cs">{sub}</div></div>')


def mmss(s):
    return f"{int(s // 60):02d}:{int(s % 60):02d}"


# ── App ─────────────────────────────────────────────────────────────────
def main():
    st.markdown(CSS, unsafe_allow_html=True)
    sim, pred, m = _sim(), _pred(), _metrics()
    per = m["final"]["per_subject"] if m else None

    with st.sidebar:
        st.markdown('<div class="side-h" style="margin-top:0">Session</div>', unsafe_allow_html=True)
        subject = st.selectbox(
            "Recording", sim.subjects, index=sim.subjects.index(8) if 8 in sim.subjects else 0,
            format_func=lambda s: f"Session {s + 1:02d} · person {PERSON[s]}"
                                  + (f" · {per[s] * 100:.1f}%" if per else ""),
            label_visibility="collapsed", disabled=st.session_state.running)
        total_min = sim.n(subject) * STEP_S / 60
        start_min = st.slider("Start at minute", 0, int(total_min), 0, disabled=st.session_state.running,
                              help="Jump ahead, e.g. to a part of the session where the driver gets drowsy.")
        st.markdown('<div class="side-h">Playback</div>', unsafe_allow_html=True)
        speed = st.select_slider("Speed", [1, 2, 5, 10, 30], value=5, format_func=lambda x: f"{x}×",
                                 help="Seconds of EEG processed per real second.")
        noise = st.slider("Added noise", 0.0, 0.5, 0.0, 0.05, format="%.2f",
                          help="Gaussian noise relative to the signal, to mimic cheaper electrodes.")

        if st.session_state.loaded != (subject, start_min * 60) and not st.session_state.running:
            reset(subject, start_min * 60)
        c1, c2 = st.columns(2)
        with c1:
            if st.session_state.running:
                if st.button("⏸ Pause"):
                    st.session_state.running = False
                    st.rerun()
            elif st.button("▶ Start"):
                st.session_state.running = True
                st.rerun()
        with c2:
            if st.button("↺ Reset", disabled=st.session_state.running):
                reset(subject, start_min * 60)
                st.rerun()

        bal = sim.balance(subject)
        rows = [("Session length", f"{total_min:.0f} min"), ("Drowsy share", f"{bal['drowsy']:.0f}%"),
                ("Final-system acc", f"{per[subject] * 100:.1f}%" if per else "—"),
                ("Live model", "v2 · held-out"), ("Device", pred.device_label)]
        st.markdown('<div class="side-h">Details</div>' + "".join(
            f'<div class="row"><span class="l">{a}</span><span class="v">{b}</span></div>' for a, b in rows),
            unsafe_allow_html=True)

    running, hist = st.session_state.running, st.session_state.hist
    last = hist[-1] if hist else None
    badge = ('<span class="chip live"><span class="dot dlive"></span>LIVE</span>' if running else
             '<span class="chip idle"><span class="dot didle"></span>STANDBY</span>')
    acc_chip = f'<span class="chip">Final accuracy <b>{m["final"]["acc"] * 100:.1f}%</b></span>' if m else ""
    st.markdown(f"""
    <div class="topbar">
      <div><div class="brand">🧠 Neuro<span>Vigi</span></div>
           <div class="tag">Real-time driver drowsiness monitor · SEED-VIG · 5 EEG channels (FT7, FT8, T7, PZ, O2)</div></div>
      <div class="chips">{badge}{acc_chip}
        <span class="chip">Tested on <b>unseen sessions</b></span>
        <span class="chip">Session <b>{subject + 1:02d}</b></span></div>
    </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🟢  Live monitor", "📊  Results", "ℹ️  About"])

    with tab1:
        c1, c2, c3, c4 = st.columns([1.5, 1.2, 1, 1])
        with c1:
            if last:
                bad = last["pred"] == 1
                note = "" if last["warm"] else f" · warming up ({len(hist)}/{SMOOTH_S} s)"
                st.markdown(f"""<div class="state {'s-bad' if bad else 's-ok'}">
                  <div class="ct">Driver state</div>
                  <div class="sv" style="color:{BAD if bad else GOOD}">{'⚠ DROWSY' if bad else '● ALERT'}</div>
                  <div class="cs mono">t = {mmss(last['t'])}{note}</div></div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="state s-idle"><div class="ct">Driver state</div>
                  <div class="sv" style="color:{MUTED}">○ Waiting</div>
                  <div class="cs">Pick a session and press ▶ Start</div></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(meter(last["p_s"] if last else None), unsafe_allow_html=True)
        with c3:
            if last:
                tr = last["true"] == 1
                st.markdown(card("Ground truth", "Drowsy" if tr else "Alert", "from eye-closure (PERCLOS)",
                                 BAD if tr else GOOD), unsafe_allow_html=True)
            else:
                st.markdown(card("Ground truth", "—", "from eye-closure (PERCLOS)", MUTED), unsafe_allow_html=True)
        with c4:
            if hist:
                ok = np.mean([h["pred"] == h["true"] for h in hist]) * 100
                st.markdown(card("Agreement so far", f"{ok:.1f}%", f"over {mmss(len(hist) * STEP_S)} of EEG"),
                            unsafe_allow_html=True)
            else:
                st.markdown(card("Agreement so far", "—", "predicted vs truth", MUTED), unsafe_allow_html=True)

        if last and last["pred"] == 1 and last["warm"]:
            run = 0
            for h in reversed(hist):
                if h["pred"] != 1:
                    break
                run += 1
            st.markdown(f'<div class="alertbar">⚠ Drowsiness detected for {run} s. '
                        'In a real vehicle this would trigger a break warning.</div>', unsafe_allow_html=True)
        elif not hist:
            st.markdown('<div class="infobar">This replays a recorded session the model never saw in training. '
                        'Each second, the last 2 s of EEG is scored; the decision uses the average of the last '
                        f'{SMOOTH_S} s, as in the final system.</div>', unsafe_allow_html=True)

        st.markdown(f'<div class="sec">Live EEG <span>· 5 channels · 200 Hz · last {EEG_VIEW_S} s</span></div>',
                    unsafe_allow_html=True)
        st.plotly_chart(eeg_chart(st.session_state.eeg), width="stretch",
                        config={"displayModeBar": False}, key="eeg")
        st.markdown('<div class="sec">Drowsiness probability <span>· shaded = truly drowsy</span></div>',
                    unsafe_allow_html=True)
        st.plotly_chart(prob_chart(hist), width="stretch", config={"displayModeBar": False}, key="prob")

        if hist:
            h = hist[-180:]
            st.markdown(f"""<div class="sec">Timeline <span>· last {len(h)} s · red = drowsy</span></div>
              <div class="striplbl"><span>Model decision</span></div><div class="strip">{strip([x['pred'] for x in h])}</div>
              <div class="striplbl"><span>Ground truth (PERCLOS)</span><span>now →</span></div>
              <div class="strip">{strip([x['true'] for x in h])}</div>""", unsafe_allow_html=True)

    with tab2:
        if not m:
            st.info("Run `python -m evaluation.export_dashboard_metrics` to generate processed_data/final_metrics.json.")
        else:
            f = m["final"]
            k1, k2, k3, k4 = st.columns(4)
            k1.markdown(card("Accuracy", f"{f['acc'] * 100:.1f}%", f"{m['n_sessions']} unseen sessions", GOOD),
                        unsafe_allow_html=True)
            k2.markdown(card("F1 score", f"{f['f1']:.3f}", "macro average"), unsafe_allow_html=True)
            k3.markdown(card("AUC-ROC", f"{f['auc']:.3f}", f"single raw model {m['single_raw']['auc']:.3f}"),
                        unsafe_allow_html=True)
            k4.markdown(card("Sessions ≥ 80%", f"{f['n80']}/{m['n_sessions']}",
                             f"drowsy caught {f['drowsy_recall'] * 100:.0f}% · alert {f['alert_recall'] * 100:.0f}%"),
                        unsafe_allow_html=True)

            st.markdown('<div class="sec">How we got here <span>· every step tested on held-out sessions</span></div>',
                        unsafe_allow_html=True)
            st.plotly_chart(progress_fig(m), width="stretch", config={"displayModeBar": False})
            st.caption("Final system: average of three models (v2, v3, v4) followed by a 45 s causal moving average. "
                       "The earlier 79.7% figure came from an evaluation that let the test session influence "
                       "early stopping, and is no longer used.")

            a, b = st.columns(2)
            with a:
                st.markdown('<div class="sec">Accuracy per session <span>· final system</span></div>',
                            unsafe_allow_html=True)
                st.plotly_chart(per_session_fig(m), width="stretch", config={"displayModeBar": False})
            with b:
                st.markdown('<div class="sec">ROC curve</div>', unsafe_allow_html=True)
                st.plotly_chart(roc_fig(m), width="stretch", config={"displayModeBar": False})
            c, d = st.columns(2)
            with c:
                st.markdown('<div class="sec">Confusion matrix <span>· final system, all sessions</span></div>',
                            unsafe_allow_html=True)
                st.plotly_chart(cm_fig(m), width="stretch", config={"displayModeBar": False})
            with d:
                st.markdown('<div class="sec">Noise robustness <span>· single models, no smoothing</span></div>',
                            unsafe_allow_html=True)
                st.plotly_chart(noise_fig(m), width="stretch", config={"displayModeBar": False})

    with tab3:
        a, b = st.columns([1.1, 1])
        with a:
            st.markdown(f"""
### How it works
NeuroVigi classifies a driver as **alert** or **drowsy** using only **5 EEG channels**:
FT7, FT8, T7 (temporal) and PZ, O2 (parietal/occipital).

1. **Input:** 2 s windows of raw EEG at 200 Hz, one every second.
2. **SEN (U-Net denoiser):** cleans the signal.
3. **EEGNet + BiLSTM + attention:** learns patterns within each window.
4. **Ensemble:** averages three trained variants (2 s windows; per-session normalisation; 4 s windows).
5. **Smoothing:** averages the drowsiness probability over the last {SMOOTH_S} s, since drowsiness
   changes slowly. This means decisions lag by up to about {SMOOTH_S} s.

### Data and validation
- **SEED-VIG:** 23 simulated-driving sessions (~2 h each) from 21 people, obtained from the lab on request.
- **Labels:** PERCLOS (eye-closure ratio from eye tracking); above 0.5 counts as drowsy.
- **Validation:** leave-one-session-out. Each fold trains on 19 sessions, uses 3 to decide when to stop,
  and tests on 1 it has never seen. Persons 4 and 5 have two sessions each; excluding their folds changes
  accuracy by only 0.1 points.
""")
        with b:
            st.markdown("""
### Model
```
EEG  (5 channels × 400 samples)
  │
  ▼  SEN U-Net denoiser   102 K params
  ▼  EEGNet block         temporal conv
  ▼  BiLSTM ×2            hidden 64
  ▼  Temporal attention
  ▼  Classifier           alert / drowsy
                          258,707 params
```
### Limits
- Trained and tested on one lab's research-grade wet electrodes.
- Consumer headsets often use different positions (a Muse uses TP9, AF7, AF8, TP10),
  so results on those are unknown until tested.
- About a third of drowsy periods are still missed.

**Stack:** PyTorch · scikit-learn · Streamlit · Plotly. Training: AdamW, cosine LR,
up to 40 epochs per fold with early stopping, on an RTX 3050.
""")

    if running:
        t0 = time.time()
        step(sim, pred, noise, speed)
        time.sleep(max(0.0, 1.0 - (time.time() - t0)))
        st.rerun()


if __name__ == "__main__":
    main()
