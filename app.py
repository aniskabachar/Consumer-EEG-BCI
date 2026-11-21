"""NeuroVigi - Real-Time EEG Brain State Monitor."""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time, sys
from pathlib import Path
from sklearn.metrics import roc_curve, auc as sk_auc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backend.inference import BCIPredictor
from backend.simulator import EEGSimulator

st.set_page_config(
    page_title="NeuroVigi · EEG Monitor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODELS_DIR = Path("./processed_data/saved_models")
DATA_PATH  = "./processed_data/seed_vig_5ch_processed.npz"
CH_NAMES   = ["FT7", "FT8", "T7", "PZ", "O2"]
CH_COLORS  = ["#00d4ff", "#a78bfa", "#34d399", "#fb923c", "#f472b6"]
LOSO = {
    0:0.6415, 1:0.8045, 2:0.8998, 3:0.6864, 4:0.7637,
    5:0.6968, 6:0.8168, 7:0.8544, 8:0.9291, 9:0.6976,
    10:0.8635, 11:0.6894, 12:0.8044, 13:0.9159, 14:0.6997,
    15:0.7758, 16:0.7620, 17:0.8642, 18:0.8103, 19:0.8028,
    20:0.8288, 21:0.9022, 22:0.8206,
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
.stApp{background:#060a18;}
html,body,[class*=css]{font-family:'Space Grotesk',sans-serif;}
section[data-testid=stSidebar]{background:#070b1c!important;border-right:1px solid #101e34;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding-top:1.2rem!important;}
.brand{padding-bottom:14px;border-bottom:1px solid #101e34;margin-bottom:16px;}
.bt{font-size:1.4rem;font-weight:700;color:#e8f0fe;margin:0;letter-spacing:-0.02em;}
.bt span{color:#00d4ff;}
.bs{font-size:0.62rem;color:#1a2e48;letter-spacing:0.2em;text-transform:uppercase;margin-top:3px;}
.badge{display:inline-flex;align-items:center;gap:6px;border-radius:20px;padding:3px 11px;font-size:0.68rem;font-weight:700;letter-spacing:0.12em;}
.blv{background:rgba(0,230,118,.08);border:1px solid rgba(0,230,118,.25);color:#00e676;}
.bof{background:rgba(20,36,60,.2);border:1px solid rgba(20,36,60,.5);color:#1a2e48;}
.dot{width:6px;height:6px;border-radius:50%;display:inline-block;}
.dlv{background:#00e676;animation:blink 1s infinite;}
.dof{background:#1a2e48;}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.15}}
.card{background:linear-gradient(145deg,#090f1e,#0b1426);border:1px solid #111e32;border-radius:13px;padding:16px 20px;height:100%;}
.ct{font-size:0.58rem;font-weight:700;letter-spacing:0.2em;text-transform:uppercase;color:#1a2e48;margin-bottom:7px;}
.cv{font-family:'JetBrains Mono',monospace;font-size:1.75rem;font-weight:600;color:#e8f0fe;line-height:1.1;}
.cs{font-size:0.67rem;color:#1a2e48;margin-top:4px;}
.sbox{border-radius:13px;padding:22px 18px;text-align:center;display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:115px;}
.sf{background:linear-gradient(145deg,#051610,#071e16);border:1px solid rgba(0,230,118,.28);box-shadow:0 0 28px rgba(0,230,118,.05);}
.sd{background:linear-gradient(145deg,#160508,#1e070c);border:1px solid rgba(255,23,68,.28);box-shadow:0 0 28px rgba(255,23,68,.06);}
.si{background:linear-gradient(145deg,#090f1e,#0b1426);border:1px solid #111e32;}
.stf{font-size:1.7rem;font-weight:700;color:#00e676;letter-spacing:.06em;animation:ag 2s ease-in-out infinite;}
.std{font-size:1.7rem;font-weight:700;color:#ff1744;letter-spacing:.06em;animation:ar 1.3s ease-in-out infinite;}
.sti{font-size:1.2rem;font-weight:500;color:#1a2e48;letter-spacing:.06em;}
@keyframes ag{0%,100%{text-shadow:0 0 8px rgba(0,230,118,.3)}50%{text-shadow:0 0 26px rgba(0,230,118,.75),0 0 48px rgba(0,230,118,.25)}}
@keyframes ar{0%,100%{text-shadow:0 0 8px rgba(255,23,68,.3)}50%{text-shadow:0 0 30px rgba(255,23,68,.85),0 0 52px rgba(255,23,68,.3)}}
.conf{font-family:'JetBrains Mono',monospace;font-size:0.74rem;color:#1a2e48;margin-top:6px;}
.sec{font-size:0.57rem;font-weight:700;letter-spacing:0.22em;text-transform:uppercase;color:#111e32;border-bottom:1px solid #090f1e;padding-bottom:4px;margin:12px 0 8px 0;}
.row{display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #080e1c;font-size:0.75rem;}
.row .l{color:#1a2e48;}.row .v{color:#00d4ff;font-family:'JetBrains Mono',monospace;font-weight:600;}
.pb{margin:5px 0;}
.pbl{display:flex;justify-content:space-between;font-size:0.7rem;margin-bottom:3px;}
.pbt{background:#080e1c;border-radius:4px;height:5px;overflow:hidden;}
.pbf{height:100%;border-radius:4px;transition:width .35s ease;}
h1,h2,h3{color:#e8f0fe!important;}
p,label{color:#7a90b0;}
.stButton>button{background:#090f1e!important;border:1px solid #111e32!important;color:#c0d0e8!important;border-radius:10px!important;font-weight:600!important;font-size:0.8rem!important;width:100%!important;transition:all .2s!important;}
.stButton>button:hover{border-color:#00d4ff!important;color:#00d4ff!important;}
button[data-baseweb=tab]{color:#1a2e48!important;font-size:0.76rem!important;font-weight:700!important;letter-spacing:.1em!important;}
button[data-baseweb=tab][aria-selected=true]{color:#00d4ff!important;}
div[data-baseweb=tab-highlight]{background:#00d4ff!important;}
.stSelectbox>div>div{background:#090f1e!important;border-color:#111e32!important;}
</style>
"""

for k, v in {"running":False,"epoch_idx":0,"history":[],"eeg_buf":[],"subject":8,"last_subject":-1}.items():
    if k not in st.session_state:
        st.session_state[k] = v

@st.cache_resource(show_spinner="Loading 1.1 GB EEG dataset…")
def _sim():  return EEGSimulator(DATA_PATH)

@st.cache_resource(show_spinner="Initialising inference engine…")
def _pred(): return BCIPredictor()

_DK  = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#02080f",
            font=dict(color="#2a4060", family="JetBrains Mono"))
_M   = dict(l=54, r=12, t=14, b=36)   # default margin

def eeg_chart(buf):
    fig = go.Figure()
    if not buf:
        fig.update_layout(**_DK, height=275, margin=_M,
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            annotations=[dict(text="◉  Press START to stream live EEG",
                x=.5, y=.5, xref="paper", yref="paper", showarrow=False,
                font=dict(color="#111e32", size=13, family="Space Grotesk"))])
        return fig
    sig = np.concatenate(buf, axis=0)
    t   = np.arange(len(sig)) / 200.0
    OFF = 5.5
    for i, (name, col) in enumerate(zip(CH_NAMES, CH_COLORS)):
        ch = sig[:, i]
        y  = (ch - ch.mean()) / (ch.std() + 1e-8) * 1.4 + i * OFF
        fig.add_trace(go.Scatter(x=t, y=y, mode="lines",
            line=dict(color=col, width=0.9), name=name))
    fig.update_layout(**_DK, height=285, margin=_M,
        legend=dict(orientation="h", y=1.07, x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        xaxis=dict(title=dict(text="time (s)", font=dict(color="#111e32", size=10)),
                   gridcolor="#06101e", tickfont=dict(color="#111e32", size=9),
                   color="#111e32", range=[t[0], t[-1]]),
        yaxis=dict(tickmode="array", tickvals=[i*OFF for i in range(5)],
                   ticktext=CH_NAMES, gridcolor="#06101e",
                   tickfont=dict(color="#2a4060", size=10), color="#111e32"))
    return fig

def prob_chart(hist):
    fig = go.Figure()
    if not hist:
        fig.update_layout(**_DK, height=130, margin=_M,
            xaxis=dict(visible=False), yaxis=dict(visible=False))
        return fig
    h  = hist[-60:]
    xs = list(range(len(h)))
    pf = [x["probs"][0] for x in h]
    pd = [x["probs"][1] for x in h]
    fig.add_trace(go.Scatter(x=xs, y=pf, mode="lines", fill="tozeroy",
        line=dict(color="#00e676", width=1.5), fillcolor="rgba(0,230,118,.08)", name="Focused"))
    fig.add_trace(go.Scatter(x=xs, y=pd, mode="lines", fill="tozeroy",
        line=dict(color="#ff1744", width=1.5), fillcolor="rgba(255,23,68,.08)", name="Drowsy"))
    fig.add_hline(y=.5, line=dict(color="#111e32", width=1, dash="dot"))
    fig.update_layout(**_DK, height=145, margin=dict(l=40, r=12, t=8, b=30),
        xaxis=dict(gridcolor="#06101e", tickfont=dict(color="#111e32", size=9), color="#111e32"),
        yaxis=dict(range=[0,1], gridcolor="#06101e",
                   tickfont=dict(color="#111e32", size=9), color="#111e32",
                   title=dict(text="P", font=dict(color="#111e32", size=10))),
        legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)", orientation="h", y=1.1, x=0))
    return fig

def per_subject_fig():
    subs = sorted(LOSO)
    accs = [LOSO[s]*100 for s in subs]
    cols = ["#00e676" if a>=80 else "#fb923c" if a>=70 else "#ff4757" for a in accs]
    mean = float(np.mean(accs))
    fig  = go.Figure()
    fig.add_trace(go.Bar(x=[f"S{s:02d}" for s in subs], y=accs,
        marker=dict(color=cols, line=dict(width=0)),
        hovertemplate="Subject %{x}<br>Acc: %{y:.1f}%<extra></extra>"))
    fig.add_hline(y=mean, line=dict(color="#00d4ff", width=1.5, dash="dash"),
        annotation_text=f"  Mean {mean:.1f}%",
        annotation_font=dict(color="#00d4ff", size=10))
    fig.update_layout(**_DK, height=275, margin=_M, showlegend=False,
        xaxis=dict(gridcolor="#06101e", tickfont=dict(color="#2a4060", size=10), color="#111e32"),
        yaxis=dict(range=[55,100], gridcolor="#06101e",
                   tickfont=dict(color="#2a4060", size=10), color="#111e32",
                   title=dict(text="Accuracy (%)", font=dict(color="#2a4060", size=10))))
    return fig

def roc_fig():
    fig = go.Figure()
    try:
        r  = np.load("./roc_data.npz")
        b  = np.load("./baseline_roc_data.npz")
        f1,t1,_ = roc_curve(r["y_true"], r["y_prob"])
        f2,t2,_ = roc_curve(b["y_true"], b["y_prob"])
        a1,a2   = sk_auc(f1,t1), sk_auc(f2,t2)
        fig.add_trace(go.Scatter(x=f1, y=t1, mode="lines", fill="tozeroy",
            line=dict(color="#00d4ff", width=2.5), fillcolor="rgba(0,212,255,.05)",
            name=f"SEN-CNN-LSTM  AUC={a1:.3f}"))
        fig.add_trace(go.Scatter(x=f2, y=t2, mode="lines",
            line=dict(color="#f472b6", width=1.8, dash="dash"),
            name=f"Baseline  AUC={a2:.3f}"))
    except Exception:
        pass
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", showlegend=False,
        line=dict(color="#111e32", width=1, dash="dot")))
    fig.update_layout(**_DK, height=275, margin=_M,
        xaxis=dict(title="False Positive Rate", gridcolor="#06101e",
                   tickfont=dict(color="#2a4060"), color="#111e32",
                   title_font=dict(color="#2a4060", size=10)),
        yaxis=dict(title="True Positive Rate", gridcolor="#06101e",
                   tickfont=dict(color="#2a4060"), color="#111e32",
                   title_font=dict(color="#2a4060", size=10)),
        legend=dict(x=.5, y=.05, bgcolor="rgba(2,8,15,.9)", font=dict(color="#7a90b0", size=10)))
    return fig

def cm_fig():
    cm  = np.array([[93055,13255],[19803,36704]])
    txt = [[f"{v:,}" for v in row] for row in cm]
    fig = go.Figure(go.Heatmap(z=cm,
        x=["Pred: Alert","Pred: Drowsy"], y=["True: Alert","True: Drowsy"],
        colorscale=[[0,"#02080f"],[.5,"#081c38"],[1,"#004898"]],
        text=txt, texttemplate="%{text}",
        textfont=dict(size=16, color="white", family="JetBrains Mono"),
        showscale=False))
    fig.update_layout(**_DK, height=250, margin=dict(l=100,r=12,t=14,b=60),
        xaxis=dict(tickfont=dict(color="#7a90b0",size=11), color="#111e32", side="bottom"),
        yaxis=dict(tickfont=dict(color="#7a90b0",size=11), color="#111e32"))
    return fig

def noise_fig():
    nx = [0, 5, 10, 20, 30, 50]
    ny = [84.17, 84.15, 84.10, 83.84, 83.26, 81.13]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=nx, y=ny, mode="lines+markers", fill="tozeroy",
        line=dict(color="#00d4ff", width=2.5), fillcolor="rgba(0,212,255,.05)",
        marker=dict(size=8, color="#060a18", line=dict(color="#00d4ff", width=2.5)),
        hovertemplate="Noise: %{x}%<br>Acc: %{y:.2f}%<extra></extra>"))
    for x, y in zip(nx, ny):
        fig.add_annotation(x=x, y=y+.28, text=f"{y:.2f}%",
            font=dict(color="#2a4060", size=9), showarrow=False)
    fig.add_hline(y=50, line=dict(color="#111e32", width=1, dash="dot"))
    fig.update_layout(**_DK, height=250, margin=_M, showlegend=False,
        xaxis=dict(title="Noise Level (%)", gridcolor="#06101e",
                   tickfont=dict(color="#2a4060"), color="#111e32",
                   title_font=dict(color="#2a4060", size=10)),
        yaxis=dict(range=[78,87], title="Accuracy (%)", gridcolor="#06101e",
                   tickfont=dict(color="#2a4060"), color="#111e32",
                   title_font=dict(color="#2a4060", size=10)))
    return fig


def main():
    st.markdown(CSS, unsafe_allow_html=True)
    sim  = _sim()
    pred = _pred()
    running = st.session_state.running
    hist    = st.session_state.history
    buf     = st.session_state.eeg_buf
    last    = hist[-1] if hist else None

    # ── Sidebar ─────────────────────────────────────────────────────────
    with st.sidebar:
        lv = "blv" if running else "bof"
        dv = "dlv" if running else "dof"
        st.markdown(f"""
        <div class="brand">
          <div class="bt">🧠 Neuro<span>Vigi</span></div>
          <div class="bs">EEG Brain State Monitor</div>
        </div>
        <div style="margin-bottom:18px">
          <span class="badge {lv}"><span class="dot {dv}"></span>
          {"LIVE" if running else "STANDBY"}</span>
        </div>""", unsafe_allow_html=True)

        subject = st.selectbox("SUBJECT", sim.subjects,
            index=sim.subjects.index(8) if 8 in sim.subjects else 0,
            format_func=lambda x: f"Subject {x:02d}  —  {LOSO[x]*100:.1f}% acc")
        noise = st.slider("NOISE INJECTION", 0.0, 0.50, 0.0, 0.05, format="%.2f",
            help="Adds Gaussian noise to simulate cheap consumer EEG hardware")
        speed = st.select_slider("SPEED", [0.5,1.0,1.5,2.0,3.0], value=1.0,
            format_func=lambda x: f"{x}×")

        c1, c2 = st.columns(2)
        with c1:
            if not running and st.button("▶  START", use_container_width=True):
                if subject != st.session_state.last_subject:
                    st.session_state.epoch_idx = 0
                    st.session_state.history   = []
                    st.session_state.eeg_buf   = []
                st.session_state.last_subject = subject
                st.session_state.subject      = subject
                st.session_state.running      = True
                st.rerun()
        with c2:
            if running and st.button("⏹  STOP", use_container_width=True):
                st.session_state.running = False
                st.rerun()

        bal = sim.balance(subject)
        st.markdown(f"""
        <div class="sec">MODEL INFO</div>
        <div class="row"><span class="l">Subject</span><span class="v">#{subject:02d}</span></div>
        <div class="row"><span class="l">LOSO Accuracy</span><span class="v">{LOSO[subject]*100:.1f}%</span></div>
        <div class="row"><span class="l">Overall Acc</span><span class="v">79.7%</span></div>
        <div class="row"><span class="l">AUC-ROC</span><span class="v">0.817</span></div>
        <div class="row"><span class="l">Parameters</span><span class="v">258 K</span></div>
        <div class="row"><span class="l">Device</span><span class="v">{pred.device_label}</span></div>
        <div class="sec">SUBJECT DATA</div>
        <div class="row"><span class="l">Total epochs</span><span class="v">{sim.n(subject):,}</span></div>
        <div class="row"><span class="l">Alert</span><span class="v">{bal["alert"]:.0f}%</span></div>
        <div class="row"><span class="l">Drowsy</span><span class="v">{bal["drowsy"]:.0f}%</span></div>
        <div class="row"><span class="l">Noise</span><span class="v">{noise*100:.0f}%</span></div>
        <div class="row"><span class="l">Epochs seen</span><span class="v">{st.session_state.epoch_idx}</span></div>
        """, unsafe_allow_html=True)

    # ── Tabs ─────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "  🟢  LIVE MONITOR  ",
        "  📊  ANALYSIS  ",
        "  ℹ️  ABOUT  ",
    ])

    with tab1:
        c1, c2, c3, c4 = st.columns([2.2, 1.8, 1, 1])
        with c1:
            if last:
                sc = "sf" if last["pred"]==0 else "sd"
                tc = "stf" if last["pred"]==0 else "std"
                nm = "FOCUSED" if last["pred"]==0 else "DROWSY"
                ic = "◉" if last["pred"]==0 else "⚠"
                st.markdown(f"""
                <div class="sbox {sc}">
                  <div class="ct">BRAIN STATE</div>
                  <div class="{tc}">{ic} {nm}</div>
                  <div class="conf">{last["conf"]*100:.1f}% confidence · epoch #{st.session_state.epoch_idx}</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="sbox si">
                  <div class="ct">BRAIN STATE</div>
                  <div class="sti">○  AWAITING SIGNAL</div>
                  <div class="conf">press START to begin live stream</div>
                </div>""", unsafe_allow_html=True)
        with c2:
            if last:
                pf = last["probs"][0]*100
                pd = last["probs"][1]*100
                st.markdown(f"""
                <div class="card">
                  <div class="ct">PROBABILITIES</div>
                  <div class="pb">
                    <div class="pbl">
                      <span style="color:#00e676;font-size:.7rem">FOCUSED</span>
                      <span style="color:#00e676;font-family:'JetBrains Mono',monospace;font-size:.7rem">{pf:.1f}%</span>
                    </div>
                    <div class="pbt"><div class="pbf" style="width:{pf:.0f}%;background:#00e676"></div></div>
                  </div>
                  <div class="pb" style="margin-top:9px">
                    <div class="pbl">
                      <span style="color:#ff1744;font-size:.7rem">DROWSY</span>
                      <span style="color:#ff1744;font-family:'JetBrains Mono',monospace;font-size:.7rem">{pd:.1f}%</span>
                    </div>
                    <div class="pbt"><div class="pbf" style="width:{pd:.0f}%;background:#ff1744"></div></div>
                  </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""<div class="card"><div class="ct">PROBABILITIES</div>
                <div class="cv" style="color:#111e32;margin-top:8px">—</div></div>""",
                unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="card">
              <div class="ct">SUBJECT ACC</div>
              <div class="cv">{LOSO[subject]*100:.1f}%</div>
              <div class="cs">LOSO validated</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            if hist:
                ok = sum(1 for h in hist if h["pred"]==h["true"])
                sa = ok/len(hist)*100
                st.markdown(f"""
                <div class="card">
                  <div class="ct">SESSION ACC</div>
                  <div class="cv">{sa:.1f}%</div>
                  <div class="cs">{len(hist)} epochs</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""<div class="card"><div class="ct">SESSION ACC</div>
                <div class="cv" style="color:#111e32">—</div></div>""",
                unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec">5-CHANNEL EEG OSCILLOSCOPE · 200 Hz · 2-SECOND EPOCHS</div>',
            unsafe_allow_html=True)
        st.plotly_chart(eeg_chart(buf), use_container_width=True, config={"displayModeBar":False})

        st.markdown('<div class="sec">PREDICTION PROBABILITY STREAM</div>', unsafe_allow_html=True)
        st.plotly_chart(prob_chart(hist), use_container_width=True, config={"displayModeBar":False})

        if hist:
            segs = "".join([
                f'<div title="{"Focused" if h["pred"]==0 else "Drowsy"} {h["conf"]*100:.0f}%"' +
                f' style="flex:1;min-width:3px;background:{"#00e676" if h["pred"]==0 else "#ff1744"};' +
                'opacity:.8;border-radius:2px"></div>'
                for h in hist[-80:]
            ])
            fp = sum(1 for h in hist if h["pred"]==0)/len(hist)*100
            st.markdown(f"""
            <div class="sec">PREDICTION TIMELINE · LAST {min(len(hist),80)} EPOCHS</div>
            <div style="display:flex;gap:2px;height:18px;overflow:hidden;border-radius:5px;margin-bottom:5px">{segs}</div>
            <div style="display:flex;justify-content:space-between;font-size:.66rem;color:#111e32">
              <span>← earlier</span>
              <span><span style="color:#00e676">■</span> Focused {fp:.0f}%
                &nbsp;&nbsp;<span style="color:#ff1744">■</span> Drowsy {100-fp:.0f}%</span>
              <span>now →</span>
            </div>""", unsafe_allow_html=True)

    with tab2:
        st.markdown("### 📊 Model Evaluation Results")
        st.markdown("<br>", unsafe_allow_html=True)
        m1,m2,m3,m4 = st.columns(4)
        with m1: st.markdown('<div class="card"><div class="ct">ACCURACY</div><div class="cv">79.7%</div><div class="cs">LOSO-CV · 23 subjects</div></div>', unsafe_allow_html=True)
        with m2: st.markdown('<div class="card"><div class="ct">F1 SCORE</div><div class="cv">0.769</div><div class="cs">macro average</div></div>', unsafe_allow_html=True)
        with m3: st.markdown('<div class="card"><div class="ct">AUC-ROC</div><div class="cv">0.817</div><div class="cs">baseline 0.751</div></div>', unsafe_allow_html=True)
        with m4: st.markdown('<div class="card"><div class="ct">MEAN ± STD</div><div class="cv">±8.0%</div><div class="cs">per-subject</div></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        ca, cb = st.columns(2)
        with ca:
            st.markdown('<div class="sec">PER-SUBJECT LOSO ACCURACY</div>', unsafe_allow_html=True)
            st.plotly_chart(per_subject_fig(), use_container_width=True, config={"displayModeBar":False})
        with cb:
            st.markdown('<div class="sec">ROC CURVE · SEN-CNN-LSTM vs BASELINE</div>', unsafe_allow_html=True)
            st.plotly_chart(roc_fig(), use_container_width=True, config={"displayModeBar":False})
        cc, cd = st.columns(2)
        with cc:
            st.markdown('<div class="sec">CONFUSION MATRIX</div>', unsafe_allow_html=True)
            st.plotly_chart(cm_fig(), use_container_width=True, config={"displayModeBar":False})
        with cd:
            st.markdown('<div class="sec">NOISE ROBUSTNESS · SEN DENOISER EFFECT</div>', unsafe_allow_html=True)
            st.plotly_chart(noise_fig(), use_container_width=True, config={"displayModeBar":False})

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
### 🧠 How It Works
Classifies **Alert vs Drowsy** from just **5 EEG channels** —
compatible with consumer headsets like Muse or OpenBCI.

**Pipeline:**
1. Raw EEG (200 Hz, 5 temporal/posterior channels: FT7, FT8, T7, PZ, O2)
2. **SEN** (U-Net denoiser) — removes cheap-electrode noise
3. **EEGNet CNN** — spatial + temporal feature extraction
4. **BiLSTM** — models temporal dynamics over 2-second epoch
5. **Temporal Attention** — focuses on informative moments
6. **Classifier** — Alert or Drowsy probability

**Dataset:** SEED-VIG — 23 subjects, ~2 hrs each, simulated driving  
**Validation:** LOSO-CV — trained on 22, tested on 1, repeated 23×  
**Labels:** PERCLOS — % eye closure per 8-second window (>50% = drowsy)
            """)
        with c2:
            st.markdown("""
### 🏗️ Architecture
```
Input  (B x 5ch x 400 samples)
  |
  v
+-------------------------------+
|  SEN   U-Net Denoiser         |
|  101,968 params               |
+-------------+-----------------+
              |
  v
+-------------------------------+
|  EEGNet Block                 |
|  temporal + depthwise conv    |
|  output: (B, 16, 12)          |
+-------------+-----------------+
              |
  v
+-------------------------------+
|  BiLSTM x2   hidden=64        |
|  output: (B, 12, 128)         |
+-------------+-----------------+
              |
  v
+-------------------------------+
|  Temporal Attention           |
|  output: (B, 128)             |
+-------------+-----------------+
              |
  v
+-------------------------------+
|  FC -> Dropout -> Softmax     |
|  {Alert, Drowsy}              |
+-------------------------------+

Total: 258,707 parameters
```
            """)
        st.markdown("""---
**Tech:** PyTorch 2.0 · scikit-learn · Streamlit · Plotly  
**Training:** AdamW · CosineAnnealing LR · 80 epochs/fold · AMP  
**Inference:** <5 ms per 2-second epoch on RTX 3050
        """)

    # ── Real-time loop ────────────────────────────────────────────────────
    if running:
        subj  = st.session_state.subject
        mpath = str(MODELS_DIR / f"fold_{subj+1}_subject_{subj}.pth")
        epoch, true_label = sim.epoch(subj, st.session_state.epoch_idx)
        p_cls, conf, probs = pred.predict(epoch, mpath, noise)
        st.session_state.history.append(
            {"pred": p_cls, "conf": conf, "true": true_label, "probs": probs}
        )
        st.session_state.eeg_buf.append(epoch)
        st.session_state.epoch_idx += 1
        if len(st.session_state.eeg_buf)  > 7:  st.session_state.eeg_buf.pop(0)
        if len(st.session_state.history) > 250:  st.session_state.history.pop(0)
        time.sleep(2.0 / speed)
        st.rerun()


if __name__ == "__main__":
    main()
