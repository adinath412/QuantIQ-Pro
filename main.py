"""
QuantIQ Pro — One Project, All Roles
Author: Adinath Vitthal More
Run:    streamlit run main.py

BUGS FIXED v2:
  1. idx_data crash on startup - safe init + empty guard
  2+3. Search results disappear on interaction - session_state persistence
  4. Valuation labels wrong - correct Graham/DCF/PE/PB labels shown
  5. Linear Regression not shown - added to ML tab
  6. Scanner text wrong - now shows correct stock count
  7. st.columns(0) crash - guarded
  8+9. Strategy tabs restructured - always visible, cached results
"""
import os, time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv
load_dotenv()

from data       import (get_indices, get_info, get_history, get_ema_signal,
                         calc_val, get_news, scan_ema, ALL_STOCKS, SECTORS,
                         LOTS, fmtN, fmtP, sf, dash, ns, NIFTY50)
from strategies import run_all_strategies, optimise_sma
from ml_models  import (random_walk_test, linear_regression, run_regression,
                         kmeans_regimes, run_all_ml)
from agents     import run_agents

st.set_page_config(page_title="QuantIQ Pro",page_icon="📈",
                   layout="wide",initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
*{font-family:'Inter',sans-serif!important}
#MainMenu,footer,header{visibility:hidden}
.stApp{background:#0a0e1a}
.block-container{padding:0.8rem 1.5rem!important;max-width:100%!important}
.stTabs [data-baseweb="tab-list"]{background:#111827;border-radius:8px;gap:3px;padding:4px}
.stTabs [data-baseweb="tab"]{background:transparent;color:#6b7280;border-radius:6px;font-weight:500;font-size:13px;padding:6px 12px}
.stTabs [aria-selected="true"]{background:#1f2937!important;color:#f9fafb!important}
.stTextInput input{background:#111827!important;border:1px solid #374151!important;color:#f9fafb!important;border-radius:8px!important;font-size:15px!important;padding:10px 14px!important}
.stButton button{background:#10b981;color:#fff;border:none;border-radius:8px;font-weight:700;transition:all 0.2s}
.stButton button:hover{background:#059669}
div[data-testid="stMetricValue"]{color:#f9fafb;font-size:1.1rem;font-weight:700}
div[data-testid="stMetricLabel"]{color:#6b7280;font-size:11px}
div[data-testid="stExpander"]{background:#111827;border:1px solid #1f2937;border-radius:10px}
hr{border-color:#1f2937!important}
.stSelectbox>div>div{background:#111827!important;border:1px solid #374151!important;color:#f9fafb!important}
.stMultiSelect>div>div{background:#111827!important;border:1px solid #374151!important}
</style>""",unsafe_allow_html=True)

def card(html,bg="#111827",br="#1f2937",pad="14px 16px",r="10px"):
    return f'<div style="background:{bg};border:1px solid {br};border-radius:{r};padding:{pad}">{html}</div>'
def row(k,v,vc="#f9fafb"):
    return (f'<div style="display:flex;justify-content:space-between;padding:6px 0;'
            f'border-bottom:1px solid #1f2937;font-size:13px"><span style="color:#6b7280">{k}</span>'
            f'<span style="color:{vc};font-weight:600">{v}</span></div>')
def sec(title):
    st.markdown(f'<div style="font-size:12px;font-weight:700;color:#3b82f6;text-transform:uppercase;'
                f'letter-spacing:0.8px;padding:8px 0 6px;border-bottom:2px solid #1f2937;'
                f'margin-bottom:10px">{title}</div>',unsafe_allow_html=True)
def badge(txt,bg,col):
    return f'<span style="background:{bg};color:{col};font-size:11px;padding:3px 9px;border-radius:20px;font-weight:700">{txt}</span>'

DARK="#0a0e1a"; CARD="#111827"; BORDER="#1f2937"
G="#10b981"; R="#ef4444"; B="#3b82f6"; Y="#f59e0b"

# ══ SESSION STATE INIT ══════════════════════════════════════════
for k,v in [("idx_data",[]),("idx_t",0),("active_sym",""),
            ("run_search",False),("q2_sym",""),("run_q2",False)]:
    if k not in st.session_state: st.session_state[k]=v

# ══ NAVBAR ══════════════════════════════════════════════════════
st.markdown(f'<div style="background:linear-gradient(135deg,#0f172a,#1e293b);border-bottom:2px solid {G};'
            f'padding:12px 20px;margin-bottom:12px;border-radius:12px;display:flex;align-items:center;'
            f'justify-content:space-between"><div><span style="font-size:1.6rem;font-weight:800;color:{G}">📈 QuantIQ</span>'
            f'<span style="color:{G};font-weight:700"> Pro</span>'
            f'<span style="color:#6b7280;font-size:12px;margin-left:10px">AI · ML · Quant · Data Science — NSE India</span></div>'
            f'<div style="font-size:11px;color:#6b7280;display:flex;gap:14px">'
            f'<span>🤖 5 AI Agents</span><span>|</span><span>📊 4 Strategies</span><span>|</span>'
            f'<span>🧠 4 ML Models</span><span>|</span><span>🎯 EMA Scanner</span><span>|</span>'
            f'<a href="https://trendlyne.com" target="_blank" style="color:{B}">trendlyne ↗</a>'
            f'</div></div>',unsafe_allow_html=True)

# ══ INDEX BAR — SAFE INIT ════════════════════════════════════════
if time.time()-st.session_state.idx_t>60:
    try: st.session_state.idx_data=get_indices()
    except: st.session_state.idx_data=[]
    st.session_state.idx_t=time.time()

if st.session_state.idx_data:
    idx_cols=st.columns(len(st.session_state.idx_data))
    for i,idx in enumerate(st.session_state.idx_data):
        with idx_cols[i]:
            up=idx["cp"]>=0; col=G if up else R; arr="▲" if up else "▼"
            st.markdown(f'<div style="background:{CARD};border:1px solid {BORDER};border-left:3px solid {col};'
                        f'border-radius:8px;padding:10px 12px;text-align:center">'
                        f'<div style="font-size:10px;color:#6b7280;font-weight:700;text-transform:uppercase;letter-spacing:0.5px">{idx["name"]}</div>'
                        f'<div style="font-size:1.1rem;font-weight:800;color:#f9fafb;margin:3px 0">{idx["price"]:,.2f}</div>'
                        f'<div style="font-size:12px;font-weight:700;color:{col}">{arr} {abs(idx["cp"]):.2f}%</div></div>',
                        unsafe_allow_html=True)
st.markdown("<br>",unsafe_allow_html=True)

# ══ SEARCH BAR — SESSION STATE PERSISTENCE ═══════════════════════
s1,s2=st.columns([5,1])
with s1: typed=st.text_input("Stock Search",placeholder="🔍  RELIANCE · TCS · HDFCBANK · INFY · TATAMOTORS...",label_visibility="collapsed",key="si")
with s2:
    if st.button("Search",type="primary",use_container_width=True):
        if typed.strip(): st.session_state.active_sym=typed.upper().strip().replace(" ",""); st.session_state.run_search=True

QUICK=["RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK","TATAMOTORS","SBIN","WIPRO","BAJFINANCE","ZOMATO"]
qc_=st.columns(10)
for i,s in enumerate(QUICK):
    with qc_[i]:
        if st.button(s,key=f"q_{s}",use_container_width=True):
            st.session_state.active_sym=s; st.session_state.run_search=True
st.markdown("<hr>",unsafe_allow_html=True)

# ══ MAIN TABS ════════════════════════════════════════════════════
T1,T2,T3,T4=st.tabs(["📊  Stock Analysis","📈  Quant Strategies + ML","🎯  200 EMA Scanner","🏆  Market Overview"])

# ════════════ TAB 1 ═════════════════════════════════════════════
with T1:
    sym=st.session_state.active_sym
    if sym and st.session_state.run_search:
        # FIX BUG 2: reset flag IMMEDIATELY so widget interactions don't retrigger full fetch
        st.session_state.run_search = False
        # FIX BUG 2: store fetched data in session_state keyed by symbol
        with st.spinner(f"Loading {sym}..."):
            info=get_info(sym); df=get_history(sym,"1y","1d")
            ema_r=get_ema_signal(sym); val=calc_val(info) if info else None
        st.session_state[f"cache_info_{sym}"] = info
        st.session_state[f"cache_df_{sym}"]   = df
        st.session_state[f"cache_ema_{sym}"]  = ema_r
        st.session_state[f"cache_val_{sym}"]  = val

    # Load from cache if available
    sym=st.session_state.active_sym
    if sym and f"cache_info_{sym}" in st.session_state:
        info  = st.session_state[f"cache_info_{sym}"]
        df    = st.session_state[f"cache_df_{sym}"]
        ema_r = st.session_state[f"cache_ema_{sym}"]
        val   = st.session_state[f"cache_val_{sym}"]

        # FIX: yfinance .info is unreliable for NSE stocks in newer versions.
        # A stock is VALID if we have price > 0 OR non-empty history.
        # Never reject based on missing longName/symbol keys alone.
        price_check = sf(info.get("currentPrice") or info.get("regularMarketPrice"))
        has_history = (df is not None and not df.empty)
        if price_check <= 0 and not has_history:
            st.error(f"❌ **{sym}** not found or no data available. "
                     f"Check the symbol and try again (e.g. RELIANCE, TCS, HDFCBANK).")
            st.stop()

        def g(k,d="—"):
            v=info.get(k)
            if v is None: return d
            try:
                if np.isnan(float(v)): return d
            except: pass
            return v

        price=sf(g("currentPrice",g("regularMarketPrice",0)))
        # FIX: if .info gave no price, fall back to last close from history
        if price <= 0 and df is not None and not df.empty:
            try: price = float(df["Close"].squeeze().iloc[-1])
            except: pass
        prev=sf(g("regularMarketPreviousClose",price))
        # prev fallback: second-last bar from history
        if prev <= 0 and df is not None and len(df) >= 2:
            try: prev = float(df["Close"].squeeze().iloc[-2])
            except: pass
        if prev <= 0: prev = price
        chg=round(price-prev,2); chgp=round(chg/prev*100 if prev else 0,2)
        up=chg>=0; cg=G if up else R; company=g("longName",g("shortName",sym))

        hc1,hc2=st.columns([3,2])
        with hc1:
            st.markdown(f'<div style="margin-bottom:4px"><span style="font-size:1.7rem;font-weight:800;color:#f9fafb">{company}</span></div>'
                        f'<div style="font-size:12px;color:#6b7280;margin-bottom:12px">'
                        f'<span style="background:#1f2937;color:#9ca3af;padding:2px 8px;border-radius:4px;font-weight:700">{sym} · NSE</span>'
                        f' &nbsp; {g("sector","—")} · {g("industry","")[:28] if g("industry","")!="—" else "—"}'
                        f' &nbsp; <a href="https://www.screener.in/company/{sym}/" target="_blank" style="color:{B};font-size:11px">screener.in ↗</a>'
                        f' &nbsp; <a href="https://trendlyne.com/equity/stock/{sym}/" target="_blank" style="color:{B};font-size:11px">trendlyne ↗</a>'
                        f' &nbsp; <a href="https://www.tickertape.in/stocks/{sym}" target="_blank" style="color:{B};font-size:11px">tickertape ↗</a></div>'
                        f'<div style="font-size:3rem;font-weight:900;color:#f9fafb;letter-spacing:-1.5px;line-height:1">₹{price:,.2f}</div>'
                        f'<div style="font-size:1rem;font-weight:700;color:{cg};margin-top:6px">{"▲" if up else "▼"} ₹{abs(chg):.2f} ({abs(chgp):.2f}%) Today</div>',
                        unsafe_allow_html=True)
        with hc2:
            m1,m2,m3=st.columns(3)
            with m1: st.metric("Day High",f"₹{g('dayHigh','—')}"); st.metric("52W High",f"₹{g('fiftyTwoWeekHigh','—')}")
            with m2: st.metric("Day Low",f"₹{g('dayLow','—')}"); st.metric("52W Low",f"₹{g('fiftyTwoWeekLow','—')}")
            with m3: st.metric("Volume",fmtN(g("volume","—"),"").replace("₹","")); st.metric("Mkt Cap",fmtN(g("marketCap","—")))
        st.markdown("<hr>",unsafe_allow_html=True)

        D1,D2,D3,D4,D5=st.tabs(["🤖 AI Research","📈 Chart","💰 Buy/Sell","📊 Fundamentals","📰 News"])

        # D1 AI
        with D1:
            has_key=bool(os.getenv("GROQ_API_KEY","")) and "your_" not in os.getenv("GROQ_API_KEY","")
            if not has_key:
                st.markdown(f'<div style="background:#1c1a0a;border:1.5px solid #ca8a04;border-radius:10px;padding:12px 16px;margin-bottom:12px">'
                            f'<div style="color:#fbbf24;font-weight:700;font-size:14px;margin-bottom:3px">⚡ Works without Groq key — add key for AI reports</div>'
                            f'<div style="color:#9ca3af;font-size:12px">Get FREE key at <a href="https://console.groq.com" target="_blank" style="color:{B}">console.groq.com</a> → .env.example → .env → Add key → Restart</div>'
                            f'</div>',unsafe_allow_html=True)
            run_btn=st.button("🤖  Run 5 AI Agents — Get Full Research Report",type="primary")
            if run_btn:
                with st.status(f"🤖 5 AI Agents analysing {sym}...",expanded=True) as status:
                    st.write("📰 Agent 1 — Fetching & analysing news...")
                    news_data=get_news(sym); st.write(f"   ✅ {len(news_data)} articles fetched")
                    st.write("📊 Agent 2 — Analysing fundamentals...")
                    st.write("📈 Agent 3 — Reading technical signals...")
                    st.write("⚠️  Agent 4 — Assessing risk factors...")
                    st.write("💰 Agent 5 — Computing fair value...")
                    st.write("🧠 Master Agent — Synthesising report...")
                    ai=run_agents(sym,info,df,news_data,ema_r,val)
                    status.update(label=f"✅ Complete! {'🤖 Groq AI' if ai['has_groq'] else '⚙️ Rule-Based'}",state="complete",expanded=False)
                st.session_state[f"ai_{sym}"]=ai

            ai=st.session_state.get(f"ai_{sym}")
            if ai:
                m_=ai["master"]; fs=m_["score"]; vc=m_["color"]
                bgm={"green":"#052e16","orange":"#1c1a0a","red":"#1e0c0c"}
                brm={"green":"#059669","orange":"#ca8a04","red":"#dc2626"}
                txm={"green":G,"orange":"#fbbf24","red":R}
                ai_lbl=badge("🤖 AI-Powered","#1e3a5f",B) if ai["has_groq"] else badge("⚙️ Rule-Based","#1c1a0a","#fbbf24")
                st.markdown(f'<div style="background:{bgm.get(vc,"#111827")};border:2px solid {brm.get(vc,"#374151")};border-radius:14px;'
                            f'padding:20px 28px;text-align:center;margin-bottom:18px">'
                            f'<div style="font-size:1.6rem;font-weight:900;color:{txm.get(vc,"#f9fafb")};margin-bottom:6px">{m_["verdict"]} &nbsp; {ai_lbl}</div>'
                            f'<div style="font-size:2.8rem;font-weight:900;color:{txm.get(vc,"#f9fafb")}">{fs}<span style="font-size:1.4rem">/100</span></div>'
                            f'<div style="color:#6b7280;font-size:13px;margin-top:4px">AI Research Score — NSE India</div></div>',unsafe_allow_html=True)
                sc_=m_["scores"]
                ags=[("📰 News",sc_["news"],ai["news"]["sent"]),("📊 Fundamental",sc_["fundamental"],ai["fund"]["verdict"]),
                     ("📈 Technical",sc_["technical"],ai["tech"]["direction"]),("⚠️ Risk",sc_["risk"],ai["risk"]["level"]+" Risk"),
                     ("💰 Valuation",sc_["valuation"],ai["val"]["verdict"])]
                ac=st.columns(5)
                for i,(nm,sv,lb) in enumerate(ags):
                    bc=G if sv>=60 else R if sv<40 else Y
                    with ac[i]:
                        st.markdown(f'<div style="background:{CARD};border:1px solid {BORDER};border-radius:10px;padding:12px;text-align:center">'
                                    f'<div style="font-size:11px;color:#6b7280;margin-bottom:5px">{nm}</div>'
                                    f'<div style="font-size:1.5rem;font-weight:900;color:{bc}">{sv}</div>'
                                    f'<div style="font-size:11px;color:#6b7280;margin-top:3px">{lb}</div>'
                                    f'<div style="background:#1f2937;height:4px;border-radius:2px;margin-top:8px;overflow:hidden">'
                                    f'<div style="background:{bc};width:{sv}%;height:100%;border-radius:2px"></div></div></div>',unsafe_allow_html=True)
                st.markdown("<br>",unsafe_allow_html=True)
                sec("🧠 Master Agent — Research Report")
                st.markdown(f'<div style="background:{CARD};border:1px solid {BORDER};border-radius:10px;padding:18px 20px;color:#e5e7eb;font-size:14px;line-height:1.8">{m_["report"].replace(chr(10),"<br>")}</div>',unsafe_allow_html=True)
                st.markdown("<br>",unsafe_allow_html=True)
                sec("📋 Individual Agent Reports")
                ca,cb=st.columns(2)
                with ca:
                    with st.expander(f"📰 News Agent — {ai['news']['sent']} ({ai['news']['score']}/100)"):
                        st.write(f"**Theme:** {ai['news']['theme']}")
                        for b_ in ai["news"].get("pos",[]): st.success(f"✅ {b_}")
                        for b_ in ai["news"].get("neg",[]): st.error(f"⚠️ {b_}")
                        st.caption("🤖 AI" if ai["news"]["powered"] else "⚙️ Rule-based")
                    with st.expander(f"📊 Fundamental Agent — {ai['fund']['verdict']} ({ai['fund']['score']}/100)"):
                        if ai["fund"]["summary"]: st.write(ai["fund"]["summary"])
                        for s_ in ai["fund"]["strengths"]: st.success(f"✅ {s_}")
                        for w_ in ai["fund"]["weaknesses"]: st.error(f"⚠️ {w_}")
                        st.caption("🤖 AI" if ai["fund"]["powered"] else "⚙️ Rule-based")
                    with st.expander(f"⚠️ Risk Agent — {ai['risk']['level']} ({ai['risk']['score']}/100)"):
                        st.write(f"**Volatility:** {ai['risk']['vol']}")
                        for r_ in ai["risk"]["risks"]: st.error(f"🔴 {r_}")
                        for s_ in ai["risk"]["safe"]: st.success(f"🟢 {s_}")
                        st.caption("🤖 AI" if ai["risk"]["powered"] else "⚙️ Rule-based")
                with cb:
                    with st.expander(f"📈 Technical Agent — {ai['tech']['direction']} ({ai['tech']['score']}/100)"):
                        st.write(f"**RSI:** {ai['tech']['rsi']} | **EMA Signal:** {ai['tech']['ema_signal']}")
                        for sg_ in ai["tech"]["signals"]: st.write(f"• {sg_}")
                        st.caption("🤖 AI" if ai["tech"]["powered"] else "⚙️ Rule-based")
                    with st.expander(f"💰 Valuation Agent — {ai['val']['verdict']} | {ai['val']['action']}"):
                        st.write(f"**Thesis:** {ai['val']['thesis']}")
                        st.write(f"**Horizon:** {ai['val']['horizon']}")
                        st.write(f"**Action:** {ai['val']['action']}")
                        st.caption("🤖 AI" if ai["val"]["powered"] else "⚙️ Rule-based")
            else:
                st.markdown(f'<div style="text-align:center;padding:50px 20px"><div style="font-size:3.5rem;margin-bottom:14px">🤖</div>'
                            f'<div style="font-size:1.3rem;font-weight:800;color:#f9fafb;margin-bottom:10px">5 AI Agents ready to analyse {sym}</div>'
                            f'<div style="font-size:13px;color:#6b7280;line-height:1.8">📰 News · 📊 Fundamental · 📈 Technical · ⚠️ Risk · 💰 Valuation · 🧠 Master</div></div>',unsafe_allow_html=True)

        # D2 CHART
        with D2:
            if not df.empty:
                cc1,cc2,cc3=st.columns(3)
                with cc1: period=st.selectbox("Period",["1M","3M","6M","1Y"],index=3,key="prd")
                with cc2: ctype=st.selectbox("Chart",["Candles","Line","Area"],key="ct")
                with cc3: inds=st.multiselect("Indicators",["EMA20","EMA50","EMA200","Bollinger","Volume","RSI","MACD"],default=["EMA50","EMA200"],key="ind")
                nmap={"1M":21,"3M":63,"6M":126,"1Y":252}
                dfp=df.tail(nmap.get(period,252)).copy(); c_=dfp["Close"].squeeze()
                nr=1+("Volume" in inds)+("RSI" in inds)+("MACD" in inds)
                fig=make_subplots(rows=nr,cols=1,shared_xaxes=True,vertical_spacing=0.025,row_heights=[0.55]+[0.15]*(nr-1))
                if ctype=="Candles":
                    fig.add_trace(go.Candlestick(x=dfp.index,open=dfp["Open"].squeeze(),high=dfp["High"].squeeze(),
                        low=dfp["Low"].squeeze(),close=c_,name="Price",
                        increasing=dict(line_color=G,fillcolor="#052e16"),decreasing=dict(line_color=R,fillcolor="#1e0c0c")),row=1,col=1)
                elif ctype=="Line":
                    fig.add_trace(go.Scatter(x=dfp.index,y=c_,name="Price",line=dict(color=G,width=2)),row=1,col=1)
                else:
                    fig.add_trace(go.Scatter(x=dfp.index,y=c_,name="Price",fill="tozeroy",fillcolor="rgba(16,185,129,0.07)",line=dict(color=G,width=2)),row=1,col=1)
                for ek,ec_ in {"EMA20":Y,"EMA50":"#f97316","EMA200":R}.items():
                    if ek in inds and ek in dfp.columns:
                        fig.add_trace(go.Scatter(x=dfp.index,y=dfp[ek].squeeze(),name=ek,line=dict(color=ec_,width=1.8,dash="dash")),row=1,col=1)
                if "Bollinger" in inds and "BB_up" in dfp:
                    fig.add_trace(go.Scatter(x=dfp.index,y=dfp["BB_up"].squeeze(),name="BB Up",line=dict(color="#6366f1",width=1,dash="dot")),row=1,col=1)
                    fig.add_trace(go.Scatter(x=dfp.index,y=dfp["BB_lo"].squeeze(),name="BB Lo",fill="tonexty",fillcolor="rgba(99,102,241,0.05)",line=dict(color="#6366f1",width=1,dash="dot")),row=1,col=1)
                if val and val.get("buy"): fig.add_hline(y=float(val["buy"]),line_dash="dot",line_color=G,line_width=1.5,annotation_text=f"BUY ₹{val['buy']}",annotation_font_color=G,row=1,col=1)
                if val and val.get("fair"): fig.add_hline(y=float(val["fair"]),line_dash="dot",line_color=B,line_width=1.5,annotation_text=f"FAIR ₹{val['fair']}",annotation_font_color=B,row=1,col=1)
                cur=2
                if "Volume" in inds and "Volume" in dfp:
                    vol_=dfp["Volume"].squeeze()
                    vc2_=[G if float(c__)>=float(o) else R for c__,o in zip(dfp["Close"].squeeze(),dfp["Open"].squeeze())]
                    fig.add_trace(go.Bar(x=dfp.index,y=vol_,name="Volume",marker_color=vc2_,opacity=0.7),row=cur,col=1); cur+=1
                if "RSI" in inds and "RSI" in dfp:
                    fig.add_trace(go.Scatter(x=dfp.index,y=dfp["RSI"].squeeze(),name="RSI",line=dict(color="#a855f7",width=2)),row=cur,col=1)
                    fig.add_hline(y=70,line_dash="dash",line_color=R,line_width=0.8,row=cur,col=1)
                    fig.add_hline(y=30,line_dash="dash",line_color=G,line_width=0.8,row=cur,col=1); cur+=1
                if "MACD" in inds and "MACD" in dfp:
                    h_=dfp["MACD_h"].squeeze()
                    fig.add_trace(go.Bar(x=dfp.index,y=h_,name="Hist",marker_color=[G if v>=0 else R for v in h_],opacity=0.8),row=cur,col=1)
                    fig.add_trace(go.Scatter(x=dfp.index,y=dfp["MACD"].squeeze(),name="MACD",line=dict(color=B,width=1.5)),row=cur,col=1)
                    fig.add_trace(go.Scatter(x=dfp.index,y=dfp["MACD_sig"].squeeze(),name="Signal",line=dict(color="#f97316",width=1.5)),row=cur,col=1)
                fig.update_layout(height=560+(nr-1)*100,paper_bgcolor=DARK,plot_bgcolor=DARK,font=dict(color="#f9fafb"),
                    xaxis_rangeslider_visible=False,legend=dict(bgcolor="rgba(0,0,0,0)",font_color="#6b7280",orientation="h",y=1.01,yanchor="bottom"),
                    margin=dict(l=0,r=0,t=8,b=0),hovermode="x unified")
                for i in range(1,nr+1): fig.update_xaxes(gridcolor="#1f2937",row=i,col=1); fig.update_yaxes(gridcolor="#1f2937",row=i,col=1)
                st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":True,"displaylogo":False})
                rsi_v=round(float(df["RSI"].iloc[-1]),1) if "RSI" in df.columns else None
                macd_v=float(df["MACD"].iloc[-1]) if "MACD" in df.columns else 0
                msig_v=float(df["MACD_sig"].iloc[-1]) if "MACD_sig" in df.columns else 0
                e200_v=float(df["EMA200"].iloc[-1]) if "EMA200" in df.columns else None
                atr_v=round(float(df["ATR"].iloc[-1]),2) if "ATR" in df.columns else None
                ic2=st.columns(5)
                with ic2[0]: st.metric("RSI (14)",str(rsi_v) if rsi_v else "—","Oversold 🟢" if rsi_v and rsi_v<30 else "Overbought 🔴" if rsi_v and rsi_v>70 else "Neutral ⚪")
                with ic2[1]: st.metric("MACD","Bullish 🟢" if macd_v>msig_v else "Bearish 🔴")
                with ic2[2]: st.metric("Trend","Above EMA200 🟢" if e200_v and price>e200_v else "Below EMA200 🔴")
                with ic2[3]: st.metric("ATR (14)",f"₹{atr_v}" if atr_v else "—")
                with ic2[4]: st.metric("Beta",str(round(sf(g("beta")),2)) if g("beta")!="—" else "—")
            else: st.error("No price data available.")

        # D3 BUY/SELL — FIXED BUG 4 (correct valuation labels)
        with D3:
            if val:
                sb=val["strong_buy"]; bp=val["buy"]; fv=val["fair"]; bkp=val["book_profit"]
                if price<=sb: stxt,scls="🔥 DEEP VALUE — Strong Buy Now","green"
                elif price<=bp: stxt,scls="✅ UNDERVALUED — Good Buy","green"
                elif price<=fv: stxt,scls="⚪ FAIRLY VALUED — Hold","orange"
                elif price<=bkp: stxt,scls="🟡 OVERVALUED — Wait","orange"
                else: stxt,scls="🔴 AVOID — Book Profits","red"
                bgm={"green":"#052e16","orange":"#1c1a0a","red":"#1e0c0c"}
                brm={"green":"#059669","orange":"#ca8a04","red":"#dc2626"}
                txm={"green":G,"orange":"#fbbf24","red":R}
                st.markdown(f'<div style="background:{bgm[scls]};border:2px solid {brm[scls]};border-radius:12px;padding:16px 24px;'
                            f'text-align:center;font-size:1.2rem;font-weight:800;color:{txm[scls]};margin-bottom:18px">'
                            f'{stxt} &nbsp;|&nbsp; Current: ₹{price:,.2f}</div>',unsafe_allow_html=True)
                lc1,lc2=st.columns(2)
                def lvl(label,pv,desc,bg_,br_,tx_,here=False):
                    now=f' &nbsp;{badge("● HERE","#065f46",G)}' if here else ""
                    st.markdown(f'<div style="background:{bg_};border:1.5px solid {br_};border-radius:10px;padding:14px 18px;margin:7px 0">'
                                f'<div style="font-size:11px;color:#6b7280;text-transform:uppercase;letter-spacing:0.5px;font-weight:700">{label}{now}</div>'
                                f'<div style="font-size:1.7rem;font-weight:900;color:{tx_};margin:5px 0">₹{pv:,.2f}</div>'
                                f'<div style="font-size:12px;color:#6b7280">{desc}</div></div>',unsafe_allow_html=True)
                with lc1:
                    sec("🟢 Buy / Entry Levels")
                    lvl("🔥 Strong Buy",sb,"65% of fair value — deep value zone","#052e16","#059669",G,price<=sb*1.02)
                    lvl("✅ Buy Price",bp,"80% of fair value — 20% margin of safety","#081f14","#10b981","#6ee7b7",sf(sb)<price<=bp*1.02)
                    lvl("⚪ Fair Value",fv,"Fully priced — accumulate on dips","#0d1e36","#3b82f6","#60a5fa",bp<price<=fv*1.02)
                    st.markdown("<br>",unsafe_allow_html=True)
                    # FIX BUG 4 — correct labels
                    sec("📐 Valuation Methods Used (to compute Fair Value)")
                    eps_=sf(info.get("trailingEps")); bv_=sf(info.get("bookValue"))
                    roe_=sf(info.get("returnOnEquity")); rev_=sf(info.get("revenueGrowth",0.1))
                    g__=min(max(rev_*100,0),25); sp_=val.get("sp",20)
                    gn_v=round(np.sqrt(22.5*eps_*bv_),2) if eps_>0 and bv_>0 else None
                    dcf_v=round(eps_*(8.5+2*g__)*(4.4/7.5),2) if eps_>0 else None
                    pe_v=round(sp_*eps_,2) if eps_>0 else None
                    pb_v=round((roe_/0.12)*bv_,2) if roe_>0 and bv_>0 else None
                    html_=""
                    for nm_,vv_ in [("Graham Number",gn_v),(f"DCF (Graham Revised)",dcf_v),
                                     (f"PE Fair Value (Sector PE: {sp_}x)",pe_v),("PB Fair Value",pb_v),
                                     ("Median = Fair Value",fv)]:
                        vs_=f"₹{vv_:,.2f}" if vv_ else "—"
                        html_+=row(nm_,vs_,"#60a5fa")
                    st.markdown(card(html_),unsafe_allow_html=True)
                with lc2:
                    sec("🔴 Sell / Target Levels")
                    lvl("📤 Book Profit",bkp,"120% of fair value — start selling","#1e0c0c","#dc2626",R,price>=bkp*.98)
                    if ema_r:
                        lvl("🎯 Target 1",ema_r["t1"],"1.5× ATR — short-term target","#1a150a","#d97706","#fbbf24")
                        lvl("🎯 Target 2",ema_r["t2"],"3× ATR — medium-term target","#1a150a","#f59e0b","#fde68a")
                        lvl("🛑 Stop Loss",ema_r["sl"],"Below 200 EMA — always respect","#1e0c0c","#ef4444","#fca5a5")
                    at=sf(g("targetMeanPrice")); ac_=str(g("recommendationKey","—")).upper(); an_=g("numberOfAnalystOpinions","—")
                    if at>0: lvl(f"👥 Analyst Target ({an_} analysts)",at,f"Consensus: {ac_}","#0d1e36","#3b82f6","#60a5fa")
                st.markdown("<hr>",unsafe_allow_html=True); sec("📦 Futures Trade Setup")
                lot=LOTS.get(sym,500); ep=ema_r["entry"] if ema_r else price
                sl_f=ema_r["sl"] if ema_r else round(price*.97,2); t1_=ema_r["t1"] if ema_r else round(price*1.05,2)
                mg=int(price*lot*.15); pf_=int((t1_-ep)*lot); lf_=int(abs(ep-sl_f)*lot); rr_=round(pf_/lf_,2) if lf_>0 else 0
                fc_=st.columns(6)
                for col_,lb_,vl_,cl_ in [(fc_[0],"Lot Size",f"{lot} shares","#f9fafb"),(fc_[1],"Margin (~15%)",fmtN(mg),"#f9fafb"),
                                           (fc_[2],"Entry",f"₹{ep:,.2f}","#60a5fa"),(fc_[3],"Stop Loss",f"₹{sl_f:,.2f}","#fca5a5"),
                                           (fc_[4],"Max Profit",fmtN(pf_),"#6ee7b7"),(fc_[5],"Max Loss",fmtN(lf_),"#fca5a5")]:
                    with col_: st.markdown(f'<div style="background:{CARD};border:1px solid {BORDER};border-radius:8px;padding:12px;text-align:center">'
                                           f'<div style="font-size:11px;color:#6b7280;margin-bottom:5px">{lb_}</div>'
                                           f'<div style="font-size:1rem;font-weight:800;color:{cl_}">{vl_}</div></div>',unsafe_allow_html=True)
                st.caption(f"R:R Ratio: 1:{rr_}")
            else:
                st.warning("Valuation data unavailable (EPS / Book Value missing for this stock).")
                st.markdown(f"[View on screener.in →](https://www.screener.in/company/{sym}/)")

        # D4 FUNDAMENTALS
        with D4:
            st.markdown(f'<div style="background:{CARD};border:1px solid {BORDER};border-radius:8px;padding:10px 14px;font-size:12px;color:#6b7280;margin-bottom:14px">'
                        f'Data: <b style="color:#f9fafb">yfinance (NSE)</b> &nbsp;·&nbsp; '
                        f'<a href="https://www.screener.in/company/{sym}/" target="_blank" style="color:{B}">screener.in ↗</a> &nbsp;·&nbsp; '
                        f'<a href="https://trendlyne.com/equity/stock/{sym}/" target="_blank" style="color:{B}">trendlyne ↗</a> &nbsp;·&nbsp; '
                        f'<a href="https://www.tickertape.in/stocks/{sym}" target="_blank" style="color:{B}">tickertape ↗</a></div>',unsafe_allow_html=True)
            def pct_(v):
                try: return f"{float(v)*100:.2f}%"
                except: return "—"
            def gc_(v):
                try: return G if float(v)>0 else R
                except: return "#f9fafb"
            fa_,fb_,fc2_,fd_=st.columns(4)
            def fsec_(title,items):
                sec(title); html_=""
                for k_,v_,c__ in items: html_+=row(k_,v_,c__)
                st.markdown(card(html_),unsafe_allow_html=True)
            with fa_: fsec_("📊 Valuation",[("PE Ratio",str(round(sf(g("trailingPE")),2)) if g("trailingPE")!="—" else "—","#f9fafb"),("Forward PE",str(round(sf(g("forwardPE")),2)) if g("forwardPE")!="—" else "—","#f9fafb"),("PB Ratio",str(round(sf(g("priceToBook")),2)) if g("priceToBook")!="—" else "—","#f9fafb"),("EV/EBITDA",str(round(sf(g("enterpriseToEbitda")),2)) if g("enterpriseToEbitda")!="—" else "—","#f9fafb"),("EPS (TTM)",f"₹{round(sf(g('trailingEps')),2)}" if g("trailingEps")!="—" else "—","#f9fafb"),("Book Value",f"₹{round(sf(g('bookValue')),2)}" if g("bookValue")!="—" else "—","#f9fafb"),("Mkt Cap",fmtN(g("marketCap")),"#f9fafb"),("Beta",str(round(sf(g("beta")),2)) if g("beta")!="—" else "—","#f9fafb")])
            with fb_: fsec_("📈 Profitability",[("ROE",pct_(g("returnOnEquity")),gc_(g("returnOnEquity"))),("ROA",pct_(g("returnOnAssets")),gc_(g("returnOnAssets"))),("Profit Margin",pct_(g("profitMargins")),gc_(g("profitMargins"))),("Gross Margin",pct_(g("grossMargins")),gc_(g("grossMargins"))),("Op. Margin",pct_(g("operatingMargins")),gc_(g("operatingMargins"))),("EBITDA",fmtN(g("ebitda")),"#f9fafb"),("Free Cash Flow",fmtN(g("freeCashflow")),"#f9fafb"),("Div Yield",pct_(g("dividendYield")),"#f9fafb")])
            with fc2_: fsec_("🚀 Growth",[("Rev Growth",pct_(g("revenueGrowth")),gc_(g("revenueGrowth"))),("Earn Growth",pct_(g("earningsGrowth")),gc_(g("earningsGrowth"))),("Revenue TTM",fmtN(g("totalRevenue")),"#f9fafb"),("Net Income",fmtN(g("netIncomeToCommon")),"#f9fafb"),("52W High",f"₹{g('fiftyTwoWeekHigh','—')}",G),("52W Low",f"₹{g('fiftyTwoWeekLow','—')}",R),("Avg Volume",fmtN(g("averageVolume","—"),"").replace("₹",""),"#f9fafb"),("Sector",g("sector","—"),"#f9fafb")])
            with fd_: fsec_("⚠️ Risk",[("Debt/Equity",str(round(sf(g("debtToEquity")),2)) if g("debtToEquity")!="—" else "—","#f9fafb"),("Current Ratio",str(round(sf(g("currentRatio")),2)) if g("currentRatio")!="—" else "—","#f9fafb"),("Total Debt",fmtN(g("totalDebt")),"#fca5a5"),("Total Cash",fmtN(g("totalCash")),G),("Analyst Rec",str(g("recommendationKey","—")).upper(),"#60a5fa"),("Analysts",str(g("numberOfAnalystOpinions","—")),"#f9fafb"),("Target",f"₹{g('targetMeanPrice','—')}","#60a5fa"),("Payout Ratio",pct_(g("payoutRatio")),"#f9fafb")])

        # D5 NEWS
        with D5:
            news_=get_news(sym)
            if news_:
                for n_ in news_:
                    if n_.get("title"):
                        st.markdown(f'<div style="background:{CARD};border:1px solid {BORDER};border-radius:10px;padding:14px 18px;margin:8px 0">'
                                    f'<div style="font-size:14px;font-weight:700;color:#f9fafb;margin-bottom:6px">{n_["title"]}</div>'
                                    f'<div style="font-size:12px;color:#6b7280;line-height:1.5;margin-bottom:8px">{n_.get("body","")[:240]}</div>'
                                    f'<div style="font-size:11px;color:#6b7280"><span style="color:{B}">{n_.get("source","")}</span>'
                                    f' &nbsp;·&nbsp; {n_.get("date","")} &nbsp;&nbsp; '
                                    f'<a href="{n_.get("url","#")}" target="_blank" style="color:{G};font-weight:700">Read full ↗</a></div></div>',unsafe_allow_html=True)
            else: st.info("No news found.")
    else:
        st.markdown(f'<div style="text-align:center;padding:70px 20px"><div style="font-size:4rem;margin-bottom:14px">📊</div>'
                    f'<div style="font-size:2rem;font-weight:900;color:#f9fafb;margin-bottom:10px">Search any Indian Stock</div>'
                    f'<div style="font-size:15px;color:#6b7280;max-width:500px;margin:0 auto;line-height:1.8">'
                    f'🤖 5 AI Agents &nbsp;·&nbsp; 📈 Live Charts &nbsp;·&nbsp; 💰 Buy/Sell Levels<br>'
                    f'🎯 200 EMA Signals &nbsp;·&nbsp; 📦 Futures Setup &nbsp;·&nbsp; 📰 News</div></div>',unsafe_allow_html=True)

# ════════════ TAB 2 ═════════════════════════════════════════════
with T2:
    st.markdown(f'<div style="background:{CARD};border:1px solid {BORDER};border-radius:8px;padding:10px 16px;font-size:13px;color:#6b7280;margin-bottom:14px">'
                f'Run <b style="color:#f9fafb">4 strategies + 4 ML models + statistical tests</b> on any NSE stock. '
                f'Includes 0.15% transaction costs. TimeSeriesSplit CV prevents data leakage.</div>',unsafe_allow_html=True)
    qc1,qc2=st.columns([3,1])
    with qc1: q_sym_input=st.text_input("Quant Analysis Symbol",placeholder="RELIANCE, TCS, HDFCBANK...",label_visibility="collapsed",key="q_sym")
    with qc2:
        if st.button("▶ Run Analysis",type="primary",use_container_width=True):
            if q_sym_input.strip(): st.session_state.q2_sym=q_sym_input.upper().strip().replace(" ",""); st.session_state.run_q2=True

    # TABS ALWAYS VISIBLE — FIX BUG 9
    q1,q2,q3,q4=st.tabs(["📊 4 Strategies","⚡ Optimisation","🎲 Statistics + Regimes","🤖 ML Models"])
    sym2=st.session_state.q2_sym

    if sym2 and st.session_state.run_q2:
        # FIX BUG 3: reset flag immediately so tab/widget interactions don't retrigger
        st.session_state.run_q2 = False
        key_df=f"df2_{sym2}"
        if key_df not in st.session_state:
            with st.spinner(f"Loading {sym2} data..."):
                # FIX: try 5y first, fall back to 3y, 2y, 1y if empty
                df2 = pd.DataFrame()
                for period_try in ["5y", "3y", "2y", "1y"]:
                    df2 = get_history(sym2, period_try, "1d")
                    if not df2.empty:
                        break
            st.session_state[key_df] = df2 if not df2.empty else None

    # Always render from cache when data is available
    key_df=f"df2_{sym2}" if sym2 else ""
    if sym2 and key_df in st.session_state:
        df2=st.session_state.get(key_df)
        if df2 is None:
            for tab in [q1,q2,q3,q4]:
                with tab: st.error(f"No data for {sym2}")
        else:
            with q1:
                key_s=f"strats_{sym2}"
                if key_s not in st.session_state:
                    with st.spinner("Running 4 strategies..."): st.session_state[key_s]=run_all_strategies(df2)
                strats=st.session_state[key_s]
                sec("📊 All Strategy Results (NSE Transaction Costs Included)")
                rows_s=[{"Strategy":sn,"Sharpe":sr["metrics"]["sharpe"],"Sortino":sr["metrics"]["sortino"],
                          "Annual Return":sr["metrics"]["annual"],"Total Return":sr["metrics"]["total"],
                          "Max Drawdown":sr["metrics"]["max_dd"],"Calmar":sr["metrics"]["calmar"],"Win Rate":sr["metrics"]["win"]}
                         for sn,sr in strats.items()]
                st.dataframe(pd.DataFrame(rows_s),use_container_width=True,hide_index=True)
                st.caption("⚠️ Realistic Sharpe 0.5–2.0 (0.15% NSE cost applied). Sharpe > 4 = overfitting.")
                fig_s=go.Figure()
                colors_s=[G,"#3b82f6","#f59e0b","#a855f7"]
                bh_=strats["SMA Crossover"]["data"]["BH"].dropna()
                fig_s.add_trace(go.Scatter(x=(1+bh_).cumprod().index,y=(1+bh_).cumprod().values,name="Buy & Hold",line=dict(color="#6b7280",width=2,dash="dash")))
                for i,(sn,sr) in enumerate(strats.items()):
                    cum_=(1+sr["data"]["Strat"].dropna()).cumprod()
                    fig_s.add_trace(go.Scatter(x=cum_.index,y=cum_.values,name=sn,line=dict(color=colors_s[i],width=2)))
                fig_s.update_layout(height=380,paper_bgcolor=DARK,plot_bgcolor=DARK,font=dict(color="#f9fafb"),
                    title=dict(text=f"Strategy vs Buy & Hold — {sym2}",font=dict(color="#f9fafb",size=13)),
                    legend=dict(bgcolor="rgba(0,0,0,0)",font_color="#6b7280"),margin=dict(l=0,r=0,t=40,b=0))
                fig_s.update_xaxes(gridcolor="#1f2937"); fig_s.update_yaxes(gridcolor="#1f2937",title="Portfolio Growth (₹1 invested)")
                st.plotly_chart(fig_s,use_container_width=True,config={"displayModeBar":False})

            with q2:
                key_o=f"opt_{sym2}"
                if key_o not in st.session_state:
                    with st.spinner("Optimising SMA parameters..."): st.session_state[key_o]=optimise_sma(df2,step=5)
                opt_r,best=st.session_state[key_o]
                if best is not None:
                    oc1,oc2,oc3=st.columns(3)
                    with oc1: st.metric("Best Short Window",int(best.Short))
                    with oc2: st.metric("Best Long Window",int(best.Long))
                    with oc3: st.metric("Best Sharpe",round(best.Sharpe,3))
                    st.caption(f"Best Total Return: {best.Return:.2%} | Max Drawdown: {best.MaxDD:.2%}")
                try:
                    pivot=opt_r.pivot(index="Short",columns="Long",values="Sharpe")
                    fig_h=go.Figure(go.Heatmap(z=pivot.values,x=pivot.columns.tolist(),y=pivot.index.tolist(),colorscale="RdYlGn",zmid=0,colorbar=dict(title="Sharpe")))
                    fig_h.update_layout(height=380,paper_bgcolor=DARK,plot_bgcolor=DARK,font=dict(color="#f9fafb"),
                        title=dict(text=f"Sharpe Ratio Heatmap — {sym2}",font=dict(color="#f9fafb",size=13)),margin=dict(l=0,r=0,t=40,b=0))
                    fig_h.update_xaxes(title="Long Window"); fig_h.update_yaxes(title="Short Window")
                    st.plotly_chart(fig_h,use_container_width=True,config={"displayModeBar":False})
                except: st.dataframe(opt_r.nlargest(10,"Sharpe"),use_container_width=True,hide_index=True)

            with q3:
                key_r=f"rw_{sym2}"
                if key_r not in st.session_state:
                    with st.spinner("Statistical tests..."): st.session_state[key_r]=random_walk_test(df2)
                rw=st.session_state[key_r]
                sc1_,sc2_,sc3_=st.columns(3)
                with sc1_: st.metric("Max Autocorrelation",f"{rw['max_ac']:.4f}","Predictable ✅" if rw["predictable"] else "Random ❌")
                with sc2_: st.metric("Jarque-Bera Stat",f"{rw['jb_stat']:.2f}",f"p={rw['jb_pval']:.4f}")
                with sc3_: st.metric("Normal Returns","No (Fat Tails) ⚠️" if not rw["normal"] else "Yes ✅","Typical for Indian stocks" if not rw["normal"] else "")
                fig_ac=go.Figure(go.Bar(x=list(range(1,len(rw["autocorr"])+1)),y=rw["autocorr"],marker_color=[G if abs(a)>0.05 else "#374151" for a in rw["autocorr"]]))
                fig_ac.add_hline(y=0.05,line_dash="dash",line_color=R,line_width=1,annotation_text="±0.05 significance")
                fig_ac.add_hline(y=-0.05,line_dash="dash",line_color=R,line_width=1)
                fig_ac.update_layout(height=300,paper_bgcolor=DARK,plot_bgcolor=DARK,font=dict(color="#f9fafb"),
                    title=dict(text=f"Return Autocorrelation — {sym2}",font=dict(color="#f9fafb",size=13)),
                    xaxis=dict(title="Lag (Days)",gridcolor="#1f2937"),yaxis=dict(title="Autocorrelation",gridcolor="#1f2937"),margin=dict(l=0,r=0,t=40,b=0))
                st.plotly_chart(fig_ac,use_container_width=True,config={"displayModeBar":False})
                key_cl=f"cl_{sym2}"
                if key_cl not in st.session_state:
                    with st.spinner("K-Means regime detection..."): st.session_state[key_cl]=kmeans_regimes(df2)
                cl=st.session_state[key_cl]
                st.markdown("<br>",unsafe_allow_html=True); sec("🎯 Market Regime Detection (K-Means Clustering)")
                rc1,rc2=st.columns(2)
                vc_=cl["data"]["Regime"].value_counts()
                with rc1:
                    fig_r=go.Figure(go.Bar(x=vc_.index.tolist(),y=vc_.values.tolist(),
                        marker_color=[G if "Bull" in x else R if "Bear" in x else Y if "Vol" in x else "#6b7280" for x in vc_.index],
                        text=vc_.values.tolist(),textposition="outside"))
                    fig_r.update_layout(height=280,paper_bgcolor=DARK,plot_bgcolor=DARK,font=dict(color="#f9fafb"),
                        title=dict(text="Days per Market Regime",font=dict(color="#f9fafb",size=12)),margin=dict(l=0,r=0,t=40,b=0),
                        xaxis=dict(gridcolor="#1f2937"),yaxis=dict(title="Days",gridcolor="#1f2937"))
                    st.plotly_chart(fig_r,use_container_width=True,config={"displayModeBar":False})
                with rc2:
                    for regime,count in vc_.items():
                        pct__=count/len(cl["data"])*100; rc_=G if "Bull" in regime else R if "Bear" in regime else Y if "Vol" in regime else "#6b7280"
                        st.markdown(f'<div style="display:flex;justify-content:space-between;align-items:center;padding:8px 0;border-bottom:1px solid {BORDER}">'
                                    f'<span style="color:#6b7280;font-size:13px">{regime}</span>'
                                    f'<span style="color:{rc_};font-weight:700;font-size:13px">{count} days ({pct__:.1f}%)</span></div>',unsafe_allow_html=True)

            # ML — FIX BUG 5: Linear Regression added
            with q4:
                key_ml=f"ml_{sym2}"
                if key_ml not in st.session_state:
                    with st.spinner("Training Random Forest, Gradient Boosting, Neural Network, Linear Regression..."):
                        ml_r=run_all_ml(df2); lr_r=run_regression(df2)
                    st.session_state[key_ml]=(ml_r,lr_r)
                ml_r,lr_r=st.session_state[key_ml]

                sec("🤖 Classification Models — Direction Prediction (Up/Down)")
                if ml_r:
                    ml_cols_=st.columns(len(ml_r))
                    for i,(mname,mr) in enumerate(ml_r.items()):
                        with ml_cols_[i]:
                            acc=mr["accuracy"]; base=mr.get("baseline",0.5); beat=acc>base; col_=G if beat else R
                            st.markdown(f'<div style="background:{CARD};border:1px solid {"#059669" if beat else "#dc2626"};border-radius:10px;padding:14px;text-align:center">'
                                        f'<div style="font-size:12px;color:#6b7280;margin-bottom:6px">{mname}</div>'
                                        f'<div style="font-size:1.8rem;font-weight:900;color:{col_}">{acc:.3f}</div>'
                                        f'<div style="font-size:11px;color:#6b7280;margin-top:4px">Baseline: {base:.3f} | {"✅ Beat" if beat else "❌ Below"}</div></div>',unsafe_allow_html=True)
                    st.caption("TimeSeriesSplit CV (5-fold) prevents data leakage. Beat baseline = model found signal.")

                st.markdown("<br>",unsafe_allow_html=True); sec("📈 Linear Regression — Return Prediction (R²)")
                if lr_r:
                    lrc1,lrc2,lrc3=st.columns(3)
                    with lrc1: st.metric("Train R²",f"{lr_r.get('train_r2',0):.4f}")
                    with lrc2: st.metric("Test R²",f"{lr_r.get('test_r2',0):.4f}")
                    with lrc3: st.metric("Features",len(lr_r.get("feature_importance",[])))
                    if "feature_importance" in lr_r:
                        fi=lr_r["feature_importance"].head(10)
                        fig_lr=go.Figure(go.Bar(x=fi["Abs"].values,y=fi["Feature"].values,orientation="h",marker_color=B))
                        fig_lr.update_layout(height=300,paper_bgcolor=DARK,plot_bgcolor=DARK,font=dict(color="#f9fafb"),
                            title=dict(text=f"Top Features — Linear Regression | {sym2}",font=dict(color="#f9fafb",size=12)),
                            xaxis=dict(title="Abs Coefficient",gridcolor="#1f2937"),yaxis=dict(gridcolor="#1f2937"),margin=dict(l=0,r=0,t=40,b=0))
                        st.plotly_chart(fig_lr,use_container_width=True,config={"displayModeBar":False})
                    st.caption("R² near 0 expected — stock returns are near-random. Coefficients show relative importance of signals.")
                else: st.info("Linear Regression data unavailable for this stock.")

                rf_r=ml_r.get("Random Forest",{})
                if rf_r and "feature_importance" in rf_r:
                    st.markdown("<br>",unsafe_allow_html=True); sec("🌲 Random Forest — Feature Importance + Scores")
                    fi=rf_r["feature_importance"].head(10)
                    fig_fi=go.Figure(go.Bar(x=fi["Importance"].values,y=fi["Feature"].values,orientation="h",marker_color=G))
                    fig_fi.update_layout(height=300,paper_bgcolor=DARK,plot_bgcolor=DARK,font=dict(color="#f9fafb"),
                        xaxis=dict(title="Importance",gridcolor="#1f2937"),yaxis=dict(gridcolor="#1f2937"),margin=dict(l=0,r=0,t=10,b=0))
                    st.plotly_chart(fig_fi,use_container_width=True,config={"displayModeBar":False})
                    oc_=st.columns(3)
                    with oc_[0]: st.metric("Accuracy",f"{rf_r['accuracy']:.3f}")
                    with oc_[1]: st.metric("ROC-AUC",f"{rf_r.get('roc_auc',0):.3f}")
                    with oc_[2]: st.metric("OOB Score",f"{rf_r.get('oob',0):.3f}")
    else:
        with q1: st.info("Enter a stock symbol above and click ▶ Run Analysis")
        with q2: st.info("Enter a stock symbol above and click ▶ Run Analysis")
        with q3: st.info("Enter a stock symbol above and click ▶ Run Analysis")
        with q4: st.info("Enter a stock symbol above and click ▶ Run Analysis")

# ════════════ TAB 3 — SCANNER (FIX BUG 6) ════════════════════
with T3:
    total_stocks=len(ALL_STOCKS)
    st.markdown(f'<div style="background:{CARD};border:1px solid {BORDER};border-radius:8px;padding:10px 16px;font-size:13px;color:#6b7280;margin-bottom:14px">'
                f'🎯 Scans <b style="color:#f9fafb">all {total_stocks} NSE F&O eligible stocks</b> for 200 EMA proximity. '
                f'Within proximity + Above = <span style="color:{G}"><b>BUY</b></span>. '
                f'Within proximity + Below = <span style="color:{R}"><b>SELL</b></span>. Cash + Futures levels shown.</div>',unsafe_allow_html=True)
    ec1,ec2,ec3=st.columns([2,2,2])
    with ec1: prox=st.slider("EMA Proximity %",0.3,3.0,1.0,0.1)
    with ec2: filt=st.selectbox("Show",["BUY + SELL","BUY Only","SELL Only"])
    with ec3: scan_btn=st.button("🚀 Scan All Stocks",type="primary",use_container_width=True)
    if scan_btn:
        prog=st.progress(0,"Starting scan..."); stat_=st.empty()
        def pcb(done,total,sym_s): prog.progress(int(done/total*100)); stat_.text(f"Scanning {sym_s}... {done}/{total}")
        scan_r=scan_ema(prox,pcb); prog.empty(); stat_.empty()
        buys=scan_r["buy"]; sells=scan_r["sell"]
        sm=st.columns(4)
        with sm[0]: st.metric("🟢 BUY",len(buys))
        with sm[1]: st.metric("🔴 SELL",len(sells))
        with sm[2]: st.metric("📊 Scanned",scan_r["total"])
        with sm[3]: st.metric("📏 Proximity",f"{prox}%")
        def show_scan(stocks,kind):
            if not stocks: st.info(f"No {kind} signals within {prox}% of 200 EMA right now."); return
            is_buy=kind=="BUY"; tc=G if is_buy else R
            st.markdown(f'<div style="font-size:1rem;font-weight:800;color:{tc};margin:14px 0 8px">{"🟢" if is_buy else "🔴"} {kind} Signals ({len(stocks)} stocks)</div>',unsafe_allow_html=True)
            rows_=[{"Symbol":s.get("sym",""),"Sector":s.get("sec","—"),"Signal":s.get("fin",""),"Conf":s.get("conf",""),
                    "Price ₹":s.get("price",""),"200 EMA ₹":s.get("ema200",""),"Dist %":s.get("dist",""),
                    "Entry ₹":s.get("entry",""),"Target1 ₹":s.get("t1",""),"SL ₹":s.get("sl",""),"Lot":s.get("lot",""),"Margin ₹":s.get("margin","")} for s in stocks]
            st.dataframe(pd.DataFrame(rows_),use_container_width=True,hide_index=True)
            for s in stocks[:5]:
                with st.expander(f'{"🟢" if is_buy else "🔴"} {s.get("fin","")} — {s.get("sym","")} | ₹{s.get("price","?")} | EMA:₹{s.get("ema200","?")} | Dist:{s.get("dist","?")}%'):
                    dc_=st.columns(3)
                    with dc_[0]: st.markdown("**EMA Data**"); st.write(f"Price: ₹{s.get('price')}"); st.write(f"200 EMA Daily: ₹{s.get('ema200')}"); st.write(f"200 EMA 4HR: ₹{s.get('ema4h')}"); st.write(f"Distance: {s.get('dist')}%"); st.write(f"ATR: ₹{s.get('atr')}")
                    with dc_[1]: st.markdown("**Cash Trade**"); st.write(f"Entry: ₹{s.get('entry')}"); st.write(f"Target 1: ₹{s.get('t1')}"); st.write(f"Target 2: ₹{s.get('t2')}"); st.write(f"Stop Loss: ₹{s.get('sl')}")
                    with dc_[2]: st.markdown("**Futures**"); st.write(f"Lot: {s.get('lot')} shares"); st.write(f"Margin: {fmtN(s.get('margin',0))}"); st.write(f"Max Profit: {fmtN(s.get('profit',0))}"); st.write(f"Max Loss: {fmtN(s.get('loss',0))}")
        if filt in ["BUY + SELL","BUY Only"]: show_scan(buys,"BUY")
        if filt in ["BUY + SELL","SELL Only"]: show_scan(sells,"SELL")
    else:
        st.markdown(f'<div style="text-align:center;padding:50px"><div style="font-size:2.5rem">🎯</div>'
                    f'<div style="font-size:1.1rem;font-weight:700;color:#f9fafb;margin-top:10px">Click Scan — find stocks near 200 EMA</div>'
                    f'<div style="color:#6b7280;font-size:13px;margin-top:5px">{total_stocks} NSE F&O stocks · BUY/SELL signals · Cash + Futures levels</div></div>',unsafe_allow_html=True)

# ════════════ TAB 4 ══════════════════════════════════════════
with T4:
    if st.button("🔄 Load Nifty 50 Live Data",type="primary",key="mbt"):
        with st.spinner("Loading Nifty 50..."):
            rows_m=[]
            import yfinance as yf_m
            tickers=[s+".NS" for s in NIFTY50]
            try:
                batch=yf_m.download(tickers,period="2d",interval="1d",progress=False,auto_adjust=True,group_by="ticker")
                for s_ in NIFTY50:
                    try:
                        tk=s_+".NS"
                        cl=None
                        if isinstance(batch.columns,pd.MultiIndex):
                            lvl0=list(batch.columns.get_level_values(0).unique())
                            lvl1=list(batch.columns.get_level_values(1).unique())
                            # new yfinance: (field, ticker) → batch["Close"][tk]
                            if "Close" in lvl0 and tk in lvl1:
                                cl=batch["Close"][tk]
                            # old yfinance: (ticker, field) → batch[tk]["Close"]
                            elif tk in lvl0 and "Close" in lvl1:
                                cl=batch[tk]["Close"]
                        else:
                            # single ticker fallback
                            if "Close" in batch.columns:
                                cl=batch["Close"]
                        if cl is None or len(cl)<2: continue
                        cl=cl.dropna()
                        if len(cl)<2: continue
                        p_=float(cl.iloc[-1]); pp_=float(cl.iloc[-2]); chg_=round((p_-pp_)/pp_*100,2)
                        rows_m.append({"Symbol":s_,"Sector":SECTORS.get(s_,"—"),"Price ₹":round(p_,2),"Change%":chg_,"Dir":"▲" if chg_>=0 else "▼"})
                    except: pass
            except Exception as e: st.error(f"Error: {e}")
        if rows_m:
            df_m=pd.DataFrame(rows_m).sort_values("Change%",ascending=False)
            fig_m=go.Figure(go.Bar(x=df_m["Symbol"],y=df_m["Change%"],marker_color=[G if v>=0 else R for v in df_m["Change%"]],
                text=[f"{v:+.2f}%" for v in df_m["Change%"]],textposition="outside",textfont=dict(size=10,color="#f9fafb")))
            fig_m.update_layout(title="Nifty 50 — Today's Performance",height=380,paper_bgcolor=DARK,plot_bgcolor=DARK,font=dict(color="#f9fafb"),
                showlegend=False,xaxis=dict(tickangle=-45,gridcolor="#1f2937"),yaxis=dict(title="Change%",gridcolor="#1f2937"),margin=dict(l=0,r=0,t=50,b=0))
            st.plotly_chart(fig_m,use_container_width=True,config={"displayModeBar":False})
            st.dataframe(df_m[["Symbol","Sector","Price ₹","Change%","Dir"]],use_container_width=True,hide_index=True)
    else:
        st.markdown(f'<div style="text-align:center;padding:50px"><div style="font-size:2.5rem">🏆</div><div style="font-size:1.1rem;font-weight:700;color:#f9fafb;margin-top:10px">Click Load for live Nifty 50 performance</div></div>',unsafe_allow_html=True)

st.markdown(f'<div style="background:{DARK};border:1px solid {BORDER};border-radius:8px;padding:12px 16px;font-size:11px;color:#6b7280;margin-top:20px;text-align:center">'
            f'⚠️ QuantIQ Pro is for educational purposes only. Not SEBI-registered financial advice. Data from yfinance/NSE. '
            f'&nbsp;|&nbsp; Python · Streamlit · yfinance · Groq AI · Scikit-learn · Plotly</div>',unsafe_allow_html=True)
