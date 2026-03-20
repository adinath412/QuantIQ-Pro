"""agents.py — 5 AI Agents + Master Agent (Groq LLM or Rule-Based)"""
import os, numpy as np

def _client():
    key=os.getenv("GROQ_API_KEY","")
    if not key or "your_" in key: return None
    try:
        from groq import Groq
        return Groq(api_key=key)
    except: return None

def _ask(c, sys, usr, tokens=300):
    if not c: return None
    try:
        r=c.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role":"system","content":sys},
                      {"role":"user","content":usr}],
            max_tokens=tokens,temperature=0.3)
        return r.choices[0].message.content.strip()
    except: return None

def _parse(text, key):
    if not text: return ""
    for line in text.split("\n"):
        if key.upper()+":" in line.upper():
            return line.split(":",1)[1].strip()
    return ""

# ── Agent 1: News ──────────────────────────────────────────────
def agent_news(c, symbol, news_list):
    headlines="\n".join(f"- {n.get('title','')}" for n in (news_list or [])[:6])
    if not headlines:
        return {"sent":"Neutral","score":50,"theme":"No news found.","pos":[],"neg":[],"powered":False,"name":"📰 News Agent"}
    result=_ask(c,"You are an Indian stock market news analyst. Be concise.",
        f"Stock:{symbol}\nHeadlines:\n{headlines}\n\nReply EXACTLY:\nSENTIMENT: Positive/Negative/Neutral\nSCORE: 0-100\nTHEME: one sentence\nBULLISH: positives or None\nBEARISH: negatives or None",200)
    text=headlines.lower()
    pos_w=["growth","profit","beat","record","strong","buy","upgrade","rally","surge"]
    neg_w=["loss","miss","decline","fall","drop","sell","downgrade","weak","crash","debt"]
    pc=sum(1 for w in pos_w if w in text); nc=sum(1 for w in neg_w if w in text)
    score=int(pc/(pc+nc or 1)*100)
    sent="Positive" if score>55 else "Negative" if score<45 else "Neutral"
    theme=f"Based on {len(news_list or [])} news articles"; pos=[]; neg=[]
    if result:
        try: score=int(_parse(result,"SCORE"))
        except: pass
        rs=_parse(result,"SENTIMENT")
        if rs: sent=rs
        rt=_parse(result,"THEME")
        if rt: theme=rt
        bp=_parse(result,"BULLISH"); np_=_parse(result,"BEARISH")
        pos=[x.strip() for x in bp.split(",") if x.strip() and x.lower()!="none"]
        neg=[x.strip() for x in np_.split(",") if x.strip() and x.lower()!="none"]
    return {"sent":sent,"score":min(100,max(0,score)),"theme":theme,
            "pos":pos[:2],"neg":neg[:2],"powered":result is not None,
            "name":"📰 News Agent"}

# ── Agent 2: Fundamental ───────────────────────────────────────
def agent_fundamental(c, symbol, info):
    def g(k): return info.get(k,"N/A")
    def p(v):
        try: return f"{float(v)*100:.1f}%"
        except: return "N/A"
    data=f"PE:{g('trailingPE')} ROE:{p(g('returnOnEquity'))} RevGrowth:{p(g('revenueGrowth'))} D/E:{g('debtToEquity')} ProfitMargin:{p(g('profitMargins'))} Sector:{g('sector')}"
    result=_ask(c,"You are an Indian equity fundamental analyst. Be direct.",
        f"Analyse {symbol}:\n{data}\n\nReply EXACTLY:\nSCORE: 0-100\nVERDICT: Strong Buy/Buy/Hold/Sell/Avoid\nSTRENGTHS: comma list\nWEAKNESSES: comma list\nSUMMARY: one sentence",250)
    score=50; verdict="Hold"; strengths=[]; weaknesses=[]; summary=""
    try:
        pe=float(info.get("trailingPE",30) or 30); roe=float(info.get("returnOnEquity",0) or 0)
        de=float(info.get("debtToEquity",50) or 50); rev=float(info.get("revenueGrowth",0) or 0)
        if pe<12: score+=15
        elif pe<20: score+=8
        elif pe>40: score-=15
        if roe>0.20: score+=15
        elif roe>0.12: score+=8
        elif roe<0.05: score-=10
        if de<30: score+=10
        elif de>150: score-=15
        if rev>0.15: score+=10
        elif rev<0: score-=10
        score=max(0,min(100,score))
        verdict="Strong Buy" if score>75 else "Buy" if score>60 else "Hold" if score>40 else "Avoid"
    except: pass
    if result:
        try: score=int(_parse(result,"SCORE"))
        except: pass
        v=_parse(result,"VERDICT")
        if v: verdict=v
        s=_parse(result,"STRENGTHS")
        if s: strengths=[x.strip() for x in s.split(",") if x.strip()]
        w=_parse(result,"WEAKNESSES")
        if w: weaknesses=[x.strip() for x in w.split(",") if x.strip()]
        su=_parse(result,"SUMMARY")
        if su: summary=su
    return {"score":min(100,max(0,score)),"verdict":verdict,
            "strengths":strengths[:2],"weaknesses":weaknesses[:2],
            "summary":summary,"powered":result is not None,"name":"📊 Fundamental Agent"}

# ── Agent 3: Technical ─────────────────────────────────────────
def agent_technical(c, symbol, df, ema_r):
    if df is None or df.empty:
        return {"score":50,"direction":"Unknown","signals":[],"powered":False,"name":"📈 Technical Agent"}
    try:
        price=float(df["Close"].squeeze().iloc[-1])
        rsi=float(df["RSI"].iloc[-1]) if "RSI" in df else 50
        macd=float(df["MACD"].iloc[-1]) if "MACD" in df else 0
        msig=float(df["MACD_sig"].iloc[-1]) if "MACD_sig" in df else 0
        ema200=float(df["EMA200"].iloc[-1]) if "EMA200" in df else price
    except: rsi=50; macd=0; msig=0; ema200=price; price=0
    ema_fin=ema_r.get("fin","WAIT") if ema_r else "WAIT"
    data=f"Price:₹{price:.0f} RSI:{rsi:.1f} MACD:{'Bullish' if macd>msig else 'Bearish'} EMA200:{'Above' if price>ema200 else 'Below'} EMASignal:{ema_fin}"
    result=_ask(c,"You are an Indian stock technical analyst. Be direct.",
        f"Technical for {symbol}:\n{data}\n\nReply EXACTLY:\nSCORE: 0-100\nDIRECTION: Bullish/Bearish/Sideways\nSIGNALS: comma list",180)
    score=50; signals=[]; direction="Sideways"
    if rsi<30: score+=15; signals.append(f"RSI oversold {rsi:.0f}")
    elif rsi>70: score-=15; signals.append(f"RSI overbought {rsi:.0f}")
    else: signals.append(f"RSI neutral {rsi:.0f}")
    if macd>msig: score+=10; signals.append("MACD bullish")
    else: score-=10; signals.append("MACD bearish")
    if price>ema200: score+=10; signals.append("Above 200 EMA")
    else: score-=10; signals.append("Below 200 EMA")
    if "STRONG BUY" in ema_fin: score+=15
    elif "STRONG SELL" in ema_fin: score-=15
    elif "BUY" in ema_fin: score+=8
    elif "SELL" in ema_fin: score-=8
    score=max(0,min(100,score))
    direction="Bullish" if score>60 else "Bearish" if score<40 else "Sideways"
    if result:
        try: score=int(_parse(result,"SCORE"))
        except: pass
        d=_parse(result,"DIRECTION")
        if d: direction=d
        ss=_parse(result,"SIGNALS")
        if ss: signals=[x.strip() for x in ss.split(",") if x.strip()]
    return {"score":max(0,min(100,score)),"direction":direction,
            "signals":signals[:4],"rsi":round(rsi,1),
            "ema_signal":ema_fin,"powered":result is not None,"name":"📈 Technical Agent"}

# ── Agent 4: Risk ──────────────────────────────────────────────
def agent_risk(c, symbol, info, df):
    vol="N/A"
    try:
        r=df["Returns"].dropna()
        vol=f"{float(r.std()*np.sqrt(252)*100):.1f}%"
    except: pass
    de=float(info.get("debtToEquity",50) or 50); pe=float(info.get("trailingPE",25) or 25)
    beta=float(info.get("beta",1) or 1); rev=float(info.get("revenueGrowth",0) or 0)
    result=_ask(c,"You are an Indian equity risk analyst.",
        f"Risk for {symbol}: D/E:{de} PE:{pe} Beta:{beta} RevGrowth:{rev*100:.1f}% Vol:{vol}\n\nReply EXACTLY:\nRISK_SCORE: 0-100\nRISK_LEVEL: Low/Moderate/High/Very High\nRISKS: comma list\nSAFE: comma list",180)
    rs=30; risks=[]; safe=[]
    if pe>50: rs+=20; risks.append(f"High valuation PE:{pe:.0f}")
    if de>150: rs+=20; risks.append(f"High debt D/E:{de:.0f}")
    if beta>1.5: rs+=10; risks.append(f"High beta {beta:.1f}")
    if rev<0: rs+=15; risks.append("Revenue declining")
    if de<30: safe.append("Low debt")
    if rev>0.1: safe.append("Growing revenue")
    if pe<20: safe.append("Reasonable valuation")
    rs=max(0,min(100,rs))
    level="Low" if rs<30 else "Moderate" if rs<55 else "High" if rs<75 else "Very High"
    if result:
        try: rs=int(_parse(result,"RISK_SCORE"))
        except: pass
        l=_parse(result,"RISK_LEVEL")
        if l: level=l
        r_=_parse(result,"RISKS")
        if r_: risks=[x.strip() for x in r_.split(",") if x.strip()]
        s=_parse(result,"SAFE")
        if s: safe=[x.strip() for x in s.split(",") if x.strip()]
    return {"score":max(0,min(100,rs)),"level":level,"risks":risks[:3],
            "safe":safe[:3],"vol":vol,"powered":result is not None,"name":"⚠️ Risk Agent"}

# ── Agent 5: Valuation ─────────────────────────────────────────
def agent_valuation(c, symbol, info, val):
    price=float(info.get("currentPrice",0) or 0)
    if not val: return {"verdict":"No Data","action":"Check manually","thesis":"Insufficient data.","horizon":"N/A","powered":False,"name":"💰 Valuation Agent"}
    data=f"Price:₹{price:.2f} StrongBuy:₹{val.get('strong_buy','N/A')} Buy:₹{val.get('buy','N/A')} Fair:₹{val.get('fair','N/A')} BookProfit:₹{val.get('book_profit','N/A')}"
    result=_ask(c,"You are a value investing expert for Indian equities.",
        f"Valuation for {symbol}:\n{data}\n\nReply EXACTLY:\nVERDICT: Deep Value/Undervalued/Fair/Overvalued/Avoid\nHORIZON: Short/Medium/Long-term\nACTION: Buy Now/Accumulate/Hold/Wait/Avoid\nTHESIS: one sentence",250)
    sb=val.get("strong_buy",0) or 0; bp=val.get("buy",0) or 0
    fv=val.get("fair",0) or 0; bkp=val.get("book_profit",0) or 0
    try:
        if price<=float(sb):   verdict,action="Deep Value","Buy Now"
        elif price<=float(bp): verdict,action="Undervalued","Buy Now"
        elif price<=float(fv): verdict,action="Fair Value","Accumulate"
        elif price<=float(bkp):verdict,action="Overvalued","Wait"
        else:                  verdict,action="Avoid","Book Profits"
    except: verdict,action="Fair Value","Hold"
    thesis=f"At ₹{price:.0f}, stock is {verdict.lower()} vs fair value ₹{fv}."
    horizon="Long-term (2+ years)"
    if result:
        v=_parse(result,"VERDICT")
        if v: verdict=v
        a=_parse(result,"ACTION")
        if a: action=a
        t=_parse(result,"THESIS")
        if t: thesis=t
        h=_parse(result,"HORIZON")
        if h: horizon=h
    return {"verdict":verdict,"action":action,"thesis":thesis,
            "horizon":horizon,"powered":result is not None,"name":"💰 Valuation Agent"}

# ── Master Agent ───────────────────────────────────────────────
def agent_master(c, symbol, n, f, t, r, v):
    ns=n["score"]; fs=f["score"]; ts=t["score"]; rs=100-r["score"]
    vs={"Deep Value":92,"Undervalued":78,"Fair Value":58,
        "Overvalued":32,"Avoid":15,"No Data":50}.get(v["verdict"],55)
    final=int(ns*.15+fs*.30+ts*.20+rs*.15+vs*.20)
    final=max(0,min(100,final))
    if   final>=75: verdict,vc="✅ STRONG BUY","green"
    elif final>=62: verdict,vc="🟢 BUY","green"
    elif final>=48: verdict,vc="⚪ ACCUMULATE","orange"
    elif final>=35: verdict,vc="🟡 HOLD","orange"
    else:           verdict,vc="🔴 AVOID","red"
    report=None
    if c:
        ctx=(f"Stock:{symbol} Score:{final}/100 Verdict:{verdict}\n"
             f"News:{n['sent']}({ns}) Fundamental:{f['verdict']}({fs}) "
             f"Technical:{t['direction']}({ts}) Risk:{r['level']}({r['score']}) "
             f"Valuation:{v['verdict']} Action:{v['action']}\n"
             f"Thesis:{v['thesis']}")
        report=_ask(c,"You are a senior equity research analyst at a top Indian firm. "
            "Write a clear research note for retail investors. Plain English. 150-200 words.",
            f"Write research note:\n{ctx}\nStart with verdict. Cover positives, risks, recommendation.",400)
    if not report:
        report=(f"**{verdict}** — Score: {final}/100\n\n"
                f"**Investment Thesis:** {v['thesis']}\n\n"
                f"**Fundamentals:** {f['summary'] or f['verdict']}\n\n"
                f"**Technical:** {t['direction']} | EMA: {t['ema_signal']}\n\n"
                f"**Key Risks:** {', '.join(r['risks'][:2]) or 'None identified'}\n\n"
                f"**Action:** {v['action']}")
    return {"score":final,"verdict":verdict,"color":vc,"report":report,
            "scores":{"news":ns,"fundamental":fs,"technical":ts,
                      "risk":r["score"],"valuation":vs,"final":final},
            "powered":c is not None,"name":"🧠 Master Agent"}

def run_agents(symbol, info, df, news, ema_r, val):
    c=_client()
    n=agent_news(c,symbol,news); f=agent_fundamental(c,symbol,info)
    t=agent_technical(c,symbol,df,ema_r); r=agent_risk(c,symbol,info,df)
    v=agent_valuation(c,symbol,info,val)
    m=agent_master(c,symbol,n,f,t,r,v)
    return {"news":n,"fund":f,"tech":t,"risk":r,"val":v,
            "master":m,"has_groq":c is not None}
