"""strategies.py — 4 Trading Strategies for NSE India"""
import numpy as np
import pandas as pd
from data import RISK_FREE, COST

def _metrics(ret, name):
    r=ret.dropna()
    if len(r)==0:
        return {"name":name,"sharpe":0,"sortino":0,"annual":"0.00%",
                "total":"0.00%","vol":"0.00%","max_dd":"0.00%",
                "calmar":0,"win":"0.00%","_sh":0,"_tot":0,"_mdd":0,
                "_cum":pd.Series([1.0])}
    ann=r.mean()*252; vol=r.std()*np.sqrt(252)+1e-10
    sharpe=(ann-RISK_FREE)/vol
    dn=r[r<0].std()*np.sqrt(252)+1e-10
    sortino=(ann-RISK_FREE)/dn
    cum=(1+r).cumprod()
    dd=(cum-cum.expanding().max())/cum.expanding().max()
    mdd=dd.min()
    calmar=ann/abs(mdd) if mdd<0 else 0
    tot=cum.iloc[-1]-1
    return {"name":name,"sharpe":round(sharpe,3),
            "sortino":round(sortino,3),"annual":f"{ann:.2%}",
            "total":f"{tot:.2%}","vol":f"{vol:.2%}",
            "max_dd":f"{mdd:.2%}","calmar":round(calmar,3),
            "win":f"{(r>0).mean():.2%}","_sh":sharpe,
            "_tot":tot,"_mdd":mdd,"_cum":cum}

def _cost(sig, ret):
    return sig.shift(1)*ret - sig.diff().abs().fillna(0)*COST

def sma_crossover(df, short=20, long=50):
    d=df.copy(); c=d["Close"].squeeze()
    d["SMA_S"]=c.rolling(short).mean()
    d["SMA_L"]=c.rolling(long).mean()
    d["Signal"]=np.where(d["SMA_S"]>d["SMA_L"],1,0)
    d["Returns"]=c.pct_change()
    d["Strat"]=_cost(d["Signal"],d["Returns"])
    d["BH"]=d["Returns"]
    d["Position"]=d["Signal"].diff()
    return {"data":d,"metrics":_metrics(d["Strat"],"SMA Crossover"),
            "bh":_metrics(d["BH"],"Buy & Hold")}

def rsi_strategy(df, oversold=35, overbought=65):
    d=df.copy(); c=d["Close"].squeeze()
    d["Returns"]=c.pct_change()
    sig=pd.Series(0,index=d.index); pos=False
    for i in range(1,len(d)):
        rsi=d["RSI"].iloc[i]
        if not pos and rsi<oversold: pos=True
        elif pos and rsi>overbought: pos=False
        sig.iloc[i]=1 if pos else 0
    d["Signal"]=sig
    d["Strat"]=_cost(d["Signal"],d["Returns"])
    d["BH"]=d["Returns"]   # FIX: BH always present for chart
    return {"data":d,"metrics":_metrics(d["Strat"],"RSI Mean Reversion"),
            "bh":_metrics(d["BH"],"Buy & Hold")}

def macd_strategy(df):
    d=df.copy(); c=d["Close"].squeeze()
    d["Returns"]=c.pct_change()
    cross=np.where((d["MACD"]>d["MACD_sig"])&
                   (d["MACD"].shift(1)<=d["MACD_sig"].shift(1)),1,0)
    sig=pd.Series(cross,index=d.index)
    sig=sig.replace(0,np.nan).ffill().fillna(0)
    sig=np.where(d["MACD"]<d["MACD_sig"],0,sig)
    d["Signal"]=sig
    d["Strat"]=_cost(d["Signal"],d["Returns"])
    d["BH"]=d["Returns"]   # FIX: BH always present for chart
    return {"data":d,"metrics":_metrics(d["Strat"],"MACD Momentum"),
            "bh":_metrics(d["BH"],"Buy & Hold")}

def bollinger_strategy(df):
    d=df.copy(); c=d["Close"].squeeze()
    d["Returns"]=c.pct_change()
    d["Signal"]=np.where(c>d["BB_up"],1,0)
    d["Signal"]=np.where(c<d["BB_mid"],0,d["Signal"])
    d["Strat"]=_cost(d["Signal"],d["Returns"])
    d["BH"]=d["Returns"]   # FIX: BH always present for chart
    return {"data":d,"metrics":_metrics(d["Strat"],"Bollinger Breakout"),
            "bh":_metrics(d["BH"],"Buy & Hold")}

def run_all_strategies(df):
    return {
        "SMA Crossover":     sma_crossover(df),
        "RSI Mean Reversion":rsi_strategy(df),
        "MACD Momentum":     macd_strategy(df),
        "Bollinger Breakout":bollinger_strategy(df),
    }

def optimise_sma(df, step=5):
    results=[]
    for s in range(5,50,step):
        for l in range(20,200,step):
            if s>=l: continue
            try:
                r=sma_crossover(df,s,l)
                m=r["metrics"]
                # FIX: only add if _sh is a valid number
                sh=m["_sh"]
                if sh is not None and not (isinstance(sh,float) and np.isnan(sh)):
                    results.append({"Short":s,"Long":l,
                        "Sharpe":sh,"Return":m["_tot"],"MaxDD":m["_mdd"]})
            except: continue
    df_r=pd.DataFrame(results)
    if df_r.empty:
        return df_r, None
    # FIX: drop NaN Sharpe before idxmax
    df_r_clean=df_r.dropna(subset=["Sharpe"])
    if df_r_clean.empty:
        return df_r, None
    best=df_r_clean.loc[df_r_clean["Sharpe"].idxmax()]
    return df_r, best
