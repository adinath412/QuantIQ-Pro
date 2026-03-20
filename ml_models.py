"""ml_models.py — ML Models for NSE India Stock Prediction"""
import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.stats import jarque_bera

SPLIT=0.70; RS=42
MIN_ROWS=100   # Minimum rows needed for ML

def _prep(df, targets=["Direction"], extra=[]):
    feats=["Ret_Lag1","Ret_Lag2","Ret_Lag3","Ret_Lag4","Ret_Lag5",
           "RSI","MACD_h","BB_pct","ATR_pct","Mom5","Mom20",
           "VolRatio","MACD","BB_up","BB_lo"]
    feats=[f for f in feats+extra if f in df.columns]
    cols=feats+[t for t in targets if t in df.columns]
    d=df[cols].dropna()
    if len(d) < MIN_ROWS:
        raise ValueError(f"Need at least {MIN_ROWS} clean rows, got {len(d)}")
    split=int(len(d)*SPLIT)
    sc=StandardScaler()
    X=d[feats].values
    Xtr=sc.fit_transform(X[:split]); Xte=sc.transform(X[split:])
    return Xtr,Xte,d,feats,split,sc

# ── Random Walk Test ──────────────────────────────────────
def random_walk_test(df):
    r=df["Returns"].dropna()
    if len(r) < 20:
        return {"autocorr":[0]*15,"jb_stat":0,"jb_pval":1,
                "normal":True,"predictable":False,"max_ac":0}
    ac=[r.autocorr(lag=i) for i in range(1,16)]
    # Replace NaN autocorr values with 0
    ac=[0 if (isinstance(v,float) and np.isnan(v)) else v for v in ac]
    jb_s,jb_p=jarque_bera(r)
    max_ac=max(map(abs,ac)) if ac else 0
    return {"autocorr":ac,"jb_stat":jb_s,"jb_pval":jb_p,
            "normal":jb_p>0.05,"predictable":max_ac>0.05,"max_ac":max_ac}

# ── Linear Regression ─────────────────────────────────────
def linear_regression(df):
    feats=["Ret_Lag1","Ret_Lag2","Ret_Lag3","RSI","MACD_h",
           "BB_pct","ATR_pct","Mom5","VolRatio"]
    feats=[f for f in feats if f in df.columns]
    if "Returns" not in df.columns or len(feats)==0: return None
    d=df[feats+["Returns"]].dropna()
    d=d.copy(); d["Target"]=d["Returns"].shift(-1); d=d.dropna()
    if len(d) < MIN_ROWS: return None   # FIX: guard insufficient data
    split=int(len(d)*SPLIT)
    sc=StandardScaler()
    Xtr=sc.fit_transform(d[feats].values[:split])
    Xte=sc.transform(d[feats].values[split:])
    ytr=d["Target"].values[:split]; yte=d["Target"].values[split:]
    m=LinearRegression(); m.fit(Xtr,ytr)
    fi=pd.DataFrame({"Feature":feats,"Coef":m.coef_}).assign(
        Abs=lambda x: x["Coef"].abs()
    ).sort_values("Abs",ascending=False)
    return {"train_r2":m.score(Xtr,ytr),"test_r2":m.score(Xte,yte),
            "feature_importance":fi,"model":m}

def run_regression(df):
    """Wrapper — always returns safe result dict."""
    try:
        r=linear_regression(df)
        return r if r else {}
    except: return {}

# ── K-Means Regime Detection ──────────────────────────────
def kmeans_regimes(df, k=4):
    cols=["Returns","VolRatio","ATR_pct","RSI","Mom5","MACD_h"]
    cols=[c for c in cols if c in df.columns]
    d=df[cols].dropna()
    if len(d) < max(k*10, MIN_ROWS):   # need at least k*10 rows
        # Return dummy result
        d2=df[cols].dropna()
        if len(d2)==0: d2=pd.DataFrame({"Returns":[0],"Regime":["⚪ Sideways"]})
        d2=d2.copy(); d2["Regime"]="⚪ Sideways"; d2["Cluster"]=0
        return {"data":d2,"summary":pd.DataFrame(),"labels":{0:"⚪ Sideways"},
                "inertias":[0],"k":1}
    sc=StandardScaler(); X=sc.fit_transform(d)
    iner=[KMeans(n_clusters=i,random_state=RS,n_init=10).fit(X).inertia_
          for i in range(2,min(8,len(d)//10))]
    km=KMeans(n_clusters=k,random_state=RS,n_init=10)
    d=d.copy(); d["Cluster"]=km.fit_predict(X)
    summ=d.groupby("Cluster")[cols].mean()
    labels={}
    for cl in range(k):
        ret=summ.loc[cl,"Returns"] if "Returns" in summ else 0
        vol=summ.loc[cl,"ATR_pct"] if "ATR_pct" in summ else 0
        if ret>0.001 and vol<0.02:    labels[cl]="🟢 Bull Market"
        elif ret<-0.001 and vol>0.02: labels[cl]="🔴 Bear Market"
        elif vol>0.025:               labels[cl]="🟡 High Volatility"
        else:                         labels[cl]="⚪ Sideways"
    d["Regime"]=d["Cluster"].map(labels)
    return {"data":d,"summary":summ,"labels":labels,
            "inertias":iner,"k":k}


# ── Random Forest ─────────────────────────────────────────
def random_forest(df):
    Xtr,Xte,d,feats,split,sc=_prep(df)
    ytr=d["Direction"].values[:split]; yte=d["Direction"].values[split:]
    tscv=TimeSeriesSplit(n_splits=5); cv=[]
    for ti,vi in tscv.split(Xtr):
        if len(ti)<10: continue
        rf_=RandomForestClassifier(n_estimators=100,random_state=RS,n_jobs=-1)
        rf_.fit(Xtr[ti],ytr[ti])
        cv.append(accuracy_score(ytr[vi],rf_.predict(Xtr[vi])))
    rf=RandomForestClassifier(n_estimators=100,random_state=RS,
                               oob_score=True,n_jobs=-1)
    rf.fit(Xtr,ytr); yp=rf.predict(Xte)
    acc=accuracy_score(yte,yp)
    try: roc=roc_auc_score(yte,rf.predict_proba(Xte)[:,1])
    except: roc=0.5
    fi=pd.DataFrame({"Feature":feats,"Importance":rf.feature_importances_}
                    ).sort_values("Importance",ascending=False)
    base=max(yte.mean(),1-yte.mean())
    return {"accuracy":acc,"roc_auc":roc,"baseline":base,
            "cv_mean":float(np.mean(cv)) if cv else 0.5,
            "oob":rf.oob_score_,
            "feature_importance":fi,"model":rf,"scaler":sc}

# ── Gradient Boosting ─────────────────────────────────────
def gradient_boosting(df):
    Xtr,Xte,d,feats,split,sc=_prep(df)
    ytr=d["Direction"].values[:split]; yte=d["Direction"].values[split:]
    gb=GradientBoostingClassifier(n_estimators=200,learning_rate=0.05,
                                   max_depth=4,subsample=0.8,random_state=RS)
    gb.fit(Xtr,ytr); yp=gb.predict(Xte)
    acc=accuracy_score(yte,yp); base=max(yte.mean(),1-yte.mean())
    fi=pd.DataFrame({"Feature":feats,"Importance":gb.feature_importances_}
                    ).sort_values("Importance",ascending=False)
    return {"accuracy":acc,"baseline":base,"feature_importance":fi,
            "model":gb,"scaler":sc}

# ── Neural Network ────────────────────────────────────────
def neural_network(df):
    Xtr,Xte,d,feats,split,sc=_prep(df)
    ytr=d["Direction"].values[:split]; yte=d["Direction"].values[split:]
    nn=MLPClassifier(hidden_layer_sizes=(64,32),activation="relu",
                     solver="adam",alpha=0.001,max_iter=500,
                     random_state=RS,early_stopping=True,
                     validation_fraction=0.15,n_iter_no_change=20)
    nn.fit(Xtr,ytr); yp=nn.predict(Xte)
    acc=accuracy_score(yte,yp); base=max(yte.mean(),1-yte.mean())
    return {"accuracy":acc,"baseline":base,
            "loss_curve":getattr(nn,'loss_curve_',[]),
            "model":nn,"scaler":sc}

# ── Run all ───────────────────────────────────────────────
def run_all_ml(df):
    results={}
    try: results["Random Forest"]=random_forest(df)
    except: pass
    try: results["Gradient Boosting"]=gradient_boosting(df)
    except: pass
    try: results["Neural Network"]=neural_network(df)
    except: pass
    return results
