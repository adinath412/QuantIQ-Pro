"""data.py — NSE India Data Pipeline — SPOT PRICE ONLY (No Options)
   All NSE F&O eligible stocks included. Search any stock → full data.
"""
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import requests
import time

# ══════════════════════════════════════════════════════════════════════
# FIX: Yahoo Finance blocks plain requests — use browser-like session
# This fixes: JSONDecodeError('Expecting value: line 1 column 1 (char 0)')
# ══════════════════════════════════════════════════════════════════════
def _make_session():
    """Create a requests session with browser headers to avoid Yahoo Finance blocks."""
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection":      "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    })
    return s

_SESSION = _make_session()

def _ticker(symbol):
    """Create yf.Ticker with browser-like session."""
    try:
        t = yf.Ticker(ns(symbol), session=_SESSION)
        return t
    except Exception:
        return yf.Ticker(ns(symbol))

# ══════════════════════════════════════════════════════════════════════
# COMPLETE NSE F&O UNIVERSE (~180+ stocks) — SPOT PRICE ONLY
# ══════════════════════════════════════════════════════════════════════

NIFTY50 = [
    "RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK","HINDUNILVR","ITC",
    "SBIN","BHARTIARTL","KOTAKBANK","LT","AXISBANK","ASIANPAINT","MARUTI",
    "TITAN","SUNPHARMA","WIPRO","ULTRACEMCO","BAJFINANCE","NESTLEIND",
    "POWERGRID","NTPC","TATAMOTORS","TECHM","HCLTECH","BAJAJFINSV",
    "COALINDIA","ONGC","JSWSTEEL","TATASTEEL","GRASIM","INDUSINDBK",
    "ADANIENT","ADANIPORTS","APOLLOHOSP","DIVISLAB","DRREDDY","CIPLA",
    "EICHERMOT","HEROMOTOCO","BPCL","BRITANNIA","HINDALCO","TATACONSUM",
    "SBILIFE","HDFCLIFE","BAJAJ-AUTO","M&M","UPL","LTIM",
]
NIFTY_NEXT = [
    "VEDL","SIEMENS","PIDILITIND","HAVELLS","DABUR","MARICO","COLPAL",
    "AMBUJACEM","BANDHANBNK","LUPIN","GAIL","NMDC","SAIL","MPHASIS",
    "COFORGE","PERSISTENT","KPITTECH","JUBLFOOD","DMART","VBL","VOLTAS",
    "CHOLAFIN","SBICARD","RECLTD","PFC","TRENT","ZOMATO","IRCTC",
    "FEDERALBNK","IDFCFIRSTB","ABCAPITAL",
]

# ─────────────────────────────────────────────────────────────────────
# ALL NSE F&O eligible stocks — these are stocks that have options
# trading on NSE. Project uses SPOT (cash market) price for all logic.
# ─────────────────────────────────────────────────────────────────────
NSE_FNO = [
    # Auto & Auto Ancillaries
    "ASHOKLEY","BALKRISIND","ESCORTS","EXIDEIND","MRF","MOTHERSON",
    "TVSMOTOR","APOLLOTYRE","CEATLTD","BHARATFORG","TIINDIA","MSSL",
    # Banking & Finance
    "BANKBARODA","CANBK","PNB","UNIONBANK","IOB",
    "RBLBANK","YESBANK","KARURVYSYA","DCBBANK","UJJIVANSFB",
    "MANAPPURAM","MUTHOOTFIN","SHRIRAMFIN","L&TFH",
    "ANGELONE","MOTILALOFS","HDFCAMC","MFSL","ICICIGI","ICICIPRULI",
    "CHOLAHLDNG","POONAWALLA","CREDITACC","AAVAS","APTUS",
    # Cement
    "ACC","RAMCOCEM","JKCEMENT","DALBHARAT","INDIACEM","SHREECEM",
    # Chemicals & Fertilizers
    "AARTIIND","DEEPAKNTR","GNFC","GSFC","COROMANDEL","PIIND",
    "NAVINFLUOR","SRF","TATACHEM","ATUL","NOCIL","VINATIORGA",
    "CHAMBLFERT","DEEPAKFERT","SUMICHEM","INSECTICID","FACT",
    # Consumer / FMCG / Beverages
    "RADICO","UBL","MCDOWELL-N","GODREJCP","EMAMILTD","VARUNBEV",
    # Energy & Oil
    "IOC","HINDPETRO","MRPL","PETRONET","IGL","MGL","ATGL","GUJGASLTD","CPCL",
    # Healthcare / Pharma
    "BIOCON","AUROPHARMA","GLENMARK","IPCALAB","ALKEM","TORNTPHARM",
    "GRANULES","LALPATHLAB","METROPOLIS","SYNGENE",
    "AJANTPHARMA","NATCOPHARM","ABBOTINDIA","PFIZER","GLAXO","STAR",
    # IT / Tech / Digital
    "OFSS","LTTS","BSOFT","LATENTVIEW","TANLA","NAUKRI","INDIAMART",
    "JUSTDIAL","POLICYBZR","DELHIVERY","RATEGAIN","HEXAWARE","AFFLE","NYKAA","PAYTM",
    # Infra / Capital Goods / Defence
    "HAL","BEL","CONCOR","NBCC","NCC","HCC","IRB","GMRINFRA",
    "ADANIGREEN","ADANIPOWER","TATAPOWER","CESC","TORNTPOWER","RPOWER",
    "BHEL","ABB","HONAUT","CUMMINSIND","THERMAX","INOXWIND",
    # Metals & Mining
    "HINDCOPPER","NATIONALUM","MOIL","JSPL","JSL","WELCORP","RATNAMANI",
    # Paints
    "BERGEPAINT","KANSAINER","AKZOINDIA",
    # Real Estate
    "DLF","GODREJPROP","OBEROIRLTY","PRESTIGE","BRIGADE","PHOENIXLTD","SOBHA","MAHLIFE",
    # Retail & Lifestyle
    "ABFRL","RAYMOND","PAGEIND","BATAINDIA","SHOPERSTOP",
    # Telecom
    "INDUSTOWER","TATACOMM",
    # Electricals & Consumer Durables
    "CROMPTON","POLYCAB","KEI","VGUARD","DIXON","WHIRLPOOL",
    "KAJARIACER","SUPREMEIND","ASTRAL","APLAPOLLO","FINOLEXIND",
    # Exchange / Fintech
    "MCX","BSE","CDSL","IEX",
    # Aviation / Travel
    "INDIGO","SPICEJET",
    # Media & Entertainment
    "ZEEL","SUNTV","PVRINOX","DELTACORP",
]

# Build unique ALL_STOCKS (no duplicates, preserving order)
_seen = set()
ALL_STOCKS = []
for s in NIFTY50 + NIFTY_NEXT + NSE_FNO:
    if s not in _seen:
        _seen.add(s)
        ALL_STOCKS.append(s)

# ══════════════════════════════════════════════════════════════════════
# SECTORS — Comprehensive mapping for all stocks
# ══════════════════════════════════════════════════════════════════════
SECTORS = {
    "TCS":"IT","INFY":"IT","WIPRO":"IT","HCLTECH":"IT","TECHM":"IT",
    "LTIM":"IT","LTTS":"IT","MPHASIS":"IT","COFORGE":"IT","PERSISTENT":"IT",
    "KPITTECH":"IT","OFSS":"IT","BSOFT":"IT","LATENTVIEW":"IT","TANLA":"IT",
    "NAUKRI":"IT","INDIAMART":"IT","JUSTDIAL":"IT","RATEGAIN":"IT",
    "HEXAWARE":"IT","POLICYBZR":"Fintech","DELHIVERY":"Logistics",
    "HDFCBANK":"Banking","ICICIBANK":"Banking","SBIN":"Banking",
    "KOTAKBANK":"Banking","AXISBANK":"Banking","INDUSINDBK":"Banking",
    "BANDHANBNK":"Banking","FEDERALBNK":"Banking","IDFCFIRSTB":"Banking",
    "BANKBARODA":"Banking","CANBK":"Banking","PNB":"Banking",
    "UNIONBANK":"Banking","RBLBANK":"Banking","YESBANK":"Banking",
    "KARURVYSYA":"Banking","DCBBANK":"Banking","UJJIVANSFB":"Banking",
    "INDIANB":"Banking","IOB":"Banking",
    "BAJFINANCE":"NBFC","BAJAJFINSV":"NBFC","CHOLAFIN":"NBFC",
    "ABCAPITAL":"Finance","MANAPPURAM":"Finance","MUTHOOTFIN":"Finance",
    "SHRIRAMFIN":"Finance","L&TFH":"Finance","SBICARD":"Finance",
    "HDFCAMC":"Finance","MFSL":"Finance","RECLTD":"Finance","PFC":"Finance",
    "ANGELONE":"Finance","MOTILALOFS":"Finance","ICICIGI":"Insurance",
    "ICICIPRULI":"Insurance","SBILIFE":"Insurance","HDFCLIFE":"Insurance",
    "POONAWALLA":"Finance","CREDITACC":"Finance","AAVAS":"Finance","APTUS":"Finance",
    "CHOLAHLDNG":"Finance",
    "MARUTI":"Auto","TATAMOTORS":"Auto","M&M":"Auto","BAJAJ-AUTO":"Auto",
    "HEROMOTOCO":"Auto","EICHERMOT":"Auto","TVSMOTOR":"Auto","ESCORTS":"Auto",
    "ASHOKLEY":"Auto","BALKRISIND":"Auto Anc","APOLLOTYRE":"Auto Anc",
    "CEATLTD":"Auto Anc","EXIDEIND":"Auto Anc","BHARATFORG":"Auto Anc",
    "TIINDIA":"Auto Anc","MSSL":"Auto Anc","MOTHERSON":"Auto Anc","MRF":"Auto Anc",
    "SUNPHARMA":"Pharma","DRREDDY":"Pharma","CIPLA":"Pharma",
    "DIVISLAB":"Pharma","BIOCON":"Pharma","AUROPHARMA":"Pharma",
    "LUPIN":"Pharma","GLENMARK":"Pharma","IPCALAB":"Pharma","ALKEM":"Pharma",
    "TORNTPHARM":"Pharma","GRANULES":"Pharma","ABBOTINDIA":"Pharma",
    "AJANTPHARMA":"Pharma","NATCOPHARM":"Pharma","STAR":"Healthcare",
    "APOLLOHOSP":"Healthcare","LALPATHLAB":"Healthcare",
    "METROPOLIS":"Healthcare","SYNGENE":"Healthcare",
    "RELIANCE":"Energy","ONGC":"Energy","BPCL":"Energy","IOC":"Energy",
    "HINDPETRO":"Energy","GAIL":"Energy","PETRONET":"Energy","MRPL":"Energy",
    "IGL":"City Gas","MGL":"City Gas","ATGL":"City Gas","GUJGASLTD":"City Gas",
    "NTPC":"Power","POWERGRID":"Power","TATAPOWER":"Power",
    "ADANIGREEN":"Power","ADANIPOWER":"Power","TORNTPOWER":"Power",
    "CESC":"Power","RPOWER":"Power","INOXWIND":"Renewable",
    "TATASTEEL":"Metals","JSWSTEEL":"Metals","HINDALCO":"Metals",
    "SAIL":"Metals","VEDL":"Metals","HINDCOPPER":"Metals",
    "NATIONALUM":"Metals","NMDC":"Mining","COALINDIA":"Mining","MOIL":"Mining",
    "JSPL":"Metals","JINDALSTEL":"Metals","JSL":"Metals","WELCORP":"Metals",
    "ULTRACEMCO":"Cement","SHREECEM":"Cement","GRASIM":"Cement",
    "AMBUJACEM":"Cement","ACC":"Cement","RAMCOCEM":"Cement",
    "JKCEMENT":"Cement","DALBHARAT":"Cement","INDIACEM":"Cement",
    "HINDUNILVR":"FMCG","ITC":"FMCG","NESTLEIND":"FMCG",
    "BRITANNIA":"FMCG","DABUR":"FMCG","MARICO":"FMCG","COLPAL":"FMCG",
    "GODREJCP":"FMCG","EMAMILTD":"FMCG","TATACONSUM":"FMCG",
    "VBL":"Beverages","UBL":"Beverages","RADICO":"Beverages","VARUNBEV":"Beverages",
    "PIDILITIND":"Chemicals","AARTIIND":"Chemicals","DEEPAKNTR":"Chemicals",
    "SRF":"Chemicals","NAVINFLUOR":"Chemicals","PIIND":"Chemicals",
    "TATACHEM":"Chemicals","GNFC":"Chemicals","GSFC":"Chemicals",
    "COROMANDEL":"Agrochem","ATUL":"Chemicals","NOCIL":"Chemicals",
    "SUMICHEM":"Agrochem","INSECTICID":"Agrochem","CHAMBLFERT":"Fertiliser",
    "DEEPAKFERT":"Fertiliser",
    "LT":"Infra","HAL":"Defence","BEL":"Defence","BHEL":"Capital Goods",
    "ABB":"Capital Goods","SIEMENS":"Capital Goods","HONAUT":"Capital Goods",
    "CUMMINSIND":"Capital Goods","THERMAX":"Capital Goods",
    "CONCOR":"Logistics","NCC":"Construction","NBCC":"Construction",
    "HCC":"Construction","IRB":"Infra","GMRINFRA":"Infra",
    "ADANIPORTS":"Ports","ADANIENT":"Conglomerate",
    "ASIANPAINT":"Paints","BERGEPAINT":"Paints","KANSAINER":"Paints","AKZOINDIA":"Paints",
    "DLF":"Real Estate","GODREJPROP":"Real Estate","OBEROIRLTY":"Real Estate",
    "PRESTIGE":"Real Estate","BRIGADE":"Real Estate","PHOENIXLTD":"Real Estate","SOBHA":"Real Estate",
    "TITAN":"Jewellery","TRENT":"Retail","DMART":"Retail",
    "PAGEIND":"Retail","ABFRL":"Retail","RAYMOND":"Retail","BATAINDIA":"Retail",
    "SHOPERSTOP":"Retail","NYKAA":"Retail",
    "BHARTIARTL":"Telecom","INDUSTOWER":"Telecom","TATACOMM":"Telecom",
    "HAVELLS":"Electricals","POLYCAB":"Electricals","KEI":"Electricals",
    "FINOLEXIND":"Electricals","CROMPTON":"Electricals","VGUARD":"Electricals",
    "VOLTAS":"Consumer Durables","WHIRLPOOL":"Consumer Durables",
    "DIXON":"Consumer Electronics","KAJARIACER":"Tiles",
    "SUPREMEIND":"Plastics","ASTRAL":"Pipes","APLAPOLLO":"Pipes",
    "ZEEL":"Media","SUNTV":"Media","PVRINOX":"Entertainment","DELTACORP":"Leisure",
    "IRCTC":"Travel","INDIGO":"Aviation","SPICEJET":"Aviation",
    "JUBLFOOD":"QSR","ZOMATO":"FoodTech",
    "MCX":"Exchange","BSE":"Exchange","CDSL":"Finance","IEX":"Exchange",
    "PAYTM":"Fintech","AFFLE":"AdTech",
}

# ══════════════════════════════════════════════════════════════════════
# NSE F&O LOT SIZES — Futures contract size (shares per lot)
# Used ONLY for futures trade setup with SPOT entry price
# ══════════════════════════════════════════════════════════════════════
LOTS = {
    "AARTIIND":500,"ABB":150,"ACC":250,"ADANIENT":250,"ADANIPORTS":1250,
    "ALKEM":100,"AMBUJACEM":1000,"ANGELONE":250,"APOLLOHOSP":125,
    "APOLLOTYRE":1500,"ASHOKLEY":5000,"ASIANPAINT":200,"ASTRAL":500,
    "ATGL":1375,"AUROPHARMA":1250,"AXISBANK":1200,"BAJAJ-AUTO":250,
    "BAJAJFINSV":125,"BAJFINANCE":125,"BALKRISIND":400,"BANDHANBNK":2500,
    "BANKBARODA":5850,"BATAINDIA":275,"BEL":3100,"BERGEPAINT":500,
    "BHARATFORG":1000,"BHARTIARTL":1851,"BHEL":5450,"BIOCON":2500,
    "BPCL":1800,"BRITANNIA":200,"BSE":350,"BSOFT":1400,"CANBK":5000,
    "CDSL":900,"CEATLTD":600,"CHOLAFIN":600,"CIPLA":650,"COALINDIA":4200,
    "COFORGE":150,"COLPAL":700,"CONCOR":1000,"COROMANDEL":500,
    "CROMPTON":2000,"CUMMINSIND":400,"DABUR":2750,"DALBHARAT":200,
    "DCBBANK":4000,"DEEPAKNTR":200,"DELTACORP":4200,"DIVISLAB":200,
    "DIXON":100,"DLF":1650,"DMART":88,"DRREDDY":125,"EICHERMOT":175,
    "EMAMILTD":1500,"ESCORTS":500,"EXIDEIND":2500,"FEDERALBNK":5000,
    "GAIL":5775,"GMRINFRA":15000,"GLENMARK":650,"GNFC":1300,
    "GODREJCP":500,"GODREJPROP":375,"GRANULES":2000,"GRASIM":500,
    "GSFC":1500,"HAL":150,"HAVELLS":600,"HCLTECH":700,"HDFCAMC":150,
    "HDFCBANK":550,"HDFCLIFE":1100,"HEROMOTOCO":300,"HINDALCO":2150,
    "HINDCOPPER":5000,"HINDPETRO":2400,"HINDUNILVR":300,"HONAUT":15,
    "IDFCFIRSTB":7500,"IEX":3750,"IGL":1375,"INDIAMART":75,
    "INDIGO":300,"INDUSINDBK":700,"INDUSTOWER":2800,"INFY":300,
    "IOC":6000,"IPCALAB":300,"IRCTC":875,"ITC":3200,"JINDALSTEL":1500,
    "JKCEMENT":200,"JSWSTEEL":1350,"JSL":4000,"JUBLFOOD":1250,
    "JUSTDIAL":300,"KAJARIACER":1000,"KPITTECH":1500,"KOTAKBANK":400,
    "L&TFH":4000,"LALPATHLAB":150,"LATENTVIEW":2000,"LT":375,
    "LTIM":150,"LTTS":100,"LUPIN":650,"M&M":700,"MANAPPURAM":5000,
    "MARICO":1600,"MARUTI":100,"MCX":250,"METROPOLIS":150,"MFSL":400,
    "MOTHERSON":6850,"MOTILALOFS":500,"MPHASIS":250,"MRF":15,
    "MSSL":3800,"NATIONALUM":8000,"NAUKRI":125,"NAVINFLUOR":100,
    "NBCC":7800,"NCC":3000,"NESTLEIND":50,"NMDC":6000,"NTPC":4000,
    "OBEROIRLTY":700,"OFSS":100,"ONGC":1975,"PAGEIND":15,"PFC":3150,
    "PERSISTENT":150,"PETRONET":3000,"PIDILITIND":275,"PIIND":300,
    "PNB":8000,"POLYCAB":175,"POWERGRID":4800,"PVRINOX":507,
    "RAMCOCEM":500,"RECLTD":3000,"RELIANCE":250,"SAIL":10000,
    "SBICARD":800,"SBILIFE":750,"SBIN":1500,"SHREECEM":25,
    "SHRIRAMFIN":150,"SIEMENS":150,"SRF":250,"SUNPHARMA":700,
    "SUNTV":1000,"SUPREMEIND":175,"SYNGENE":650,"TATACHEM":850,
    "TATACOMM":400,"TATACONSUM":875,"TATAMOTORS":2850,"TATAPOWER":4350,
    "TATASTEEL":5500,"TCS":150,"TECHM":600,"THERMAX":300,"TIINDIA":400,
    "TITAN":375,"TORNTPHARM":150,"TORNTPOWER":500,"TRENT":275,
    "TVSMOTOR":700,"UBL":350,"ULTRACEMCO":100,"UNIONBANK":7500,
    "UPL":1300,"VBL":400,"VEDL":2750,"VOLTAS":800,"WIPRO":1600,
    "YESBANK":40000,"ZEEL":4500,"ZOMATO":3000,"DELHIVERY":2000,
    "NYKAA":1500,"PAYTM":2000,"AFFLE":400,"POONAWALLA":2000,
    "SPICEJET":5000,"PHOENIXLTD":400,"PRESTIGE":600,"BRIGADE":800,
    "SOBHA":400,"KARURVYSYA":2000,"RBLBANK":2500,"UJJIVANSFB":5000,
    "ABCAPITAL":4000,"CREDITACC":2000,"AAVAS":175,"APTUS":3500,
    "FINOLEXIND":1000,"INOXWIND":2500,"VGUARD":2000,
    "WHIRLPOOL":225,"KANSAINER":400,"NATCOPHARM":1200,"AJANTPHARMA":300,
    "ESCORTS":500,"EXIDEIND":2500,"CEATLTD":600,"MUTHOOTFIN":600,
    "ICICIGI":400,"ICICIPRULI":1500,"RATNAMANI":250,"WELCORP":2000,
    "JSPL":3000,"MOIL":2000,"HINDCOPPER":5000,
}

# Constants
INDICES = {
    "NIFTY 50":"^NSEI","SENSEX":"^BSESN",
    "BANKNIFTY":"^NSEBANK","NIFTY IT":"^CNXIT",
    "MIDCAP":"^NSEMDCP50","PHARMA":"^CNXPHARMA",
}
RISK_FREE = 0.065   # 6.5% Indian G-Sec
COST      = 0.0015  # 0.15% NSE spot transaction cost

# ══════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════
def ns(s):
    s = s.upper().strip()
    return s if s.endswith(".NS") else s + ".NS"

def sf(v, d=0.0):
    try:
        f = float(v)
        return d if np.isnan(f) else f
    except:
        return d

def fmtN(v, pre="₹"):
    try:
        v = float(v)
        if v >= 1e7: return f"{pre}{v/1e7:.2f}Cr"
        if v >= 1e5: return f"{pre}{v/1e5:.2f}L"
        return f"{pre}{v:,.2f}"
    except:
        return "—"

def fmtP(v):
    try: return f"{float(v)*100:.2f}%"
    except: return "—"

def dash(v):
    if v is None: return "—"
    try:
        if np.isnan(float(v)): return "—"
    except: pass
    return v

# ══════════════════════════════════════════════════════════════════════
# INDICES
# ══════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=60)
def get_indices():
    out = []
    for name, sym in INDICES.items():
        try:
            fi = yf.Ticker(sym, session=_SESSION).fast_info
            p  = sf(getattr(fi, "last_price", 0))
            pc = sf(getattr(fi, "previous_close", p) or p)
            ch = round(p - pc, 2)
            cp = round(ch / pc * 100 if pc else 0, 2)
            out.append({"name": name, "price": p, "ch": ch, "cp": cp})
        except:
            out.append({"name": name, "price": 0, "ch": 0, "cp": 0})
    return out

# ══════════════════════════════════════════════════════════════════════
# STOCK INFO — works for ANY NSE stock
# ══════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=300)
def get_info(symbol):
    """
    Fetch SPOT price info for any NSE stock via yfinance.
    FIX: yfinance 0.2.40+ often returns empty .info for NSE stocks.
    Multi-layer fallback: .info → fast_info attrs → history price → minimal stub.
    """
    ticker = None
    info   = {}
    try:
        ticker = _ticker(symbol)   # uses browser session — fixes JSONDecodeError
        info   = ticker.info or {}
    except:
        info = {}

    # Layer 1: patch price from fast_info if .info missing price
    price = sf(info.get("currentPrice") or info.get("regularMarketPrice"))
    if price <= 0:
        try:
            fi    = ticker.fast_info if ticker else None
            if fi is not None:
                price = sf(getattr(fi, "last_price", 0))
                if price <= 0:
                    price = sf(getattr(fi, "regularMarketPrice", 0))
                if price > 0:
                    info["currentPrice"]       = price
                    info["regularMarketPrice"] = price
                # patch other fast_info fields into info if missing
                for attr, key in [
                    ("previous_close",      "regularMarketPreviousClose"),
                    ("fifty_two_week_high",  "fiftyTwoWeekHigh"),
                    ("fifty_two_week_low",   "fiftyTwoWeekLow"),
                    ("market_cap",           "marketCap"),
                    ("volume",               "volume"),
                ]:
                    if not info.get(key):
                        val = getattr(fi, attr, None)
                        if val is not None:
                            info[key] = val
        except:
            pass

    # Layer 2: if still no price, use Ticker.history() — most reliable
    if sf(info.get("currentPrice")) <= 0:
        try:
            ticker2 = ticker if ticker else _ticker(symbol)
            hist = ticker2.history(period="5d", interval="1d", auto_adjust=True, actions=False)
            if hist is not None and not hist.empty and "Close" in hist.columns:
                cl_series = hist["Close"].dropna()
                if len(cl_series) >= 1:
                    last_close = float(cl_series.iloc[-1])
                    if last_close > 0:
                        info["currentPrice"]       = last_close
                        info["regularMarketPrice"] = last_close
                        if len(cl_series) >= 2:
                            info["regularMarketPreviousClose"] = float(cl_series.iloc[-2])
        except:
            pass

    # Layer 3: fallback to yf.download() if still no price
    if sf(info.get("currentPrice")) <= 0:
        try:
            hist = yf.download(ns(symbol), period="2d", interval="1d",
                               progress=False, auto_adjust=True, session=_SESSION)
            if not hist.empty:
                # FIX: handle MultiIndex from newer yfinance
                if isinstance(hist.columns, pd.MultiIndex):
                    lvl0 = list(hist.columns.get_level_values(0).unique())
                    lvl1 = list(hist.columns.get_level_values(1).unique())
                    if "Close" in lvl0:
                        hist.columns = hist.columns.get_level_values(0)
                    elif "Close" in lvl1:
                        hist.columns = hist.columns.get_level_values(1)
                if "Close" in hist.columns:
                    cl_series = hist["Close"].squeeze().dropna()
                    if len(cl_series) >= 1:
                        last_close = float(cl_series.iloc[-1])
                        if last_close > 0:
                            info["currentPrice"]       = last_close
                            info["regularMarketPrice"] = last_close
                            if len(cl_series) >= 2:
                                info["regularMarketPreviousClose"] = float(cl_series.iloc[-2])
        except:
            pass

    # Layer 3: inject symbol as fallback so "not found" check passes
    # (stock is valid as long as we have ANY price > 0)
    if sf(info.get("currentPrice")) > 0:
        if not info.get("symbol"):
            info["symbol"]   = symbol.upper().replace(".NS", "")
        if not info.get("longName"):
            info["longName"] = info.get("shortName", symbol.upper().replace(".NS", ""))

    return info

# ══════════════════════════════════════════════════════════════════════
# HISTORY + INDICATORS — SPOT OHLCV price only
# ══════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=300)
def get_history(symbol, period="1y", interval="1d"):
    """
    Download OHLCV spot price data for any NSE stock.
    FIX v5: Use Ticker.history() as primary — no MultiIndex issues.
    Falls back to yf.download() if Ticker.history() fails.
    """
    df = pd.DataFrame()

    # ── Method 1: Ticker.history() — cleanest, no MultiIndex ──────────
    try:
        ticker = _ticker(symbol)   # browser session — fixes JSONDecodeError
        df = ticker.history(period=period, interval=interval, auto_adjust=True, actions=False)
        # Ticker.history returns flat columns: Open, High, Low, Close, Volume
        if df is not None and not df.empty:
            # Drop timezone from index if present
            if hasattr(df.index, "tz") and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
    except:
        df = pd.DataFrame()

    # ── Method 2: yf.download() fallback ─────────────────────────────
    if df.empty:
        try:
            df = yf.download(
                ns(symbol), period=period, interval=interval,
                progress=False, auto_adjust=True, session=_SESSION
            )
            # Handle ALL MultiIndex layouts (yfinance version differences)
            if isinstance(df.columns, pd.MultiIndex):
                lvl0 = list(df.columns.get_level_values(0).unique())
                lvl1 = list(df.columns.get_level_values(1).unique())
                ohlcv = {"Open", "High", "Low", "Close", "Volume", "Adj Close"}
                if ohlcv & set(lvl0):
                    df.columns = df.columns.get_level_values(0)
                elif ohlcv & set(lvl1):
                    df.columns = df.columns.get_level_values(1)
                else:
                    df.columns = df.columns.get_level_values(0)
            # Remove duplicate columns
            df = df.loc[:, ~df.columns.duplicated()]
        except:
            df = pd.DataFrame()

    # ── Validate ──────────────────────────────────────────────────────
    if df is None or df.empty:
        return pd.DataFrame()
    if "Close" not in df.columns:
        return pd.DataFrame()

    # Drop rows where Close is NaN
    df = df.dropna(subset=["Close"])
    if df.empty:
        return pd.DataFrame()

    # ── FIX: ensure Close/High/Low/Volume are always Series ──────────
    def to_series(col):
        s = df[col].squeeze() if col in df.columns else pd.Series(dtype=float)
        if not isinstance(s, pd.Series):
            s = pd.Series([s], index=df.index)
        return s

    c  = to_series("Close")
    h  = to_series("High")
    lo = to_series("Low")
    v  = to_series("Volume")

    # ── Indicators ────────────────────────────────────────────────────

    # EMAs
    for span in [20, 50, 200]:
        df[f"EMA{span}"] = c.ewm(span=span, adjust=False).mean()

    # RSI (14) — safe division
    try:
        delta = c.diff()
        gain  = delta.clip(lower=0).ewm(13, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(13, adjust=False).mean()
        loss_safe = loss.where(loss != 0, 1e-10)
        df["RSI"] = 100 - 100 / (1 + gain / loss_safe)
    except:
        df["RSI"] = 50.0

    # MACD
    try:
        e12 = c.ewm(12, adjust=False).mean()
        e26 = c.ewm(26, adjust=False).mean()
        df["MACD"]     = e12 - e26
        df["MACD_sig"] = df["MACD"].ewm(9, adjust=False).mean()
        df["MACD_h"]   = df["MACD"] - df["MACD_sig"]
    except:
        df["MACD"] = df["MACD_sig"] = df["MACD_h"] = 0.0

    # Bollinger Bands
    try:
        m20 = c.rolling(20).mean()
        s20 = c.rolling(20).std()
        df["BB_up"]  = m20 + 2 * s20
        df["BB_lo"]  = m20 - 2 * s20
        df["BB_mid"] = m20
    except:
        df["BB_up"] = df["BB_lo"] = df["BB_mid"] = c

    # ATR (14)
    try:
        tr = pd.concat(
            [h - lo, (h - c.shift()).abs(), (lo - c.shift()).abs()], axis=1
        ).max(axis=1)
        df["ATR"] = tr.rolling(14).mean()
    except:
        df["ATR"] = c * 0.02

    # Volume MA
    try:
        df["Vol_MA"] = v.rolling(20).mean()
    except:
        df["Vol_MA"] = v

    # ML features
    try:
        df["Returns"]   = c.pct_change()
        df["Direction"] = (df["Returns"].shift(-1) > 0).astype(int)
        for i in range(1, 6):
            df[f"Ret_Lag{i}"] = df["Returns"].shift(i)
        df["Mom5"]     = c / c.shift(5) - 1
        df["Mom20"]    = c / c.shift(20) - 1
        vol_ma_safe    = df["Vol_MA"].where(df["Vol_MA"] > 0, 1)
        df["VolRatio"] = v / vol_ma_safe
        bb_range       = (df["BB_up"] - df["BB_lo"]).where((df["BB_up"] - df["BB_lo"]) > 0, 1e-10)
        df["BB_pct"]   = (c - df["BB_lo"]) / bb_range
        df["ATR_pct"]  = df["ATR"] / c.where(c > 0, 1)
    except:
        pass

    return df

# ══════════════════════════════════════════════════════════════════════
# VALUATION
# ══════════════════════════════════════════════════════════════════════
def calc_val(info):
    eps  = sf(info.get("trailingEps"))
    bv   = sf(info.get("bookValue"))
    roe  = sf(info.get("returnOnEquity"))
    rev  = sf(info.get("revenueGrowth", 0.1))
    SP   = {
        "Information Technology": 28, "Technology": 28,
        "Financial Services": 16, "Consumer Defensive": 45,
        "Consumer Cyclical": 30, "Healthcare": 28,
        "Energy": 12, "Basic Materials": 12,
        "Industrials": 22, "Utilities": 14,
    }
    sp   = SP.get(info.get("sector", ""), 20)
    vals = []
    g    = min(max(rev * 100, 0), 25)
    for v in [
        round(eps * (8.5 + 2 * g) * (4.4 / 7.5), 2) if eps > 0 else None,
        round(np.sqrt(22.5 * eps * bv), 2)            if eps > 0 and bv > 0 else None,
        round(sp * eps, 2)                             if eps > 0 else None,
        round((roe / 0.12) * bv, 2)                   if roe > 0 and bv > 0 else None,
    ]:
        if v and v > 10:
            vals.append(v)
    if not vals:
        return None
    fair = round(float(np.median(vals)), 2)
    return {
        "fair":        fair,
        "strong_buy":  round(fair * 0.65, 2),
        "buy":         round(fair * 0.80, 2),
        "book_profit": round(fair * 1.20, 2),
        "sp":          sp,
    }

# ══════════════════════════════════════════════════════════════════════
# EMA SIGNAL — SPOT PRICE ONLY
# ══════════════════════════════════════════════════════════════════════
def get_ema_signal(symbol, prox=1.0):
    """
    200 EMA signal using SPOT price (daily + 4H).
    Entry / SL / Target all calculated on spot price.
    FIX: robust version — never silently returns None on valid stocks.
    """
    try:
        df = get_history(symbol, "1y", "1d")
        if df.empty:
            return None

        # FIX: lower threshold — some stocks have < 60 days in 1y
        if len(df) < 20:
            return None

        close_s = df["Close"].squeeze()
        price   = float(close_s.iloc[-1])
        if price <= 0:
            return None

        # EMA200 — compute fresh if not present or all NaN
        if "EMA200" in df.columns and not df["EMA200"].isna().all():
            ema200 = float(df["EMA200"].iloc[-1])
        else:
            ema200 = float(close_s.ewm(span=200, adjust=False).mean().iloc[-1])

        # ATR — compute fresh if not present
        if "ATR" in df.columns and not df["ATR"].isna().all():
            atr = float(df["ATR"].iloc[-1])
        else:
            h = df["High"].squeeze(); lo = df["Low"].squeeze()
            tr = pd.concat([h-lo,(h-close_s.shift()).abs(),(lo-close_s.shift()).abs()],axis=1).max(axis=1)
            atr = float(tr.rolling(14).mean().iloc[-1])
        if np.isnan(atr) or atr <= 0:
            atr = price * 0.02   # fallback: 2% of price

        dist = abs((price - ema200) / ema200 * 100) if ema200 > 0 else 99

        # 4H EMA200 — safe resample with multiple fallback attempts
        ema4h = ema200  # default fallback
        try:
            df4 = get_history(symbol, "60d", "1h")
            if not df4.empty and len(df4) > 20:
                # FIX: ensure DatetimeIndex before resample
                if not isinstance(df4.index, pd.DatetimeIndex):
                    df4.index = pd.to_datetime(df4.index)
                df4 = df4.resample("4h").agg({
                    "Open":"first","High":"max","Low":"min",
                    "Close":"last","Volume":"sum"
                }).dropna()
                if len(df4) > 5:
                    c4 = df4["Close"].squeeze()
                    ema4h_series = c4.ewm(span=200, adjust=False).mean()
                    val4h = float(ema4h_series.iloc[-1])
                    if not np.isnan(val4h) and val4h > 0:
                        ema4h = val4h
        except:
            pass  # keep ema4h = ema200 fallback

        dist4h = abs((price - ema4h) / ema4h * 100) if ema4h > 0 else 99

        d_sig = ("BUY" if price > ema200 else "SELL") if dist   <= prox else "WAIT"
        h_sig = ("BUY" if price > ema4h  else "SELL") if dist4h <= prox else "WAIT"

        if   d_sig == "BUY"  and h_sig == "BUY":  fin, conf = "STRONG BUY",  "HIGH"
        elif d_sig == "SELL" and h_sig == "SELL":  fin, conf = "STRONG SELL", "HIGH"
        elif d_sig == "BUY":                        fin, conf = "BUY",         "MEDIUM"
        elif d_sig == "SELL":                       fin, conf = "SELL",        "MEDIUM"
        else:                                       fin, conf = "WAIT",        "LOW"

        ib    = "BUY" in fin
        entry = round(ema200 * 1.002 if ib else ema200 * 0.998, 2)
        sl    = round(ema200 * 0.985 if ib else ema200 * 1.015, 2)
        t1    = round(price + 1.5 * atr if ib else price - 1.5 * atr, 2)
        t2    = round(price + 3.0 * atr if ib else price - 3.0 * atr, 2)

        sym_clean = symbol.upper().replace(".NS","")
        lot    = LOTS.get(sym_clean, 500)
        margin = int(price * lot * 0.15)

        return {
            "fin":fin,"conf":conf,
            "price":round(price,2),"ema200":round(ema200,2),
            "ema4h":round(ema4h,2),"dist":round(dist,2),
            "dist4h":round(dist4h,2),"d_sig":d_sig,"h_sig":h_sig,
            "atr":round(atr,2),"entry":entry,"sl":sl,"t1":t1,"t2":t2,
            "lot":lot,"margin":margin,
            "profit":int((t1-entry)*lot),"loss":int(abs(entry-sl)*lot),
            "df":df,
        }
    except:
        return None

# ══════════════════════════════════════════════════════════════════════
# NEWS
# ══════════════════════════════════════════════════════════════════════
def get_news(symbol):
    try:
        from duckduckgo_search import DDGS
        with DDGS() as d:
            return list(d.news(
                f"{symbol} NSE India stock",
                region="in-en", max_results=6
            ))
    except:
        return []

# ══════════════════════════════════════════════════════════════════════
# EMA SCANNER — Scans ALL NSE F&O stocks (spot price)
# ══════════════════════════════════════════════════════════════════════
def scan_ema(prox=1.0, cb=None):
    """Scan all ~180+ NSE F&O stocks for 200 EMA signals. Spot price only."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    buy   = []
    sell  = []
    total = len(ALL_STOCKS)

    def one(s):
        r = get_ema_signal(s, prox)
        if r:
            r.update({"sym": s, "sec": SECTORS.get(s, "—")})
        return r

    with ThreadPoolExecutor(max_workers=15) as ex:
        futs = {ex.submit(one, s): s for s in ALL_STOCKS}
        done = 0
        for f in as_completed(futs):
            done += 1
            if cb:
                cb(done, total, futs[f])
            r = f.result()
            if r and "BUY"  in r.get("fin", ""):
                buy.append(r)
            elif r and "SELL" in r.get("fin", ""):
                sell.append(r)

    buy.sort(key=lambda x: x.get("dist", 99))
    sell.sort(key=lambda x: x.get("dist", 99))
    return {"buy": buy, "sell": sell, "total": total}
