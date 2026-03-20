<div align="center">

# 📈 QuantIQ Pro

### AI-Powered Quantitative Analysis Platform — NSE India

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B?style=flat-square&logo=streamlit)](https://streamlit.io)
[![Groq](https://img.shields.io/badge/Groq-LLM-orange?style=flat-square)](https://groq.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?style=flat-square&logo=scikit-learn)](https://scikit-learn.org)
[![NSE](https://img.shields.io/badge/Exchange-NSE%20India-green?style=flat-square)](https://nseindia.com)
[![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)](LICENSE)

**One platform. Every role.**

*Quant Researcher · Algo Trader · ML Engineer · AI Engineer · Data Scientist · Data Analyst*

[Features](#features) · [Quick Start](#quick-start) · [Architecture](#architecture) · [Screenshots](#screenshots) · [Tech Stack](#tech-stack)

</div>

---

## Overview

QuantIQ Pro is a full-stack quantitative finance application built on **NSE India** data. It combines systematic trading strategies, machine learning models, and autonomous AI agents into a single Streamlit interface — deployable locally or on Hugging Face Spaces with zero cost.

**Works without any API key.** Add a free Groq key for AI-powered research reports.

```
Search any NSE stock → get institutional-grade analysis in seconds
```

---

## Features

### Tab 1 — Stock Analysis

**AI Research** — 5 autonomous agents analyse any stock and produce a research report:

| Agent | What it does |
|-------|-------------|
| 📰 News Agent | Fetches live headlines, scores sentiment (Positive / Neutral / Negative) |
| 📊 Fundamental Agent | Analyses PE, ROE, debt ratios, margins — gives Buy/Hold/Avoid verdict |
| 📈 Technical Agent | Reads RSI, MACD, EMA200 signals — gives Bullish/Bearish/Sideways direction |
| ⚠️ Risk Agent | Computes annualised volatility, identifies top 3 risk factors |
| 💰 Valuation Agent | Calculates Graham Number, DCF, PE Fair Value, PB Fair Value |
| 🧠 Master Agent | Synthesises all 5 agents → weighted score (0–100) → research report |

**Live Charts** — Interactive Plotly charts with 7 technical indicators:
- Candlestick / Line / Area
- EMA 20 / 50 / 200 overlaid
- Bollinger Bands
- Volume bars
- RSI (14)
- MACD with histogram

**Buy / Sell Levels** — Valuation levels with four calculation methods:
- Graham Number = √(22.5 × EPS × Book Value)
- DCF (Graham Revised) = EPS × (8.5 + 2g) × (4.4 / 7.5)
- PE Fair Value = Sector PE × EPS
- PB Fair Value = (ROE / 12%) × Book Value
- Futures trade setup — lot size, margin, entry, stop-loss, targets, R:R ratio

**Fundamentals** — 15 financial metrics across Valuation, Profitability, Growth, Risk panels.

---

### Tab 2 — Quant Strategies + ML

**4 Systematic Trading Strategies** — all with realistic NSE transaction costs (0.15%):

| Strategy | Logic | Performance Metric |
|----------|-------|--------------------|
| SMA Crossover | Long when SMA20 > SMA50 | Sharpe, Sortino, Calmar |
| RSI Mean Reversion | Buy RSI < 35, Exit RSI > 65 | Max Drawdown, Win Rate |
| MACD Momentum | Long on MACD–Signal crossover | Total Return, Annual Return |
| Bollinger Breakout | Buy when price > Upper Band | vs Buy & Hold benchmark |

**SMA Optimisation** — Grid search across 300+ parameter combinations. Sharpe ratio heatmap (Short × Long window).

**Statistical Tests**
- Autocorrelation test (15 lags) — detects predictable patterns
- Jarque-Bera normality test — proves fat tails in Indian stock returns
- K-Means Market Regime Detection — classifies every day as Bull / Bear / Sideways / High Volatility

**4 ML Models** — all with TimeSeriesSplit cross-validation (no data leakage):

| Model | Type | Key Metric |
|-------|------|------------|
| Linear Regression | Return prediction | Train R² / Test R² + feature coefficients |
| Random Forest | Direction classification | Accuracy + OOB Score + ROC-AUC |
| Gradient Boosting | Direction classification | Accuracy + baseline comparison |
| Neural Network (MLP) | Direction classification | Accuracy + early stopping |

---

### Tab 3 — 200 EMA Scanner

Scans all **253 NSE F&O eligible stocks** simultaneously:
- Proximity slider (0.3% – 3.0%) — find stocks touching the 200 EMA
- Daily + 4-Hour EMA alignment for high-confidence signals
- BUY / SELL / STRONG BUY / STRONG SELL with confidence level
- Entry price, Stop Loss, Target 1, Target 2 for every signal
- Futures trade details — lot size, margin required, max profit, max loss

---

### Tab 4 — Market Overview

Live Nifty 50 performance dashboard — all 50 stocks, % change, sector colour-coding.

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/quantiq-pro.git
cd quantiq-pro

# 2. Install
pip install -r requirements.txt

# 3. Run
streamlit run main.py
```

Open **http://localhost:8501**

### Optional — Add Groq AI key (free)

```bash
cp .env.example .env
# Edit .env → add your key from console.groq.com (free, no credit card)
```

Without key → rule-based agents (fully functional).
With key → Groq Llama 3 powers all 5 agents for GPT-quality reports.

---

## Architecture

```
quantiq-pro/
│
├── main.py          663 lines   Streamlit UI — all 4 tabs, session state, charts
├── data.py          623 lines   NSE data pipeline — yfinance + retry logic + 20 indicators
├── agents.py        243 lines   5 AI agents + Master agent (Groq LLM or rule-based)
├── ml_models.py     165 lines   4 ML models + K-Means + statistical tests
├── strategies.py    110 lines   4 trading strategies + SMA optimisation
│
├── requirements.txt             Pinned dependencies
├── .env.example                 Environment variable template
└── README.md
```

**Data Flow**

```
NSE Stock Symbol
       │
       ▼
 data.py ──► yfinance download (retry + period fallback)
       │
       ▼
 20+ Technical Indicators computed
 (RSI · MACD · EMA · Bollinger · ATR · Volume Ratio · Momentum)
       │
       ├──► agents.py    ──► 5 AI agents ──► Master Agent ──► Research Report
       ├──► strategies.py ──► 4 strategies ──► Sharpe / Drawdown / Win Rate
       ├──► ml_models.py  ──► 4 ML models ──► Accuracy / OOB / ROC-AUC
       └──► main.py       ──► Streamlit UI ──► Interactive Dashboard
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **UI Framework** | Streamlit 1.32 |
| **Data Source** | yfinance (NSE India — free, no API key) |
| **Charts** | Plotly — candlestick, heatmap, bar, scatter |
| **AI / LLM** | Groq API — Llama 3 8B (free tier) |
| **ML Models** | scikit-learn — Random Forest, GBM, MLP, LinearRegression, KMeans |
| **Statistics** | SciPy — Jarque-Bera, autocorrelation |
| **News Search** | DuckDuckGo Search (no API key) |
| **Language** | Python 3.10+ |

---

## Design Decisions

**Why NSE India?**
Indian equities have distinct microstructure — higher volatility, fat-tailed returns, and sector rotations driven by RBI policy and FII flows. All parameters (transaction cost, risk-free rate, lot sizes) reflect actual NSE conditions, not US market defaults.

**Why Sharpe 0.5–2.0 instead of 4.0+?**
Every strategy includes 0.15% per-trade transaction cost. Strategies without costs show Sharpe 4+ — this is overfitting, not alpha. Realistic costs bring Sharpe to honest levels. A Sharpe of 0.8 with real NSE costs is solid institutional-grade performance.

**Why TimeSeriesSplit instead of K-Fold?**
Financial data is sequential. Standard K-Fold randomly shuffles data, which means the model trains on future data to predict past data — a form of data leakage. TimeSeriesSplit always trains on past and tests on future, which mirrors how live trading works.

**Why rule-based fallback for agents?**
The platform works without any API key. The rule-based agents compute the same scores from the same financial data — the Groq LLM only adds natural-language narrative around those numbers. This makes the project accessible without account setup.

---

## Key Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Risk-Free Rate | 6.5% | Indian 10-Year G-Sec yield |
| Transaction Cost | 0.15% per trade | NSE standard (brokerage + STT + GST) |
| EMA Scanner Universe | 253 stocks | Complete NSE F&O eligible list |
| ML Cross-Validation | TimeSeriesSplit, 5 folds | Prevents look-ahead bias |
| K-Means Regimes | 4 clusters | Bull / Bear / Sideways / High Volatility |
| Autocorrelation Lags | 15 | Standard for daily equity returns |

---

## Deployment

**Hugging Face Spaces (free)**

```bash
# Create new Space → SDK: Streamlit → Python 3.10
# Upload all files
# Add secret: GROQ_API_KEY (optional)
```

**Local**

```bash
streamlit run main.py --server.port 8501
```

---

## Author

**Adinath Vitthal More**
Quantitative Researcher · QFI Capital

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://linkedin.com/in/adi-more-1a5b34210)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat-square&logo=github)]([https://github.com/YOUR_USERNAME](https://github.com/adinath412))

---

## Disclaimer

> This project is built for educational and portfolio demonstration purposes only.
> It is **not** SEBI-registered investment advice.
> All analysis is algorithmic and does not constitute a recommendation to buy or sell any security.
> Past strategy performance does not guarantee future results.
> Always consult a SEBI-registered investment advisor before making financial decisions.

---

<div align="center">

Built with Python · Streamlit · Groq AI · scikit-learn · yfinance · Plotly

⭐ Star this repo if it helped you

</div>
