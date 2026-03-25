# NIFTY 50 STOCK TREND ANALYZER

A Python tool that fetches live Nifty 50 data and produces
technical analysis charts + an Excel report.

---

## SETUP (one-time)

```bash
pip install yfinance pandas numpy matplotlib seaborn openpyxl
```

---

## RUN

### Basic (3-month analysis, all 50 stocks)
```bash
python nifty50_analyzer.py
```

### Change period
```bash
python nifty50_analyzer.py --period 1mo   # 1 month
python nifty50_analyzer.py --period 3mo   # 3 months (default)
python nifty50_analyzer.py --period 6mo   # 6 months
python nifty50_analyzer.py --period 1y    # 1 year
```

### Add a single-stock deep-dive
```bash
python nifty50_analyzer.py --stock RELIANCE
python nifty50_analyzer.py --stock TCS --period 6mo
```

### Custom output folder
```bash
python nifty50_analyzer.py --out my_reports
```

---

## OUTPUTS (saved to `nifty50_output/` by default)

| File | Description |
|------|-------------|
| `01_nifty50_overview.png`     | Full dashboard: returns, sector heatmap, RSI/MACD distribution, 52W scatter |
| `02_gainers_losers.png`       | Top 10 gainers & losers bar chart |
| `03_technical_heatmap.png`    | Per-stock RSI / Momentum / Bollinger Band heatmap |
| `04_deepdive_<SYMBOL>.png`    | Price + MAs + BB + Volume + RSI + MACD for top gainer, top loser, user pick |
| `05_52week_tracker.png`       | 52-week range tracker for all 50 stocks |
| `06_sector_grid.png`          | Sector-wise return breakdown |
| `nifty50_analysis.xlsx`       | 4-sheet Excel: Summary, Price History, Technical Data, Sector Analysis |

---

## WHAT GETS ANALYZED

### Technical Indicators
- **RSI (14)** — Relative Strength Index; flags overbought (>70) and oversold (<30)
- **MACD (12/26/9)** — Momentum; BUY when MACD crosses above signal, SELL below
- **Moving Averages** — MA20, MA50, EMA12 with trend classification (Bullish/Neutral/Bearish)
- **Bollinger Bands (20, 2σ)** — Volatility bands; BB Position shows where price sits in the band
- **10-Day Momentum** — Raw price momentum over 10 trading days
- **ATR (14)** — Average True Range (volatility measure)
- **Volume Ratio** — Last 5-day avg vs 20-day avg volume

### 52-Week Tracking
- 52-week high and low from available data
- % distance from each extreme
- "Near High" / "Deep Correction" flags in charts

---

## REQUIREMENTS
- Python 3.8+
- Internet connection (fetches data from Yahoo Finance)
- ~2–3 minutes runtime for full 50-stock analysis

---

## DATA SOURCE
Yahoo Finance via the `yfinance` library.
Tickers use `.NS` suffix (NSE-listed).
