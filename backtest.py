import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

OUT = "docs/data"
os.makedirs(OUT, exist_ok=True)

# ANNAHME: 0,06 % Gebuehr je Seite (Bitget Taker) + 0,02 % Slippage je Seite
# = 0,16 % pro Round-Trip. Funding ist hier NICHT enthalten (aktuell 0).
KOSTEN_RT = 0.0016

h = yf.Ticker("GC=F").history(period="max", interval="1d", auto_adjust=False)
h = h.dropna(subset=["Open", "Close"])
h.index = pd.to_datetime(h.index).tz_localize(None).normalize()
h["d"] = h.index

iso = h.index.isocalendar()
h["wk"] = iso.year.astype(str) + "-" + iso.week.astype(str).str.zfill(2)

g = h.groupby("wk")
w = pd.DataFrame({
    "open": g["Open"].first(),
    "close": g["Close"].last(),
    "first_day": g["d"].first(),
    "last_day": g["d"].last(),
    "n": g["Close"].size(),
}).sort_values("first_day").reset_index()

# laufende Woche und unvollstaendige Wochen entfernen
heute = pd.Timestamp.now()
cw = heute.isocalendar()
aktuell = str(cw.year) + "-" + str(cw.week).zfill(2)
w = w[(w["wk"] != aktuell) & (w["n"] >= 4)].reset_index(drop=True)

w["ret"] = w["close"] / w["open"] - 1

# Signale: nur Daten vor Wochenstart (shift(1) = Vorwoche)
prev_close = w["close"].shift(1)
sma10 = w["close"].rolling(10).mean().shift(1)
mom4 = w["close"].shift(1) / w["close"].shift(5) - 1
w["sig_long"] = 1
w["sig_short"] = -1
w["sig_trend"] = np.where(prev_close > sma10, 1, -1)
w["sig_mom"] = np.where(mom4 > 0, 1, -1)

gueltig = sma10.notna() & mom4.notna()
w = w[gueltig].reset_index(drop=True)

strategien = {
    "Immer LONG": "sig_long",
    "Immer SHORT": "sig_short",
    "Trend (SMA 10 Wochen)": "sig_trend",
    "Momentum (4 Wochen)": "sig_mom",
}


def kennzahlen(r):
    r = pd.Series(r).dropna()
    n = len(r)
    if n < 10:
        return None
    gew = r[r > 0]
    verl = r[r <= 0]
    eq = (1 + r).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    sd = r.std(ddof=1)
    t = float(r.mean() / (sd / np.sqrt(n))) if sd > 0 else 0.0
    return {
        "trades": int(n),
        "trefferquote_pct": round(float((r > 0).mean() * 100), 1),
        "avg_gewinn_pct": round(float(gew.mean() * 100), 2) if len(gew) else None,
        "avg_verlust_pct": round(float(verl.mean() * 100), 2) if len(verl) else None,
        "erwartung_pct": round(float(r.mean() * 100), 3),
        "t_wert": round(t, 2),
        "gesamt_pct": round(float((eq.iloc[-1] - 1) * 100), 1),
        "max_drawdown_pct": round(float(dd * 100), 1),
    }


jahr = w["first_day"].dt.year
ende = w["first_day"].max()
perioden = {
    "Gesamt": pd.Series(True, index=w.index),
    "2001-2009": (jahr >= 2001) & (jahr <= 2009),
    "2010-2019": (jahr >= 2010) & (jahr <= 2019),
    "2020 bis heute": jahr >= 2020,
    "Letzte 3 Jahre": w["first_day"] >= ende - pd.DateOffset(years=3),
}

ergebnisse = {}
for pname, maske in perioden.items():
    ergebnisse[pname] = {}
    for sname, col in strategien.items():
        netto = w[col] * w["ret"] - KOSTEN_RT
        ergebnisse[pname][sname] = kennzahlen(netto[maske])

info = {
    "wochen": int(len(w)),
    "erste_woche": w["first_day"].min().strftime("%Y-%m-%d"),
    "letzte_woche": w["last_day"].max().strftime("%Y-%m-%d"),
    "wochenrendite_std_pct": round(float(w["ret"].std() * 100), 2),
    "anteil_steigende_wochen_pct": round(float((w["ret"] > 0).mean() * 100), 1),
    "ausreisser_ueber_8pct": int((w["ret"].abs() > 0.08).sum()),
}

out = {
    "erstellt_utc": datetime.now(timezone.utc).isoformat(),
    "kosten_roundtrip_pct": KOSTEN_RT * 100,
    "info": info,
    "ergebnisse": ergebnisse,
    "grenzen": [
        "GC=F ist eine Front-Month-Reihe: Kontraktwechsel koennen Wochenrenditen verzerren.",
        "Einstieg = erster Handelstag der Woche (Open), Ausstieg = letzter Handelstag (Close). Real kann der Kurs abweichen.",
        "Kosten sind eine Annahme. Funding und Stop-Loss/Take-Profit sind hier nicht simuliert.",
        "Bitget-XAUUSDT hat eine andere Historie. Dies ist ein Test auf Gold-Futures, nicht auf Bitget.",
        "Ein t-Wert unter 2 bedeutet: Ergebnis ist von Zufall kaum zu unterscheiden.",
        "Ergebnisse der Vergangenheit sagen nichts sicher ueber die Zukunft.",
    ],
}
with open(OUT + "/backtest.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print("Wochen:", info["wochen"], info["erste_woche"], "bis", info["letzte_woche"])
for sname in strategien:
    k = ergebnisse["Gesamt"][sname]
    print(sname, "| Trades", k["trades"], "| Erwartung/Trade %", k["erwartung_pct"],
          "| t", k["t_wert"], "| MaxDD %", k["max_drawdown_pct"])
