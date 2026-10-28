import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = "docs/data"
os.makedirs(OUT, exist_ok=True)

# ---- Vorab festgelegte Parameter (nicht aendern) ----
P_LONG, P_SHORT = 0.55, 0.45
KOSTEN = {"0,16 %": 0.0016, "0,06 %": 0.0006}
START_TEST = 2010

key = os.environ.get("FRED_API_KEY")
if not key:
    raise SystemExit("FRED_API_KEY fehlt")

# ---- Wochendaten aus GC=F ----
h = yf.Ticker("GC=F").history(period="max", interval="1d", auto_adjust=False)
h = h.dropna(subset=["Open", "Close"])
h.index = pd.to_datetime(h.index).tz_localize(None).normalize()
h["d"] = h.index
iso = h.index.isocalendar()
h["wk"] = iso.year.astype(str) + "-" + iso.week.astype(str).str.zfill(2)
g = h.groupby("wk")
w = pd.DataFrame({
    "open": g["Open"].first(), "close": g["Close"].last(),
    "first_day": g["d"].first(), "last_day": g["d"].last(), "n": g["Close"].size(),
}).sort_values("first_day").reset_index()
cw = pd.Timestamp.now().isocalendar()
aktuell = str(cw.year) + "-" + str(cw.week).zfill(2)
w = w[(w["wk"] != aktuell) & (w["n"] >= 4)].reset_index(drop=True)
w["ret"] = w["close"] / w["open"] - 1

# ---- FRED: 10J-Rendite, nur Werte bis Freitag der Vorwoche werden genutzt ----
r = requests.get(
    "https://api.stlouisfed.org/fred/series/observations",
    params={"series_id": "DGS10", "api_key": key, "file_type": "json",
            "observation_start": "1999-01-01"},
    timeout=30,
)
r.raise_for_status()
f = pd.DataFrame(r.json()["observations"])
f = f[f["value"] != "."]
f = pd.DataFrame({"date": pd.to_datetime(f["date"]).astype("datetime64[ns]"),
                  "y10": f["value"].astype(float).values}).sort_values("date")
w["last_day"] = w["last_day"].astype("datetime64[ns]")
w = pd.merge_asof(w.sort_values("last_day"), f, left_on="last_day", right_on="date")
w = w.sort_values("first_day").reset_index(drop=True)

# ---- Merkmale: alles verschoben, nur Wissen bis zur Vorwoche ----
c = w["close"]
feat = pd.DataFrame({
    "mom1": c / c.shift(1) - 1,
    "mom4": c / c.shift(4) - 1,
    "dist": c / c.rolling(10).mean() - 1,
    "vol4": w["ret"].rolling(4).std(),
    "dy4": w["y10"] - w["y10"].shift(4),
}).shift(1)
w = pd.concat([w, feat], axis=1).dropna(subset=list(feat.columns)).reset_index(drop=True)
w["up"] = (w["ret"] > 0).astype(int)
w["year"] = w["first_day"].dt.year
cols = list(feat.columns)

# ---- Walk-forward: bis Jahr Y-1 trainieren, Jahr Y testen ----
teile = []
for y in range(START_TEST, int(w["year"].max()) + 1):
    tr = w[w["year"] < y]
    te = w[w["year"] == y]
    if len(te) == 0 or len(tr) < 200:
        continue
    m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
    m.fit(tr[cols], tr["up"])
    t = te.copy()
    t["p"] = m.predict_proba(te[cols])[:, 1]
    t["basis"] = tr["up"].mean()
    teile.append(t)
oos = pd.concat(teile).reset_index(drop=True)

oos["pos_A"] = np.where(oos["p"] > P_LONG, 1, np.where(oos["p"] < P_SHORT, -1, 0))
oos["pos_B"] = np.where(oos["dist"] > 0, 1, 0)
oos["pos_L"] = 1


def kennzahlen(pos, kosten):
    pos = pos.values
    brutto = pos * oos["ret"].values
    netto = brutto - kosten * (pos != 0)
    trades = netto[pos != 0]
    n = len(trades)
    eq = np.cumprod(1 + netto)
    dd = float((eq / np.maximum.accumulate(eq) - 1).min())
    if n >= 2 and trades.std(ddof=1) > 0:
        t = float(trades.mean() / (trades.std(ddof=1) / np.sqrt(n)))
    else:
        t = 0.0
    return {
        "wochen": int(len(netto)),
        "trades": int(n),
        "long": int((pos == 1).sum()),
        "short": int((pos == -1).sum()),
        "trefferquote_pct": round(float((trades > 0).mean() * 100), 1) if n else None,
        "erwartung_pro_trade_pct": round(float(trades.mean() * 100), 3) if n else None,
        "t_wert": round(t, 2),
        "gesamt_pct": round(float((eq[-1] - 1) * 100), 1),
        "max_drawdown_pct": round(dd * 100, 1),
    }


ergebnis = {}
for kname, k in KOSTEN.items():
    ergebnis[kname] = {
        "Modell A (logistisch)": kennzahlen(oos["pos_A"], k),
        "Modell B (Trend nur Long)": kennzahlen(oos["pos_B"], k),
        "Immer LONG": kennzahlen(oos["pos_L"], k),
    }

# ---- Qualitaet der Wahrscheinlichkeiten ----
p = oos["p"].clip(1e-6, 1 - 1e-6)
yv = oos["up"]
b0 = oos["basis"].clip(1e-6, 1 - 1e-6)
brier_m = float(((p - yv) ** 2).mean())
brier_b = float(((b0 - yv) ** 2).mean())
ll_m = float(-(yv * np.log(p) + (1 - yv) * np.log(1 - p)).mean())
ll_b = float(-(yv * np.log(b0) + (1 - yv) * np.log(1 - b0)).mean())
bins = [0, 0.40, 0.45, 0.50, 0.55, 0.60, 1.0]
oos["bin"] = pd.cut(oos["p"], bins=bins)
kalib = []
for b, grp in oos.groupby("bin", observed=True):
    kalib.append({"bereich": str(b), "wochen": int(len(grp)),
                  "mittlere_prognose": round(float(grp["p"].mean()), 3),
                  "tatsaechlich_steigend": round(float(grp["up"].mean()), 3)})

# ---- Jahre ----
jahre = []
for y, grp in oos.groupby("year"):
    netto = grp["pos_A"] * grp["ret"] - 0.0016 * (grp["pos_A"] != 0)
    jahre.append({"jahr": int(y), "trades": int((grp["pos_A"] != 0).sum()),
                  "summe_netto_pct": round(float(netto.sum() * 100), 1)})

# ---- Freigabe-Pruefung (Kosten 0,16 %) ----
a = ergebnis["0,16 %"]["Modell A (logistisch)"]
lg = ergebnis["0,16 %"]["Immer LONG"]
pruefungen = {
    "mind. 100 Trades": a["trades"] >= 100,
    "Erwartung pro Trade > 0": (a["erwartung_pro_trade_pct"] or -1) > 0,
    "t-Wert > 2": a["t_wert"] > 2,
    "besser als Immer LONG": (a["erwartung_pro_trade_pct"] or -1) > (lg["erwartung_pro_trade_pct"] or 0),
    "Brier besser als Basisrate": brier_m < brier_b,
}
freigabe = all(pruefungen.values())

out = {
    "erstellt_utc": datetime.now(timezone.utc).isoformat(),
    "testzeitraum": [str(oos["first_day"].min().date()), str(oos["last_day"].max().date())],
    "wochen_im_test": int(len(oos)),
    "parameter": {"p_long": P_LONG, "p_short": P_SHORT, "merkmale": cols},
    "ergebnis": ergebnis,
    "brier_modell": round(brier_m, 4), "brier_basisrate": round(brier_b, 4),
    "logloss_modell": round(ll_m, 4), "logloss_basisrate": round(ll_b, 4),
    "kalibrierung": kalib, "jahre_modell_A": jahre,
    "pruefungen": pruefungen, "freigabe_modell_A": freigabe,
}
with open(OUT + "/modelltest.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=2)

print("Test:", out["testzeitraum"][0], "bis", out["testzeitraum"][1], "|", len(oos), "Wochen")
for kname, d in ergebnis.items():
    print("== Kosten", kname)
    for name, k in d.items():
        print(name, "| Trades", k["trades"], "(L", k["long"], "S", k["short"], ")",
              "| Erw %", k["erwartung_pro_trade_pct"], "| t", k["t_wert"],
              "| Gesamt %", k["gesamt_pct"], "| MaxDD %", k["max_drawdown_pct"])
print("Brier Modell", round(brier_m, 4), "vs Basisrate", round(brier_b, 4))
print("LogLoss Modell", round(ll_m, 4), "vs Basisrate", round(ll_b, 4))
print("Kalibrierung:")
for kz in kalib:
    print(" ", kz["bereich"], kz["wochen"], "Wochen | Prognose", kz["mittlere_prognose"],
          "| real", kz["tatsaechlich_steigend"])
print("Pruefungen:", pruefungen)
print("FREIGABE MODELL A:", freigabe)
