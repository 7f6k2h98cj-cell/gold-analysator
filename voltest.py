import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = "docs/data"
os.makedirs(OUT, exist_ok=True)

# ---- Vorab festgelegt (nicht aendern) ----
START_TEST = 2010
Q_STARK = 0.80
FENSTER = 260

h = yf.Ticker("GC=F").history(period="max", interval="1d", auto_adjust=False)
h = h.dropna(subset=["Open", "High", "Low", "Close"])
h.index = pd.to_datetime(h.index).tz_localize(None).normalize()
h["d"] = h.index
iso = h.index.isocalendar()
h["wk"] = iso.year.astype(str) + "-" + iso.week.astype(str).str.zfill(2)
g = h.groupby("wk")
w = pd.DataFrame({
    "open": g["Open"].first(), "high": g["High"].max(), "low": g["Low"].min(),
    "close": g["Close"].last(), "first_day": g["d"].first(),
    "last_day": g["d"].last(), "n": g["Close"].size(),
}).sort_values("first_day").reset_index()

heute = pd.Timestamp.now()
cw = heute.isocalendar()
aktuell = str(cw.year) + "-" + str(cw.week).zfill(2)
if heute.weekday() < 5:  # unter der Woche: laufende Woche ist unvollstaendig
    w = w[w["wk"] != aktuell]
w = w[w["n"] >= 4].reset_index(drop=True)

w["range"] = (w["high"] - w["low"]) / w["open"]

# Zeile t = Wissensstand am Ende der Woche t, Ziel = Woche t+1
d = w.copy()
d["l1"] = np.log(d["range"])
d["l4"] = np.log(d["range"].rolling(4).mean())
d["l13"] = np.log(d["range"].rolling(13).mean())
d["l26"] = np.log(d["range"].rolling(26).mean())
d["thr"] = d["range"].rolling(FENSTER).quantile(Q_STARK)
nxt = d["range"].shift(-1)
d["y_log"] = np.log(nxt)
d["strong"] = np.where(nxt.notna() & d["thr"].notna(), (nxt > d["thr"]).astype(float), np.nan)
d["target_year"] = d["first_day"].shift(-1).dt.year
cols = ["l1", "l4", "l13", "l26"]

valid = d.dropna(subset=cols + ["thr", "y_log", "strong", "target_year"]).copy()
valid["target_year"] = valid["target_year"].astype(int)

# ---- Walk-forward ----
teile = []
for y in range(START_TEST, int(valid["target_year"].max()) + 1):
    tr = valid[valid["target_year"] < y]
    te = valid[valid["target_year"] == y]
    if len(te) == 0 or len(tr) < 300:
        continue
    lin = LinearRegression().fit(tr[cols], tr["y_log"])
    res = tr["y_log"] - lin.predict(tr[cols])
    q10, q90 = np.quantile(res, [0.10, 0.90])
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    clf.fit(tr[cols], tr["strong"])
    t = te.copy()
    pred = lin.predict(te[cols])
    t["pred_range"] = np.exp(pred)
    t["lo"] = np.exp(pred + q10)
    t["hi"] = np.exp(pred + q90)
    t["p_strong"] = clf.predict_proba(te[cols])[:, 1]
    t["basis"] = tr["strong"].mean()
    teile.append(t)
oos = pd.concat(teile).reset_index(drop=True)

actual = np.exp(oos["y_log"])
naive = np.exp(oos["l26"])
mae_m = float((oos["pred_range"] - actual).abs().mean())
mae_n = float((naive - actual).abs().mean())
cover = float(((actual >= oos["lo"]) & (actual <= oos["hi"])).mean())

p = oos["p_strong"].clip(1e-6, 1 - 1e-6)
y = oos["strong"]
b0 = oos["basis"].clip(1e-6, 1 - 1e-6)
brier_m = float(((p - y) ** 2).mean())
brier_b = float(((b0 - y) ** 2).mean())
ll_m = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
ll_b = float(-(y * np.log(b0) + (1 - y) * np.log(1 - b0)).mean())

kalib = []
oos["bin"] = pd.cut(oos["p_strong"], bins=[0, 0.1, 0.2, 0.3, 0.4, 1.0])
for b, grp in oos.groupby("bin", observed=True):
    kalib.append({"bereich": str(b), "wochen": int(len(grp)),
                  "mittlere_prognose": round(float(grp["p_strong"].mean()), 3),
                  "tatsaechlich": round(float(grp["strong"].mean()), 3)})

pruefungen = {
    "Brier besser als Basisrate": brier_m < brier_b,
    "80-%-Bereich trifft zwischen 72 und 88 %": 0.72 <= cover <= 0.88,
    "Spannen-Fehler kleiner als 26-Wochen-Durchschnitt": mae_m < mae_n,
}
freigabe = all(pruefungen.values())

# ---- Aktuelle Prognose fuer die kommende Woche ----
live = d.iloc[[-1]]
tr = valid
lin = LinearRegression().fit(tr[cols], tr["y_log"])
res = tr["y_log"] - lin.predict(tr[cols])
q10, q90 = np.quantile(res, [0.10, 0.90])
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
clf.fit(tr[cols], tr["strong"])
prog = None
if live[cols + ["thr"]].notna().all(axis=1).iloc[0]:
    pr = float(lin.predict(live[cols])[0])
    prog = {
        "basis_woche_bis": str(live["last_day"].iloc[0].date()),
        "letzter_wochenschluss_gc": round(float(live["close"].iloc[0]), 2),
        "erwartete_spanne_pct": round(float(np.exp(pr) * 100), 2),
        "spanne_80_von_pct": round(float(np.exp(pr + q10) * 100), 2),
        "spanne_80_bis_pct": round(float(np.exp(pr + q90) * 100), 2),
        "schwelle_starker_ausschlag_pct": round(float(live["thr"].iloc[0] * 100), 2),
        "p_starker_ausschlag": round(float(clf.predict_proba(live[cols])[0, 1]), 3),
        "basisrate_starker_ausschlag": round(float(tr["strong"].mean()), 3),
    }

out = {
    "erstellt_utc": datetime.now(timezone.utc).isoformat(),
    "definition": "Starker Ausschlag = Wochenspanne (Hoch-Tief)/Open ueber dem "
                  "80. Perzentil der letzten 260 Wochen. Nur Vergangenheit.",
    "testzeitraum": [str(oos["first_day"].min().date()), "bis " + str(int(oos["target_year"].max()))],
    "wochen_im_test": int(len(oos)),
    "mae_modell_pct": round(mae_m * 100, 3), "mae_naiv_pct": round(mae_n * 100, 3),
    "abdeckung_80_pct": round(cover * 100, 1),
    "brier_modell": round(brier_m, 4), "brier_basisrate": round(brier_b, 4),
    "logloss_modell": round(ll_m, 4), "logloss_basisrate": round(ll_b, 4),
    "kalibrierung": kalib, "pruefungen": pruefungen,
    "freigabe": freigabe, "prognose": prog,
    "hinweis": "Basis ist GC=F (Futures). Bitget-XAUUSDT kann leicht abweichen.",
}
with open(OUT + "/volatilitaet.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=2)

print("Testwochen:", len(oos))
print("Spannen-Fehler % Modell", round(mae_m * 100, 3), "vs 26-Wo-Schnitt", round(mae_n * 100, 3))
print("Abdeckung 80-%-Bereich:", round(cover * 100, 1), "%")
print("Brier", round(brier_m, 4), "vs Basisrate", round(brier_b, 4))
print("LogLoss", round(ll_m, 4), "vs Basisrate", round(ll_b, 4))
print("Kalibrierung:")
for k in kalib:
    print(" ", k["bereich"], k["wochen"], "Wochen | Prognose", k["mittlere_prognose"], "| real", k["tatsaechlich"])
print("Pruefungen:", pruefungen)
print("FREIGABE:", freigabe)
print("Prognose:", prog)
