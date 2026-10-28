import json
import os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import requests
import yfinance as yf

VERSION = "0.1"
OUT = "docs/data"
os.makedirs(OUT + "/history", exist_ok=True)

jetzt = datetime.now(timezone.utc)
quellen = {}
warnungen = []


def melde(name, ok, detail, stand=None):
    quellen[name] = {"ok": ok, "detail": detail, "datenstand_utc": stand}
    if not ok:
        warnungen.append(name + ": " + detail)


# 1) Yahoo GC=F (lange Historie, nur fuer spaetere Modelle)
gc = None
try:
    h = yf.Ticker("GC=F").history(period="max", interval="1d", auto_adjust=False)
    h = h.dropna(subset=["Close"])
    if len(h) < 1000:
        raise ValueError("zu wenige Tageskerzen: " + str(len(h)))
    letzter = h.index[-1]
    alter = (jetzt.date() - letzter.date()).days
    gc = {
        "letzter_schlusskurs": round(float(h["Close"].iloc[-1]), 2),
        "datum": letzter.strftime("%Y-%m-%d"),
        "anzahl_tage": len(h),
        "erstes_datum": h.index[0].strftime("%Y-%m-%d"),
    }
    melde("Yahoo GC=F", alter <= 4,
          str(len(h)) + " Tageskerzen, letzte vom " + gc["datum"]
          + " (kann unvollstaendig sein)", gc["datum"])
except Exception as e:
    melde("Yahoo GC=F", False, "Abruf fehlgeschlagen: " + repr(e))

# 2) Bitget XAUUSDT (Kurs fuer den Handel)
bitget = None
try:
    r = requests.get(
        "https://api.bitget.com/api/v2/mix/market/ticker",
        params={"symbol": "XAUUSDT", "productType": "USDT-FUTURES"},
        timeout=20,
    )
    r.raise_for_status()
    d = r.json()["data"][0]
    ts = datetime.fromtimestamp(int(d["ts"]) / 1000, timezone.utc)
    alter_min = (jetzt - ts).total_seconds() / 60
    bitget = {
        "letzter_kurs": float(d["lastPr"]),
        "markpreis": float(d["markPrice"]),
        "indexpreis": float(d["indexPrice"]),
        "funding_rate": d["fundingRate"],
        "zeitstempel_utc": ts.isoformat(),
    }
    melde("Bitget XAUUSDT", alter_min < 10,
          "Kurs ist " + str(round(alter_min, 1)) + " Minuten alt",
          ts.isoformat())
except Exception as e:
    melde("Bitget XAUUSDT", False, "Abruf fehlgeschlagen: " + repr(e))

# 3) FRED (Makrodaten)
fred = {}
key = os.environ.get("FRED_API_KEY")
serien = {
    "DFII10": "Reale Rendite 10J (TIPS)",
    "DGS10": "Rendite 10J Treasury",
    "DGS2": "Rendite 2J Treasury",
    "DFF": "Fed Funds Rate",
    "DTWEXBGS": "US-Dollar-Index (breit)",
}
if not key:
    melde("FRED", False, "Kein API-Schluessel (Secret FRED_API_KEY) gefunden")
else:
    fehler = []
    for sid, name in serien.items():
        try:
            r = requests.get(
                "https://api.stlouisfed.org/fred/series/observations",
                params={"series_id": sid, "api_key": key, "file_type": "json",
                        "sort_order": "desc", "limit": 10},
                timeout=20,
            )
            r.raise_for_status()
            obs = [o for o in r.json()["observations"] if o["value"] != "."]
            letzte = obs[0]
            fred[sid] = {"name": name, "wert": float(letzte["value"]),
                         "datum": letzte["date"]}
            alter = (jetzt.date() - datetime.strptime(letzte["date"], "%Y-%m-%d").date()).days
            if alter > 10:
                fehler.append(sid + " veraltet (" + letzte["date"] + ")")
        except Exception as e:
            fehler.append(sid + ": " + repr(e))
    melde("FRED", not fehler,
          "alle " + str(len(serien)) + " Serien ok" if not fehler else "; ".join(fehler))

# 4) Entscheidung (Version 0.1: noch kein Modell)
kritisch = [n for n in ("Yahoo GC=F", "Bitget XAUUSDT")
            if not quellen.get(n, {}).get("ok")]
grund = ["Version 0.1: Es ist noch kein Prognosemodell eingebaut. "
         "Diese Version prueft nur die Datenanbindung."]
if kritisch:
    grund.append("Kritische Datenquelle fehlerhaft: " + ", ".join(kritisch))

bericht = {
    "version": VERSION,
    "erstellt_utc": jetzt.isoformat(),
    "erstellt_berlin": jetzt.astimezone(ZoneInfo("Europe/Berlin")).strftime("%d.%m.%Y %H:%M"),
    "entscheidung": "KEIN TRADE",
    "grund": grund,
    "bitget": bitget,
    "gc_yahoo": gc,
    "fred": fred,
    "quellen": quellen,
    "warnungen": warnungen,
    "hinweis": "Entscheidungshilfe, keine Gewinngarantie.",
}

# 5) Speichern: nie ueberschreiben, jeder Lauf bekommt eine eigene Datei
name = jetzt.strftime("%Y%m%d-%H%M%S") + ".json"
with open(OUT + "/history/" + name, "w", encoding="utf-8") as f:
    json.dump(bericht, f, ensure_ascii=False, indent=2)
with open(OUT + "/latest.json", "w", encoding="utf-8") as f:
    json.dump(bericht, f, ensure_ascii=False, indent=2)

pfad = OUT + "/history.json"
verlauf = []
if os.path.exists(pfad):
    with open(pfad, encoding="utf-8") as f:
        verlauf = json.load(f)
verlauf.append({
    "zeit": bericht["erstellt_berlin"],
    "entscheidung": bericht["entscheidung"],
    "bitget_kurs": bitget["letzter_kurs"] if bitget else None,
    "datei": "history/" + name,
})
with open(pfad, "w", encoding="utf-8") as f:
    json.dump(verlauf, f, ensure_ascii=False, indent=2)

print("Fertig:", bericht["entscheidung"])
for n, q in quellen.items():
    print(("OK    " if q["ok"] else "FEHLER"), n, "-", q["detail"])
