# app.py
import streamlit as st
import requests
import pandas as pd
from typing import Any, Optional

st.set_page_config(page_title="FPL Team Rater (Auth + Predictions)", layout="wide")

BASE = "https://fantasy.premierleague.com/api"
BOOTSTRAP = f"{BASE}/bootstrap-static/"
ENTRY = f"{BASE}/entry/{{}}/"
PICKS = f"{BASE}/entry/{{}}/event/{{}}/picks/"
MY_TEAM = f"{BASE}/my-team/{{}}/"
PLAYER_SUMMARY = f"{BASE}/element-summary/{{}}/"
PREDICTIONS_URL = "https://www.fantasyfootballpundit.com/fpl-points-predictor/"

TEAM_COLORS = {
    "Arsenal": "#EF0107", "Aston Villa": "#95BFE5", "Bournemouth": "#DA291C",
    "Brentford": "#D20000", "Brighton": "#0057B8", "Burnley": "#6C1D45",
    "Chelsea": "#034694", "Crystal Palace": "#1B458F", "Everton": "#003399",
    "Fulham": "#000000", "Liverpool": "#C8102E", "Luton Town": "#FF6600",
    "Man City": "#6CABDD", "Man United": "#DA291C", "Newcastle United": "#241F20",
    "Nottingham Forest": "#E51937", "Sheffield United": "#EE2737",
    "Tottenham Hotspur": "#132257", "West Ham": "#7A263A",
    "Wolves": "#F5A300"
}

# ---------------- Helpers ----------------
@st.cache_data(show_spinner=False)
def fetch_bootstrap() -> dict:
    r = requests.get(BOOTSTRAP, timeout=10)
    r.raise_for_status()
    return r.json()

def safe_get_json(url: str, headers: dict = None) -> Any:
    """Return response.json() if 200; else return dict describing status/error."""
    try:
        merged_headers = {"User-Agent": "Mozilla/5.0"}
        if headers:
            merged_headers.update(headers)
        r = requests.get(url, headers=merged_headers, timeout=12)
    except Exception as e:
        return {"__error__": str(e)}
    if r.status_code == 200:
        try:
            return r.json()
        except Exception as e:
            return {"__error__": f"Invalid JSON: {e}"}
    else:
        return {"__status__": r.status_code, "__text__": r.text[:800]}

def normalize_picks(raw):
    if raw is None:
        return None
    if isinstance(raw, dict) and "picks" in raw:
        raw = raw["picks"]
    if isinstance(raw, list):
        return [{"element": int(it["element"]), "is_captain": bool(it.get("is_captain", False))}
                for it in raw if isinstance(it, dict) and "element" in it]
    return None

def render_lineup(df):
    """Render the lineup in football formation style."""
    gk = df[df["position"] == "GK"]
    defs = df[df["position"] == "DEF"]
    mids = df[df["position"] == "MID"]
    fwds = df[df["position"] == "FWD"]
    subs = df[df["position"].isin(["GK","DEF","MID","FWD"])].iloc[11:]  # assume last 4 as subs

    def player_html(row):
        cap = " 🧢" if row["is_captain"] else ""
        return f"<div style='background:{row['colour']};padding:4px 8px;border-radius:6px;color:#fff;margin:4px;display:inline-block;'>{row['web_name']}{cap}</div>"

    st.markdown("### Lineup")
    lineup_html = "<div style='text-align:center;'>"

    for _, r in gk.iterrows():
        lineup_html += player_html(r) + "<br><br>"

    for _, r in defs.iterrows():
        lineup_html += player_html(r)
    lineup_html += "<br><br>"

    for _, r in mids.iterrows():
        lineup_html += player_html(r)
    lineup_html += "<br><br>"

    for _, r in fwds.iterrows():
        lineup_html += player_html(r)
    lineup_html += "<br><br>"

    lineup_html += "<div style='margin-top:20px;'>Subs:<br>"
    for _, r in subs.iterrows():
        lineup_html += player_html(r)
    lineup_html += "</div></div>"

    st.markdown(lineup_html, unsafe_allow_html=True)


# ---------------- UI ----------------
st.title("⚽ FPL Team Rater — Formation View")
sidebar = st.sidebar
entry_id_input = sidebar.text_input("FPL Entry ID (numeric)", "2792859")
gw_input = sidebar.number_input("Gameweek (0 = auto)", min_value=0, value=0)

if st.button("Fetch squad & show lineup"):
    entry_id = entry_id_input.strip()
    bootstrap = fetch_bootstrap()
    elements = {e["id"]: e for e in bootstrap["elements"]}
    element_types = {et["id"]: et["singular_name_short"] for et in bootstrap["element_types"]}
    teams = {t["id"]: t for t in bootstrap["teams"]}
    events = bootstrap.get("events", [])

    # pick GW
    if gw_input > 0:
        gw = gw_input
    else:
        gw = next((e["id"] for e in events if e.get("is_current")), None)

    resp = safe_get_json(PICKS.format(entry_id, gw))
    picks = normalize_picks(resp.get("picks")) if isinstance(resp, dict) else None
    if not picks:
        st.error("Could not fetch picks")
        st.stop()

    rows = []
    for p in picks:
        player = elements.get(p["element"])
        if not player:
            continue
        team_id = player.get("team")
        team_name = teams.get(team_id, {}).get("name", "Unknown")
        rows.append({
            "id": p["element"],
            "web_name": player["web_name"],
            "position": element_types[player["element_type"]],
            "team_name": team_name,
            "colour": TEAM_COLORS.get(team_name, "#666"),
            "is_captain": p.get("is_captain", False)
        })
    df = pd.DataFrame(rows)

    render_lineup(df)
