# app.py - robust FPL Team Rater with pre-season fallback + diagnostics + visuals
import streamlit as st
import requests
import pandas as pd
from typing import Any, Optional

st.set_page_config(page_title="FPL Team Rater", layout="wide")

BASE = "https://fantasy.premierleague.com/api"
BOOTSTRAP = f"{BASE}/bootstrap-static/"
ENTRY = f"{BASE}/entry/{{}}/"
PICKS = f"{BASE}/entry/{{}}/event/{{}}/picks/"
MY_TEAM = f"{BASE}/my-team/{{}}/"
PLAYER_SUMMARY = f"{BASE}/element-summary/{{}}/"

# Team colours by team name (adjust/complete as you like)
TEAM_COLORS = {
    "Arsenal": "#EF0107",
    "Aston Villa": "#95BFE5",
    "Bournemouth": "#E31B23",
    "Brentford": "#ED1B2E",
    "Brighton": "#0057B7",
    "Chelsea": "#034694",
    "Crystal Palace": "#1B458F",
    "Everton": "#003399",
    "Fulham": "#000000",
    "Liverpool": "#C8102E",
    "Luton": "#FF5F00",
    "Man City": "#6CABDD",
    "Man United": "#DA291C",
    "Newcastle": "#241F20",
    "Nottingham Forest": "#E51937",
    "Sheffield United": "#EE2737",
    "Spurs": "#132257",
    "West Ham": "#7A263A",
    "Wolves": "#F5A300",
    "Burnley": "#6C1D45"
}

# --- Helpers ---
@st.cache_data(show_spinner=False)
def fetch_bootstrap():
    r = requests.get(BOOTSTRAP, timeout=10)
    r.raise_for_status()
    return r.json()

def try_get(url: str) -> Optional[Any]:
    """Return response.json() on 200, otherwise None; propagate some status codes as messages."""
    try:
        r = requests.get(url, timeout=10)
    except Exception as e:
        return {"__error__": str(e)}
    if r.status_code == 200:
        try:
            return r.json()
        except Exception as e:
            return {"__error__": f"Invalid JSON: {e}"}
    else:
        return {"__status__": r.status_code, "__text__": r.text[:500]}

def find_picks_recursive(obj):
    """
    Recursively search JSON structure for a list of dicts where items contain 'element' keys.
    Returns first found list or None.
    """
    if isinstance(obj, dict):
        # direct hits
        if "picks" in obj and isinstance(obj["picks"], list):
            return obj["picks"]
        for v in obj.values():
            found = find_picks_recursive(v)
            if found:
                return found
    elif isinstance(obj, list) and obj:
        # if list-of-dicts and they contain 'element', assume this is picks
        first = obj[0]
        if isinstance(first, dict) and "element" in first:
            return obj
        # else search inside items
        for item in obj:
            found = find_picks_recursive(item)
            if found:
                return found
    return None

def normalize_picks(raw_picks):
    """
    Normalize pick items so each pick is a dict with at least key 'element' (player id) and optional 'is_captain'.
    Handles lists of ints, lists of dicts, etc.
    """
    if raw_picks is None:
        return None
    if isinstance(raw_picks, dict):
        # sometimes a JSON wrapper was returned
        raw_picks = find_picks_recursive(raw_picks) or []
    if isinstance(raw_picks, list):
        normalized = []
        for item in raw_picks:
            if isinstance(item, dict) and "element" in item:
                normalized.append({"element": int(item["element"]), "is_captain": bool(item.get("is_captain", False))})
            elif isinstance(item, int):
                normalized.append({"element": int(item), "is_captain": False})
            else:
                # ignore unknown item shapes
                continue
        return normalized
    return None

# --- UI ---
st.title("⚽ FPL Team Rater — Robust (pre-season friendly)")
st.write("Enter your Entry ID; the app will try multiple endpoints and provide a CSV upload fallback if needed.")

col1, col2 = st.columns([3,1])
with col1:
    entry_id = st.text_input("Enter your FPL Entry ID (numeric)", value="")
with col2:
    gw_input = st.number_input("Optional: Gameweek (leave 0 for auto-detect)", min_value=0, value=0)

if st.button("Show my team"):
    if not entry_id or not entry_id.strip().isdigit():
        st.error("Please enter a numeric Entry ID (the number in your FPL URL).")
        st.stop()

    entry_id = entry_id.strip()
    # fetch bootstrap
    try:
        bootstrap = fetch_bootstrap()
    except Exception as e:
        st.error(f"Could not fetch bootstrap-static: {e}")
        st.stop()

    # build lookups
    elements = {e["id"]: e for e in bootstrap["elements"]}
    element_types = {et["id"]: et["singular_name_short"] for et in bootstrap["element_types"]}
    teams = {t["id"]: t for t in bootstrap["teams"]}
    teams_by_name = {t["name"]: t for t in bootstrap["teams"]}

    # detect current gw if not provided
    events = bootstrap.get("events", [])
    if gw_input > 0:
        gw = int(gw_input)
    else:
        gw = next((e["id"] for e in events if e.get("is_current")), None)

    st.info(f"Trying to load picks for Entry {entry_id} (detected GW = {gw})")

    # 1) Try the canonical picks endpoint
    picks_raw = None
    resolved_from = None

    if gw:
        st.write("Attempting: /entry/{id}/event/{gw}/picks/")
        picks_resp = try_get(PICKS.format(entry_id, gw))
        if picks_resp and isinstance(picks_resp, dict) and "picks" in picks_resp:
            picks_raw = picks_resp["picks"]
            resolved_from = f"PICKS endpoint (GW {gw})"
        else:
            st.write(f"Picks endpoint: status/info: {picks_resp.get('__status__') if isinstance(picks_resp, dict) else 'none'}")

    # 2) Try /my-team/{id}/
    if picks_raw is None:
        st.write("Attempting: /my-team/{id}/ (some public teams expose picks here)")
        my_team_resp = try_get(MY_TEAM.format(entry_id))
        if isinstance(my_team_resp, dict) and "picks" in my_team_resp:
            picks_raw = my_team_resp["picks"]
            resolved_from = "MY_TEAM endpoint"
        else:
            # if we got a dict but no picks, note keys for diagnostics
            if isinstance(my_team_resp, dict):
                st.write("my-team keys:", list(my_team_resp.keys())[:20])

    # 3) Try /entry/{id}/ and search recursively for picks-like lists
    if picks_raw is None:
        st.write("Attempting: /entry/{id}/ and recursive search for 'element' lists")
        entry_resp = try_get(ENTRY.format(entry_id))
        if isinstance(entry_resp, dict):
            # show top-level keys for debugging
            st.write("Entry endpoint top-level keys (for debugging):", list(entry_resp.keys()))
            found = find_picks_recursive(entry_resp)
            if found:
                picks_raw = found
                resolved_from = "ENTRY endpoint (recursive find)"
            else:
                # common place: entry_resp.get('entry', {}).get('squad') or entry_resp.get('squad')
                squad = None
                if isinstance(entry_resp.get("entry"), dict):
                    squad = entry_resp["entry"].get("squad")
                    if squad:
                        st.write("Found 'entry' -> 'squad' with length", len(squad))
                if not squad and "squad" in entry_resp:
                    squad = entry_resp.get("squad")
                    st.write("Found top-level 'squad' with length", len(squad) if isinstance(squad, list) else 'no')
                if isinstance(squad, list) and squad and isinstance(squad[0], dict) and "element" in squad[0]:
                    picks_raw = squad
                    resolved_from = "ENTRY -> squad"
                # else: nothing found
        else:
            st.write("entry endpoint returned:", entry_resp)

    # If still nothing, provide CSV upload fallback
    if picks_raw is None:
        st.warning("Could not automatically find your picks/squad from the public endpoints.")
        st.info("Two options:\n• Wait until GW1 starts and try again (the picks endpoint will become available),\n• Or upload a CSV with your squad/picks now.")
        uploaded = st.file_uploader("Upload picks CSV (column 'element' = player id)", type=["csv"])
        if uploaded is None:
            st.stop()
        try:
            csv_df = pd.read_csv(uploaded)
            if "element" not in csv_df.columns:
                st.error("CSV must contain an 'element' column with player ids.")
                st.stop()
            picks_raw = csv_df.to_dict(orient="records")
            resolved_from = "User CSV upload"
        except Exception as e:
            st.error(f"Could not read uploaded CSV: {e}")
            st.stop()

    # Normalize picks
    picks = normalize_picks(picks_raw)
    if not picks:
        st.error("No valid picks list could be constructed.")
        st.stop()

    st.success(f"Picks loaded from: {resolved_from} (players found: {len(picks)})")

    # Build DataFrame of players
    rows = []
    for p in picks:
        pid = int(p["element"])
        player = elements.get(pid)
        if not player:
            # skip unknown player id but report
            rows.append({"id": pid, "web_name": f"Unknown ({pid})", "position": "?", "team_id": None, "team_name": "Unknown", "total_points": 0, "form": 0.0, "now_cost_m": 0.0, "is_captain": p.get("is_captain", False)})
            continue
        team_id = player.get("team")
        team_name = teams.get(team_id, {}).get("name", "Unknown")
        rows.append({
            "id": pid,
            "web_name": player.get("web_name"),
            "first_name": player.get("first_name"),
            "second_name": player.get("second_name"),
            "position": element_types.get(player.get("element_type")),
