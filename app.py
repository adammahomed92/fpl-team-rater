"""
FPL Team Rater - Streamlit App
Single-file Streamlit app that rates an FPL team out of 100 using public FPL endpoints.

Files included in this document:
- app code (below)
- requirements.txt (at the end)
- quick deploy instructions (end)

How it works:
- User enters FPL Team ID (entry ID) and optionally a Gameweek.
- App fetches `bootstrap-static` for player data and events, and
  `https://fantasy.premierleague.com/api/entry/{entry_id}/event/{gw}/picks/` for picks.
- Scores the team using weighted factors and displays breakdown, charts, and transfer suggestions.

Note: Some private teams won't return public picks (the picks endpoint returns 403). If that
happens you'll need to use a public team or share a CSV export of your team picks.

"""

import streamlit as st
import requests
import pandas as pd
from functools import lru_cache
import math

st.set_page_config(page_title="FPL Team Rater", layout="wide")

# --- Configuration / Weights ---
WEIGHTS = {
    "points": 0.4,
    "form": 0.3,
    "value": 0.2,
    "top_players": 0.1
}

# Normalization constants (tweakable)
MAX_AVG_POINTS = 200  # used to scale average player total points
MAX_FORM = 10         # player form scale
MAX_VALUE = 12        # million (typical top-end player value)
TOP_PLAYER_THRESHOLD_PERCENT = 20.0  # selected_by_percent > this counts as "top player"

# --- Helpers ---
@lru_cache(maxsize=2)
def fetch_bootstrap():
    """Fetches the FPL bootstrap-static JSON (cached)
    Returns the JSON dict or raises an exception on failure."""
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.json()

@lru_cache(maxsize=128)
def fetch_entry_picks(entry_id: str, event: int):
    """Fetches public picks for the given entry (team) and event (gameweek).
    Uses public endpoint; will fail for private teams.
    Returns JSON dict from the endpoint."""
    url = f"https://fantasy.premierleague.com/api/entry/{entry_id}/event/{event}/picks/"
    r = requests.get(url, timeout=10)
    # propagate status for caller to handle
    if r.status_code == 404:
        raise ValueError("Team not found (404). Check the Entry/Team ID.")
    if r.status_code == 403:
        raise PermissionError("Picks are not public for this team (403).")
    r.raise_for_status()
    return r.json()


def detect_current_event(bootstrap_json: dict) -> int:
    # bootstrap has 'events' list with 'is_current' boolean; fallback to 'current_event' key
    events = bootstrap_json.get("events", [])
    for ev in events:
        if ev.get("is_current"):
            return ev.get("id")
    # fallback
    if "current_event" in bootstrap_json and bootstrap_json["current_event"]:
        return bootstrap_json["current_event"]
    # last resort: choose next upcoming event id
    for ev in events:
        if ev.get("is_next"):
            return ev.get("id")
    raise RuntimeError("Could not detect current gameweek from bootstrap-static")


def score_team(picks_json: dict, player_map: dict):
    """Calculate the 0-100 score and a breakdown given picks JSON and a map of player id -> player data."""
    picks = picks_json.get("picks", [])
    if not picks:
        raise ValueError("No picks found for this selection")

    total_points = 0.0
    total_form = 0.0
    total_value = 0.0
    top_players = 0
    num_players = len(picks)

    # Collect player rows for display
    rows = []

    for p in picks:
        pid = p.get("element")
        is_captain = p.get("is_captain", False)
        pdata = player_map.get(pid)
        if not pdata:
            # skip unknown player id (shouldn't happen)
            continue
        tp = pdata.get("total_points", 0)
        form = float(pdata.get("form") or 0.0)
        now_cost = pdata.get("now_cost", 0) / 10.0  # convert to millions
        selected_pct = float(pdata.get("selected_by_percent") or 0.0)

        total_points += tp
        total_form += form
        total_value += now_cost
        if selected_pct > TOP_PLAYER_THRESHOLD_PERCENT:
            top_players += 1

        rows.append({
            "id": pid,
            "web_name": pdata.get("web_name"),
            "position": pdata.get("element_type"),
            "total_points": tp,
            "form": form,
            "now_cost_m": now_cost,
            "selected_pct": selected_pct,
            "is_captain": is_captain
        })

    avg_points = total_points / num_players
    avg_form = total_form / num_players
    avg_value = total_value / num_players
    top_ratio = top_players / num_players if num_players else 0.0

    # Normalized sub-scores (0-100)
    score_points = min((avg_points / MAX_AVG_POINTS) * 100.0, 100.0)
    score_form = min((avg_form / MAX_FORM) * 100.0, 100.0)
    score_value = min((avg_value / MAX_VALUE) * 100.0, 100.0)
    score_top = top_ratio * 100.0

    final_score = (
        score_points * WEIGHTS["points"] +
        score_form * WEIGHTS["form"] +
        score_value * WEIGHTS["value"] +
        score_top * WEIGHTS["top_players"]
    )

    breakdown = {
        "Average Points (per player)": round(avg_points, 1),
        "Average Form": round(avg_form, 2),
        "Average Value (m)": round(avg_value, 2),
        "Top Player %": round(top_ratio * 100, 1),
        "Points Subscore": round(score_points, 1),
        "Form Subscore": round(score_form, 1),
        "Value Subscore": round(score_value, 1),
        "Top Players Subscore": round(score_top, 1)
    }

    players_df = pd.DataFrame(rows)

    return round(final_score, 1), breakdown, players_df


def suggest_changes(players_df: pd.DataFrame, player_map: dict, n=5):
    """Simple suggestions: list the n lowest form players and players underperforming for their value."""
    suggestions = {}
    if players_df.empty:
        return suggestions

    low_form = players_df.sort_values("form").head(n)
    cheap_underperform = players_df.copy()
    # compute points per million
    cheap_underperform["points_per_m"] = cheap_underperform["total_points"] / (cheap_underperform["now_cost_m"] + 0.01)
    low_ppm = cheap_underperform.sort_values("points_per_m").head(n)

    suggestions["low_form"] = low_form[["web_name", "form", "total_points"]].to_dict(orient="records")
    suggestions["low_ppm"] = low_ppm[["web_name", "now_cost_m", "total_points", "points_per_m"]].to_dict(orient="records")
    return suggestions


# --- Streamlit UI ---
st.title("⚽ FPL Team Rater — Rate your team out of 100")
st.caption("Uses public FPL endpoints — enter your Entry (Team) ID. If picks are private, use a public team or upload a CSV of your picks.")

col1, col2 = st.columns([2, 1])
with col1:
    entry_id = st.text_input("Enter your FPL Entry (Team) ID:")
    with st.expander("How to find your Entry ID"):
        st.write("The Entry ID is the numeric ID for your team. If you view your team page on the FPL site, the URL usually contains `/entry/<ID>/`. Copy that number here.")

with col2:
    # fetch bootstrap and detect current GW
    try:
        bootstrap = fetch_bootstrap()
        current_gw = detect_current_event(bootstrap)
        gw = st.number_input("Gameweek:", min_value=1, max_value=50, value=current_gw)
        st.write(f"Detected current gameweek: {current_gw}")
    except Exception as e:
        st.error(f"Could not fetch FPL bootstrap data: {e}")
        st.stop()

if st.button("Rate my team"):
    if not entry_id:
        st.error("Please enter a valid Entry (Team) ID.")
        st.stop()
    try:
        picks_json = fetch_entry_picks(entry_id.strip(), int(gw))
    except PermissionError as pe:
        st.error(str(pe))
        st.info("If your team is private you can: 1) use a public team's ID, or 2) export your picks as a CSV and upload it via the 'Upload picks CSV' tool below.")
        # show CSV upload option
        uploaded = st.file_uploader("Upload CSV of your picks (columns: element (player id), position, is_captain)", type=["csv"]) 
        if uploaded:
            df = pd.read_csv(uploaded)
            # emulate picks_json structure minimally
            picks_json = {"picks": df.to_dict(orient="records")}
        else:
            st.stop()
    except Exception as e:
        st.error(f"Could not fetch team picks: {e}")
        st.stop()

    try:
        bootstrap = fetch_bootstrap()
        elements = bootstrap.get("elements", [])
        player_map = {p["id"]: p for p in elements}

        score, breakdown, players_df = score_team(picks_json, player_map)

        # main results
        st.success(f"Your Team Score: **{score} / 100**")
        st.markdown("### Score Breakdown")
        st.table(pd.DataFrame.from_dict(breakdown, orient='index', columns=['Value']).rename_axis('Metric').reset_index())

        # bar chart for subscores
        subscores = {
            'Points': breakdown['Points Subscore'],
            'Form': breakdown['Form Subscore'],
            'Value': breakdown['Value Subscore'],
            'Top Players': breakdown['Top Players Subscore']
        }
        st.bar_chart(pd.DataFrame.from_dict(subscores, orient='index', columns=['Score']))

        # show players
        st.markdown("### Players in selected GW picks")
        # make web_name clickable? keep simple
        show_cols = [c for c in ['web_name','total_points','form','now_cost_m','selected_pct','is_captain'] if c in players_df.columns]
        st.dataframe(players_df[show_cols].sort_values('position'))

        # suggestions
        suggestions = suggest_changes(players_df, player_map, n=5)
        st.markdown("### Suggestions")
        if suggestions:
            with st.expander("Low form players (consider transferring)"):
                for r in suggestions.get('low_form', []):
                    st.write(f"{r['web_name']} — form: {r['form']} — total points: {r['total_points']}")
            with st.expander("Low points per million (value for money)"):
                for r in suggestions.get('low_ppm', []):
                    st.write(f"{r['web_name']} — £{r['now_cost_m']}m — points: {r['total_points']} — ppm: {round(r['points_per_m'],2)}")
        else:
            st.write("No suggestions available.")

        st.markdown("---")
        st.markdown("#### Notes")
        st.write("• Scores are a heuristic — tweak the weights/constants in the app to match how you value different factors.")
        st.write("• If the entry picks endpoint returns 403, the team is private and picks aren't publicly visible.")

    except Exception as e:
        st.error(f"Error calculating score: {e}")


# --- Requirements and deployment notes (printed at bottom of file) ---

# requirements.txt (copy to a file named requirements.txt when deploying)
REQUIREMENTS = '''
streamlit
requests
pandas
'''

# Deploy instructions (also copy to your repo README):
DEPLOY = '''
1) Create a GitHub repo and add this file named `app.py` (or `fpl_team_rater_app.py`).
2) Add a `requirements.txt` file with the following contents:

streamlit
requests
pandas

3) Commit and push to GitHub.
4) Sign in to Streamlit Cloud (https://streamlit.io/cloud) and click "New app" -> connect your GitHub repo and choose the app file.
5) Click Deploy — Streamlit will provision and give you a shareable URL.

Notes:
- If you prefer another hosting (Heroku, Railway, Vercel with a python adapter, etc.) the repo and requirements will work similarly.
- If your team picks are private, the app cannot fetch them from the public endpoint (403). You can either use a public team or upload a CSV of your picks.
'''

# show the requirements and deploy instructions in a collapsible area in app if user scrolls to bottom
with st.expander("Requirements & Deploy Notes"):
    st.code(REQUIREMENTS)
    st.text(DEPLOY)

# End of file
