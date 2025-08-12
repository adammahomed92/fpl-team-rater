import streamlit as st
import requests
import pandas as pd

# Mapping of team IDs to primary colour (you can tweak)
TEAM_COLOURS = {
    1: "#EF0107",   # Arsenal
    2: "#0057B8",   # Aston Villa
    3: "#6CABDD",   # Bournemouth
    4: "#1B458F",   # Brentford
    5: "#6A263D",   # Brighton
    6: "#034694",   # Chelsea
    7: "#DA291C",   # Crystal Palace
    8: "#001C58",   # Everton
    9: "#EE2737",   # Fulham
    10: "#D71920",  # Liverpool
    11: "#241F20",  # Luton Town
    12: "#1B458F",  # Man City
    13: "#DA291C",  # Man United
    14: "#FDB913",  # Newcastle
    15: "#003090",  # Nottingham Forest
    16: "#005BAC",  # Sheffield United
    17: "#FDB913",  # Spurs
    18: "#FBEE23",  # West Ham
    19: "#FFCD00",  # Wolves
    20: "#005BAC"   # Burnley
}

BASE_URL = "https://fantasy.premierleague.com/api"

def get_bootstrap():
    return requests.get(f"{BASE_URL}/bootstrap-static/").json()

def get_entry(entry_id):
    return requests.get(f"{BASE_URL}/entry/{entry_id}/").json()

def get_picks(entry_id, gw):
    url = f"{BASE_URL}/entry/{entry_id}/event/{gw}/picks/"
    r = requests.get(url)
    if r.status_code == 404:
        return None
    return r.json()

def render_team(players_df):
    for _, row in players_df.iterrows():
        team_colour = TEAM_COLOURS.get(row["team_id"], "#CCCCCC")
        st.markdown(
            f"<div style='background-color:{team_colour};color:white;padding:4px;border-radius:6px'>"
            f"{row['name']} - {row['team_name']}</div>",
            unsafe_allow_html=True
        )

st.title("FPL Team Rater")
entry_id = st.text_input("Enter your FPL Entry ID")

if st.button("Show my team") and entry_id:
    data = get_bootstrap()
    teams_info = {t["id"]: t["name"] for t in data["teams"]}
    elements_df = pd.DataFrame(data["elements"])

    # Detect gameweek
    events = data["events"]
    current_gw = next((e["id"] for e in events if e["is_current"]), None)

    picks_data = None
    if current_gw:
        picks_data = get_picks(entry_id, current_gw)

    if not picks_data:
        st.info("Pre-season: showing pre-season squad.")
        entry_data = get_entry(entry_id)
        # This won't include picks — so show initial squad as blank list for now
        # You'd replace this with correct preseason squad extraction if needed
        squad_ids = [p["element"] for p in entry_data.get("squad", [])] or []
    else:
        squad_ids = [p["element"] for p in picks_data["picks"]]

    if not squad_ids:
        st.error("No squad data found.")
    else:
        squad_df = elements_df[elements_df["id"].isin(squad_ids)]
        squad_df["team_name"] = squad_df["team"].map(teams_info)
        squad_df.rename(columns={"web_name": "name", "team": "team_id"}, inplace=True)
        render_team(squad_df[["name", "team_name", "team_id"]])
