import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="FPL Team Rater", layout="centered")
st.title("⚽ Fantasy Premier League — Team Rater")

entry_id = st.text_input("Enter your FPL Entry ID", value="", help="Find this in your FPL team URL")

def get_current_gameweek():
    """Fetches current gameweek and status."""
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    data = requests.get(url).json()
    events = data["events"]
    current_gw = next((e["id"] for e in events if e["is_current"]), None)
    next_gw = next((e["id"] for e in events if e["is_next"]), None)
    return current_gw, next_gw, events

def get_preseason_squad(entry_id):
    """Gets the squad before GW1 starts."""
    url = f"https://fantasy.premierleague.com/api/entry/{entry_id}/"
    data = requests.get(url).json()
    picks_url = f"https://fantasy.premierleague.com/api/my-team/{entry_id}/"
    squad_data = requests.get(picks_url).json()
    return squad_data["picks"], squad_data

def get_gw_picks(entry_id, gw):
    """Gets team picks for a given gameweek."""
    url = f"https://fantasy.premierleague.com/api/entry/{entry_id}/event/{gw}/picks/"
    r = requests.get(url)
    if r.status_code == 404:
        return None
    return r.json()["picks"]

def rate_team(picks):
    """Dummy rating function — replace with real logic."""
    return round(len(picks) / 15 * 100, 1)

if entry_id:
    try:
        current_gw, next_gw, events = get_current_gameweek()

        if current_gw is None and next_gw == 1:
            # Season not started yet — use pre-season squad
            st.info("Season hasn't started yet — showing pre-season squad.")
            picks, raw_data = get_preseason_squad(entry_id)
        else:
            # Try to fetch this GW's picks
            picks = get_gw_picks(entry_id, current_gw)
            if picks is None:
                st.warning("No picks found for this Gameweek — showing pre-season squad.")
                picks, raw_data = get_preseason_squad(entry_id)

        # Rate team
        rating = rate_team(picks)
        st.metric(label="Team Rating", value=f"{rating}/100")

        # Show picks in table
        df = pd.DataFrame(picks)
        st.dataframe(df)

    except Exception as e:
        st.error(f"Could not fetch data: {e}")
