import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="FPL Team Rater", layout="wide")
st.title("⚽ Fantasy Premier League - Team Rater")

# Get gameweek info from API
def get_gameweeks():
    try:
        r = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/")
        r.raise_for_status()
        events = r.json()['events']

        current = next((e for e in events if e['is_current']), None)
        upcoming = next((e for e in events if e['is_next']), None)

        if current:
            return events, current['id'], "current"
        elif upcoming:
            return events, upcoming['id'], "pre-season"
        else:
            return events, None, "unknown"
    except Exception as e:
        st.error(f"Error fetching gameweek data: {e}")
        return [], None, "error"

# Fetch picks for a given team & GW
def get_team_picks(team_id, gw):
    url = f"https://fantasy.premierleague.com/api/entry/{team_id}/event/{gw}/picks/"
    try:
        r = requests.get(url)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.HTTPError as e:
        st.error(f"HTTP error: {e}")
    except Exception as e:
        st.error(f"Error fetching picks: {e}")
    return None

# Get all player info
def get_player_info():
    try:
        r = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/")
        r.raise_for_status()
        return pd.DataFrame(r.json()['elements'])
    except Exception as e:
        st.error(f"Error fetching player info: {e}")
        return pd.DataFrame()

# Sidebar controls
st.sidebar.header("FPL Settings")
team_id = st.sidebar.number_input("Enter your FPL Team ID", min_value=1, value=123456)

events, default_gw, season_state = get_gameweeks()

gw_list = [e['id'] for e in events]
gw = st.sidebar.selectbox("Select Gameweek", gw_list, index=(default_gw - 1) if default_gw else 0)

if st.sidebar.button("Fetch My Squad"):
    picks_data = get_team_picks(team_id, gw)
    if picks_data and "picks" in picks_data:
        players_df = get_player_info()

        squad = []
        for pick in picks_data["picks"]:
            pid = pick["element"]
            player = players_df[players_df["id"] == pid].iloc[0]
            squad.append({
                "Player": f"{player['first_name']} {player['second_name']}",
                "Position": player['element_type'],
                "Team ID": player['team'],
                "Price": player['now_cost'] / 10,
                "Captain": pick['is_captain'],
                "Vice": pick['is_vice_captain']
            })

        st.subheader(f"Squad for Gameweek {gw} ({season_state})")
        st.dataframe(pd.DataFrame(squad))
    else:
        st.warning("No squad data found for this gameweek.")
