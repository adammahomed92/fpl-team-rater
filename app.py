import streamlit as st
import requests
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="FPL Team Rater", layout="wide")

st.title("⚽ Fantasy Premier League - Team Rater")

# Get current gameweek from API
def get_current_gameweek():
    try:
        r = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/")
        r.raise_for_status()
        data = r.json()
        events = data['events']
        current_gw = next(e['id'] for e in events if e['is_current'])
        return current_gw
    except Exception as e:
        st.error(f"Error fetching gameweek: {e}")
        return None

# Get team picks for a given GW
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

# Get player details from bootstrap data
def get_player_info():
    try:
        r = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/")
        r.raise_for_status()
        return r.json()['elements']
    except Exception as e:
        st.error(f"Error fetching player info: {e}")
        return []

# Sidebar inputs
team_id = st.sidebar.number_input("Enter your FPL Team ID", min_value=1, value=123456)
if st.sidebar.button("Fetch My Squad"):
    current_gw = get_current_gameweek()
    if current_gw:
        picks_data = get_team_picks(team_id, current_gw)
        if picks_data and "picks" in picks_data:
            players_info = get_player_info()
            player_df = pd.DataFrame(players_info)

            picks_list = []
            for pick in picks_data["picks"]:
                pid = pick["element"]
                player = player_df[player_df["id"] == pid].iloc[0]
                picks_list.append({
                    "Player": f"{player['first_name']} {player['second_name']}",
                    "Position": player['element_type'],
                    "Team": player['team'],
                    "Now Cost": player['now_cost'] / 10,
                    "Multiplier": pick['multiplier']
                })

            picks_df = pd.DataFrame(picks_list)
            st.subheader(f"Gameweek {current_gw} Squad")
            st.dataframe(picks_df)
        else:
            st.error("Could not auto-detect picks/squad. Ensure your team ID is correct and public.")
