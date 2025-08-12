import streamlit as st
import requests
import pandas as pd

# --- Constants ---
BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
MY_TEAM_URL = "https://fantasy.premierleague.com/api/my-team/{team_id}/"
PICKS_URL = "https://fantasy.premierleague.com/api/entry/{team_id}/event/{gw}/picks/"
LOGIN_URL = "https://users.premierleague.com/accounts/login/"

# --- Helper: FPL Login ---
def fpl_login(email, password):
    session = requests.Session()
    payload = {
        "login": email,
        "password": password,
        "app": "plfpl-web",
        "redirect_uri": "https://fantasy.premierleague.com"
    }
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    r = session.post(LOGIN_URL, data=payload, headers=headers)
    if r.status_code != 200:
        st.error(f"Login failed: {r.status_code}")
        return None
    return session

# --- Helper: Get Team Data ---
def get_team(session, team_id, gw):
    # Try live picks first
    picks_url = PICKS_URL.format(team_id=team_id, gw=gw)
    r = session.get(picks_url)
    if r.status_code == 200:
        return r.json(), "live"
    
    # Fallback to pre-season squad
    my_team_url = MY_TEAM_URL.format(team_id=team_id)
    r = session.get(my_team_url)
    if r.status_code == 200:
        return r.json(), "pre"
    
    return None, None

# --- Helper: Shirt color mapping ---
def get_team_colors():
    r = requests.get(BOOTSTRAP_URL)
    data = r.json()
    team_map = {t["id"]: {"name": t["name"], "code": t["code"], "short": t["short_name"]} for t in data["teams"]}
    return team_map, data["elements"]

# --- Helper: Display Squad ---
def display_squad(squad_data, player_data, team_map, mode):
    st.subheader("Your Squad")
    rows = []
    
    if mode == "live":
        for pick in squad_data["picks"]:
            pid = pick["element"]
            pinfo = next(p for p in player_data if p["id"] == pid)
            tinfo = team_map[pinfo["team"]]
            rows.append({
                "Player": pinfo["web_name"],
                "Team": tinfo["name"],
                "Position": pinfo["element_type"],
                "Now Cost": pinfo["now_cost"] / 10
            })
    else:
        for pick in squad_data["picks"]:
            pid = pick["element"]
            pinfo = next(p for p in player_data if p["id"] == pid)
            tinfo = team_map[pinfo["team"]]
            rows.append({
                "Player": pinfo["web_name"],
                "Team": tinfo["name"],
                "Position": pinfo["element_type"],
                "Now Cost": pinfo["now_cost"] / 10
            })
    
    df = pd.DataFrame(rows)
    st.dataframe(df)

# --- Streamlit UI ---
st.title("⚽ Fantasy Premier League - Team Rater")

email = st.sidebar.text_input("FPL Email")
password = st.sidebar.text_input("FPL Password", type="password")
team_id = st.sidebar.number_input("Team ID", step=1)
gw = st.sidebar.number_input("Gameweek", step=1, min_value=1, max_value=38)

if st.sidebar.button("Fetch Team"):
    session = fpl_login(email, password)
    if session:
        squad_data, mode = get_team(session, team_id, gw)
        if squad_data:
            team_map, player_data = get_team_colors()
            display_squad(squad_data, player_data, team_map, mode)
        else:
            st.error("Could not fetch your team data.")

