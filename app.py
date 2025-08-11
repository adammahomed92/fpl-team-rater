import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="FPL Team Rater", layout="wide")

# Constants
BOOTSTRAP = "https://fantasy.premierleague.com/api/bootstrap-static/"
ENTRY = "https://fantasy.premierleague.com/api/entry/{}/"
PICKS = "https://fantasy.premierleague.com/api/entry/{}/event/{}/picks/"
PLAYER_SUMMARY = "https://fantasy.premierleague.com/api/element-summary/{}/"

# Manual mapping of club hex colors (via TeamColorCodes.com) :contentReference[oaicite:1]{index=1}
CLUB_COLORS = {
    "Arsenal": "#EF0107",
    "Man City": "#6CABDD",
    "Liverpool": "#C8102E",
    # Add all clubs similarly...
}

@st.cache_data
def fetch_bootstrap():
    return requests.get(BOOTSTRAP).json()

def get_current_and_next_gw(events):
    current = next((e["id"] for e in events if e["is_current"]), None)
    nxt = next((e["id"] for e in events if e["is_next"]), None)
    return current, nxt

def get_preseason_squad(entry_id):
    data = requests.get(ENTRY.format(entry_id)).json()
    squad = data.get("entry", {}).get("squad", []) or []
    return [{"element": p["element"], "is_captain": False} for p in squad]

def get_gw_picks(entry_id, gw):
    r = requests.get(PICKS.format(entry_id, gw))
    return r.json().get("picks") if r.status_code == 200 else None

def rate_team(df):
    avg_pts = df["total_points"].mean()
    avg_form = df["form"].mean()
    val_efficiency = (df["total_points"] / (df["now_cost_m"] + 0.01)).mean()
    # Composite score
    score = (avg_pts * 0.4 + avg_form * 0.3 + val_efficiency * 0.3)
    return round(min(score, 100), 1)

def get_recommendations(out_df, elements):
    # Out: low form and low ppm
    low_form = out_df.nsmallest(5, "form")

    # In: sample pool from all players
    potential = pd.DataFrame(elements).T
    potential['ppm'] = potential['total_points'] / (potential['now_cost'] / 10 + 0.01)
    top_in = potential.nlargest(10, 'ppm').head(5)

    # Fixture difficulty scaling
    top_in['upcoming_fdr'] = top_in['id'].apply(lambda pid: requests.get(PLAYER_SUMMARY.format(pid)).json()['fixtures'][0]['difficulty'])
    top_in = top_in.nsmallest(5, 'upcoming_fdr')
    return low_form, top_in

entry_id = st.text_input("Enter your FPL Entry ID")
if st.button("Show my team"):
    try:
        bootstrap = fetch_bootstrap()
        events = bootstrap["events"]
        current, nxt = get_current_and_next_gw(events)

        if current is None and nxt == 1:
            st.info("Pre-season: showing pre-season squad.")
            picks = get_preseason_squad(entry_id)
        else:
            picks = get_gw_picks(entry_id, current) or get_preseason_squad(entry_id)

        players = {p["id"]: p for p in bootstrap["elements"]}
        types = {t["id"]: t["singular_name_short"] for t in bootstrap["element_types"]}

        rows = []
        for p in picks:
            pl = players[p["element"]]
            rows.append({
                "id": pl["id"],
                "name": pl["web_name"],
                "pos": types[pl["element_type"]],
                "club": pl["team"],
                "club_name": bootstrap["teams"][pl["team"]-1]["name"],
                "total_points": pl["total_points"],
                "form": float(pl["form"] or 0),
                "now_cost_m": pl["now_cost"]/10,
                "is_captain": p.get("is_captain", False)
            })

        df = pd.DataFrame(rows)
        df["color"] = df["club_name"].map(CLUB_COLORS).fillna("#FFFFFF")
        rating = rate_team(df)

        st.metric("Team Score", f"{rating}/100")
        st.markdown("### Lineup")
        for _, r in df.iterrows():
            cap = " 🧢" if r["is_captain"] else ""
            st.markdown(f"<div style='background:{r['color']}; padding:8px; margin:2px; border-radius:4px;'>"
                        f"{r['name']} ({r['pos']}) - {r['club_name']}{cap}</div>", unsafe_allow_html=True)

        low_form, top_in = get_recommendations(df, players)
        st.markdown("### Suggestions — Players to consider transferring out")
        for _, r in low_form.iterrows():
            st.write(f"{r['name']} — Form: {r['form']} — Points: {r['total_points']}")

        st.markdown("### Suggestions — Players to consider bringing in")
        for _, r in top_in.iterrows():
            st.write(f"{r['web_name']} — PPM: {round(r['ppm'],2)} — Next Fixture Difficulty: {r['upcoming_fdr']}")

    except Exception as e:
        st.error(f"Error: {e}")
