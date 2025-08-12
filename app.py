# app.py
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

# Team colours by official name (tweak as desired)
TEAM_COLORS = {
    "Arsenal": "#EF0107",
    "Aston Villa": "#95BFE5",
    "Bournemouth": "#DA291C",
    "Brentford": "#D20000",
    "Brighton": "#0057B8",
    "Burnley": "#6C1D45",
    "Chelsea": "#034694",
    "Crystal Palace": "#1B458F",
    "Everton": "#003399",
    "Fulham": "#000000",
    "Liverpool": "#C8102E",
    "Luton": "#FF6600",
    "Man City": "#6CABDD",
    "Man United": "#DA291C",
    "Newcastle": "#241F20",
    "Nottingham Forest": "#E51937",
    "Sheffield United": "#EE2737",
    "Spurs": "#132257",
    "Tottenham Hotspur": "#132257",  # some APIs call them this
    "West Ham": "#7A263A",
    "Wolves": "#F5A300"
}

# ---------- Helpers ----------
@st.cache_data(show_spinner=False)
def fetch_bootstrap():
    r = requests.get(BOOTSTRAP, timeout=10)
    r.raise_for_status()
    return r.json()

def try_get(url: str) -> Any:
    """Return response.json() on 200, else return dict with status/text or error."""
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
        return {"__status__": r.status_code, "__text__": r.text[:800]}

def find_picks_recursive(obj):
    """Recursively search JSON for a list containing dicts with 'element' key (picks)."""
    if isinstance(obj, dict):
        if "picks" in obj and isinstance(obj["picks"], list):
            return obj["picks"]
        for v in obj.values():
            found = find_picks_recursive(v)
            if found:
                return found
    elif isinstance(obj, list) and obj:
        first = obj[0]
        if isinstance(first, dict) and "element" in first:
            return obj
        for item in obj:
            found = find_picks_recursive(item)
            if found:
                return found
    return None

def normalize_picks(raw):
    """Return list of dicts with keys 'element' (int) and 'is_captain' (bool)."""
    if raw is None:
        return None
    if isinstance(raw, dict):
        raw = find_picks_recursive(raw) or raw.get("picks") or []
    if isinstance(raw, list):
        normalized = []
        for it in raw:
            if isinstance(it, dict) and "element" in it:
                normalized.append({"element": int(it["element"]), "is_captain": bool(it.get("is_captain", False))})
            elif isinstance(it, int):
                normalized.append({"element": int(it), "is_captain": False})
        return normalized
    return None

# ---------- UI ----------
st.title("⚽ FPL Team Rater — Visual Lineup & Transfer Suggestions")
st.write("Enter your Entry ID (the number in your FPL URL). The app will try multiple endpoints and offer CSV fallback.")

col1, col2 = st.columns([3,1])
with col1:
    entry_id = st.text_input("FPL Entry ID", value="", placeholder="e.g. 2792859")
with col2:
    gw_input = st.number_input("Optional: Gameweek (0 = auto)", min_value=0, value=0)

if st.button("Show my team"):
    # basic validation
    if not entry_id or not entry_id.strip().isdigit():
        st.error("Please enter a numeric Entry ID.")
        st.stop()
    entry_id = entry_id.strip()

    # fetch bootstrap
    try:
        bootstrap = fetch_bootstrap()
    except Exception as exc:
        st.error(f"Could not fetch FPL static data: {exc}")
        st.stop()

    elements = {e["id"]: e for e in bootstrap["elements"]}
    element_types = {et["id"]: et["singular_name_short"] for et in bootstrap["element_types"]}
    teams = {t["id"]: t for t in bootstrap["teams"]}

    # determine gameweek
    events = bootstrap.get("events", [])
    gw = gw_input if gw_input > 0 else next((e["id"] for e in events if e.get("is_current")), None)
    st.info(f"Detected GW = {gw} (you can set manually)")

    resolved_from = None
    picks_raw = None

    # 1) Try /entry/{id}/event/{gw}/picks/
    if gw:
        st.write("Trying picks endpoint...")
        r = try_get(PICKS.format(entry_id, gw))
        if isinstance(r, dict) and "picks" in r:
            picks_raw = r["picks"]
            resolved_from = f"PICKS endpoint (GW {gw})"
        else:
            st.write(f"Picks endpoint result: {r.get('__status__') if isinstance(r, dict) else 'unknown'}")

    # 2) Try /my-team/{id}/
    if picks_raw is None:
        st.write("Trying my-team endpoint...")
        r = try_get(MY_TEAM.format(entry_id))
        if isinstance(r, dict) and "picks" in r:
            picks_raw = r["picks"]
            resolved_from = "MY_TEAM endpoint"
        else:
            # show top-level keys for debugging
            if isinstance(r, dict):
                st.write("my-team keys:", list(r.keys()))

    # 3) Try /entry/{id}/ and recursive find
    if picks_raw is None:
        st.write("Trying entry endpoint and searching for squad/picks...")
        r = try_get(ENTRY.format(entry_id))
        if isinstance(r, dict):
            st.write("Entry top-level keys (debug):", list(r.keys()))
            found = find_picks_recursive(r)
            if found:
                picks_raw = found
                resolved_from = "ENTRY endpoint (recursive found picks)"
            else:
                # check common 'entry'->'squad'
                if isinstance(r.get("entry"), dict):
                    squad = r["entry"].get("squad")
                    if squad:
                        picks_raw = squad
                        resolved_from = "ENTRY -> entry.squad"
                # also check top-level 'squad'
                if picks_raw is None and "squad" in r and isinstance(r["squad"], list):
                    picks_raw = r["squad"]
                    resolved_from = "ENTRY -> squad (top-level)"

    # 4) If still none, let user upload CSV
    if picks_raw is None:
        st.warning("Could not auto-detect picks/squad from public endpoints.")
        st.info("Upload a CSV with a column named 'element' (player id) as a fallback, or wait until GW1 starts.")
        uploaded = st.file_uploader("Upload CSV with column 'element'", type=["csv"])
        if uploaded is None:
            st.stop()
        try:
            df_csv = pd.read_csv(uploaded)
            if "element" not in df_csv.columns:
                st.error("CSV must include an 'element' column containing FPL player ids.")
                st.stop()
            picks_raw = df_csv.to_dict(orient="records")
            resolved_from = "User CSV upload"
        except Exception as e:
            st.error(f"Could not read uploaded CSV: {e}")
            st.stop()

    # normalize picks
    picks = normalize_picks(picks_raw)
    if not picks:
        st.error("Could not construct picks list from the data found.")
        st.stop()

    st.success(f"Loaded picks from: {resolved_from} — {len(picks)} players")

    # Build dataframe of squad players
    rows = []
    for p in picks:
        pid = int(p["element"])
        player = elements.get(pid)
        if not player:
            rows.append({
                "id": pid,
                "web_name": f"Unknown ({pid})",
                "position": "Unknown",
                "team_id": None,
                "team_name": "Unknown",
                "total_points": 0,
                "form": 0.0,
                "now_cost_m": 0.0,
                "selected_by_percent": 0.0,
                "is_captain": p.get("is_captain", False)
            })
            continue
        team_id = player.get("team")
        team_name = teams.get(team_id, {}).get("name", "Unknown")
        rows.append({
            "id": pid,
            "web_name": player.get("web_name"),
            "first_name": player.get("first_name"),
            "second_name": player.get("second_name"),
            "position": element_types.get(player.get("element_type")),
            "team_id": team_id,
            "team_name": team_name,
            "total_points": int(player.get("total_points", 0)),
            "form": float(player.get("form") or 0.0),
            "now_cost_m": player.get("now_cost", 0) / 10.0,
            "selected_by_percent": float(player.get("selected_by_percent") or 0.0),
            "is_captain": bool(p.get("is_captain", False))
        })

    df = pd.DataFrame(rows)
    if df.empty:
        st.error("No valid players found.")
        st.stop()

    # Visual lineup - simple card grid
    st.markdown("## Squad Lineup")
    df["colour"] = df["team_name"].map(TEAM_COLORS).fillna("#777777")
    # show 4 columns of cards
    ncols = 4
    cols = st.columns(ncols)
    for i, r in df.iterrows():
        col = cols[i % ncols]
        initials = "".join([s[0] for s in str(r["web_name"]).split()][:2]).upper()
        captain = " 🧢" if r["is_captain"] else ""
        with col:
            st.markdown(
                f"""
                <div style="background:{r['colour']}; padding:10px; border-radius:8px; color:#fff; text-align:center; margin-bottom:8px;">
                    <div style="font-size:20px; font-weight:700">{initials}</div>
                    <div style="font-weight:600">{r['web_name']}{captain}</div>
                    <div style="font-size:12px">{r['position']} — {r['team_name']}</div>
                    <div style="font-size:12px">Pts: {r['total_points']} • Form: {r['form']} • £{r['now_cost_m']}m</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # Team Rating (heuristic composite)
    avg_points = df["total_points"].mean()
    avg_form = df["form"].mean()
    val_eff = (df["total_points"] / (df["now_cost_m"] + 0.01)).mean()
    # normalize and weight safely (digit-by-digit style internally done by Python)
    score_points = min(avg_points / 150.0, 1.0) * 100.0
    score_form = min(avg_form / 8.0, 1.0) * 100.0
    score_val = min(val_eff / 8.0, 1.0) * 100.0
    final_score = round(score_points * 0.4 + score_form * 0.35 + score_val * 0.25, 1)
    st.metric("Team Rating", f"{final_score} / 100")

    # Recommendations: OUT = lowest form, IN = best points-per-million not in squad
    st.markdown("## Recommendations")

    # Out suggestions
    out_df = df.sort_values("form").head(5)
    st.markdown("### Players to consider transferring OUT (lowest form)")
    for _, r in out_df.iterrows():
        st.write(f"- {r['web_name']} — Form: {r['form']} — Points: {r['total_points']} — £{r['now_cost_m']}m")

    # In suggestions (best ppm not currently in squad)
    all_players = pd.DataFrame(bootstrap["elements"])
    all_players["now_cost_m"] = all_players["now_cost"] / 10.0
    all_players["ppm"] = all_players["total_points"] / (all_players["now_cost_m"] + 0.01)
    squad_ids = set(df["id"].tolist())
    candidates = all_players[~all_players["id"].isin(squad_ids)].copy()
    # prefer those with reasonable form and high ppm
    candidates = candidates[(candidates["minutes"] > 0)].sort_values(["ppm", "form"], ascending=[False, False])
    top_in = candidates.head(10)[["id", "web_name", "total_points", "now_cost_m", "ppm"]].head(5)

    st.markdown("### Players to consider transferring IN (best points-per-million and minutes played)")
    if top_in.empty:
        st.write("No incoming candidates found.")
    else:
        for _, r in top_in.iterrows():
            st.write(f"- {r['web_name']} — PPM: {round(r['ppm'],2)} — Points: {int(r['total_points'])} — £{r['now_cost_m']}m")

    # Diagnostics / troubleshooting
    with st.expander("Diagnostics / Raw info"):
        st.write("Resolved from:", resolved_from)
        st.write("Number of picks loaded:", len(picks))
        # show top-level keys of /entry/ for debugging
        entry_resp = try_get(ENTRY.format(entry_id))
        if isinstance(entry_resp, dict):
            st.write("Entry endpoint keys:", list(entry_resp.keys()))
        else:
            st.write("Entry endpoint response:", entry_resp)
