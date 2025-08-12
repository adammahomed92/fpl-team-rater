# app.py
import streamlit as st
import requests
import pandas as pd
from typing import Any, Optional
from io import StringIO

st.set_page_config(page_title="FPL Team Rater + Predictions", layout="wide")

BASE = "https://fantasy.premierleague.com/api"
BOOTSTRAP = f"{BASE}/bootstrap-static/"
ENTRY = f"{BASE}/entry/{{}}/"
PICKS = f"{BASE}/entry/{{}}/event/{{}}/picks/"
MY_TEAM = f"{BASE}/my-team/{{}}/"
PLAYER_SUMMARY = f"{BASE}/element-summary/{{}}/"
PREDICTIONS_URL = "https://www.fantasyfootballpundit.com/fpl-points-predictor/"

# Team colours by official name (tweak if needed)
TEAM_COLORS = {
    "Arsenal": "#EF0107", "Aston Villa": "#95BFE5", "Bournemouth": "#DA291C",
    "Brentford": "#D20000", "Brighton": "#0057B8", "Burnley": "#6C1D45",
    "Chelsea": "#034694", "Crystal Palace": "#1B458F", "Everton": "#003399",
    "Fulham": "#000000", "Liverpool": "#C8102E", "Luton Town": "#FF6600",
    "Man City": "#6CABDD", "Man United": "#DA291C", "Newcastle": "#241F20",
    "Nottingham Forest": "#E51937", "Sheffield United": "#EE2737", "Spurs": "#132257",
    "Tottenham Hotspur": "#132257", "West Ham": "#7A263A", "Wolves": "#F5A300",
    "BURNLEY": "#6C1D45"  # some variations
}

# ---------- Helpers ----------
@st.cache_data(show_spinner=False)
def fetch_bootstrap() -> dict:
    r = requests.get(BOOTSTRAP, timeout=10)
    r.raise_for_status()
    return r.json()

def safe_get_json(url: str) -> Any:
    """Return JSON on success, else return dict describing error."""
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
        return {"__status__": r.status_code, "__text__": r.text[:1000]}

def find_picks_recursive(obj):
    """Find list containing dicts with 'element' key recursively."""
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
    """Return list of dicts with keys 'element' and 'is_captain'"""
    if raw is None:
        return None
    if isinstance(raw, dict):
        if "picks" in raw and isinstance(raw["picks"], list):
            raw = raw["picks"]
        else:
            found = find_picks_recursive(raw)
            if found:
                raw = found
    if isinstance(raw, list):
        out = []
        for it in raw:
            if isinstance(it, dict) and "element" in it:
                out.append({"element": int(it["element"]), "is_captain": bool(it.get("is_captain", False))})
            elif isinstance(it, int):
                out.append({"element": it, "is_captain": False})
        return out
    return None

@st.cache_data(show_spinner=False)
def fetch_predictions() -> Optional[pd.DataFrame]:
    """
    Scrape predicted points table from FantasyFootballPundit.
    Returns a DataFrame with columns including 'Name' and 'Predicted Points' if successful.
    """
    headers = {"User-Agent": "Mozilla/5.0 (compatible; FPL-Team-Rater/1.0)"}
    try:
        r = requests.get(PREDICTIONS_URL, headers=headers, timeout=15)
        r.raise_for_status()
        # pandas will parse the first table usually containing predictions
        tables = pd.read_html(r.text)
        if not tables:
            return None
        # heuristics: find a table with a 'Predicted' or 'Pred' column
        for t in tables:
            cols_lower = [c.lower() for c in t.columns.astype(str)]
            if any("pred" in c for c in cols_lower) or any("predicted" in c for c in cols_lower):
                df = t.copy()
                # try to standardize columns
                name_col = next((c for c in df.columns if 'name' in str(c).lower()), df.columns[0])
                pred_col = next((c for c in df.columns if 'pred' in str(c).lower()), None)
                if pred_col is None:
                    # fallback: try numeric columns
                    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                    pred_col = numeric_cols[0] if numeric_cols else None
                df = df[[name_col, pred_col]].rename(columns={name_col: "name", pred_col: "pred_pts"})
                df["pred_pts"] = pd.to_numeric(df["pred_pts"], errors="coerce")
                # cleanup name column (strip excess whitespace)
                df["name"] = df["name"].astype(str).str.strip()
                return df
        # final fallback: return first table
        df = tables[0]
        return df
    except Exception:
        return None

# ---------- UI ----------
st.title("⚽ FPL Team Rater — Predictions & Transfer Suggestions")
st.write("This app fetches your squad (pre-season friendly), merges predicted points from FantasyFootballPundit, shows a visual lineup, rates your team, and suggests transfers.")

left, right = st.columns([3,1])
with left:
    entry_id = st.text_input("Enter your FPL Entry ID", placeholder="e.g. 2792859")
    gw_input = st.number_input("Optional: Gameweek (0 = auto-detect)", min_value=0, value=0)
    show_diagnostics = st.checkbox("Show diagnostics", value=False)
with right:
    st.markdown("### Quick tips")
    st.markdown("- Use the numeric Entry ID from your FPL URL.")
    st.markdown("- If picks are private the app will ask you to upload a CSV.")
    st.markdown("- Predictions are scraped from an external site; if that fails the app will still run without predicted points.")

if st.button("Fetch squad & run analysis"):
    if not entry_id or not entry_id.strip().isdigit():
        st.error("Please enter a numeric Entry ID.")
        st.stop()
    entry_id = entry_id.strip()

    # fetch bootstrap
    try:
        bootstrap = fetch_bootstrap()
    except Exception as e:
        st.error(f"Could not fetch bootstrap-static: {e}")
        st.stop()

    elements = {e["id"]: e for e in bootstrap["elements"]}
    element_types = {et["id"]: et["singular_name_short"] for et in bootstrap["element_types"]}
    teams = {t["id"]: t for t in bootstrap["teams"]}
    events = bootstrap.get("events", [])

    # detect gameweek
    if gw_input > 0:
        gw = gw_input
    else:
        gw = next((e["id"] for e in events if e.get("is_current")), None)
        if gw is None:
            # pre-season: use next if exists
            gw = next((e["id"] for e in events if e.get("is_next")), None)

    resolved_from = None
    picks_raw = None

    # 1) Try canonical picks endpoint if we have a gw
    if gw:
        if show_diagnostics:
            st.write(f"Trying picks endpoint for GW {gw}...")
        picks_resp = safe_get_json(PICKS.format(entry_id, gw))
        if isinstance(picks_resp, dict) and "picks" in picks_resp:
            picks_raw = picks_resp["picks"]
            resolved_from = f"PICKS endpoint (GW {gw})"
        else:
            if show_diagnostics:
                st.write("picks endpoint response keys/status:", list(picks_resp.keys()) if isinstance(picks_resp, dict) else str(picks_resp))

    # 2) Try /my-team/{id}/
    if picks_raw is None:
        if show_diagnostics:
            st.write("Trying my-team endpoint...")
        my_team_resp = safe_get_json(MY_TEAM.format(entry_id))
        if isinstance(my_team_resp, dict) and "picks" in my_team_resp:
            picks_raw = my_team_resp["picks"]
            resolved_from = "MY_TEAM endpoint"
        else:
            if show_diagnostics and isinstance(my_team_resp, dict):
                st.write("my-team keys:", list(my_team_resp.keys()))

    # 3) Try /entry/{id}/ and recursive search for squad/picks
    if picks_raw is None:
        if show_diagnostics:
            st.write("Trying entry endpoint and recursive search...")
        entry_resp = safe_get_json(ENTRY.format(entry_id))
        if isinstance(entry_resp, dict):
            if show_diagnostics:
                st.write("entry top-level keys:", list(entry_resp.keys()))
            found = find_picks_recursive(entry_resp)
            if found:
                picks_raw = found
                resolved_from = "ENTRY endpoint (recursive found picks)"
            else:
                # try common nested 'entry' -> 'squad'
                if isinstance(entry_resp.get("entry"), dict):
                    squad = entry_resp["entry"].get("squad")
                    if squad:
                        picks_raw = squad
                        resolved_from = "ENTRY -> entry.squad"
                if picks_raw is None and "squad" in entry_resp and isinstance(entry_resp["squad"], list):
                    picks_raw = entry_resp["squad"]
                    resolved_from = "ENTRY -> squad (top-level)"
        else:
            if show_diagnostics:
                st.write("entry endpoint returned:", entry_resp)

    # 4) CSV fallback
    if picks_raw is None:
        st.warning("Could not auto-detect picks/squad from public endpoints.")
        st.info("Upload a CSV fallback with an 'element' column (player ids), or wait until GW1 starts.")
        uploaded = st.file_uploader("Upload CSV with 'element' column", type=["csv"])
        if uploaded is None:
            st.stop()
        try:
            df_csv = pd.read_csv(uploaded)
            if "element" not in df_csv.columns:
                st.error("CSV must include 'element' column.")
                st.stop()
            picks_raw = df_csv.to_dict(orient="records")
            resolved_from = "User CSV upload"
        except Exception as e:
            st.error(f"CSV read error: {e}")
            st.stop()

    # normalize picks
    picks = normalize_picks(picks_raw)
    if not picks:
        st.error("No valid picks could be constructed from the data.")
        st.stop()

    if resolved_from is None:
        resolved_from = "unknown"

    st.success(f"Loaded picks from: {resolved_from} ({len(picks)} players)")

    # build squad DataFrame
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

    # Visual lineup
    st.markdown("## Squad Lineup")
    df["colour"] = df["team_name"].map(TEAM_COLORS).fillna("#777777")
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

    # Fetch predictions (best-effort)
    pred_df = fetch_predictions()
    if pred_df is None:
        st.info("Predicted points not available (external site could not be scraped). Recommendations will use historical stats only.")
    else:
        st.success("Fetched predicted points (from FantasyFootballPundit)")

    # compute team rating (composite)
    avg_points = df["total_points"].mean()
    avg_form = df["form"].mean()
    val_eff = (df["total_points"] / (df["now_cost_m"] + 0.01)).mean()
    # make predicted component: average predicted points for squad if predictions available
    if pred_df is not None:
        # match by player name heuristically: pred names usually full names; match by web_name or second_name
        def lookup_pred(name):
            # try exact match
            match = pred_df[pred_df["name"].str.contains(str(name), case=False, na=False)]
            if not match.empty:
                return match["pred_pts"].iloc[0]
            # try last name match
            last = str(name).split()[-1]
            match2 = pred_df[pred_df["name"].str.contains(last, case=False, na=False)]
            if not match2.empty:
                return match2["pred_pts"].iloc[0]
            return None
        preds = []
        for _, r in df.iterrows():
            p = lookup_pred(r["web_name"])
            if p is None:
                p = lookup_pred(f"{r['first_name']} {r['second_name']}")
            preds.append(p if p is not None else 0.0)
        df["pred_pts"] = preds
        avg_pred = float(df["pred_pts"].mean())
    else:
        avg_pred = 0.0

    # Normalized sub-scores
    score_points = min(avg_points / 150.0, 1.0) * 100.0
    score_form = min(avg_form / 8.0, 1.0) * 100.0
    score_val = min(val_eff / 8.0, 1.0) * 100.0
    score_pred = min(avg_pred / 5.0, 1.0) * 100.0  # assume 5 predicted pts is high
    # weights (you can tune)
    WEIGHTS = {"points": 0.35, "form": 0.25, "value": 0.2, "pred": 0.2}
    final_score = round(score_points * WEIGHTS["points"] + score_form * WEIGHTS["form"] + score_val * WEIGHTS["value"] + score_pred * WEIGHTS["pred"], 1)
    st.metric("Team Rating", f"{final_score} / 100")

    # Recommendations
    st.markdown("## Recommendations")

    # OUT: lowest form and low ppm
    df["ppm"] = df["total_points"] / (df["now_cost_m"] + 0.01)
    out_candidates = df.sort_values(["form", "ppm"], ascending=[True, True]).head(5)
    st.markdown("### Players to consider transferring OUT (low form / low value)")
    for _, r in out_candidates.iterrows():
        st.write(f"- {r['web_name']} ({r['team_name']}) — Form: {r['form']} — Pts: {r['total_points']} — £{r['now_cost_m']}m — PPM: {round(r['ppm'],2)}")

    # IN: top predicted players (not in squad) with minutes > 0 and good ppm
    all_players = pd.DataFrame(bootstrap["elements"])
    all_players["now_cost_m"] = all_players["now_cost"] / 10.0
    all_players["ppm"] = all_players["total_points"] / (all_players["now_cost"] / 10.0 + 0.01)
    squad_ids = set(df["id"].tolist())
    candidates = all_players[~all_players["id"].isin(squad_ids)].copy()
    # filter realistic ones who have minutes > 0 (not brand-new zero-minute players)
    candidates = candidates[candidates["minutes"] > 0]
    # merge predicted points if available
    if pred_df is not None:
        # normalize names in pred_df to lower for matching
        pred_lookup = pred_df.set_index(pred_df["name"].str.lower())["pred_pts"].to_dict()
        def get_pred_for(row):
            # try several name variations
            name_variants = [row["web_name"], f"{row['first_name']} {row['second_name']}", row["second_name"]]
            for v in name_variants:
                if not v or pd.isna(v): 
                    continue
                val = pred_lookup.get(str(v).lower())
                if val is not None:
                    return val
            return None
        candidates["pred_pts"] = candidates.apply(get_pred_for, axis=1)
        # fill missing preds with 0 for sorting
        candidates["pred_pts"] = pd.to_numeric(candidates["pred_pts"], errors="coerce").fillna(0.0)
    else:
        candidates["pred_pts"] = 0.0

    # Score candidate by predicted points and ppm
    # Normalize pred and ppm into simple score
    candidates["score"] = candidates["pred_pts"] * 2.0 + candidates["ppm"] * 0.5
    top_in = candidates.sort_values("score", ascending=False).head(12)
    st.markdown("### Players to consider transferring IN (predicted points + value)")
    for _, r in top_in.head(6).iterrows():
        st.write(f"- {r['web_name']} ({bootstrap['teams'][r['team']-1]['name']}) — Pred: {round(r['pred_pts'],2) if 'pred_pts' in r else 'N/A'} — PPM: {round(r['ppm'],2)} — Pts: {int(r['total_points'])} — £{r['now_cost_m']}m")

    # Diagnostics
    if show_diagnostics:
        st.markdown("### Diagnostics")
        st.write("Resolved from:", resolved_from)
        st.write("Entries found:", len(picks))
        entry_resp = safe_get_json(ENTRY.format(entry_id))
        st.write("Entry endpoint keys (top-level):", list(entry_resp.keys()) if isinstance(entry_resp, dict) else entry_resp)
        st.write("Bootstrap top keys:", list(bootstrap.keys()))
