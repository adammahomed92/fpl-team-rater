# app.py
import streamlit as st
import requests
import pandas as pd
from typing import Any, Optional

st.set_page_config(page_title="FPL Team Rater (Auth + Predictions)", layout="wide")

BASE = "https://fantasy.premierleague.com/api"
BOOTSTRAP = f"{BASE}/bootstrap-static/"
ENTRY = f"{BASE}/entry/{{}}/"
PICKS = f"{BASE}/entry/{{}}/event/{{}}/picks/"
MY_TEAM = f"{BASE}/my-team/{{}}/"
PLAYER_SUMMARY = f"{BASE}/element-summary/{{}}/"
PREDICTIONS_URL = "https://www.fantasyfootballpundit.com/fpl-points-predictor/"

# Basic mapping of team colors by official name (tweak as desired)
TEAM_COLORS = {
    "Arsenal": "#EF0107", "Aston Villa": "#95BFE5", "Bournemouth": "#DA291C",
    "Brentford": "#D20000", "Brighton": "#0057B8", "Burnley": "#6C1D45",
    "Chelsea": "#034694", "Crystal Palace": "#1B458F", "Everton": "#003399",
    "Fulham": "#000000", "Liverpool": "#C8102E", "Luton Town": "#FF6600",
    "Man City": "#6CABDD", "Man United": "#DA291C", "Newcastle United": "#241F20",
    "Nottingham Forest": "#E51937", "Sheffield United": "#EE2737",
    "Tottenham Hotspur": "#132257", "Spurs": "#132257", "West Ham": "#7A263A",
    "Wolves": "#F5A300"
}

# ---------------- Helpers ----------------
@st.cache_data(show_spinner=False)
def fetch_bootstrap() -> dict:
    """Get bootstrap-static (players, teams, events, etc)."""
    r = requests.get(BOOTSTRAP, timeout=10)
    r.raise_for_status()
    return r.json()

def safe_get_json(url: str, headers: dict = None) -> Any:
    """Return response.json() if 200; else return dict describing status/error."""
    try:
        r = requests.get(url, headers=headers, timeout=12, headers={"User-Agent":"Mozilla/5.0", **(headers or {})})
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
    """Search JSON recursively for a list containing dicts with 'element' key."""
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
    """Normalize different shapes of picks/squad JSON into [{'element': id, 'is_captain': bool, 'position'?, 'multiplier'?}, ...]."""
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
                out.append({
                    "element": int(it["element"]),
                    "is_captain": bool(it.get("is_captain", False)),
                    "position": it.get("position"),
                    "multiplier": it.get("multiplier"),
                })
            elif isinstance(it, int):
                out.append({"element": int(it), "is_captain": False})
        return out
    return None

@st.cache_data(show_spinner=False)
def fetch_predictions() -> Optional[pd.DataFrame]:
    """Scrape predicted points table from FantasyFootballPundit (best-effort)."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; FPL-Team-Rater/1.0)"}
    try:
        r = requests.get(PREDICTIONS_URL, headers=headers, timeout=15)
        r.raise_for_status()
        tables = pd.read_html(r.text)
        if not tables:
            return None
        # heuristics: find table with a 'Pred' or 'Predicted' column
        for t in tables:
            cols = [str(c).lower() for c in t.columns.astype(str)]
            if any("pred" in c for c in cols):
                # Find name and predicted columns
                name_col = next((c for c in t.columns if 'name' in str(c).lower()), t.columns[0])
                pred_col = next((c for c in t.columns if 'pred' in str(c).lower()), None)
                if pred_col is None:
                    numeric_cols = [c for c in t.columns if pd.api.types.is_numeric_dtype(t[c])]
                    pred_col = numeric_cols[0] if numeric_cols else None
                df = t[[name_col, pred_col]].rename(columns={name_col: "name", pred_col: "pred_pts"})
                df["pred_pts"] = pd.to_numeric(df["pred_pts"], errors="coerce").fillna(0.0)
                df["name"] = df["name"].astype(str).str.strip()
                return df
        return tables[0]
    except Exception:
        return None

# --- Formation helpers (NEW) ---
def _make_shirt_html(name: str, colour: str, position: str, team: str, pts: int, form: float, cost_m: float, is_captain: bool) -> str:
    initials = "".join([s[0] for s in str(name).split()][:2]).upper()
    cap = " 🧢" if is_captain else ""
    return f"""
    <div style="background:{colour}; padding:12px; border-radius:10px; color:#fff; text-align:center; min-width:90px;">
        <div style="font-size:22px; font-weight:800; line-height:1">{initials}</div>
        <div style="font-weight:700; margin-top:2px">{name}{cap}</div>
        <div style="font-size:12px; opacity:.95">{position} — {team}</div>
        <div style="font-size:11px; opacity:.9">Pts: {pts} • Form: {form} • £{cost_m}m</div>
    </div>
    """

def _centered_row(n_total_cols: int, cards_html: list):
    """Render a centered row by padding empty columns left/right."""
    TOTAL = n_total_cols  # 9 is a good default
    n = min(len(cards_html), TOTAL)
    pad_left = max((TOTAL - n) // 2, 0)
    pad_right = TOTAL - n - pad_left
    cols = st.columns(TOTAL)
    for i in range(pad_left):
        with cols[i]:
            st.write("")
    for i, html in enumerate(cards_html):
        with cols[pad_left + i]:
            st.markdown(html, unsafe_allow_html=True)
    for i in range(pad_right):
        with cols[pad_left + n + i]:
            st.write("")

def _norm_pos(p):
    if not p: return "UNK"
    p = str(p).upper()
    if p in ("GKP", "GK"): return "GK"
    if p in ("DEF", "DEFENDER"): return "DEF"
    if p in ("MID", "MIDFIELDER"): return "MID"
    if p in ("FWD", "FOR", "FORWARD", "ST"): return "FWD"
    return p

# -------------- UI --------------
st.title("⚽ FPL Team Rater — Auth-friendly + Predictions")
st.write("Enter your Entry ID (the number in your FPL URL). For pre-season picks you can paste FPL cookies (pl_profile & pl_session) in the sidebar. If you prefer not to share cookies, upload a CSV with an 'element' column as a fallback.")

sidebar = st.sidebar
sidebar.header("Settings & Auth (optional)")
entry_id_input = sidebar.text_input("FPL Entry ID (numeric)", "")
gw_input = sidebar.number_input("Gameweek (0 = auto)", min_value=0, value=0)
cookie_input = sidebar.text_area("Optional: Cookie header value (paste 'pl_profile=...; pl_session=...')", placeholder="pl_profile=xxx; pl_session=yyy")
show_diag = sidebar.checkbox("Show diagnostics", value=False)
sidebar.markdown("---")
sidebar.write("Security note: cookies are only used for the current requests and not stored by the app. Do not paste passwords.")

if st.button("Fetch squad & analyze"):
    if not entry_id_input or not entry_id_input.strip().isdigit():
        st.error("Please enter a numeric Entry ID.")
        st.stop()
    entry_id = entry_id_input.strip()

    # bootstrap
    try:
        bootstrap = fetch_bootstrap()
    except Exception as e:
        st.error(f"Could not fetch bootstrap-static: {e}")
        st.stop()

    elements = {e["id"]: e for e in bootstrap["elements"]}
    element_types = {et["id"]: et["singular_name_short"] for et in bootstrap["element_types"]}
    teams = {t["id"]: t for t in bootstrap["teams"]}
    events = bootstrap.get("events", [])

    # determine gw
    if gw_input > 0:
        gw = gw_input
    else:
        gw = next((e["id"] for e in events if e.get("is_current")), None)
        if gw is None:
            gw = next((e["id"] for e in events if e.get("is_next")), None)

    # Prepare optional headers for authenticated endpoints
    headers = None
    if cookie_input and isinstance(cookie_input, str) and '=' in cookie_input:
        headers = {"Cookie": cookie_input}

    resolved_from = None
    picks_raw = None

    # 1) Try public picks endpoint if gw available
    if gw:
        if show_diag:
            st.write(f"Trying public picks endpoint for GW {gw}...")
        resp = safe_get_json(PICKS.format(entry_id, gw))
        if isinstance(resp, dict) and "picks" in resp:
            picks_raw = resp["picks"]
            resolved_from = f"PICKS endpoint (GW {gw})"
        else:
            if show_diag:
                st.write("picks response status/keys:", list(resp.keys()) if isinstance(resp, dict) else str(resp))

    # 2) Try my-team (auth required) if the public picks not found and cookie provided
    if picks_raw is None and headers:
        if show_diag:
            st.write("Trying authenticated /my-team/ endpoint...")
        resp = safe_get_json(MY_TEAM.format(entry_id), headers=headers)
        if isinstance(resp, dict) and "picks" in resp:
            picks_raw = resp["picks"]
            resolved_from = "MY_TEAM endpoint (auth)"
        else:
            if show_diag and isinstance(resp, dict):
                st.write("my-team keys:", list(resp.keys()))

    # 3) Try /entry/{id}/ and recursive search for squad/picks
    if picks_raw is None:
        if show_diag:
            st.write("Trying entry endpoint and recursive search for picks/squad...")
        resp = safe_get_json(ENTRY.format(entry_id))
        if isinstance(resp, dict):
            if show_diag:
                st.write("entry top-level keys:", list(resp.keys()))
            found = find_picks_recursive(resp)
            if found:
                picks_raw = found
                resolved_from = "ENTRY endpoint (recursive found picks)"
            else:
                # common nested spot: entry -> squad
                if isinstance(resp.get("entry"), dict):
                    squad = resp["entry"].get("squad")
                    if squad:
                        picks_raw = squad
                        resolved_from = "ENTRY -> entry.squad"
                if picks_raw is None and "squad" in resp and isinstance(resp["squad"], list):
                    picks_raw = resp["squad"]
                    resolved_from = "ENTRY -> squad (top-level)"

    # 4) CSV fallback
    if picks_raw is None:
        st.warning("Could not auto-detect picks/squad from public or authenticated endpoints.")
        st.info("Upload a CSV with column 'element' (player ids) as fallback, or provide correct cookie to access /my-team/ during pre-season.")
        uploaded = st.file_uploader("Upload CSV with 'element' column", type=["csv"])
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
            st.error(f"Error reading CSV: {e}")
            st.stop()

    picks = normalize_picks(picks_raw)
    if not picks:
        st.error("Could not construct picks list from the data found.")
        st.stop()

    if resolved_from is None:
        resolved_from = "unknown"
    st.success(f"Loaded picks from: {resolved_from} ({len(picks)} players)")

    # Build squad dataframe
    rows = []
    for p in picks:
        pid = int(p["element"])
        player = elements.get(pid)
        if not player:
            rows.append({
                "id": pid, "web_name": f"Unknown ({pid})", "position": "Unknown",
                "team_id": None, "team_name": "Unknown", "total_points": 0,
                "form": 0.0, "now_cost_m": 0.0, "selected_by_percent": 0.0,
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
            "is_captain": bool(p.get("is_captain", False)),
            "position_idx": p.get("position"),
            "multiplier": p.get("multiplier"),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        st.error("No valid players found.")
        st.stop()

    # Assign colours
    df["colour"] = df["team_name"].map(TEAM_COLORS).fillna("#777777")

    # --------- STARTERS/FORMATION DETECTION (NEW) ----------
    # Build quick lookup for position index & multiplier from the raw picks
    pos_index = {}
    mult_map = {}
    if isinstance(picks_raw, list):
        for it in picks_raw:
            if isinstance(it, dict) and "element" in it:
                eid = int(it["element"])
                if "position" in it:
                    pos_index[eid] = it.get("position", None)
                if "multiplier" in it:
                    try:
                        mult_map[eid] = int(it.get("multiplier", 0))
                    except Exception:
                        pass

    def _is_starter(row):
        eid = int(row["id"])
        if eid in mult_map:
            return mult_map[eid] > 0
        if eid in pos_index:
            return (pos_index[eid] is not None) and (int(pos_index[eid]) <= 11)
        return True  # fallback

    df["is_starter"] = df.apply(_is_starter, axis=1)
    df["pos4"] = df["position"].map(_norm_pos)

    # ------------- FORMATION LAYOUT (NEW) -------------
    st.markdown("## Current Lineup")
    starters = df[df["is_starter"]].copy()
    subs = df[~df["is_starter"]].copy()

    starters.sort_values(["pos4", "now_cost_m", "total_points"], ascending=[True, False, False], inplace=True)
    subs.sort_values(["now_cost_m", "total_points"], ascending=[False, False], inplace=True)

    gks  = starters[starters["pos4"] == "GK"]
    defs = starters[starters["pos4"] == "DEF"]
    mids = starters[starters["pos4"] == "MID"]
    fwds = starters[starters["pos4"] == "FWD"]

    def _cards(df_subset):
        cards = []
        for _, r in df_subset.iterrows():
            cards.append(_make_shirt_html(
                name=r["web_name"],
                colour=r["colour"],
                position=r["pos4"],
                team=r["team_name"],
                pts=int(r["total_points"]),
                form=float(r["form"]),
                cost_m=float(r["now_cost_m"]),
                is_captain=bool(r["is_captain"]),
            ))
        return cards

    # Each row centrally justified; TOTAL grid columns = 9 for nice centering
    st.markdown("#### GK")
    _centered_row(9, _cards(gks))

    st.markdown("#### DEF")
    _centered_row(9, _cards(defs))

    st.markdown("#### MID")
    _centered_row(9, _cards(mids))

    st.markdown("#### FWD")
    _centered_row(9, _cards(fwds))

    st.markdown("#### SUBS")
    _centered_row(9, _cards(subs))

    # -------- Rest of your app: predictions, rating, recommendations --------
    # Fetch predictions (best-effort)
    pred_df = fetch_predictions()
    if pred_df is None:
        st.info("Predicted points unavailable (external site scrape failed). Recommendations will use historical stats.")
    else:
        st.success("Fetched predicted points (external source).")

    # Team rating
    avg_points = df["total_points"].mean()
    avg_form = df["form"].mean()
    val_eff = (df["total_points"] / (df["now_cost_m"] + 0.01)).mean()
    if pred_df is not None:
        # try to map predicted points to squad heuristically
        def find_pred(name):
            m = pred_df[pred_df["name"].str.contains(str(name), case=False, na=False)]
            if not m.empty:
                return float(m["pred_pts"].iloc[0])
            last = str(name).split()[-1]
            m2 = pred_df[pred_df["name"].str.contains(last, case=False, na=False)]
            if not m2.empty:
                return float(m2["pred_pts"].iloc[0])
            return 0.0
        preds = []
        for _, r in df.iterrows():
            pval = find_pred(r["web_name"])
            if pval == 0.0:
                pval = find_pred(f"{r['first_name']} {r['second_name']}")
            preds.append(pval)
        df["pred_pts"] = preds
        avg_pred = float(df["pred_pts"].mean())
    else:
        avg_pred = 0.0

    score_points = min(avg_points / 150.0, 1.0) * 100.0
    score_form = min(avg_form / 8.0, 1.0) * 100.0
    score_val = min(val_eff / 8.0, 1.0) * 100.0
    score_pred = min(avg_pred / 5.0, 1.0) * 100.0
    WEIGHTS = {"points": 0.35, "form": 0.25, "value": 0.2, "pred": 0.2}
    final_score = round(score_points * WEIGHTS["points"] + score_form * WEIGHTS["form"] + score_val * WEIGHTS["value"] + score_pred * WEIGHTS["pred"], 1)
    st.metric("Team Rating", f"{final_score} / 100")

    # Recommendations
    st.markdown("## Recommendations")
    df["ppm"] = df["total_points"] / (df["now_cost_m"] + 0.01)
    out_candidates = df.sort_values(["form", "ppm"], ascending=[True, True]).head(5)
    st.markdown("### Players to consider transferring OUT (low form / low value)")
    for _, r in out_candidates.iterrows():
        st.write(f"- {r['web_name']} ({r['team_name']}) — Form: {r['form']} — Pts: {r['total_points']} — £{r['now_cost_m']}m — PPM: {round(r['ppm'],2)}")

    # Build candidate list for IN suggestions
    all_players = pd.DataFrame(bootstrap["elements"])
    all_players["now_cost_m"] = all_players["now_cost"] / 10.0
    all_players["ppm"] = all_players["total_points"] / (all_players["now_cost_m"] + 0.01)
    squad_ids = set(df["id"].tolist())
    candidates = all_players[~all_players["id"].isin(squad_ids)].copy()
    candidates = candidates[candidates["minutes"] > 0]
    if pred_df is not None:
        pred_lookup = pred_df.set_index(pred_df["name"].str.lower())["pred_pts"].to_dict()
        def get_pred_for(row):
            for v in [row.get("web_name"), f"{row.get('first_name','')} {row.get('second_name','')}", row.get("second_name")]:
                if not v: 
                    continue
                val = pred_lookup.get(str(v).lower())
                if val is not None:
                    return val
            return 0.0
        candidates["pred_pts"] = candidates.apply(get_pred_for, axis=1)
    else:
        candidates["pred_pts"] = 0.0
    candidates["score"] = candidates["pred_pts"] * 2.0 + candidates["ppm"] * 0.5
    top_in = candidates.sort_values("score", ascending=False).head(10)

    st.markdown("### Players to consider transferring IN (predicted points + value)")
    for _, r in top_in.head(6).iterrows():
        team_name = bootstrap["teams"][int(r["team"])-1]["name"]
        st.write(f"- {r['web_name']} ({team_name}) — Pred: {round(r.get('pred_pts',0),2)} — PPM: {round(r['ppm'],2)} — Pts: {int(r['total_points'])} — £{r['now_cost_m']}m")

    # Diagnostics
    if show_diag:
        st.markdown("### Diagnostics")
        st.write("Resolved from:", resolved_from)
        st.write("Number of picks loaded:", len(picks))
        st.write("Bootstrap keys:", list(bootstrap.keys()))
        entry_resp = safe_get_json(ENTRY.format(entry_id))
        st.write("Entry endpoint keys (top-level):", list(entry_resp.keys()) if isinstance(entry_resp, dict) else entry_resp)
