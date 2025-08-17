# app.py
import streamlit as st
import requests
import pandas as pd
from typing import Any, Optional, Dict

st.set_page_config(page_title="FPL Team Analyst", layout="wide")

BASE = "https://fantasy.premierleague.com/api"
BOOTSTRAP = f"{BASE}/bootstrap-static/"
ENTRY = f"{BASE}/entry/{{}}/"
PICKS = f"{BASE}/entry/{{}}/event/{{}}/picks/"
MY_TEAM = f"{BASE}/my-team/{{}}/"
PLAYER_SUMMARY = f"{BASE}/element-summary/{{}}/"
PREDICTIONS_URL = "https://www.fantasyfootballpundit.com/fpl-points-predictor/"

# Hard-coded Entry/Team ID (always used)
DEFAULT_ENTRY_ID = "2792859"

# ---------------- Helpers ----------------
@st.cache_data(show_spinner=False)
def fetch_bootstrap() -> dict:
    r = requests.get(BOOTSTRAP, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return r.json()

def safe_get_json(url: str, headers: dict = None) -> Any:
    try:
        merged_headers = {"User-Agent": "Mozilla/5.0"}
        if headers:
            merged_headers.update(headers)
        r = requests.get(url, headers=merged_headers, timeout=12)
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
    headers = {"User-Agent": "Mozilla/5.0 (compatible; FPL-Team-Rater/1.0)"}
    try:
        r = requests.get(PREDICTIONS_URL, headers=headers, timeout=15)
        r.raise_for_status()
        tables = pd.read_html(r.text)
        if not tables:
            return None
        for t in tables:
            cols = [str(c).lower() for c in t.columns.astype(str)]
            if any("pred" in c for c in cols):
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

@st.cache_data(show_spinner=False)
def fetch_element_history(player_id: int) -> Optional[Dict]:
    url = PLAYER_SUMMARY.format(player_id)
    r = requests.get(url, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
    if r.status_code != 200:
        return None
    try:
        return r.json()
    except Exception:
        return None

def get_recent_points(player_id: int, current_gw: Optional[int]) -> Dict[str, int]:
    hist = fetch_element_history(player_id)
    if not hist or "history" not in hist or not isinstance(hist["history"], list):
        return {"gw_points": 0, "last3_points": 0}
    h = hist["history"]
    gw_pts = 0
    if current_gw:
        for row in h:
            if int(row.get("round", 0)) == int(current_gw):
                gw_pts = int(row.get("total_points", 0))
                break
    h_sorted = sorted(h, key=lambda x: int(x.get("round", 0)), reverse=True)
    last3 = sum(int(x.get("total_points", 0)) for x in h_sorted[:3])
    return {"gw_points": gw_pts, "last3_points": last3}

# ---------------- UI ----------------
st.title("⚽ FPL Team Rater — Auth-friendly + Predictions")
st.write("This tool reads your FPL squad (hard-coded to your team) and gives a rating, context, and transfer ideas. It works with public picks (in-season), cookie fallback (pre-season), or a simple CSV upload.")

with st.sidebar:
    st.header("Settings & Auth (optional)")
    gw_input = st.number_input("Gameweek (0 = auto)", min_value=0, value=0)
    cookie_input = st.text_area(
        "Optional: Cookie header value (paste 'pl_profile=...; pl_session=...')",
        placeholder="pl_profile=xxx; pl_session=yyy"
    )
    show_diag = st.checkbox("Show diagnostics", value=False)
    st.markdown("---")
    st.write("Security note: cookies are only used for the current requests and not stored by the app. Do not paste passwords.")

with st.expander("ℹ️ Keys & Scales (what you’re looking at)"):
    st.markdown(
        """
**Columns**
- **Pos** – GK / DEF / MID / FWD  
- **Player** – Player name  
- **Team** – Club  
- **GW Pts (Cur)** – Points this current gameweek (if match data exists)  
- **Pts (Last 3 GWs)** – Sum of the last 3 gameweeks in the player's history  
- **Points** – Total season points  
- **Form** – FPL form metric (recent average)  
- **Price £m** – Current price  
- **PPM** – Points Per £m (value metric)  
- **Selected %** – Ownership  
- **C** – Captain flag

**Team Rating guide**
- **90–100**: Elite – title-winning material  
- **80–89**: Great – you’re set up very well  
- **65–79**: Good – competitive with a few tweaks  
- **50–64**: Fair – several upgrades worth considering  
- **0–49**: Needs work – time to refresh core picks  
        """
    )

if st.button("Fetch my squad & analyze"):
    entry_id = DEFAULT_ENTRY_ID  # always your team

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

    # optional headers for /my-team
    headers = {"Cookie": cookie_input} if (cookie_input and '=' in cookie_input) else None

    resolved_from = None
    picks_raw = None

    # 1) public picks
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

    # 2) /my-team (auth) if needed
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

    # 3) /entry recursive search
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

    # Build squad dataframe (+ current GW and last3 points via element-summary)
    rows = []
    for p in picks:
        pid = int(p["element"])
        player = elements.get(pid)
        if not player:
            rows.append({
                "id": pid, "web_name": f"Unknown ({pid})", "position": "Unknown",
                "team_id": None, "team_name": "Unknown", "total_points": 0,
                "form": 0.0, "now_cost_m": 0.0, "selected_by_percent": 0.0,
                "is_captain": p.get("is_captain", False),
                "position_idx": p.get("position"),
                "multiplier": p.get("multiplier"),
                "gw_points": 0, "last3_points": 0
            })
            continue

        team_id = player.get("team")
        team_name = teams.get(team_id, {}).get("name", "Unknown")
        recent = get_recent_points(pid, gw)

        rows.append({
            "id": pid,
            "web_name": player.get("web_name"),
            "first_name": player.get("first_name"),
            "second_name": player.get("second_name"),
            "position": {et["id"]: et["singular_name_short"] for et in bootstrap["element_types"]}.get(player.get("element_type")),
            "team_id": team_id,
            "team_name": team_name,
            "total_points": int(player.get("total_points", 0)),
            "form": float(player.get("form") or 0.0),
            "now_cost_m": player.get("now_cost", 0) / 10.0,
            "selected_by_percent": float(player.get("selected_by_percent") or 0.0),
            "is_captain": bool(p.get("is_captain", False)),
            "position_idx": p.get("position"),
            "multiplier": p.get("multiplier"),
            "gw_points": int(recent.get("gw_points", 0)),
            "last3_points": int(recent.get("last3_points", 0)),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        st.error("No valid players found.")
        st.stop()

    # --------- Plain tables by position ----------
    pos_order = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}
    df["pos_order"] = df["position"].map(lambda p: pos_order.get(str(p).upper(), 99))

    # Detect starters (multiplier>0 or position<=11), else pick first 11 by (pos, price, points)
    if "multiplier" in df.columns or "position_idx" in df.columns:
        def _is_starter_row(r):
            if pd.notna(r.get("multiplier", None)):
                try:
                    return int(r["multiplier"]) > 0
                except Exception:
                    pass
            if pd.notna(r.get("position_idx", None)):
                try:
                    return int(r["position_idx"]) <= 11
                except Exception:
                    pass
            return False
        df["is_starter"] = df.apply(_is_starter_row, axis=1)
        if df["is_starter"].sum() == 0:
            df = df.sort_values(["pos_order", "now_cost_m", "total_points"], ascending=[True, False, False])
            df["is_starter"] = False
            df.loc[df.index[:11], "is_starter"] = True
    else:
        df = df.sort_values(["pos_order", "now_cost_m", "total_points"], ascending=[True, False, False]).reset_index(drop=True)
        df["is_starter"] = False
        df.loc[:10, "is_starter"] = True

    df["ppm"] = df["total_points"] / (df["now_cost_m"] + 0.01)

    def _prep_display(dfx):
        out = dfx.copy().sort_values(["pos_order", "web_name"]).rename(columns={
            "position": "Pos",
            "web_name": "Player",
            "team_name": "Team",
            "gw_points": "GW Pts (Cur)",
            "last3_points": "Pts (Last 3 GWs)",
            "total_points": "Points",
            "form": "Form",
            "now_cost_m": "Price £m",
            "selected_by_percent": "Selected %",
            "is_captain": "C"
        })
        out["Form"] = out["Form"].astype(float).round(2)
        out["Price £m"] = out["Price £m"].astype(float).round(1)
        out["PPM"] = (out["Points"] / (out["Price £m"] + 0.01)).astype(float).round(2)
        return out[["Pos", "Player", "Team", "GW Pts (Cur)", "Pts (Last 3 GWs)", "Points", "Form", "Price £m", "PPM", "Selected %", "C"]]

    starters_tbl = _prep_display(df[df["is_starter"]])
    subs_tbl     = _prep_display(df[~df["is_starter"]])

    st.markdown("### Starting XI (GK → DEF → MID → FWD)")
    st.table(starters_tbl)

    st.markdown("### Substitutes")
    st.table(subs_tbl)

    # -------- Predictions + Team rating + Recommendations --------
    pred_df = fetch_predictions()
    if pred_df is None:
        st.info("Predicted points unavailable (external site scrape failed). Recommendations will use historical stats.")
    else:
        st.success("Fetched predicted points (external source).")

    avg_points = df["total_points"].mean()
    avg_form = df["form"].mean()
    val_eff = (df["total_points"] / (df["now_cost_m"] + 0.01)).mean()

    if pred_df is not None:
        def find_pred(name):
            m = pred_df[pred_df["name"].str.contains(str(name), case=False, na=False)]
            if not m.empty:
                return float(m["pred_pts"].iloc[0])
            last = str(name).split()[-1]
            m2 = pred_df[pred_df["name"].str.contains(last, case=False, na=False)]
            if not m2.empty:
                return float(m2["pred_pts"].iloc[0])
            return 0.0
        df["pred_pts"] = [find_pred(n) for n in df["web_name"].astype(str)]
        # try first+last if direct fails
        zero_mask = df["pred_pts"] == 0.0
        df.loc[zero_mask, "pred_pts"] = [
            find_pred(f"{fn} {sn}") for fn, sn in zip(df.loc[zero_mask, "first_name"].fillna(""), df.loc[zero_mask, "second_name"].fillna(""))
        ]
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
    out_candidates = df[df["is_starter"]].sort_values(["form", "ppm"], ascending=[True, True]).head(5)
    st.markdown("### Players to consider transferring OUT (low form / low value)")
    for _, r in out_candidates.iterrows():
        st.write(
            f"- {r['web_name']} ({r['team_name']}) — Form: {r['form']} — "
            f"GW Pts: {r['gw_points']} — Last 3: {r['last3_points']} — "
            f"Pts: {int(r['total_points'])} — £{r['now_cost_m']}m — PPM: {round(r['ppm'],2)}"
        )

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

    if show_diag:
        st.markdown("### Diagnostics")
        st.write("Resolved from:", resolved_from)
        st.write("Number of picks loaded:", len(picks))
        st.write("Bootstrap keys:", list(bootstrap.keys()))
        entry_resp = safe_get_json(ENTRY.format(entry_id))
        st.write("Entry endpoint keys (top-level):", list(entry_resp.keys()) if isinstance(entry_resp, dict) else entry_resp)
