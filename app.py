# app.py
import streamlit as st
import requests
import pandas as pd
import numpy as np
from typing import Any, Optional, Dict

# ---------------- Config ----------------
st.set_page_config(page_title="FPL Team Analyst (Pro)", layout="wide")

BASE = "https://fantasy.premierleague.com/api"
BOOTSTRAP = f"{BASE}/bootstrap-static/"
ENTRY = f"{BASE}/entry/{{}}/"
PICKS = f"{BASE}/entry/{{}}/event/{{}}/picks/"
MY_TEAM = f"{BASE}/my-team/{{}}/"
PLAYER_SUMMARY = f"{BASE}/element-summary/{{}}/"
PREDICTIONS_URL = "https://www.fantasyfootballpundit.com/fpl-points-predictor/"

# Hard-coded Entry/Team ID (always used)
DEFAULT_ENTRY_ID = "2792859"

# Abbreviation map (populated after bootstrap)
TEAM_SHORT: Dict[int, str] = {}

# ---------------- Helpers ----------------
@st.cache_data(show_spinner=False)
def fetch_bootstrap() -> dict:
    r = requests.get(BOOTSTRAP, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return r.json()

def safe_get_json(url: str, headers: dict = None) -> Any:
    try:
        merged_headers = {"User-Agent": "Mozilla/5.0"}
        if headers:
            merged_headers.update(headers)
        r = requests.get(url, headers=merged_headers, timeout=15)
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
                    "is_vice_captain": bool(it.get("is_vice_captain", False)),
                    "position": it.get("position"),
                    "multiplier": it.get("multiplier"),
                })
            elif isinstance(it, int):
                out.append({"element": int(it), "is_captain": False, "is_vice_captain": False})
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
        return None
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def fetch_element_history(player_id: int) -> Optional[Dict]:
    url = PLAYER_SUMMARY.format(player_id)
    try:
        r = requests.get(url, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def get_recent_points(player_id: int, target_rounds: list, current_gw: Optional[int]) -> Dict[str, int]:
    hist = fetch_element_history(player_id)
    pts_map = {}
    gw_cur = 0
    if hist and "history" in hist and isinstance(hist["history"], list):
        for row in hist["history"]:
            rnd = int(row.get("round", 0))
            pts_map[rnd] = int(row.get("total_points", 0))
        if current_gw:
            gw_cur = pts_map.get(int(current_gw), 0)
    out = {"gw_points": gw_cur, "gwm1": 0, "gwm2": 0, "gwm3": 0}
    for key, rnd in zip(["gwm1", "gwm2", "gwm3"], target_rounds):
        out[key] = pts_map.get(int(rnd), 0) if rnd else 0
    return out

# ---------- Advanced metrics & reasoning ----------
@st.cache_data(show_spinner=False)
def fetch_element_summary(player_id: int) -> Optional[Dict]:
    url = PLAYER_SUMMARY.format(player_id)
    try:
        r = requests.get(url, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def _last_n(values, n=5):
    arr = [v for v in values if v is not None]
    return arr[-n:] if len(arr) > n else arr

def _abbr_for_team_id(team_id: Optional[int]) -> str:
    if team_id is None:
        return "UNK"
    return TEAM_SHORT.get(int(team_id), "UNK")

def compute_player_adv_metrics(player_id: int, n_hist: int = 5, n_fix: int = 3) -> Dict[str, float]:
    """
    Returns:
        xg/xa/xgi last n, mins avg, starts%, pts std, avg FDR next n,
        next_opps (abbr + H/A), next3_abbr (e.g., 'MCI(H), LIV(A), EVE(H)')
    """
    data = fetch_element_summary(player_id)
    if not data:
        return {
            "xg_lastn": 0.0, "xa_lastn": 0.0, "xgi_lastn": 0.0,
            "mins_avg_lastn": 0.0, "starts_pct_lastn": 0.0,
            "pts_std_lastn": 0.0, "avg_fdr_nextn": 3.0,
            "next_opps": "", "next3_abbr": ""
        }
    hist = data.get("history", []) or []
    fixtures = data.get("fixtures", []) or []

    rounds = sorted(hist, key=lambda r: r.get("round", 0))
    mins = _last_n([int(h.get("minutes", 0)) for h in rounds], n_hist)
    pts  = _last_n([int(h.get("total_points", 0)) for h in rounds], n_hist)
    xg   = _last_n([float(h.get("expected_goals", 0.0) or 0.0) for h in rounds], n_hist)
    xa   = _last_n([float(h.get("expected_assists", 0.0) or 0.0) for h in rounds], n_hist)

    starts = _last_n([1 if int(h.get("minutes",0)) >= 60 else 0 for h in rounds], n_hist)
    mins_avg = float(np.mean(mins)) if mins else 0.0
    starts_pct = (sum(starts) / len(starts) * 100.0) if starts else 0.0
    pts_std = float(np.std(pts)) if len(pts) >= 2 else 0.0

    xg_sum = float(np.sum(xg)) if xg else 0.0
    xa_sum = float(np.sum(xa)) if xa else 0.0
    xgi_sum = xg_sum + xa_sum

    next_n = fixtures[:n_fix]
    fdrs = [int(f.get("difficulty", 3)) for f in next_n]
    avg_fdr = float(np.mean(fdrs)) if fdrs else 3.0

    # Opponents using bootstrap short names
    abbrs = []
    opps_for_reason = []
    for f in next_n:
        opp_id = f.get("opponent_team")
        is_home = bool(f.get("is_home"))
        ab = _abbr_for_team_id(opp_id)
        side = "H" if is_home else "A"
        abbrs.append(f"{ab}({side})")
        opps_for_reason.append(f"{ab} ({side})")
    next3_abbr = ", ".join(abbrs)
    next_opps = ", ".join(opps_for_reason)

    return {
        "xg_lastn": round(xg_sum, 2),
        "xa_lastn": round(xa_sum, 2),
        "xgi_lastn": round(xgi_sum, 2),
        "mins_avg_lastn": round(mins_avg, 1),
        "starts_pct_lastn": round(starts_pct, 1),
        "pts_std_lastn": round(pts_std, 2),
        "avg_fdr_nextn": round(avg_fdr, 2),
        "next_opps": next_opps,
        "next3_abbr": next3_abbr
    }

def reason_out(row) -> str:
    reasons = []
    if row.get("form", 0) < 2.5:
        reasons.append(f"low form ({row.get('form',0):.2f})")
    if row.get("ppm", 0) < 10 and row.get("now_cost_m", 0) >= 7.0:
        reasons.append(f"poor value (PPM {row.get('ppm',0):.1f} at £{row.get('now_cost_m',0):.1f}m)")
    if row.get("mins_avg5", 0) < 60 or row.get("starts_pct5", 0) < 60:
        reasons.append(f"rotation risk ({row.get('mins_avg5',0):.0f} mins avg, {row.get('starts_pct5',0):.0f}% starts)")
    if row.get("avg_fdr3", 3.0) >= 4.0:
        reasons.append(f"tough fixtures (avg FDR {row.get('avg_fdr3',0):.1f})")
    try:
        own_delta = float(row.get("own_delta_event", 0) or 0)
    except Exception:
        own_delta = 0.0
    if own_delta < 0:
        reasons.append("ownership falling")
    if row.get("pts_std5", 0) > 4.0:
        reasons.append(f"erratic returns (std {row.get('pts_std5',0):.1f})")
    if not reasons:
        reasons.append("underperforming relative to price")
    opps = row.get("next_opps", "")
    if opps:
        reasons.append(f"next: {opps}")
    return "; ".join(reasons)

def reason_in(row) -> str:
    reasons = []
    if row.get("pred_pts", 0) > 0:
        reasons.append(f"predicted {row.get('pred_pts',0):.1f} pts next GW")
    if row.get("xgi_last5", 0) >= 1.5:
        reasons.append(f"strong underlying (xGI {row.get('xgi_last5',0):.2f} in last 5)")
    if row.get("avg_fdr3", 3.0) <= 3.0:
        reasons.append(f"good fixtures (avg FDR {row.get('avg_fdr3',0):.1f})")
    if row.get("mins_avg5", 0) >= 70 and row.get("starts_pct5", 0) >= 70:
        reasons.append("reliable 70+ mins starter")
    try:
        own = float(row.get("selected_by_percent", 0) or 0)
    except Exception:
        own = 0.0
    if own <= 10:
        reasons.append("differential (≤10% owned)")
    if row.get("ppm", 0) >= 15:
        reasons.append(f"value pick (PPM {row.get('ppm',0):.1f})")
    opps = row.get("next_opps", "")
    if opps:
        reasons.append(f"next: {opps}")
    return "; ".join(reasons) if reasons else "Strong balance of fixtures, role, and metrics"

# ---------------- UI ----------------
st.title("⚽ FPL Team Analyst — Pro Edition")
st.write("Reads your FPL squad (hard-coded to your team) and gives a rating, context, and transfer ideas with advanced metrics, captain markers, and fixture lookahead.")

with st.sidebar:
    st.header("Settings & Auth")
    gw_input = st.number_input("Gameweek (0 = auto)", min_value=0, value=0)
    cookie_input = st.text_area(
        "Optional: Cookie header value (paste 'pl_profile=...; pl_session=...')",
        placeholder="pl_profile=xxx; pl_session=yyy"
    )
    st.markdown("---")
    st.subheader("Suggestion Filters")
    pos_filter = st.multiselect("Positions to target (IN)", ["GKP", "GK", "DEF", "MID", "FWD"],
                                default=["DEF", "MID", "FWD", "GKP"])
    max_price_in = st.number_input("Max price for IN picks (£m)", min_value=0.0, max_value=15.0, value=12.0, step=0.5)
    price_tolerance = st.number_input("Price tolerance vs OUT (£m)", min_value=0.0, max_value=5.0, value=0.5, step=0.5)
    per_position_view = st.checkbox("Show per-position IN buckets", value=True)
    st.markdown("---")
    show_diag = st.checkbox("Show diagnostics", value=False)

with st.expander("ℹ️ Keys & Scales"):
    st.markdown(
        """
**Columns**
- **Pos** – GKP / DEF / MID / FWD  
- **Player** – Player name  
- **Team** – Club  
- **Cap** – **C** (captain), VC (vice), or blank  
- **GW Pts (Cur)** – Points this current gameweek (if any)  
- **GW X / GW Y / GW Z** – Points in each of the last three completed gameweeks  
- **Points** – Total season points  
- **Form** – FPL form metric (recent average)  
- **Price £m** – Current price  
- **PPM** – Points Per £m (value metric)  
- **Selected %** – Ownership  
- **Next 3** – Next three fixtures (abbr + H/A)  
        """
    )

if st.button("Fetch my squad & analyze", type="primary", use_container_width=True):
    entry_id = DEFAULT_ENTRY_ID

    # bootstrap
    try:
        bootstrap = fetch_bootstrap()
    except Exception as e:
        st.error(f"Could not fetch bootstrap-static: {e}")
        st.stop()

    elements = {e["id"]: e for e in bootstrap["elements"]}
    element_types = {et["id"]: et["singular_name_short"] for et in bootstrap["element_types"]}
    teams = {t["id"]: t for t in bootstrap["teams"]}

    # ✅ Ensure we update the module-level TEAM_SHORT
    global TEAM_SHORT
    TEAM_SHORT = {t["id"]: t.get("short_name", t.get("name", "")) for t in bootstrap["teams"]}

    events = bootstrap.get("events", [])

    # determine gw
    if gw_input > 0:
        gw = gw_input
    else:
        gw = next((e["id"] for e in events if e.get("is_current")), None)
        if gw is None:
            gw = next((e["id"] for e in events if e.get("is_next")), None)

    # Determine last three completed GWs for per-week columns and labels
    if gw:
        last_rounds = [gw - 1, gw - 2, gw - 3]
    else:
        finished_ids = [e["id"] for e in events if e.get("finished")]
        last_completed = max(finished_ids) if finished_ids else None
        last_rounds = [last_completed, (last_completed - 1 if last_completed else None), (last_completed - 2 if last_completed else None)]
    last_rounds = [r for r in last_rounds if r and r > 0]
    while len(last_rounds) < 3:
        last_rounds.append(None)
    r1, r2, r3 = last_rounds[:3]
    label_gw1 = f"GW {r1}" if r1 else "Prev GW 1"
    label_gw2 = f"GW {r2}" if r2 else "Prev GW 2"
    label_gw3 = f"GW {r3}" if r3 else "Prev GW 3"

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
        st.info("Upload a CSV with columns: element, is_captain (bool), is_vice_captain (bool) [optional].")
        uploaded = st.file_uploader("Upload CSV with 'element' column", type=["csv"])
        if uploaded is None:
            st.stop()
        try:
            df_csv = pd.read_csv(uploaded)
            if "element" not in df_csv.columns:
                st.error("CSV must include an 'element' column containing FPL player ids.")
                st.stop()
            if "is_captain" not in df_csv.columns:
                df_csv["is_captain"] = False
            if "is_vice_captain" not in df_csv.columns:
                df_csv["is_vice_captain"] = False
            picks_raw = df_csv.to_dict(orient="records")
            resolved_from = "User CSV upload"
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            st.stop()

    picks = normalize_picks(picks_raw)
    if not picks:
        st.error("Could not construct picks list from the data found.")
    if resolved_from is None:
        resolved_from = "unknown"
    st.success(f"Loaded picks from: {resolved_from} ({len(picks)} players)")

    # Build squad dataframe
    rows = []
    pos_map_typeshort = {et["id"]: et["singular_name_short"] for et in bootstrap["element_types"]}

    for p in picks:
        pid = int(p["element"])
        player = elements.get(pid)
        if not player:
            rows.append({
                "id": pid, "web_name": f"Unknown ({pid})", "position": "Unknown",
                "team_id": None, "team_name": "Unknown", "total_points": 0,
                "form": 0.0, "now_cost_m": 0.0, "selected_by_percent": 0.0,
                "is_captain": p.get("is_captain", False), "is_vice_captain": p.get("is_vice_captain", False),
                "position_idx": p.get("position"), "multiplier": p.get("multiplier"),
                "gw_points": 0, "gwm1": 0, "gwm2": 0, "gwm3": 0,
                "own_delta_event": 0,
                "xg_last5":0,"xa_last5":0,"xgi_last5":0,
                "mins_avg5":0,"starts_pct5":0,"pts_std5":0,"avg_fdr3":3.0,
                "next_opps":"", "next3_abbr":""
            })
            continue

        team_id = player.get("team")
        team_name = teams.get(team_id, {}).get("name", "Unknown")
        recent = get_recent_points(pid, [r1, r2, r3], gw)

        adv = compute_player_adv_metrics(pid, n_hist=5, n_fix=3)

        transfers_in_ev = int(player.get("transfers_in_event", 0) or 0)
        transfers_out_ev = int(player.get("transfers_out_event", 0) or 0)
        own_delta_event = transfers_in_ev - transfers_out_ev

        rows.append({
            "id": pid,
            "web_name": player.get("web_name"),
            "first_name": player.get("first_name"),
            "second_name": player.get("second_name"),
            "position": pos_map_typeshort.get(player.get("element_type")),
            "team_id": team_id,
            "team_name": team_name,

            "total_points": int(player.get("total_points", 0)),
            "form": float(player.get("form") or 0.0),
            "now_cost_m": player.get("now_cost", 0) / 10.0,
            "selected_by_percent": float(player.get("selected_by_percent") or 0.0),

            "is_captain": bool(p.get("is_captain", False)),
            "is_vice_captain": bool(p.get("is_vice_captain", False)),
            "position_idx": p.get("position"),
            "multiplier": p.get("multiplier"),

            "gw_points": int(recent.get("gw_points", 0)),
            "gwm1": int(recent.get("gwm1", 0)),
            "gwm2": int(recent.get("gwm2", 0)),
            "gwm3": int(recent.get("gwm3", 0)),

            "own_delta_event": own_delta_event,

            "xg_last5": adv["xg_lastn"],
            "xa_last5": adv["xa_lastn"],
            "xgi_last5": adv["xgi_lastn"],
            "mins_avg5": adv["mins_avg_lastn"],
            "starts_pct5": adv["starts_pct_lastn"],
            "pts_std5": adv["pts_std_lastn"],
            "avg_fdr3": adv["avg_fdr_nextn"],
            "next_opps": adv["next_opps"],
            "next3_abbr": adv["next3_abbr"],
        })

    df = pd.DataFrame(rows)
    if df.empty:
        st.error("No valid players found.")
        st.stop()

    # --------- Position order (ensure GK/GKP first) ----------
    def _pos_order(p: str) -> int:
        pu = (p or "").upper()
        if pu in ("GK", "GKP"):
            return 0
        if pu == "DEF":
            return 1
        if pu == "MID":
            return 2
        if pu == "FWD":
            return 3
        return 99

    df["pos_order"] = df["position"].map(_pos_order)

    # Detect starters
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

    # Value metric
    df["ppm"] = df["total_points"] / (df["now_cost_m"] + 0.01)

    # Captain/Vice display column
    def cap_marker(r):
        if r.get("is_captain"):
            return "C"
        if r.get("is_vice_captain"):
            return "VC"
        return ""
    df["Cap"] = df.apply(cap_marker, axis=1)

    # --------- Display prep (1-dp & centered) ----------
    def _prep_display(dfx):
        out = dfx.copy().sort_values(["pos_order", "web_name"]).rename(columns={
            "position": "Pos",
            "web_name": "Player",
            "team_name": "Team",
            "gw_points": "GW Pts (Cur)",
            "total_points": "Points",
            "form": "Form",
            "now_cost_m": "Price £m",
            "selected_by_percent": "Selected %",
        })
        out = out.rename(columns={"gwm1": label_gw1, "gwm2": label_gw2, "gwm3": label_gw3})
        out["Next 3"] = dfx["next3_abbr"]

        cols = ["Pos", "Player", "Team", "Cap",
                "GW Pts (Cur)", label_gw1, label_gw2, label_gw3,
                "Points", "Form", "Price £m", "PPM", "Selected %", "Next 3"]
        out = out[cols]

        # Styler: formats (1 dp where relevant), centered cells, bold C in Cap
        fmt = {
            "Form": "{:.1f}",
            "Price £m": "{:.1f}",
            "PPM": "{:.1f}",
            "Selected %": "{:.1f}",
            "GW Pts (Cur)": "{:.0f}",
            label_gw1: "{:.0f}",
            label_gw2: "{:.0f}",
            label_gw3: "{:.0f}",
            "Points": "{:.0f}",
        }
        styler = (
            out.style
            .format(fmt)
            .set_properties(**{"text-align": "center"})
            .set_table_styles([dict(selector="th", props=[("text-align", "center")])])
        )

        def bold_c(val):
            return "font-weight: 700" if val == "C" else ""
        styler = styler.applymap(bold_c, subset=pd.IndexSlice[:, ["Cap"]])

        return styler

    starters_view = _prep_display(df[df["is_starter"]])
    subs_view     = _prep_display(df[~df["is_starter"]])

    st.markdown("### Starting XI (GKP → DEF → MID → FWD)")
    st.dataframe(starters_view, use_container_width=True)

    st.markdown("### Substitutes")
    st.dataframe(subs_view, use_container_width=True)

    # -------- Predictions + Team rating --------
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
        zero_mask = df["pred_pts"] == 0.0
        df.loc[zero_mask, "pred_pts"] = [
            find_pred(f"{fn} {sn}") for fn, sn in zip(df.loc[zero_mask, "first_name"].fillna(""), df.loc[zero_mask, "second_name"].fillna(""))
        ]
        avg_pred = float(df["pred_pts"].mean())
    else:
        df["pred_pts"] = 0.0
        avg_pred = 0.0

    score_points = min(avg_points / 150.0, 1.0) * 100.0
    score_form = min(avg_form / 8.0, 1.0) * 100.0
    score_val = min(val_eff / 8.0, 1.0) * 100.0
    score_pred = min(avg_pred / 5.0, 1.0) * 100.0
    WEIGHTS = {"points": 0.35, "form": 0.25, "value": 0.2, "pred": 0.2}
    final_score = round(score_points * WEIGHTS["points"] + score_form * WEIGHTS["form"] + score_val * WEIGHTS["value"] + score_pred * WEIGHTS["pred"], 1)
    st.metric("Team Rating", f"{final_score} / 100")

    # -------- Recommendations --------
    st.markdown("## Recommendations")

    # ----- OUT candidates -----
    st.markdown("### Players to consider transferring OUT (data-driven reasons)")
    def out_score(r):
        s = 0.0
        s += (2.5 - min(r["form"], 2.5)) * 6.0
        s += max(0.0, 7.0 - r["ppm"]) * 1.5
        s += max(0.0, 60 - r["mins_avg5"]) * 0.05
        s += max(0.0, 60 - r["starts_pct5"]) * 0.03
        s += max(0.0, r["avg_fdr3"] - 3.0) * 3.0
        s += max(0, -r["own_delta_event"]) * 0.0005
        s += r["pts_std5"] * 0.4
        if r["now_cost_m"] >= 8.0: s += 1.5
        return float(s)

    out_pool = df[df["is_starter"]].copy()
    out_pool["out_score"] = out_pool.apply(out_score, axis=1)
    out_suggestions = out_pool.sort_values("out_score", ascending=False).head(5)

    for _, r in out_suggestions.iterrows():
        st.write(
            f"- **{r['web_name']}** ({r['team_name']}, {r['position']}) — "
            f"Form {r['form']:.1f} | PPM {r['ppm']:.1f} | £{r['now_cost_m']:.1f}m | "
            f"mins(avg5) {r['mins_avg5']:.0f} | starts% {r['starts_pct5']:.0f} | "
            f"Next FDR {r['avg_fdr3']:.1f} — _{reason_out(r)}_"
            f"\n  **Next 3**: {r.get('next3_abbr','')}"
        )

    # ----- IN candidates -----
    st.markdown("### Players to consider transferring IN (preds + xGI + fixtures + role)")

    all_players = pd.DataFrame(bootstrap["elements"]).copy()
    all_players["now_cost_m"] = all_players["now_cost"] / 10.0
    all_players["ppm"] = all_players["total_points"] / (all_players["now_cost_m"] + 0.01)
    all_players["position"] = all_players["element_type"].map({et["id"]: et["singular_name_short"] for et in bootstrap["element_types"]})
    team_map = {t["id"]: t["name"] for t in bootstrap.get("teams", [])}
    all_players["team_name"] = all_players["team"].map(team_map)

    squad_ids = set(df["id"].tolist())
    candidates = all_players[~all_players["id"].isin(squad_ids)].copy()
    candidates = candidates[candidates["minutes"] > 0]

    candidates["selected_by_percent"] = pd.to_numeric(
        candidates.get("selected_by_percent", 0), errors="coerce"
    ).fillna(0.0)

    # Budget & position filters
    candidates = candidates[candidates["now_cost_m"] <= max_price_in]
    if pos_filter:
        if any(p in ("GK", "GKP") for p in pos_filter):
            pos_filter = list({p if p != "GK" else "GKP" for p in pos_filter})
            candidates.loc[candidates["position"] == "GK", "position"] = "GKP"
        candidates = candidates[candidates["position"].isin(pos_filter)]

    # Attach predicted points if available
    pred_df = pred_df  # alias
    if pred_df is not None:
        pred_lookup = pred_df.set_index(pred_df["name"].str.lower())["pred_pts"].to_dict()
        def get_pred_for(row):
            for v in [row.get("web_name"), f"{row.get('first_name','')} {row.get('second_name','')}", row.get("second_name")]:
                if not v:
                    continue
                val = pred_lookup.get(str(v).lower())
                if val is not None:
                    return float(val)
            return 0.0
        candidates["pred_pts"] = candidates.apply(get_pred_for, axis=1)
    else:
        candidates["pred_pts"] = 0.0

    prefilter = candidates.sort_values(["pred_pts","ppm","total_points"], ascending=False).head(80).copy()

    adv_rows = []
    for _, r in prefilter.iterrows():
        pid = int(r["id"])
        adv = compute_player_adv_metrics(pid, n_hist=5, n_fix=3)
        adv_rows.append({
            "id": pid,
            "xg_last5": adv["xg_lastn"],
            "xa_last5": adv["xa_lastn"],
            "xgi_last5": adv["xgi_lastn"],
            "mins_avg5": adv["mins_avg_lastn"],
            "starts_pct5": adv["starts_pct_lastn"],
            "pts_std5": adv["pts_std_lastn"],
            "avg_fdr3": adv["avg_fdr_nextn"],
            "next_opps": adv["next_opps"],
            "next3_abbr": adv["next3_abbr"],
        })
    adv_df = pd.DataFrame(adv_rows)
    cand = prefilter.merge(adv_df, on="id", how="left").fillna({
        "xg_last5":0,"xa_last5":0,"xgi_last5":0,"mins_avg5":0,"starts_pct5":0,"pts_std5":0,"avg_fdr3":3.0,"next3_abbr":""
    })

    def in_score(r):
        s = 0.0
        s += r["pred_pts"] * 2.0
        s += r["xgi_last5"] * 1.5
        s += max(0.0, (3.5 - r["avg_fdr3"])) * 2.0
        s += max(0.0, (r["mins_avg5"] - 60)) * 0.05
        s += max(0.0, (r["starts_pct5"] - 60)) * 0.03
        s += r["ppm"] * 0.2
        try:
            own = float(r.get("selected_by_percent", 0) or 0)
            if own <= 10: s += 1.2
        except Exception:
            pass
        s -= max(0.0, r["pts_std5"] - 4.0) * 0.3
        return float(s)

    cand["in_score"] = cand.apply(in_score, axis=1)

    if per_position_view:
        for pos in ["GKP", "DEF", "MID", "FWD"]:
            if pos_filter and pos not in pos_filter:
                continue
            st.markdown(f"#### Top {pos} targets")
            subset = cand[cand["position"].replace({"GK":"GKP"}) == pos].sort_values("in_score", ascending=False).head(6)
            if subset.empty:
                st.write("_No suitable targets under filters._")
                continue
            for _, r in subset.iterrows():
                rec = get_recent_points(int(r["id"]), [r1, r2, r3], gw)
                st.write(
                    f"- **{r['web_name']}** ({r['team_name']}, {pos}) — "
                    f"Pred {r['pred_pts']:.1f} | xGI(5) {r['xgi_last5']:.2f} | mins(avg5) {r['mins_avg5']:.0f} | "
                    f"starts% {r['starts_pct5']:.0f} | PPM {r['ppm']:.1f} | £{r['now_cost_m']:.1f}m — "
                    f"_{reason_in(r)}_"
                    f"\n  **Next 3**: {r.get('next3_abbr','')}"
                    f"\n  GW Cur {rec.get('gw_points',0)} — {label_gw1}: {rec.get('gwm1',0)}, {label_gw2}: {rec.get('gwm2',0)}, {label_gw3}: {rec.get('gwm3',0)}"
                )
    else:
        top_in = cand.sort_values("in_score", ascending=False).head(10)
        for _, r in top_in.iterrows():
            rec = get_recent_points(int(r["id"]), [r1, r2, r3], gw)
            st.write(
                f"- **{r['web_name']}** ({r['team_name']}, {r['position']}) — "
                f"Pred {r['pred_pts']:.1f} | xGI(5) {r['xgi_last5']:.2f} | mins(avg5) {r['mins_avg5']:.0f} | "
                f"starts% {r['starts_pct5']:.0f} | PPM {r['ppm']:.1f} | £{r['now_cost_m']:.1f}m — "
                f"_{reason_in(r)}_"
                f"\n  **Next 3**: {r.get('next3_abbr','')}"
                f"\n  GW Cur {rec.get('gw_points',0)} — {label_gw1}: {rec.get('gwm1',0)}, {label_gw2}: {rec.get('gwm2',0)}, {label_gw3}: {rec.get('gwm3',0)}"
            )

    # ----- Replacement matcher -----
    st.markdown("### Like-for-like replacements (by position & price band)")
    if 'pred_pts' not in cand.columns:
        cand['pred_pts'] = 0.0
    for _, outp in out_suggestions.iterrows():
        pos = outp["position"]
        pos_norm = "GKP" if str(pos).upper() in ("GK", "GKP") else str(pos)
        max_price_for_this_out = outp["now_cost_m"] + price_tolerance
        pool = cand[(cand["position"].replace({"GK":"GKP"}) == pos_norm) & (cand["now_cost_m"] <= max_price_for_this_out)]
        if pool.empty:
            st.write(f"- **{outp['web_name']}** → _No suitable {pos_norm} under £{max_price_for_this_out:.1f}m_")
            continue
        best = pool.sort_values("in_score", ascending=False).head(3)
        repls = ", ".join([f"{r['web_name']} (£{r['now_cost_m']:.1f}m, Pred {r['pred_pts']:.1f}, Next 3: {r.get('next3_abbr','')})" for _, r in best.iterrows()])
        st.write(
            f"- **{outp['web_name']}** ({pos_norm}, £{outp['now_cost_m']:.1f}m) → {repls}"
        )

    # ----- Transparency / Advanced Metrics -----
    with st.expander("🔬 Advanced metrics for your Starting XI"):
        adv_cols = ["position","web_name","team_name","form","ppm","now_cost_m",
                    "xg_last5","xa_last5","xgi_last5","mins_avg5","starts_pct5","pts_std5",
                    "avg_fdr3","own_delta_event","next_opps","next3_abbr","Cap"]
        adv = df[df["is_starter"]][adv_cols].sort_values(["position","web_name"]).rename(
            columns={"position":"Pos","web_name":"Player","team_name":"Team","now_cost_m":"Price £m"}
        )
        sty = (
            adv.style
            .format({"form":"{:.1f}","ppm":"{:.1f}","Price £m":"{:.1f}","avg_fdr3":"{:.1f}"})
            .set_properties(**{"text-align":"center"})
            .set_table_styles([dict(selector="th", props=[("text-align","center")])])
        )
        st.dataframe(sty, use_container_width=True)

    # Diagnostics
    if show_diag:
        st.markdown("### Diagnostics")
        st.write("Resolved from:", resolved_from)
        st.write("TEAM_SHORT sample:", dict(list(TEAM_SHORT.items())[:6]))
