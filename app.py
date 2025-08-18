# app.py
# FPL Team Analyst — Dashboard Edition (fixed styler KeyError)
# - Clean dashboard (tabs, cards)
# - GK first, bold C / VC markers
# - 1-dp numeric formatting, centered tables
# - Next 3 fixtures with opponent abbr + per-fixture FDR color
# - Team logos in tables
# - Data-driven IN/OUT reasoning

import streamlit as st
import requests
import pandas as pd
import numpy as np
from typing import Any, Optional, Dict, List

# ---------------- Config ----------------
st.set_page_config(page_title="FPL Team Analyst — Dashboard", layout="wide")

BASE = "https://fantasy.premierleague.com/api"
BOOTSTRAP = f"{BASE}/bootstrap-static/"
ENTRY = f"{BASE}/entry/{{}}/"
PICKS = f"{BASE}/entry/{{}}/event/{{}}/picks/"
MY_TEAM = f"{BASE}/my-team/{{}}/"
PLAYER_SUMMARY = f"{BASE}/element-summary/{{}}/"
PREDICTIONS_URL = "https://www.fantasyfootballpundit.com/fpl-points-predictor/"

DEFAULT_ENTRY_ID = "2792859"  # your team

# Global team metadata (update in-place after bootstrap so caches see it)
TEAM_SHORT: Dict[int, str] = {}
TEAM_LOGO_URL: Dict[int, str] = {}  # code -> logo url

# ---------------- Helpers ----------------
@st.cache_data(show_spinner=False)
def fetch_bootstrap() -> dict:
    r = requests.get(BOOTSTRAP, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return r.json()

def safe_get_json(url: str, headers: dict = None) -> Any:
    try:
        merged_headers = {"User-Agent": "Mozilla/5.0"}
        if headers: merged_headers.update(headers)
        r = requests.get(url, headers=merged_headers, timeout=15)
    except Exception as e:
        return {"__error__": str(e)}
    if r.status_code == 200:
        try:    return r.json()
        except Exception as e:
            return {"__error__": f"Invalid JSON: {e}"}
    return {"__status__": r.status_code, "__text__": r.text[:800]}

def find_picks_recursive(obj):
    if isinstance(obj, dict):
        if "picks" in obj and isinstance(obj["picks"], list): return obj["picks"]
        for v in obj.values():
            found = find_picks_recursive(v)
            if found: return found
    elif isinstance(obj, list) and obj:
        first = obj[0]
        if isinstance(first, dict) and "element" in first: return obj
        for item in obj:
            found = find_picks_recursive(item)
            if found: return found
    return None

def normalize_picks(raw):
    if raw is None: return None
    if isinstance(raw, dict):
        if "picks" in raw and isinstance(raw["picks"], list):
            raw = raw["picks"]
        else:
            found = find_picks_recursive(raw)
            if found: raw = found
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
        if not tables: return None
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
    if team_id is None: return "UNK"
    return TEAM_SHORT.get(int(team_id), "UNK")

def _logo_for_team_id(team_id: Optional[int]) -> str:
    if team_id is None: return ""
    return TEAM_LOGO_URL.get(int(team_id), "")

def derive_next3_and_fdr(fixtures: List[dict], player_team_id: Optional[int], n_fix=3):
    """Return (labels list, fdr list) like ['MCI(H)',...], [4,3,2]"""
    labs, fdrs = [], []
    for f in fixtures[:n_fix]:
        opp_id = f.get("opponent_team")
        is_home = f.get("is_home")
        # Derive from team_h/team_a if missing
        if (opp_id is None or is_home is None) and player_team_id is not None:
            th, ta = f.get("team_h"), f.get("team_a")
            dh, da = f.get("team_h_difficulty"), f.get("team_a_difficulty")
            if th and ta:
                if int(th) == int(player_team_id):
                    opp_id = int(ta); is_home = True; fdr = dh if dh is not None else f.get("difficulty")
                elif int(ta) == int(player_team_id):
                    opp_id = int(th); is_home = False; fdr = da if da is not None else f.get("difficulty")
                else:
                    fdr = f.get("difficulty")
            else:
                fdr = f.get("difficulty")
        else:
            dh, da = f.get("team_h_difficulty"), f.get("team_a_difficulty")
            if is_home is True and dh is not None: fdr = dh
            elif is_home is False and da is not None: fdr = da
            else: fdr = f.get("difficulty")

        side = "H" if is_home else "A"
        ab = _abbr_for_team_id(opp_id)
        labs.append(f"{ab}({side})")
        try: fdrs.append(int(fdr) if fdr is not None else 3)
        except: fdrs.append(3)
    # pad to length 3
    while len(labs) < 3: labs.append("")
    while len(fdrs) < 3: fdrs.append(3)
    return labs, fdrs

def compute_player_adv_metrics(player_id: int, player_team_id: Optional[int]) -> Dict[str, float]:
    data = fetch_element_summary(player_id)
    if not data:
        return {"xg5":0.0,"xa5":0.0,"xgi5":0.0,"mins5":0.0,"starts5":0.0,"std5":0.0,
                "avg_fdr3":3.0,"next1":"","next2":"","next3":"","fdr1":3,"fdr2":3,"fdr3":3}

    hist = data.get("history", []) or []
    fixtures = data.get("fixtures", []) or []

    rounds = sorted(hist, key=lambda r: r.get("round", 0))
    mins = _last_n([int(h.get("minutes", 0)) for h in rounds], 5)
    pts  = _last_n([int(h.get("total_points", 0)) for h in rounds], 5)
    xg   = _last_n([float(h.get("expected_goals", 0.0) or 0.0) for h in rounds], 5)
    xa   = _last_n([float(h.get("expected_assists", 0.0) or 0.0) for h in rounds], 5)

    starts = _last_n([1 if int(h.get("minutes",0)) >= 60 else 0 for h in rounds], 5)
    mins_avg = float(np.mean(mins)) if mins else 0.0
    starts_pct = (sum(starts) / len(starts) * 100.0) if starts else 0.0
    pts_std = float(np.std(pts)) if len(pts) >= 2 else 0.0

    xg_sum = float(np.sum(xg)) if xg else 0.0
    xa_sum = float(np.sum(xa)) if xa else 0.0
    xgi_sum = xg_sum + xa_sum

    labs, fdrs = derive_next3_and_fdr(fixtures, player_team_id, n_fix=3)
    avg_fdr = float(np.mean(fdrs)) if fdrs else 3.0

    return {"xg5":round(xg_sum,2),"xa5":round(xa_sum,2),"xgi5":round(xgi_sum,2),
            "mins5":round(mins_avg,1),"starts5":round(starts_pct,1),"std5":round(pts_std,2),
            "avg_fdr3":round(avg_fdr,2),
            "next1":labs[0],"next2":labs[1],"next3":labs[2],
            "fdr1":fdrs[0],"fdr2":fdrs[1],"fdr3":fdrs[2]}

def get_recent_points(player_id: int, target_rounds: list, current_gw: Optional[int]) -> Dict[str, int]:
    data = fetch_element_summary(player_id)
    pts_map = {}; gw_cur = 0
    if data and "history" in data and isinstance(data["history"], list):
        for row in data["history"]:
            rnd = int(row.get("round", 0))
            pts_map[rnd] = int(row.get("total_points", 0))
        if current_gw: gw_cur = pts_map.get(int(current_gw), 0)
    out = {"gw_points": gw_cur, "gwm1": 0, "gwm2": 0, "gwm3": 0}
    for key, rnd in zip(["gwm1","gwm2","gwm3"], target_rounds):
        out[key] = pts_map.get(int(rnd), 0) if rnd else 0
    return out

def reason_out(row) -> str:
    rs = []
    if row.get("Form",0) < 2.5: rs.append(f"low form {row['Form']:.1f}")
    if row.get("PPM",0) < 10 and row.get("Price £m",0) >= 7.0: rs.append(f"poor value (PPM {row['PPM']:.1f})")
    if row.get("mins5",0) < 60 or row.get("starts5",0) < 60: rs.append(f"rotation risk ({row['mins5']:.0f}m avg)")
    if row.get("avg_fdr3",3.0) >= 4.0: rs.append(f"tough fixtures (FDR {row['avg_fdr3']:.1f})")
    if row.get("std5",0) > 4.0: rs.append(f"erratic (std {row['std5']:.1f})")
    return "; ".join(rs) if rs else "Underperforming relative to price"

def reason_in(row) -> str:
    rs = []
    if row.get("pred_pts",0) > 0: rs.append(f"{row['pred_pts']:.1f} predicted")
    if row.get("xgi5",0) >= 1.5: rs.append(f"xGI(5) {row['xgi5']:.2f}")
    if row.get("avg_fdr3",3.0) <= 3.0: rs.append(f"good fixtures (FDR {row['avg_fdr3']:.1f})")
    if row.get("mins5",0) >= 70 and row.get("starts5",0) >= 70: rs.append("reliable starter")
    try:
        own = float(row.get("selected_by_percent",0) or 0)
        if own <= 10: rs.append("differential")
    except: pass
    if row.get("PPM",0) >= 15: rs.append(f"value (PPM {row['PPM']:.1f})")
    return ", ".join(rs) if rs else "Balanced pick (fixtures, role, price)"

# ---------------- UI ----------------
st.markdown("""
<style>
.block-container { padding-top: 1.2rem; }
div[data-testid="stMetric"] { background: #f7f9fc; border-radius: 14px; padding: 8px 10px; }
div[data-testid="stMetric"] > label { font-weight: 600; }
.dataframe tbody tr th { display: none; }
</style>
""", unsafe_allow_html=True)

st.title("🏟️ FPL Team Analyst — Dashboard")

with st.sidebar:
    st.header("Settings")
    gw_input = st.number_input("Gameweek (0 = auto)", min_value=0, value=0)
    cookie_input = st.text_area("Optional cookie (pl_profile=…; pl_session=…)", placeholder="pl_profile=xxx; pl_session=yyy")
    st.markdown("---")
    st.subheader("IN suggestions filters")
    pos_filter = st.multiselect("Positions", ["GKP","DEF","MID","FWD"], default=["DEF","MID","FWD","GKP"])
    max_price_in = st.number_input("Max price (£m)", 0.0, 15.0, 12.0, 0.5)
    price_tolerance = st.number_input("Price wiggle vs OUT (£m)", 0.0, 5.0, 0.5, 0.5)
    st.markdown("---")
    show_diag = st.checkbox("Show diagnostics", False)

tabs = st.tabs(["🏠 Overview", "🔁 Transfers", "🔬 Advanced"])

with tabs[0]:
    st.write("Fetch your squad and see a clean Starting XI view with next fixtures and value metrics.")

if st.button("Fetch my squad & analyze", type="primary", use_container_width=True):
    # bootstrap
    try:
        bootstrap = fetch_bootstrap()
    except Exception as e:
        st.error(f"Could not fetch bootstrap-static: {e}")
        st.stop()

    elements = {e["id"]: e for e in bootstrap["elements"]}
    teams = {t["id"]: t for t in bootstrap["teams"]}

    # Update global team maps in-place
    TEAM_SHORT.clear(); TEAM_LOGO_URL.clear()
    for t in bootstrap["teams"]:
        TEAM_SHORT[t["id"]] = t.get("short_name", t.get("name",""))
        TEAM_LOGO_URL[t["id"]] = f"https://resources.premierleague.com/premierleague/badges/50/t{t['code']}.png"

    events = bootstrap.get("events", [])
    # determine gw
    if gw_input > 0:
        gw = gw_input
    else:
        gw = next((e["id"] for e in events if e.get("is_current")), None)
        if gw is None:
            gw = next((e["id"] for e in events if e.get("is_next")), None)

    # last three completed GWs for labels
    if gw:
        last_rounds = [gw-1, gw-2, gw-3]
    else:
        finished = [e["id"] for e in events if e.get("finished")]
        last_completed = max(finished) if finished else None
        last_rounds = [last_completed, (last_completed-1 if last_completed else None), (last_completed-2 if last_completed else None)]
    last_rounds = [r for r in last_rounds if r and r > 0]
    while len(last_rounds) < 3: last_rounds.append(None)
    r1, r2, r3 = last_rounds[:3]
    label_gw1 = f"GW {r1}" if r1 else "Prev GW 1"
    label_gw2 = f"GW {r2}" if r2 else "Prev GW 2"
    label_gw3 = f"GW {r3}" if r3 else "Prev GW 3"

    headers = {"Cookie": cookie_input} if (cookie_input and '=' in cookie_input) else None

    resolved_from, picks_raw = None, None
    # 1) public picks
    if gw:
        resp = safe_get_json(PICKS.format(DEFAULT_ENTRY_ID, gw))
        if isinstance(resp, dict) and "picks" in resp:
            picks_raw = resp["picks"]; resolved_from = f"PICKS endpoint (GW {gw})"
    # 2) my-team
    if picks_raw is None and headers:
        resp = safe_get_json(MY_TEAM.format(DEFAULT_ENTRY_ID), headers=headers)
        if isinstance(resp, dict) and "picks" in resp:
            picks_raw = resp["picks"]; resolved_from = "MY_TEAM endpoint (auth)"
    # 3) entry fallback
    if picks_raw is None:
        resp = safe_get_json(ENTRY.format(DEFAULT_ENTRY_ID))
        if isinstance(resp, dict):
            found = find_picks_recursive(resp)
            if found: picks_raw = found; resolved_from = "ENTRY endpoint (recursive)"

    # 4) CSV fallback
    if picks_raw is None:
        with tabs[0]:
            st.warning("Could not auto-detect picks from public or authenticated endpoints.")
            up = st.file_uploader("Upload CSV with 'element' (+ optional is_captain, is_vice_captain)", type=["csv"])
            if up is None: st.stop()
            df_csv = pd.read_csv(up)
            if "element" not in df_csv.columns: st.error("CSV must include 'element' column."); st.stop()
            if "is_captain" not in df_csv.columns: df_csv["is_captain"] = False
            if "is_vice_captain" not in df_csv.columns: df_csv["is_vice_captain"] = False
            picks_raw = df_csv.to_dict(orient="records"); resolved_from = "User CSV upload"

    picks = normalize_picks(picks_raw)
    if not picks: st.error("Could not construct picks list."); st.stop()

    # Build squad dataframe
    pos_map_typeshort = {et["id"]: et["singular_name_short"] for et in bootstrap["element_types"]}

    rows = []
    for p in picks:
        pid = int(p["element"])
        pl = elements.get(pid)
        if not pl: continue
        team_id = pl.get("team"); team_name = teams.get(team_id, {}).get("name","")
        adv = compute_player_adv_metrics(pid, player_team_id=team_id)
        recent = get_recent_points(pid, [r1, r2, r3], gw)

        rows.append({
            "id": pid,
            "Player": pl.get("web_name"),
            "First": pl.get("first_name"),
            "Second": pl.get("second_name"),
            "Pos": pos_map_typeshort.get(pl.get("element_type")),
            "Team": team_name,
            "TeamID": team_id,
            "Logo": _logo_for_team_id(team_id),

            "GW Pts (Cur)": int(recent["gw_points"]),
            label_gw1: int(recent["gwm1"]),
            label_gw2: int(recent["gwm2"]),
            label_gw3: int(recent["gwm3"]),

            "Points": int(pl.get("total_points",0)),
            "Form": float(pl.get("form") or 0.0),
            "Price £m": pl.get("now_cost",0)/10.0,
            "Selected %": float(pl.get("selected_by_percent") or 0.0),

            "is_captain": bool(p.get("is_captain", False)),
            "is_vice_captain": bool(p.get("is_vice_captain", False)),
            "multiplier": p.get("multiplier"),
            "position_idx": p.get("position"),

            # adv
            "xg5": adv["xg5"], "xa5": adv["xa5"], "xgi5": adv["xgi5"],
            "mins5": adv["mins5"], "starts5": adv["starts5"], "std5": adv["std5"],
            "avg_fdr3": adv["avg_fdr3"],
            "Next1": adv["next1"], "Next2": adv["next2"], "Next3": adv["next3"],
            "FDR1": adv["fdr1"], "FDR2": adv["fdr2"], "FDR3": adv["fdr3"],
        })

    df = pd.DataFrame(rows)
    if df.empty: st.error("No valid players found."); st.stop()

    # Position ordering: GKP first
    def _pos_order(p: str) -> int:
        u = (p or "").upper()
        return {"GKP":0,"GK":0,"DEF":1,"MID":2,"FWD":3}.get(u, 99)
    df["pos_order"] = df["Pos"].map(_pos_order)

    # Detect starters
    def _is_starter(r):
        m = r.get("multiplier"); idx = r.get("position_idx")
        try:
            if m is not None and int(m) > 0: return True
            if idx is not None and int(idx) <= 11: return True
        except: pass
        return False
    df["is_starter"] = df.apply(_is_starter, axis=1)

    # Value metric & captain marker
    df["PPM"] = (df["Points"] / (df["Price £m"] + 0.01)).astype(float)
    def cap_marker(r):
        if r["is_captain"]: return "🅒"
        if r["is_vice_captain"]: return "VC"
        return ""
    df["Cap"] = df.apply(cap_marker, axis=1)

    # ======= OVERVIEW TAB =======
    with tabs[0]:
        st.success(f"Loaded picks from: {resolved_from or 'unknown'} ({len(df)} players)")

        # KPIs
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Team Rating (quick)", f"{min(df['Form'].mean()/8*100,100):.1f} / 100")
        c2.metric("Avg Form", f"{df['Form'].mean():.1f}")
        c3.metric("Avg PPM", f"{df['PPM'].mean():.1f}")
        c4.metric("Team Value (£m)", f"{df['Price £m'].sum():.1f}")

        # Prepare starters/bench
        starters = df[df["is_starter"]].copy().sort_values(["pos_order","Player"])
        bench    = df[~df["is_starter"]].copy().sort_values(["pos_order","Player"])

        base_cols = ["Logo","Pos","Player","Team","Cap",
                     "GW Pts (Cur)", label_gw1, label_gw2, label_gw3,
                     "Points","Form","Price £m","PPM","Selected %",
                     "Next1","Next2","Next3"]
        starters = starters[base_cols + ["FDR1","FDR2","FDR3"]]
        bench    = bench[base_cols + ["FDR1","FDR2","FDR3"]]

        # Format mapping
        fmt = {"Form":"{:.1f}","Price £m":"{:.1f}","PPM":"{:.1f}","Selected %":"{:.1f}",
               "GW Pts (Cur)":"{:.0f}", label_gw1:"{:.0f}", label_gw2:"{:.0f}", label_gw3:"{:.0f}",
               "Points":"{:.0f}"}

        # ---- FIXED: Style function that reads FDRs from the *full* table while styling the visible view
        def render_table(tbl: pd.DataFrame):
            view = tbl.drop(columns=["FDR1","FDR2","FDR3"])
            # capture FDRs aligned to the view's index
            fdr_vals = tbl.loc[view.index, ["FDR1","FDR2","FDR3"]].copy()

            def _color_for(d):
                d = int(d)
                if d <= 2: return "background-color:#d7f5d7;"   # green
                if d == 3: return "background-color:#fff6bf;"   # yellow
                if d == 4: return "background-color:#ffd8b2;"   # orange
                return "background-color:#ffb3b3;"               # red

            def _style_next(df_view: pd.DataFrame):
                styles = pd.DataFrame("", index=df_view.index, columns=df_view.columns)
                for i in df_view.index:
                    d1, d2, d3 = fdr_vals.loc[i, ["FDR1","FDR2","FDR3"]]
                    styles.loc[i, "Next1"] = _color_for(d1)
                    styles.loc[i, "Next2"] = _color_for(d2)
                    styles.loc[i, "Next3"] = _color_for(d3)
                return styles

            sty = (view.style
                .format(fmt)
                .set_properties(**{"text-align":"center"})
                .set_table_styles([dict(selector="th", props=[("text-align","center")])])
                .apply(_style_next, axis=None)  # <-- apply on the visible view only
            )
            def bold_cap(val): return "font-weight:700" if val == "🅒" else ""
            sty = sty.applymap(bold_cap, subset=pd.IndexSlice[:, ["Cap"]])
            return sty

        st.markdown("### Starting XI")
        st.dataframe(
            render_table(starters),
            use_container_width=True,
            column_config={"Logo": st.column_config.ImageColumn(width="small")}
        )

        st.markdown("### Bench")
        st.dataframe(
            render_table(bench),
            use_container_width=True,
            column_config={"Logo": st.column_config.ImageColumn(width="small")}
        )

    # ======= PREDICTIONS (for Transfers) =======
    pred_df = fetch_predictions()
    df["pred_pts"] = 0.0
    if pred_df is not None:
        def find_pred(name, fn, sn):
            m = pred_df[pred_df["name"].str.contains(str(name), case=False, na=False)]
            if not m.empty: return float(m["pred_pts"].iloc[0])
            last = str(name).split()[-1]
            m2 = pred_df[pred_df["name"].str.contains(last, case=False, na=False)]
            if not m2.empty: return float(m2["pred_pts"].iloc[0])
            m3 = pred_df[pred_df["name"].str.contains(f"{fn} {sn}", case=False, na=False)]
            if not m3.empty: return float(m3["pred_pts"].iloc[0])
            return 0.0
        df["pred_pts"] = [find_pred(elements[i]["web_name"], elements[i]["first_name"], elements[i]["second_name"]) for i in df["id"]]

    # ======= TRANSFERS TAB =======
    with tabs[1]:
        st.subheader("Suggested OUT")
        def out_score(r):
            s = 0.0
            s += (2.5 - min(r["Form"], 2.5)) * 6.0
            s += max(0.0, 7.0 - r["PPM"]) * 1.5
            s += max(0.0, 60 - r["mins5"]) * 0.05
            s += max(0.0, 60 - r["starts5"]) * 0.03
            s += max(0.0, r["avg_fdr3"] - 3.0) * 3.0
            s += r["std5"] * 0.4
            if r["Price £m"] >= 8.0: s += 1.5
            return float(s)

        outs = df[df["is_starter"]].copy()
        outs["out_score"] = outs.apply(out_score, axis=1)
        outs = outs.sort_values("out_score", ascending=False).head(5)

        for _, r in outs.iterrows():
            with st.container(border=True):
                cc1, cc2 = st.columns([0.12, 0.88])
                with cc1:
                    st.image(r["Logo"])
                    st.caption(r["Team"])
                with cc2:
                    st.markdown(f"**{r['Player']}** — {r['Pos']}  |  £{r['Price £m']:.1f}m  |  Form **{r['Form']:.1f}**  |  PPM **{r['PPM']:.1f}**")
                    st.caption(f"Next: {r['Next1']}  •  {r['Next2']}  •  {r['Next3']}")
                    st.write(f"_Reasoning:_ {reason_out(r)}")

        st.subheader("Suggested IN")
        all_players = pd.DataFrame(bootstrap["elements"]).copy()
        all_players["now_cost_m"] = all_players["now_cost"] / 10.0
        all_players["ppm"] = all_players["total_points"] / (all_players["now_cost_m"] + 0.01)
        all_players["position"] = all_players["element_type"].map({et["id"]: et["singular_name_short"] for et in bootstrap["element_types"]})
        team_map = {t["id"]: t["name"] for t in bootstrap["teams"]}
        all_players["team_name"] = all_players["team"].map(team_map)

        squad_ids = set(df["id"].tolist())
        candidates = all_players[(~all_players["id"].isin(squad_ids)) & (all_players["minutes"] > 0)].copy()
        candidates["selected_by_percent"] = pd.to_numeric(candidates.get("selected_by_percent",0), errors="coerce").fillna(0.0)
        candidates = candidates[candidates["now_cost_m"] <= max_price_in]
        if pos_filter:
            candidates = candidates[candidates["position"].replace({"GK":"GKP"}).isin(pos_filter)]

        if pred_df is not None:
            pred_lookup = pred_df.set_index(pred_df["name"].str.lower())["pred_pts"].to_dict()
            def pred_for(r):
                for v in [r.get("web_name"), f"{r.get('first_name','')} {r.get('second_name','')}", r.get("second_name")]:
                    if not v: continue
                    val = pred_lookup.get(str(v).lower())
                    if val is not None: return float(val)
                return 0.0
            candidates["pred_pts"] = candidates.apply(pred_for, axis=1)
        else:
            candidates["pred_pts"] = 0.0

        sample = candidates.sort_values(["pred_pts","ppm","total_points"], ascending=False).head(100).copy()
        adv_rows = []
        for _, rr in sample.iterrows():
            adv = compute_player_adv_metrics(int(rr["id"]), player_team_id=int(rr["team"]))
            adv_rows.append({"id": int(rr["id"]), **adv})
        cand = sample.merge(pd.DataFrame(adv_rows), on="id", how="left").fillna({"xgi5":0,"mins5":0,"starts5":0,"std5":0,"avg_fdr3":3.0})

        def in_score(r):
            s = 0.0
            s += r["pred_pts"] * 2.0
            s += r["xgi5"] * 1.5
            s += max(0.0, (3.5 - r["avg_fdr3"])) * 2.0
            s += max(0.0, (r["mins5"] - 60)) * 0.05
            s += max(0.0, (r["starts5"] - 60)) * 0.03
            s += r["ppm"] * 0.2
            s -= max(0.0, r["std5"] - 4.0) * 0.3
            return float(s)

        cand["in_score"] = cand.apply(in_score, axis=1)
        top_in = cand.sort_values("in_score", ascending=False).head(12)

        cols = st.columns(3)
        for i, (_, r) in enumerate(top_in.iterrows()):
            with cols[i % 3].container(border=True):
                st.markdown(f"**{r['web_name']}** — {r['position']}  |  **£{r['now_cost_m']:.1f}m**")
                st.caption(f"{team_map.get(int(r['team']), '')}")
                st.write(f"Pred **{r['pred_pts']:.1f}** | xGI(5) **{r['xgi5']:.2f}** | PPM **{r['ppm']:.1f}**")
                st.caption(f"Next: {r.get('next1','')}  •  {r.get('next2','')}  •  {r.get('next3','')}")
                st.write(f"_Why:_ {reason_in(r)}")

        st.markdown("##### Like-for-like ideas")
        for _, outp in outs.iterrows():
            pos = outp["Pos"]; max_price_for_out = outp["Price £m"] + price_tolerance
            pool = top_in[(top_in["position"].replace({"GK":"GKP"}) == pos) & (top_in["now_cost_m"] <= max_price_for_out)]
            names = ", ".join([f"{r['web_name']} (£{r['now_cost_m']:.1f}m, Pred {r['pred_pts']:.1f})" for _, r in pool.head(3).iterrows()])
            st.write(f"- **{outp['Player']}** → {names or '_no suitable matches_'}")

    # ======= ADVANCED TAB =======
    with tabs[2]:
        st.subheader("Starter metrics")
        adv_cols = ["Logo","Pos","Player","Team","Cap","Points","Form","Price £m","PPM",
                    "xg5","xa5","xgi5","mins5","starts5","std5","avg_fdr3",
                    "Next1","Next2","Next3"]
        starters_adv = df[df["is_starter"]].copy().sort_values(["pos_order","Player"])[adv_cols]
        fmt_adv = {"Form":"{:.1f}","Price £m":"{:.1f}","PPM":"{:.1f}","xg5":"{:.2f}","xa5":"{:.2f}","xgi5":"{:.2f}",
                   "mins5":"{:.0f}","starts5":"{:.0f}","std5":"{:.1f}","avg_fdr3":"{:.1f}","Points":"{:.0f}"}
        sty = (starters_adv.style
               .format(fmt_adv)
               .set_properties(**{"text-align":"center"})
               .set_table_styles([dict(selector="th", props=[("text-align","center")])]))
        st.dataframe(sty, use_container_width=True,
                     column_config={"Logo": st.column_config.ImageColumn(width="small")})

    # Diagnostics
    if show_diag:
        with st.expander("Diagnostics"):
            st.write("TEAM_SHORT sample:", dict(list(TEAM_SHORT.items())[:6]))
