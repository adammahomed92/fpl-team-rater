import streamlit as st
import requests
import pandas as pd
from typing import Dict, List, Tuple, Optional

st.set_page_config(page_title="FPL Team Rater + FDR Transfers", layout="wide")

# -------------------- Constants --------------------
FPL_BASE = "https://fantasy.premierleague.com/api"
BOOTSTRAP = f"{FPL_BASE}/bootstrap-static/"
FIXTURES = f"{FPL_BASE}/fixtures/"
MY_TEAM = f"{FPL_BASE}/my-team/{{team_id}}/"
LOGIN = "https://users.premierleague.com/accounts/login/"

POS_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

def fdr_color(val: float) -> str:
    if pd.isna(val): return ""
    v = float(val)
    if v <= 2: return "background-color: lightgreen; font-weight: 600; text-align:center;"
    if v == 3:  return "background-color: lightgrey; font-weight: 600; text-align:center;"
    return "background-color: lightcoral; font-weight: 600; text-align:center;"

# -------------------- Data fetchers --------------------
@st.cache_data(show_spinner=False)
def load_bootstrap() -> Dict:
    r = requests.get(BOOTSTRAP, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return r.json()

@st.cache_data(show_spinner=False)
def load_fixtures() -> List[Dict]:
    r = requests.get(FIXTURES, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return r.json()

def fpl_login(email: str, password: str) -> Optional[requests.Session]:
    """Login and return an authenticated session (cookies persisted)."""
    sess = requests.Session()
    sess.headers.update({
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*",
        "Origin": "https://fantasy.premierleague.com",
        "Referer": "https://fantasy.premierleague.com/",
    })
    payload = {
        "login": email,
        "password": password,
        "app": "plfpl-web",
        "redirect_uri": "https://fantasy.premierleague.com",
        "rememberMe": "true",
    }
    resp = sess.post(LOGIN, data=payload, allow_redirects=True, timeout=20)
    if resp.status_code == 200:
        return sess
    return None

def get_my_team(sess: requests.Session, team_id: str, cookie_header: Optional[str] = None) -> Dict:
    """Fetch /my-team/ with either session cookies or an explicit Cookie header fallback."""
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://fantasy.premierleague.com/my-team",
        "Origin": "https://fantasy.premierleague.com",
    }
    if cookie_header:
        headers["Cookie"] = cookie_header
        # use a plain requests call with explicit cookie header
        r = requests.get(MY_TEAM.format(team_id=team_id), headers=headers, timeout=20)
    else:
        r = sess.get(MY_TEAM.format(team_id=team_id), headers=headers, timeout=20)
    r.raise_for_status()
    return r.json()

# -------------------- Helpers --------------------
def detect_current_or_next_gw(events: List[Dict]) -> Tuple[int, List[int]]:
    current = next((e["id"] for e in events if e.get("is_current")), None)
    nxt = next((e["id"] for e in events if e.get("is_next")), None)
    anchor = current or nxt or 1
    horizon = [g for g in [anchor, anchor + 1, anchor + 2] if 1 <= g <= 38]
    return anchor, horizon

def team_upcoming_fixture_rows(team_id: int, fixtures: List[Dict], gws: List[int]) -> List[Tuple[int, str, int]]:
    out = []
    by_event = {}
    for f in fixtures:
        ev = f.get("event")
        if ev is None: continue
        by_event.setdefault(ev, []).append(f)
    for gw in gws:
        gw_fixts = by_event.get(gw, [])
        found = None
        for f in gw_fixts:
            if f["team_h"] == team_id: found = ("H", f["team_h_difficulty"]); break
            if f["team_a"] == team_id: found = ("A", f["team_a_difficulty"]); break
        out.append((gw, found[0], found[1]) if found else (gw, "-", float("nan")))
    return out

def calc_player_fdr_summary(player_row: pd.Series, fixtures: List[Dict], gws: List[int]) -> Tuple[float, List[float], List[str]]:
    team_id = int(player_row["team"])
    runs = team_upcoming_fixture_rows(team_id, fixtures, gws)
    fdrs = [r[2] for r in runs]
    labels = [f"GW{r[0]} {r[1]}" for r in runs]
    nums = [v for v in fdrs if pd.notna(v)]
    avg = sum(nums) / len(nums) if nums else 3.0
    return avg, fdrs, labels

def build_team_df(my_team_json: Dict, elements_df: pd.DataFrame, fixtures: List[Dict], gws: List[int]) -> pd.DataFrame:
    picks = pd.DataFrame(my_team_json["picks"])
    df = picks.merge(elements_df, left_on="element", right_on="id", how="left")
    df["position"] = df["element_type"].map(POS_MAP)
    df["cost"] = df["now_cost"] / 10.0
    df["form"] = pd.to_numeric(df["form"], errors="coerce").fillna(0.0)
    df["ppg"] = pd.to_numeric(df["points_per_game"], errors="coerce").fillna(0.0)
    fdr_data = df.apply(lambda r: calc_player_fdr_summary(r, fixtures, gws), axis=1, result_type="expand")
    df["fdr_avg_next3"] = fdr_data[0]
    df["fdr1"] = fdr_data[1].apply(lambda x: x[0] if len(x) > 0 else float("nan"))
    df["fdr2"] = fdr_data[1].apply(lambda x: x[1] if len(x) > 1 else float("nan"))
    df["fdr3"] = fdr_data[1].apply(lambda x: x[2] if len(x) > 2 else float("nan"))
    df["gw1"] = fdr_data[2].apply(lambda x: x[0] if len(x) > 0 else "")
    df["gw2"] = fdr_data[2].apply(lambda x: x[1] if len(x) > 1 else "")
    df["gw3"] = fdr_data[2].apply(lambda x: x[2] if len(x) > 2 else "")
    return df

def team_rating(df: pd.DataFrame) -> float:
    avg_form = df["form"].mean()
    avg_ppg = df["ppg"].mean()
    avg_inv_fdr = (5 - df["fdr_avg_next3"]).mean()
    s_form = min(avg_form / 6.0, 1.0) * 100
    s_ppg = min(avg_ppg / 6.0, 1.0) * 100
    s_fdr = min(avg_inv_fdr / 4.0, 1.0) * 100
    return round(s_form * 0.4 + s_ppg * 0.35 + s_fdr * 0.25, 1)

def suggest_swaps(team_df, elements_df, fixtures, gws, budget_cap, money_itb) -> pd.DataFrame:
    current_ids = set(team_df["id"].tolist())
    current_total_cost = float(team_df["cost"].sum())
    elements = elements_df.copy()
    elements["position"] = elements["element_type"].map(POS_MAP)
    elements["cost"] = elements["now_cost"] / 10.0
    elements["form"] = pd.to_numeric(elements["form"], errors="coerce").fillna(0.0)
    elements["ppg"] = pd.to_numeric(elements["points_per_game"], errors="coerce").fillna(0.0)

    fdr_parts = elements.apply(lambda r: calc_player_fdr_summary(r, fixtures, gws), axis=1, result_type="expand")
    elements["fdr_avg_next3"] = fdr_parts[0]
    elements["cand_fdr1"] = fdr_parts[1].apply(lambda x: x[0] if len(x) > 0 else float("nan"))
    elements["cand_fdr2"] = fdr_parts[1].apply(lambda x: x[1] if len(x) > 1 else float("nan"))
    elements["cand_fdr3"] = fdr_parts[1].apply(lambda x: x[2] if len(x) > 2 else float("nan"))
    elements["impact"] = elements["form"] / elements["fdr_avg_next3"].replace(0, 0.01)

    suggestions = []
    for _, outp in team_df.iterrows():
        pos = outp["position"]
        out_cost = float(outp["cost"])
        out_impact = float(outp["form"] / (outp["fdr_avg_next3"] if outp["fdr_avg_next3"] else 3.0))
        max_in_cost = (budget_cap + money_itb) - (current_total_cost - out_cost)

        pool = elements[
            (elements["position"] == pos) &
            (~elements["id"].isin(current_ids)) &
            (elements["cost"] <= max_in_cost)
        ].sort_values("impact", ascending=False).head(12)

        for _, inc in pool.iterrows():
            gain = float(inc["impact"] - out_impact)
            if gain <= 0.0:
                continue
            suggestions.append({
                "Out": f"{outp['web_name']} ({pos})",
                "Out £m": round(out_cost, 1),
                "In": f"{inc['web_name']} ({pos})",
                "In £m": round(float(inc["cost"]), 1),
                "Impact Gain": round(gain, 3),
                "In Form": round(float(inc["form"]), 2),
                "In PPG": round(float(inc["ppg"]), 2),
                "FDR1": inc["cand_fdr1"],
                "FDR2": inc["cand_fdr2"],
                "FDR3": inc["cand_fdr3"],
            })

    return pd.DataFrame(suggestions).sort_values(["Impact Gain", "In Form"], ascending=[False, False]).reset_index(drop=True)

# -------------------- UI --------------------
st.title("⚽️ FPL Team Rater + Colour-coded FDR Transfers")

with st.sidebar:
    st.header("Sign in to FPL")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    # Hard-coded default Team ID; you can still override if needed
    team_id = st.text_input("Team ID", value="2792859")
    budget_cap = st.number_input("Budget cap (£m)", min_value=80.0, max_value=120.0, value=100.0, step=0.5)
    cookie_fallback = st.text_area("Optional Cookie header (fallback)", placeholder="pl_profile=...; pl_session=...")
    show_diag = st.checkbox("Show diagnostics", value=False)

if st.button("Analyze my team"):
    if not (email and password and team_id.isdigit()):
        st.error("Please enter email, password, and a numeric Team ID.")
        st.stop()

    sess = fpl_login(email, password)
    if not sess:
        st.error("Login failed — check your credentials.")
        st.stop()

    try:
        bootstrap = load_bootstrap()
        fixtures = load_fixtures()
    except Exception as e:
        st.error(f"Failed to load FPL data: {e}")
        st.stop()

    elements_df = pd.DataFrame(bootstrap["elements"])
    events = bootstrap.get("events", [])
    anchor_gw, next3 = detect_current_or_next_gw(events)

    # --- Fetch my-team with robust fallbacks ---
    try:
        my = get_my_team(sess, team_id)
    except requests.HTTPError as e:
        # 403/401 → try cookie fallback if provided
        if cookie_fallback.strip():
            try:
                my = get_my_team(sess, team_id, cookie_header=cookie_fallback.strip())
                st.info("Used cookie fallback to access /my-team/.")
            except Exception as e2:
                st.error(f"Could not fetch /my-team/ with cookie fallback: {e2}")
                st.stop()
        else:
            st.error(f"Could not fetch /my-team/: {e}. If you use Apple/Google SSO, paste the Cookie header above (from DevTools → Network → any /api/ request).")
            st.stop()
    except Exception as e:
        st.error(f"Could not fetch /my-team/: {e}")
        st.stop()

    # Build my team df
    team_df = build_team_df(my, elements_df, fixtures, next3)

    st.subheader("📋 Your Squad (with next 3 GWs)")
    cols_to_show = ["web_name", "position", "cost", "form", "ppg", "gw1", "fdr1", "gw2", "fdr2", "gw3", "fdr3"]
    disp = team_df[cols_to_show].rename(columns={
        "web_name": "Player", "ppg": "PPG",
        "gw1": "Next 1", "gw2": "Next 2", "gw3": "Next 3",
        "fdr1": "FDR 1", "fdr2": "FDR 2", "fdr3": "FDR 3",
    })
    styler = disp.style.applymap(fdr_color, subset=["FDR 1", "FDR 2", "FDR 3"])
    st.dataframe(styler, use_container_width=True)

    rating = team_rating(team_df)
    st.metric("Team Rating", f"{rating} / 100")

    money_itb = my.get("transfers", {}).get("bank", 0) / 10.0
    st.write(f"💰 In the bank (ITB): **£{money_itb:.1f}m** • Budget cap used for suggestions: **£{budget_cap:.1f}m**")

    st.subheader("🔄 Suggested Transfers (OUT → IN)")
    swaps = suggest_swaps(team_df, elements_df, fixtures, next3, budget_cap, money_itb)
    if swaps.empty:
        st.success("No clear high-impact upgrades within the budget.")
    else:
        swap_styler = swaps.style.applymap(fdr_color, subset=["FDR1", "FDR2", "FDR3"])
        st.dataframe(swap_styler, use_container_width=True)

    if show_diag:
        st.divider()
        st.caption("Diagnostics")
        st.write("Next GWs:", next3)
        st.write("Session cookies:", list(sess.cookies.get_dict().keys()))
