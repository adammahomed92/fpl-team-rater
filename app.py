import streamlit as st
import requests
import pandas as pd
from typing import Dict, List, Tuple, Optional

st.set_page_config(page_title="FPL Team Rater + Transfers (FDR)", layout="wide")

# -------------------- Constants --------------------
FPL_BASE = "https://fantasy.premierleague.com/api"
BOOTSTRAP = f"{FPL_BASE}/bootstrap-static/"
FIXTURES = f"{FPL_BASE}/fixtures/"
MY_TEAM = f"{FPL_BASE}/my-team/{{team_id}}/"
LOGIN = "https://users.premierleague.com/accounts/login/"

POS_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

# Color thresholds for FDR (official-ish convention)
def fdr_color(val: float) -> str:
    if pd.isna(val):
        return ""
    v = float(val)
    if v <= 2:   # Easy
        return "background-color: lightgreen; font-weight: 600; text-align:center;"
    if v == 3:  # Neutral
        return "background-color: lightgrey; font-weight: 600; text-align:center;"
    # Hard (4-5)
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
    sess = requests.Session()
    payload = {
        "login": email,
        "password": password,
        "app": "plfpl-web",
        "redirect_uri": "https://fantasy.premierleague.com",
    }
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = sess.post(LOGIN, data=payload, headers=headers, allow_redirects=True, timeout=20)
    if resp.status_code == 200:
        return sess
    return None

def get_my_team(sess: requests.Session, team_id: str) -> Dict:
    r = sess.get(MY_TEAM.format(team_id=team_id), headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    r.raise_for_status()
    return r.json()

# -------------------- Helpers --------------------
def detect_current_or_next_gw(events: List[Dict]) -> Tuple[int, List[int]]:
    """
    Returns (anchor_gw, [anchor, anchor+1, anchor+2]) where anchor is current if available,
    otherwise next. Filters to 1..38 range.
    """
    current = next((e["id"] for e in events if e.get("is_current")), None)
    nxt = next((e["id"] for e in events if e.get("is_next")), None)
    anchor = current or nxt or 1
    horizon = [g for g in [anchor, anchor + 1, anchor + 2] if 1 <= g <= 38]
    return anchor, horizon

def team_upcoming_fixture_rows(team_id: int, fixtures: List[Dict], gws: List[int]) -> List[Tuple[int, str, int]]:
    """
    For a given team id, return list of (gw, H/A label, fdr) for supplied gw list.
    If fixture not scheduled for a gw, returns (gw, '-', NaN).
    """
    out = []
    by_event = {}
    for f in fixtures:
        ev = f.get("event")
        if ev is None:
            continue
        by_event.setdefault(ev, []).append(f)

    for gw in gws:
        gw_fixts = by_event.get(gw, [])
        found = None
        for f in gw_fixts:
            if f["team_h"] == team_id:
                found = ("H", f["team_h_difficulty"])
                break
            if f["team_a"] == team_id:
                found = ("A", f["team_a_difficulty"])
                break
        if found:
            out.append((gw, found[0], found[1]))
        else:
            out.append((gw, "-", float("nan")))
    return out

def calc_player_fdr_summary(player_row: pd.Series, fixtures: List[Dict], gws: List[int]) -> Tuple[float, List[float], List[str]]:
    """
    Returns (avg_fdr_next3, [fdr1,fdr2,fdr3], [label1,label2,label3]) for player's team.
    """
    team_id = int(player_row["team"])
    runs = team_upcoming_fixture_rows(team_id, fixtures, gws)
    fdrs = [r[2] for r in runs]
    labels = [f"GW{r[0]} {r[1]}" for r in runs]
    # avg over available numbers
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

    # FDR for next 3
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
    """
    Composite 0–100 score using form, PPG, and fixture ease.
    """
    avg_form = df["form"].mean()
    avg_ppg = df["ppg"].mean()
    avg_inv_fdr = (5 - df["fdr_avg_next3"]).mean()

    # Normalize (heuristics)
    s_form = min(avg_form / 6.0, 1.0) * 100
    s_ppg = min(avg_ppg / 6.0, 1.0) * 100
    s_fdr = min(avg_inv_fdr / 4.0, 1.0) * 100

    return round(s_form * 0.4 + s_ppg * 0.35 + s_fdr * 0.25, 1)

def suggest_swaps(
    team_df: pd.DataFrame,
    elements_df: pd.DataFrame,
    fixtures: List[Dict],
    gws: List[int],
    budget_cap: float,
    money_itb: float,
) -> pd.DataFrame:
    """
    OUT -> IN suggestions per position, ranked by impact gain.
    Impact uses a simple metric: form / fdr_avg_next3.
    Respects the overall £100m cap using ITB + selling the OUT player.
    """
    current_ids = set(team_df["id"].tolist())
    current_total_cost = float(team_df["cost"].sum())
    elements = elements_df.copy()
    elements["position"] = elements["element_type"].map(POS_MAP)
    elements["cost"] = elements["now_cost"] / 10.0
    elements["form"] = pd.to_numeric(elements["form"], errors="coerce").fillna(0.0)
    elements["ppg"] = pd.to_numeric(elements["points_per_game"], errors="coerce").fillna(0.0)

    # Compute FDRs for candidates
    fdr_parts = elements.apply(lambda r: calc_player_fdr_summary(r, fixtures, gws), axis=1, result_type="expand")
    elements["fdr_avg_next3"] = fdr_parts[0]
    elements["cand_fdr1"] = fdr_parts[1].apply(lambda x: x[0] if len(x) > 0 else float("nan"))
    elements["cand_fdr2"] = fdr_parts[1].apply(lambda x: x[1] if len(x) > 1 else float("nan"))
    elements["cand_fdr3"] = fdr_parts[1].apply(lambda x: x[2] if len(x) > 2 else float("nan"))
    elements["impact"] = elements["form"] / elements["fdr_avg_next3"].replace(0, 0.01)

    # Build OUT->IN
    suggestions = []
    for _, outp in team_df.iterrows():
        pos = outp["position"]
        out_cost = float(outp["cost"])
        out_impact = float(outp["form"] / (outp["fdr_avg_next3"] if outp["fdr_avg_next3"] else 3.0))

        # Budget math: new_total = current_total - out_cost + in_cost
        # require new_total <= budget_cap + money_itb
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

    if not suggestions:
        return pd.DataFrame()

    df = pd.DataFrame(suggestions).sort_values(["Impact Gain", "In Form"], ascending=[False, False]).reset_index(drop=True)
    return df

# -------------------- UI --------------------
st.title("⚽️ FPL Team Rater + Colour-coded FDR Transfers")

with st.sidebar:
    st.header("Sign in to FPL")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    team_id = st.text_input("Team ID (from your FPL URL)")
    budget_cap = st.number_input("Budget cap (£m)", min_value=80.0, max_value=120.0, value=100.0, step=0.5)
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
    teams_df = pd.DataFrame(bootstrap["teams"])
    events = bootstrap.get("events", [])
    anchor_gw, next3 = detect_current_or_next_gw(events)

    try:
        my = get_my_team(sess, team_id)
    except Exception as e:
        st.error(f"Could not fetch /my-team/: {e}")
        st.stop()

    # Build my team df
    team_df = build_team_df(my, elements_df, fixtures, next3)

    # Top panel: team + rating
    st.subheader("📋 Your Squad (with next 3 GWs)")
    show_cols = ["web_name", "position", "cost", "form", "ppg", "gw1", "fdr1", "gw2", "fdr2", "gw3", "fdr3"]
    disp = team_df[show_cols].rename(columns={
        "web_name": "Player",
        "ppg": "PPG",
        "gw1": "Next 1",
        "gw2": "Next 2",
        "gw3": "Next 3",
        "fdr1": "FDR 1",
        "fdr2": "FDR 2",
        "fdr3": "FDR 3",
    })

    # Style FDR cells
    styler = disp.style.applymap(fdr_color, subset=["FDR 1", "FDR 2", "FDR 3"])
    st.dataframe(styler, use_container_width=True)

    rating = team_rating(team_df)
    st.metric("Team Rating", f"{rating} / 100")

    # Money in the bank (ITB)
    money_itb = my.get("transfers", {}).get("bank", 0) / 10.0
    st.write(f"💰 In the bank (ITB): **£{money_itb:.1f}m** • Budget cap used for suggestions: **£{budget_cap:.1f}m**")

    # Suggestions
    st.subheader("🔄 Suggested Transfers (OUT → IN)")
    swaps = suggest_swaps(team_df, elements_df, fixtures, next3, budget_cap, money_itb)
    if swaps.empty:
        st.success("No clear high-impact upgrades within the budget.")
    else:
        # Colour the incoming player's FDR columns
        swap_disp = swaps.copy()
        swap_styler = swap_disp.style.applymap(fdr_color, subset=["FDR1", "FDR2", "FDR3"])
        st.dataframe(swap_styler, use_container_width=True)

    if show_diag:
        st.divider()
        st.caption("Diagnostics")
        st.write("Anchor GW:", anchor_gw, "— Next GWs:", next3)
        st.write("Elements cols:", list(elements_df.columns)[:10], "… total:", elements_df.shape)
        st.write("My-team keys:", list(my.keys()))
