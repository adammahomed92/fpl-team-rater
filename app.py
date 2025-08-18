import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="FPL Team Analyst", layout="wide")

BASE = "https://fantasy.premierleague.com/api"
BOOTSTRAP = f"{BASE}/bootstrap-static/"
FIXTURES = f"{BASE}/fixtures/"

# ---------------- Helpers ----------------
@st.cache_data(show_spinner=False)
def fetch_bootstrap() -> dict:
    r = requests.get(BOOTSTRAP, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return r.json()

@st.cache_data(show_spinner=False)
def fetch_fixtures() -> list:
    r = requests.get(FIXTURES, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return r.json()

# Load bootstrap and fixtures
bootstrap = fetch_bootstrap()
fixtures = fetch_fixtures()

# Map for team id → short name
TEAM_SHORT = {t["id"]: t["short_name"] for t in bootstrap["teams"]}

# Example dataframe (replace with your team logic)
df = pd.DataFrame([
    {"Logo":"https://resources.premierleague.com/premierleague/photos/players/110x140/p109745.png",
     "Pos":"GKP","Player":"Alisson","Team":"LIV","Cap":"🅒",
     "Points":120,"Form":6.2,"Price £m":5.5,"PPM":21.8,
     "xg5":0.0,"xa5":0.1,"xgi5":0.1,"mins5":450,"starts5":5,"std5":0.2,
     "avg_fdr6":3.0,
     "Next1":"MCI(A)","Next2":"EVE(H)","Next3":"AVL(H)","Next4":"NEW(A)","Next5":"CHE(H)","Next6":"ARS(A)",
     "FDR1":5,"FDR2":2,"FDR3":3,"FDR4":4,"FDR5":4,"FDR6":5},
    {"Logo":"https://resources.premierleague.com/premierleague/photos/players/110x140/p103955.png",
     "Pos":"DEF","Player":"Trent","Team":"LIV","Cap":"",
     "Points":140,"Form":7.1,"Price £m":7.5,"PPM":18.6,
     "xg5":0.1,"xa5":0.4,"xgi5":0.5,"mins5":440,"starts5":5,"std5":0.4,
     "avg_fdr6":3.2,
     "Next1":"MCI(A)","Next2":"EVE(H)","Next3":"AVL(H)","Next4":"NEW(A)","Next5":"CHE(H)","Next6":"ARS(A)",
     "FDR1":5,"FDR2":2,"FDR3":3,"FDR4":4,"FDR5":4,"FDR6":5}
])

# Add starter flag for demo
df["is_starter"] = True
df["pos_order"] = df["Pos"].map({"GKP":0,"DEF":1,"MID":2,"FWD":3})

# ---------------- Layout ----------------
st.title("⚽ FPL Team Analyst Dashboard")

tabs = st.tabs(["Overview", "Transfers", "Advanced"])

# ======= OVERVIEW TAB =======
with tabs[0]:
    st.subheader("Overview")
    st.write("Summary of your team will go here...")

# ======= TRANSFERS TAB =======
with tabs[1]:
    st.subheader("Suggested Transfers")
    st.write("In/Out recommendations will go here...")

# ======= ADVANCED TAB =======
with tabs[2]:
    st.subheader("Starter metrics")

    # chip-style legend
    st.markdown("""
    <div style='display:flex; gap:10px; margin-bottom:10px;'>
      <span style='background-color:#d7f5d7; padding:3px 8px; border-radius:4px;'>FDR 1–2 Easy</span>
      <span style='background-color:#fff6bf; padding:3px 8px; border-radius:4px;'>FDR 3 Neutral</span>
      <span style='background-color:#ffd8b2; padding:3px 8px; border-radius:4px;'>FDR 4 Tough</span>
      <span style='background-color:#ffb3b3; padding:3px 8px; border-radius:4px;'>FDR 5 Very Tough</span>
    </div>
    """, unsafe_allow_html=True)

    adv_cols_visible = [
        "Logo","Pos","Player","Team","Cap","Points","Form","Price £m","PPM",
        "xg5","xa5","xgi5","mins5","starts5","std5","avg_fdr6",
        "Next1","Next2","Next3","Next4","Next5","Next6"
    ]
    adv_cols_fdr = ["FDR1","FDR2","FDR3","FDR4","FDR5","FDR6"]
    starters_adv_full = (
        df[df["is_starter"]]
        .copy()
        .sort_values(["pos_order","Player"])[adv_cols_visible + adv_cols_fdr]
    )

    view = starters_adv_full.drop(columns=adv_cols_fdr)
    fdr_vals = starters_adv_full.loc[view.index, adv_cols_fdr].copy()

    fmt_adv = {
        "Form":"{:.1f}","Price £m":"{:.1f}","PPM":"{:.1f}",
        "xg5":"{:.2f}","xa5":"{:.2f}","xgi5":"{:.2f}",
        "mins5":"{:.0f}","starts5":"{:.0f}","std5":"{:.1f}",
        "avg_fdr6":"{:.1f}","Points":"{:.0f}"
    }

    next_cols = ["Next1","Next2","Next3","Next4","Next5","Next6"]

    def _color_for(d):
        d = int(d)
        if d <= 2: return "background-color:#d7f5d7;"   # green
        if d == 3: return "background-color:#fff6bf;"   # yellow
        if d == 4: return "background-color:#ffd8b2;"   # orange
        return "background-color:#ffb3b3;"              # red

    def _style_next(df_view: pd.DataFrame):
        styles = pd.DataFrame("", index=df_view.index, columns=df_view.columns)
        for i in df_view.index:
            for j, nc in enumerate(next_cols, start=1):
                styles.loc[i, nc] = _color_for(fdr_vals.loc[i, f"FDR{j}"])
        return styles

    def _bold_cap(val):
        return "font-weight:700" if val == "🅒" else ""

    sty = (
        view.style
            .format(fmt_adv)
            .set_properties(**{"text-align":"center"})
            .set_table_styles([dict(selector="th", props=[("text-align","center")])])
            .apply(_style_next, axis=None)
            .applymap(_bold_cap, subset=pd.IndexSlice[:, ["Cap"]])
    )

    st.dataframe(
        sty,
        use_container_width=True,
        column_config={"Logo": st.column_config.ImageColumn(width="small")}
    )
