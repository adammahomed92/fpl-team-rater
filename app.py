# ======= ADVANCED TAB =======
with tabs[2]:
    st.subheader("Starter metrics")

    # Include FDR1..FDR6 so we can color Next1..Next6 using them
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

    # View shown to users (no FDR columns)
    view = starters_adv_full.drop(columns=adv_cols_fdr)
    # Keep FDRs aligned by index for styling
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
        if d <= 2: return "background-color:#d7f5d7;"   # green (easy)
        if d == 3: return "background-color:#fff6bf;"   # yellow (neutral)
        if d == 4: return "background-color:#ffd8b2;"   # orange (tough)
        return "background-color:#ffb3b3;"              # red (very tough)

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
            .apply(_style_next, axis=None)  # color-code Next1..Next6 by FDRs
            .applymap(_bold_cap, subset=pd.IndexSlice[:, ["Cap"]])  # bold captain marker
    )

    st.dataframe(
        sty,
        use_container_width=True,
        column_config={"Logo": st.column_config.ImageColumn(width="small")}
    )
