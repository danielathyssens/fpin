from pathlib import Path
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
CSV_FILE = "batch_run_outputs/experiment_master_results_all_v2.csv"
OUTDIR = Path("batch_run_outputs/latex_tables")
OUTDIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------
MAIN_RESULTS_SETTINGS = [
    ("uniform", 20, 4),
    ("uniform", 50, 7),
    ("uniform", 100, 10),
]

CROSS_SIZE_MULTI_M_TIGHT = [
    ("uniform", 20, 3),
    ("uniform", 50, 6),
    ("uniform", 100, 9),
]

CROSS_SIZE_MULTI_M_LOOSE = [
    ("uniform", 20, 4),
    ("uniform", 50, 7),
    ("uniform", 100, 10),
]

VRP60_CASE_SETTINGS = [
    ("uniform", 60, 6),
    ("uniform", 60, 7),
    ("uniform", 60, 9),
]

EXPLOSION_CASE_SETTINGS = [
    ("explosion", 60, 7),
    ("explosion", 60, 9),
]

REF_METHOD_LABELS = {"HGS", "LKH"}

LEARNED_METHOD_LABELS = {
    "AM (greedy)",
    "AM (sampling)",
    "BQ",
    "POMO, single traj.",
    "POMO",
    "PARCO",
    "PIM",
    "F-PIN (GT+Split)",
    "F-PIN (assign.)",
}

DISPLAY_ORDER = [
    "HGS",
    "LKH",
    "AM (greedy)",
    "AM (sampling)",
    "BQ",
    "POMO, single traj.",
    "POMO",
    "PARCO",
    "PIM",
    "F-PIN (GT+Split)",
    "F-PIN (assign.)",
]

# per-table row specs
MAIN_METHOD_SPECS = [
    ("HGS", "solver", "no"),
    ("LKH", "solver", "no"),
    ("AM", "greedy", "no"),
    ("AM", "sample", "no"),
    ("BQ", "greedy", "no"),
    ("POMO", "greedy", "no"),
    ("POMO", "pomo20", "no"),
    ("PARCO", "greedy", "no"),
    ("F-PIN", "giant-tour+split", "yes"),
    ("F-PIN", "vehicle-assignment", "yes"),
]

VRP60_METHOD_SPECS = [
    ("HGS", "solver", "no"),
    ("LKH", "solver", "no"),
    ("BQ", "greedy", "no"),
    ("POMO", "pomo20", "no"),
    ("AM", "greedy", "no"),
    ("PARCO", "greedy", "no"),
    ("F-PIN", "giant-tour+split", "yes"),
    ("F-PIN", "vehicle-assignment", "yes"),
]

EXPLOSION_METHOD_SPECS = [
    ("HGS", "solver", "no"),
    ("LKH", "solver", "no"),
    ("F-PIN", "giant-tour+split", "yes"),
    ("F-PIN", "vehicle-assignment", "yes"),
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def sanitize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.loc[:, ~df.columns.duplicated()]

    if "status" in df.columns:
        df = df[df["status"].fillna("").astype(str).str.strip().eq("done")].copy()

    for col in ["N", "M"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    numeric_cols = [
        "cost", "cost_v", "num_vehicles", "viol_pct",
        "delta_k", "feas_pct", "runtime"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    string_cols = ["split", "dist", "method", "decode", "+post", "checkpoint", "notes", "status"]
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()

    return df


def fmt(x, digits=2, dash="--"):
    if pd.isna(x):
        return dash
    return f"{float(x):.{digits}f}"


def fmt_intish(x, digits=2, dash="--"):
    if pd.isna(x):
        return dash
    xf = float(x)
    if abs(xf - round(xf)) < 1e-9:
        return str(int(round(xf)))
    return f"{xf:.{digits}f}"


def latex_bold(s: str) -> str:
    return rf"\textbf{{{s}}}"


def display_label(label: str) -> str:
    if label == "HGS":
        return "HGS (ref.)"
    if label == "LKH":
        return "LKH (ref.)"
    return label


def pretty_method_label(method: str, decode: str, post: str) -> str:
    method = str(method).strip()
    decode = str(decode).strip()
    post = str(post).strip()

    if method == "F-PIN":
        if decode == "vehicle-assignment":
            return "F-PIN (assign.)"
        if decode == "giant-tour+split":
            return "F-PIN (GT+Split)"
        return f"F-PIN ({decode})"

    if method == "AM":
        if decode == "sample":
            return "AM (sampling)"
        return "AM (greedy)"

    if method == "POMO":
        if decode == "pomo20":
            return "POMO"
        return "POMO, single traj."

    if method == "BQ":
        return "BQ"
    if method == "PARCO":
        return "PARCO"
    if method == "LKH":
        return "LKH"
    if method == "HGS":
        return "HGS"

    return method


def choose_row(sub: pd.DataFrame, method: str, decode: str, post: str):
    """
    Choose exactly one representative row for a given
    (method, decode, +post) inside an already filtered setting subset.

    Selection rule:
      1. exact match on method/decode/+post
      2. sort by cost_v ascending
      3. then cost ascending
      4. then runtime ascending if available
    """
    cand = sub[
        (sub["method"] == method) &
        (sub["decode"] == decode) &
        (sub["+post"] == post)
    ].copy()

    if cand.empty:
        return None

    sort_cols = []
    if "cost_v" in cand.columns:
        sort_cols.append("cost_v")
    if "cost" in cand.columns:
        sort_cols.append("cost")
    if "runtime" in cand.columns:
        sort_cols.append("runtime")

    cand = cand.sort_values(sort_cols, ascending=True, na_position="last")
    return cand.iloc[0].copy()


def collect_table_rows(
    df: pd.DataFrame,
    settings,
    method_specs,
) -> dict:
    """
    Returns:
      selected[(dist, N, M)] = DataFrame with <= 1 row per requested spec
      and a guaranteed method_label column.
    """
    selected = {}

    for dist, n, m in settings:
        sub = df[
            (df["dist"] == dist) &
            (df["N"] == n) &
            (df["M"] == m)
        ].copy()

        chosen_rows = []
        seen_labels = set()

        for method, decode, post in method_specs:
            row = choose_row(sub, method, decode, post)
            if row is None:
                continue

            label = pretty_method_label(row["method"], row["decode"], row["+post"])

            # defensive guard: avoid duplicate labels inside one block
            if label in seen_labels:
                continue

            row = row.copy()
            row["method_label"] = label
            chosen_rows.append(row)
            seen_labels.add(label)

        if chosen_rows:
            selected[(dist, n, m)] = pd.DataFrame(chosen_rows)
        else:
            selected[(dist, n, m)] = pd.DataFrame(columns=list(df.columns) + ["method_label"])

    return selected


def best_learned_per_setting(selected: dict, metric: str, smaller_is_better: bool = True) -> dict:
    out = {}
    for key, sub in selected.items():
        if sub.empty or metric not in sub.columns:
            out[key] = None
            continue

        learned = sub[sub["method_label"].isin(LEARNED_METHOD_LABELS)].copy()
        learned = learned[pd.notna(learned[metric])]

        if learned.empty:
            out[key] = None
        else:
            out[key] = learned[metric].min() if smaller_is_better else learned[metric].max()

    return out


def metric_is_best(value, best_value, atol=1e-12) -> bool:
    if best_value is None or pd.isna(value):
        return False
    return abs(float(value) - float(best_value)) <= atol


def write_tex(filename: str, lines):
    out = OUTDIR / filename
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


# ---------------------------------------------------------------------
# Table 1: Main results across sizes
# ---------------------------------------------------------------------
def make_main_results_table(df: pd.DataFrame):
    settings = MAIN_RESULTS_SETTINGS
    selected = collect_table_rows(df, settings, MAIN_METHOD_SPECS)

    best_cost_v = best_learned_per_setting(selected, "cost_v", True)
    best_viol = best_learned_per_setting(selected, "viol_pct", True)

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{3.5pt}")
    lines.append(r"\caption{Main results across problem sizes. HGS and LKH are classical non-learned reference solvers shown for orientation; boldface highlights the best learned method in each metric column.}")
    lines.append(r"\label{tab:fcvrp_multi}")
    lines.append(r"\begin{tabular}{lcc cc cc}")
    lines.append(r"\toprule")
    lines.append(r"& \multicolumn{2}{c}{VRP20 ($M=4$)} & \multicolumn{2}{c}{VRP50 ($M=7$)} & \multicolumn{2}{c}{VRP100 ($M=10$)} \\")
    lines.append(r"\cmidrule(l){2-3}\cmidrule(l){4-5}\cmidrule(l){6-7}")
    lines.append(r"Method & Cost$_v$ $\downarrow$ & Viol.\% $\downarrow$ & Cost$_v$ $\downarrow$ & Viol.\% $\downarrow$ & Cost$_v$ $\downarrow$ & Viol.\% $\downarrow$ \\")
    lines.append(r"\midrule")

    for label in DISPLAY_ORDER:
        if label == "AM (greedy)":
            lines.append(r"\midrule")
        if label == "F-PIN (GT+Split)":
            lines.append(r"\midrule")

        row_cells = [display_label(label)]

        for key in settings:
            sub = selected[key]
            r = sub[sub["method_label"] == label] if not sub.empty else pd.DataFrame()

            if r.empty:
                row_cells.extend(["--", "--"])
                continue

            r = r.iloc[0]
            cv = fmt(r["cost_v"], 2)
            vp = fmt_intish(r["viol_pct"], 2)

            if label in LEARNED_METHOD_LABELS:
                if metric_is_best(r["cost_v"], best_cost_v[key]):
                    cv = latex_bold(cv)
                if metric_is_best(r["viol_pct"], best_viol[key]):
                    vp = latex_bold(vp)

            row_cells.extend([cv, vp])

        lines.append(" & ".join(row_cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    write_tex("main_results_cross_size.tex", lines)


# ---------------------------------------------------------------------
# Table 1b: Cross-size multi-M table
# ---------------------------------------------------------------------
def make_cross_size_multi_m_table(df: pd.DataFrame):
    tight_settings = CROSS_SIZE_MULTI_M_TIGHT
    loose_settings = CROSS_SIZE_MULTI_M_LOOSE

    selected_tight = collect_table_rows(df, tight_settings, MAIN_METHOD_SPECS)
    selected_loose = collect_table_rows(df, loose_settings, MAIN_METHOD_SPECS)

    best_cost_v_tight = best_learned_per_setting(selected_tight, "cost_v", True)
    best_viol_tight = best_learned_per_setting(selected_tight, "viol_pct", True)

    best_cost_v_loose = best_learned_per_setting(selected_loose, "cost_v", True)
    best_viol_loose = best_learned_per_setting(selected_loose, "viol_pct", True)

    def render_block(lines, settings, selected, best_cost_v, best_viol):
        lines.append(r"\midrule")
        lines.append(
            rf"\multicolumn{{1}}{{c}}{{}} & "
            rf"\multicolumn{{2}}{{c}}{{VRP20 ($M={settings[0][2]}$)}} & "
            rf"\multicolumn{{2}}{{c}}{{VRP50 ($M={settings[1][2]}$)}} & "
            rf"\multicolumn{{2}}{{c}}{{VRP100 ($M={settings[2][2]}$)}} \\"
        )
        lines.append(r"\cmidrule(l){2-3}\cmidrule(l){4-5}\cmidrule(l){6-7}")

        for label in DISPLAY_ORDER:
            if label == "AM (greedy)":
                lines.append(r"\midrule")
            if label == "F-PIN (GT+Split)":
                lines.append(r"\midrule")

            row_cells = [display_label(label)]

            for key in settings:
                sub = selected[key]
                r = sub[sub["method_label"] == label] if not sub.empty else pd.DataFrame()

                if r.empty:
                    row_cells.extend(["--", "--"])
                    continue

                r = r.iloc[0]
                cv = fmt(r["cost_v"], 2)
                vp = fmt_intish(r["viol_pct"], 2)

                if label in LEARNED_METHOD_LABELS:
                    if metric_is_best(r["cost_v"], best_cost_v[key]):
                        cv = latex_bold(cv)
                    if metric_is_best(r["viol_pct"], best_viol[key]):
                        vp = latex_bold(vp)

                row_cells.extend([cv, vp])

            lines.append(" & ".join(row_cells) + r" \\")

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{3.5pt}")
    lines.append(r"\caption{Main results across tighter and looser fleet settings. HGS and LKH are classical non-learned references; boldface highlights the best learned method within each setting.}")
    lines.append(r"\label{tab:fcvrp_multi_m}")
    lines.append(r"\begin{tabular}{lcccccc}")
    lines.append(r"\toprule")
    lines.append(r"Method & Cost$_v$ & Viol.\% & Cost$_v$ & Viol.\% & Cost$_v$ & Viol.\% \\")
    render_block(lines, tight_settings, selected_tight, best_cost_v_tight, best_viol_tight)
    render_block(lines, loose_settings, selected_loose, best_cost_v_loose, best_viol_loose)
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    write_tex("main_results_cross_size_multi_m.tex", lines)


# ---------------------------------------------------------------------
# Table 2: VRP60 case study
# ---------------------------------------------------------------------
def make_vrp60_case_study_table(df: pd.DataFrame):
    settings = VRP60_CASE_SETTINGS
    selected = collect_table_rows(df, settings, VRP60_METHOD_SPECS)

    best_cost_v = best_learned_per_setting(selected, "cost_v", True)
    best_num_vehicles = best_learned_per_setting(selected, "num_vehicles", True)
    best_viol = best_learned_per_setting(selected, "viol_pct", True)

    ordered_labels = [
        "HGS",
        "LKH",
        "BQ",
        "POMO",
        "AM (greedy)",
        "PARCO",
        "F-PIN (GT+Split)",
        "F-PIN (assign.)",
    ]

    lines = []
    lines.append(r"\begin{table}[ht!]")
    lines.append(r"\centering")
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{1.5pt}")
    lines.append(r"\caption{Case study on uniform VRP60 across fleet limits. HGS and LKH are classical reference solvers; boldface highlights the best learned method in each metric column.}")
    lines.append(r"\label{tab:vrp60_case_study}")
    lines.append(r"\begin{tabular}{lccc ccc ccc}")
    lines.append(r"\toprule")
    lines.append(r"& \multicolumn{3}{c}{VRP60 ($M=6$)} & \multicolumn{3}{c}{VRP60 ($M=7$)} & \multicolumn{3}{c}{VRP60 ($M=9$)} \\")
    lines.append(r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}")
    lines.append(r"Method & Cost$_v$ $\downarrow$ & \#Veh $\downarrow$ & Viol.\% $\downarrow$ & Cost$_v$ $\downarrow$ & \#Veh $\downarrow$ & Viol.\% $\downarrow$ & Cost$_v$ $\downarrow$ & \#Veh $\downarrow$ & Viol.\% $\downarrow$ \\")
    lines.append(r"\midrule")

    for label in ordered_labels:
        if label == "BQ":
            lines.append(r"\midrule")
        if label == "F-PIN (GT+Split)":
            lines.append(r"\midrule")

        row_cells = [display_label(label)]

        for key in settings:
            sub = selected[key]
            r = sub[sub["method_label"] == label] if not sub.empty else pd.DataFrame()

            if r.empty:
                row_cells.extend(["--", "--", "--"])
                continue

            r = r.iloc[0]
            cv = fmt(r["cost_v"], 2)
            nv = fmt(r["num_vehicles"], 2)
            vp = fmt_intish(r["viol_pct"], 2)

            if label in LEARNED_METHOD_LABELS:
                if metric_is_best(r["cost_v"], best_cost_v[key]):
                    cv = latex_bold(cv)
                if metric_is_best(r["num_vehicles"], best_num_vehicles[key]):
                    nv = latex_bold(nv)
                if metric_is_best(r["viol_pct"], best_viol[key]):
                    vp = latex_bold(vp)

            row_cells.extend([cv, nv, vp])

        lines.append(" & ".join(row_cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    write_tex("vrp60_case_study.tex", lines)


# ---------------------------------------------------------------------
# Table 3: Explosion case study
# ---------------------------------------------------------------------
def make_explosion_case_study_table(df: pd.DataFrame):
    settings = EXPLOSION_CASE_SETTINGS
    selected = collect_table_rows(df, settings, EXPLOSION_METHOD_SPECS)

    learned_labels_local = {"F-PIN (GT+Split)", "F-PIN (assign.)"}
    best_cost_v = {}
    best_num_vehicles = {}
    best_viol = {}

    for key, sub in selected.items():
        learned = sub[sub["method_label"].isin(learned_labels_local)] if not sub.empty else pd.DataFrame()
        best_cost_v[key] = learned["cost_v"].min() if not learned.empty and learned["cost_v"].notna().any() else None
        best_num_vehicles[key] = learned["num_vehicles"].min() if not learned.empty and learned["num_vehicles"].notna().any() else None
        best_viol[key] = learned["viol_pct"].min() if not learned.empty and learned["viol_pct"].notna().any() else None

    ordered_labels = ["HGS", "LKH", "F-PIN (GT+Split)", "F-PIN (assign.)"]

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{3.2pt}")
    lines.append(r"\caption{Evaluation on the Explosion distribution for FC-CVRP60. HGS and LKH are classical references; boldface highlights the better F-PIN decoding variant in each setting.}")
    lines.append(r"\label{tab:explosion_case_study}")
    lines.append(r"\begin{tabular}{lccc ccc}")
    lines.append(r"\toprule")
    lines.append(r"& \multicolumn{3}{c}{VRP60 ($M=7$)} & \multicolumn{3}{c}{VRP60 ($M=9$)} \\")
    lines.append(r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}")
    lines.append(r"Method & Cost$_v$ $\downarrow$ & \#Veh $\downarrow$ & Viol.\% $\downarrow$ & Cost$_v$ $\downarrow$ & \#Veh $\downarrow$ & Viol.\% $\downarrow$ \\")
    lines.append(r"\midrule")

    for label in ordered_labels:
        row_cells = [display_label(label)]

        for key in settings:
            sub = selected[key]
            r = sub[sub["method_label"] == label] if not sub.empty else pd.DataFrame()

            if r.empty:
                row_cells.extend(["--", "--", "--"])
                continue

            r = r.iloc[0]
            cv = fmt(r["cost_v"], 2)
            nv = fmt(r["num_vehicles"], 2)
            vp = fmt_intish(r["viol_pct"], 2)

            if label in learned_labels_local:
                if metric_is_best(r["cost_v"], best_cost_v[key]):
                    cv = latex_bold(cv)
                if metric_is_best(r["num_vehicles"], best_num_vehicles[key]):
                    nv = latex_bold(nv)
                if metric_is_best(r["viol_pct"], best_viol[key]):
                    vp = latex_bold(vp)

            row_cells.extend([cv, nv, vp])

        lines.append(" & ".join(row_cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    write_tex("explosion_case_study.tex", lines)

# ---------------------------------------------------------------------
# Table 3: F-PIN ablation (construction vs post-processing)
# ---------------------------------------------------------------------
def make_fpin_post_ablation_table(
    df: pd.DataFrame,
    *,
    decoder: str = "giant-tour+split",
    decoder_caption: str | None = None,
    out_name: str | None = None,
    label: str | None = None,
):
    settings = [
        ("uniform", 20, 3),
        ("uniform", 20, 4),
        ("uniform", 50, 6),
        ("uniform", 50, 7),
        ("uniform", 100, 9),
        ("uniform", 100, 10),
    ]

    if decoder_caption is None:
        if decoder == "giant-tour+split":
            decoder_caption = "Giant-tour+split"
        elif decoder == "vehicle-assignment":
            decoder_caption = "Vehicle-assignment"
        else:
            decoder_caption = decoder

    if out_name is None:
        decoder_slug = decoder.replace("+", "_").replace("-", "_")
        out_name = f"fpin_post_ablation_{decoder_slug}.tex"

    if label is None:
        decoder_slug = decoder.replace("+", "_").replace("-", "_")
        label = f"tab:fpin_post_ablation_{decoder_slug}"

    rows = []
    for dist, n, m in settings:
        sub = df[
            (df["dist"] == dist)
            & (df["N"] == n)
            & (df["M"] == m)
            & (df["method"] == "F-PIN")
            & (df["decode"] == decoder)
        ].copy()

        # choose strongest representative row if duplicates exist
        row_no = sub[sub["+post"] == "no"].sort_values(
            ["cost_v", "cost", "runtime"], na_position="last"
        )
        row_yes = sub[sub["+post"] == "yes"].sort_values(
            ["cost_v", "cost", "runtime"], na_position="last"
        )

        r_no = row_no.iloc[0] if not row_no.empty else None
        r_yes = row_yes.iloc[0] if not row_yes.empty else None

        rows.append((n, m, r_no, r_yes))

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{3.5pt}")
    lines.append(
        rf"\caption{{Ablation of post-processing for \textsc{{F-PIN ({decoder_caption})}}. "
        rf"We compare the raw decoded solution against the post-processed variant "
        rf"across fleet-constrained settings.}}"
    )
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\begin{tabular}{cc lcccc}")
    lines.append(r"\toprule")
    lines.append(
        r"$N$ & $M$ & Variant & Cost$_v$ $\downarrow$ & \#Veh $\downarrow$ & Viol.\% $\downarrow$ & Runtime $\downarrow$ \\"
    )
    lines.append(r"\midrule")

    for n, m, r_no, r_yes in rows:
        vals_cost_v = [r["cost_v"] for r in [r_no, r_yes] if r is not None and pd.notna(r["cost_v"])]
        vals_nv = [r["num_vehicles"] for r in [r_no, r_yes] if r is not None and pd.notna(r["num_vehicles"])]
        vals_viol = [r["viol_pct"] for r in [r_no, r_yes] if r is not None and pd.notna(r["viol_pct"])]
        vals_rt = [r["runtime"] for r in [r_no, r_yes] if r is not None and pd.notna(r["runtime"])]

        best_cost_v = min(vals_cost_v) if vals_cost_v else None
        best_nv = min(vals_nv) if vals_nv else None
        best_viol = min(vals_viol) if vals_viol else None
        best_rt = min(vals_rt) if vals_rt else None

        # no-post row
        if r_no is not None:
            cv = fmt(r_no["cost_v"], 2)
            nv = fmt(r_no["num_vehicles"], 2)
            vp = fmt_intish(r_no["viol_pct"], 2)
            rt = fmt(r_no["runtime"], 2)

            if best_cost_v is not None and pd.notna(r_no["cost_v"]) and abs(float(r_no["cost_v"]) - float(best_cost_v)) < 1e-12:
                cv = latex_bold(cv)
            if best_nv is not None and pd.notna(r_no["num_vehicles"]) and abs(float(r_no["num_vehicles"]) - float(best_nv)) < 1e-12:
                nv = latex_bold(nv)
            if best_viol is not None and pd.notna(r_no["viol_pct"]) and abs(float(r_no["viol_pct"]) - float(best_viol)) < 1e-12:
                vp = latex_bold(vp)
            if best_rt is not None and pd.notna(r_no["runtime"]) and abs(float(r_no["runtime"]) - float(best_rt)) < 1e-12:
                rt = latex_bold(rt)

            lines.append(
                rf"{n} & {m} & \textsc{{F-PIN ({decoder_caption})}} & {cv} & {nv} & {vp} & {rt} \\"
            )
        else:
            lines.append(
                rf"{n} & {m} & \textsc{{F-PIN ({decoder_caption})}} & -- & -- & -- & -- \\"
            )

        # +post row
        if r_yes is not None:
            cv = fmt(r_yes["cost_v"], 2)
            nv = fmt(r_yes["num_vehicles"], 2)
            vp = fmt_intish(r_yes["viol_pct"], 2)
            rt = fmt(r_yes["runtime"], 2)

            if best_cost_v is not None and pd.notna(r_yes["cost_v"]) and abs(float(r_yes["cost_v"]) - float(best_cost_v)) < 1e-12:
                cv = latex_bold(cv)
            if best_nv is not None and pd.notna(r_yes["num_vehicles"]) and abs(float(r_yes["num_vehicles"]) - float(best_nv)) < 1e-12:
                nv = latex_bold(nv)
            if best_viol is not None and pd.notna(r_yes["viol_pct"]) and abs(float(r_yes["viol_pct"]) - float(best_viol)) < 1e-12:
                vp = latex_bold(vp)
            if best_rt is not None and pd.notna(r_yes["runtime"]) and abs(float(r_yes["runtime"]) - float(best_rt)) < 1e-12:
                rt = latex_bold(rt)

            lines.append(
                rf" &  & \textsc{{F-PIN ({decoder_caption})}} +post & {cv} & {nv} & {vp} & {rt} \\"
            )
        else:
            lines.append(
                rf" &  & \textsc{{F-PIN ({decoder_caption})}} +post & -- & -- & -- & -- \\"
            )

        lines.append(r"\midrule")

    if lines[-1] == r"\midrule":
        lines = lines[:-1]

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    out = OUTDIR / out_name
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    df = pd.read_csv(CSV_FILE)
    df = sanitize(df)

    make_main_results_table(df)
    make_cross_size_multi_m_table(df)
    make_vrp60_case_study_table(df)
    make_explosion_case_study_table(df)

    make_fpin_post_ablation_table(
        df,
        decoder="giant-tour+split",
        decoder_caption="GT+Split",
        out_name="fpin_post_ablation_gt_split.tex",
        label="tab:fpin_post_ablation_gt_split",
    )

    make_fpin_post_ablation_table(
        df,
        decoder="vehicle-assignment",
        decoder_caption="assign.",
        out_name="fpin_post_ablation_assign.tex",
        label="tab:fpin_post_ablation_assign",
    )
if __name__ == "__main__":
    main()