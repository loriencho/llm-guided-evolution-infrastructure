from pathlib import Path
import subprocess
import shutil
import math

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="LLM-Guided Evolution Analysis Dashboard",
    layout="wide",
)

st.title("LLM-Guided Evolution Analysis Dashboard")
st.caption("Upload Slurm logs and/or pickle files, run the analysis pipeline, and view the generated results.")


UPLOAD_DIR = Path("analysis/dashboard/uploads")
RESULTS_DIR = Path("analysis/results/analysis")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def run_command(cmd: list[str]) -> None:
    st.write("Running:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout:
        st.text(result.stdout)

    if result.returncode != 0:
        st.error(result.stderr)
        st.stop()


def finite_mask(df: pd.DataFrame) -> pd.Series:
    return (
        df["fitness_1_num"].apply(lambda x: pd.notna(x) and math.isfinite(x))
        & df["fitness_2_num"].apply(lambda x: pd.notna(x) and math.isfinite(x))
    )


# -------------------------
# Sidebar upload + pipeline
# -------------------------

st.sidebar.header("Upload Files")

uploaded_files = st.sidebar.file_uploader(
    "Upload Slurm, log, or pickle files",
    type=["out", "log", "pkl"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.sidebar.write("Uploaded:", [f.name for f in uploaded_files])

if st.sidebar.button("Analyze Uploaded Files"):
    if not uploaded_files:
        st.sidebar.warning("Upload at least one file first.")
    else:
        shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

        for uploaded_file in uploaded_files:
            file_path = UPLOAD_DIR / uploaded_file.name
            with file_path.open("wb") as f:
                f.write(uploaded_file.getbuffer())

        st.sidebar.success(f"Saved {len(uploaded_files)} uploaded file(s).")

        commands = [
            [
                "python3", "analysis/scripts/summarize_slurm.py",
                "--input", str(UPLOAD_DIR),
                "--output", str(RESULTS_DIR / "slurm_summary.csv"),
            ],
            [
                "python3", "analysis/scripts/inventory_runs.py",
                "--input", str(UPLOAD_DIR),
                "--output", str(RESULTS_DIR / "run_inventory.csv"),
            ],
            [
                "python3", "analysis/scripts/extract_metrics.py",
                "--input", str(UPLOAD_DIR),
                "--output", str(RESULTS_DIR / "run_metrics.csv"),
            ],
            [
                "python3", "analysis/scripts/entity_metrics_summary.py",
                "--input", str(RESULTS_DIR / "run_metrics.csv"),
                "--output", str(RESULTS_DIR / "entity_metrics_summary.csv"),
                "--report-output", str(RESULTS_DIR / "entity_metrics_report.md"),
            ],
            [
                "python3", "analysis/scripts/build_failure_report.py",
                "--input", str(RESULTS_DIR / "slurm_summary.csv"),
                "--counts-output", str(RESULTS_DIR / "failure_counts.csv"),
                "--report-output", str(RESULTS_DIR / "failure_report.md"),
            ],
            [
                "python3", "analysis/scripts/compare_runs.py",
                "--slurm", str(RESULTS_DIR / "slurm_summary.csv"),
                "--inventory", str(RESULTS_DIR / "run_inventory.csv"),
                "--metrics", str(RESULTS_DIR / "run_metrics.csv"),
                "--output", str(RESULTS_DIR / "run_comparison.csv"),
            ],
            [
                "python3", "analysis/scripts/export_dashboard_data.py",
                "--slurm-input", str(RESULTS_DIR / "slurm_summary.csv"),
                "--comparison-input", str(RESULTS_DIR / "run_comparison.csv"),
                "--failure-input", str(RESULTS_DIR / "failure_counts.csv"),
                "--metrics-input", str(RESULTS_DIR / "run_metrics.csv"),
                "--output-dir", str(RESULTS_DIR),
            ],
        ]

        for cmd in commands:
            run_command(cmd)

        st.session_state["analysis_complete"] = True
        st.success("Analysis complete. Dashboard outputs updated.")


# -------------------------
# Load generated outputs
# -------------------------

slurm_summary = read_csv(RESULTS_DIR / "slurm_summary.csv")
failure_counts = read_csv(RESULTS_DIR / "failure_counts.csv")
entity_summary = read_csv(RESULTS_DIR / "entity_metrics_summary.csv")
run_comparison = read_csv(RESULTS_DIR / "run_comparison.csv")
run_inventory = read_csv(RESULTS_DIR / "run_inventory.csv")

st.caption(f"Reading outputs from: `{RESULTS_DIR}`")


# -------------------------
# 1. Run Health
# -------------------------

st.header("1. Run Health")

if not slurm_summary.empty:
    col1, col2, col3 = st.columns(3)

    total_runs = len(slurm_summary)
    failed_runs = (
        (slurm_summary["status_guess"] == "failed").sum()
        if "status_guess" in slurm_summary.columns
        else 0
    )
    completed_runs = (
        slurm_summary["status_guess"].isin(["completed", "completed_or_unknown"]).sum()
        if "status_guess" in slurm_summary.columns
        else 0
    )

    col1.metric("Total Runs", int(total_runs))
    col2.metric("Failed Runs", int(failed_runs))
    col3.metric("Completed Runs", int(completed_runs))

    st.dataframe(slurm_summary, use_container_width=True)
else:
    st.info("Upload a Slurm `.out` or `.log` file and click Analyze Uploaded Files.")


# -------------------------
# 2. Failure Breakdown
# -------------------------

st.header("2. Failure Breakdown")

if not failure_counts.empty:
    st.dataframe(failure_counts, use_container_width=True)

    if {"group_type", "group_value", "count"}.issubset(failure_counts.columns):
        failure_df = failure_counts[failure_counts["group_type"] == "failure_reason"]

        if not failure_df.empty:
            st.bar_chart(failure_df.set_index("group_value")["count"])
        else:
            st.info("No failure reasons found.")
else:
    st.info("No failure data available.")


# -------------------------
# 3. Controller / Child Job Activity
# -------------------------

st.header("3. Controller / Child Job Activity")

if not slurm_summary.empty:
    cols = ["job_id", "submitted_child_jobs", "completed_child_jobs", "wait_events"]
    available = [c for c in cols if c in slurm_summary.columns]

    if len(available) > 1:
        controller_df = slurm_summary[available].copy()

        for c in available:
            if c != "job_id":
                controller_df[c] = pd.to_numeric(controller_df[c], errors="coerce").fillna(0)

        st.dataframe(controller_df, use_container_width=True)

        numeric_cols = [c for c in available if c != "job_id"]
        st.bar_chart(controller_df.set_index("job_id")[numeric_cols])
    else:
        st.info("No controller activity columns found.")
else:
    st.info("Upload a Slurm `.out` or `.log` file to show controller activity.")


# -------------------------
# 4. Entity Fitness Quality
# -------------------------

st.header("4. Entity Fitness Quality")

if not entity_summary.empty and {"fitness_1", "fitness_2"}.issubset(entity_summary.columns):
    df = entity_summary.copy()

    df["fitness_1_num"] = pd.to_numeric(df["fitness_1"], errors="coerce")
    df["fitness_2_num"] = pd.to_numeric(df["fitness_2"], errors="coerce")

    finite = finite_mask(df)

    infinite = (
        df["fitness_1"].astype(str).isin(["inf", "-inf"])
        | df["fitness_2"].astype(str).isin(["inf", "-inf"])
    )

    missing = ~(finite | infinite)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Entities", int(len(df)))
    col2.metric("Finite Fitness", int(finite.sum()))
    col3.metric("Infinite Fitness", int(infinite.sum()))
    col4.metric("Missing / Malformed", int(missing.sum()))

    quality_df = pd.DataFrame({
        "category": ["finite", "infinite", "missing/malformed"],
        "count": [int(finite.sum()), int(infinite.sum()), int(missing.sum())],
    })

    st.bar_chart(quality_df.set_index("category")["count"])
else:
    st.info("Upload a `.pkl` file to show entity fitness quality.")


# -------------------------
# 5. Top Individuals
# -------------------------

st.header("5. Top Individuals")

finite_df = pd.DataFrame()

if not entity_summary.empty and {"fitness_1", "fitness_2"}.issubset(entity_summary.columns):
    top_df = entity_summary.copy()

    top_df["fitness_1_num"] = pd.to_numeric(top_df["fitness_1"], errors="coerce")
    top_df["fitness_2_num"] = pd.to_numeric(top_df["fitness_2"], errors="coerce")

    finite_df = top_df[finite_mask(top_df)].copy()

    if not finite_df.empty:
        finite_df = finite_df.sort_values("fitness_1_num", ascending=False)

        show_cols = [
            "entity_id",
            "job_id",
            "results_job",
            "entity_status",
            "fitness_1",
            "fitness_2",
            "most_common_mutate_type",
        ]
        show_cols = [c for c in show_cols if c in finite_df.columns]

        st.dataframe(finite_df[show_cols].head(10), use_container_width=True)
    else:
        st.info("No finite fitness rows available for top-individual ranking.")
else:
    st.info("No top-individual data available.")


# -------------------------
# 6. Pareto Scatter
# -------------------------

st.header("6. Pareto Scatter")

if not finite_df.empty:
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(
        finite_df["fitness_2_num"],
        finite_df["fitness_1_num"],
        alpha=0.7,
        label="Finite entities",
    )

    ax.set_xlabel("fitness_2")
    ax.set_ylabel("fitness_1")
    ax.set_title("Pareto Scatter: Finite Fitness Points")
    ax.legend()

    st.pyplot(fig)

    st.caption("Only finite fitness rows are plotted. Infinite or missing fitness values are excluded.")
else:
    st.info("Upload a `.pkl` file with finite fitness values to show Pareto scatter.")


# -------------------------
# 7. Run Inventory
# -------------------------

st.header("7. Run Artifact Inventory")

if not run_inventory.empty:
    st.dataframe(run_inventory, use_container_width=True)
else:
    st.info("No run inventory available.")


# -------------------------
# 8. Run Comparison
# -------------------------

st.header("8. Run Comparison")

if not run_comparison.empty:
    st.dataframe(run_comparison, use_container_width=True)
else:
    st.info("No run comparison available.")