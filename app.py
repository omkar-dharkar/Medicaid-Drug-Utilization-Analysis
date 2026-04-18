from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_NAME = "Medicaid Drug Utilization Analysis"
SOURCE_URL = "https://www.medicaid.gov/medicaid/prescription-drugs/state-drug-utilization-data"
OUTPUT_DIR = Path("outputs")
PROCESSED_DIR = OUTPUT_DIR / "processed"
SUMMARY_DIR = OUTPUT_DIR / "summary_tables"
METADATA_PATH = OUTPUT_DIR / "run_metadata.json"


st.set_page_config(
    page_title=PROJECT_NAME,
    layout="wide",
    initial_sidebar_state="expanded",
)


def money(value: float) -> str:
    return f"${value:,.0f}"


def number(value: float) -> str:
    return f"{value:,.0f}"


def compact_money(value: float) -> str:
    abs_value = abs(value)
    if abs_value >= 1_000_000_000:
        return f"${value / 1_000_000_000:,.2f}B"
    if abs_value >= 1_000_000:
        return f"${value / 1_000_000:,.2f}M"
    if abs_value >= 1_000:
        return f"${value / 1_000:,.2f}K"
    return money(value)


def compact_number(value: float) -> str:
    abs_value = abs(value)
    if abs_value >= 1_000_000_000:
        return f"{value / 1_000_000_000:,.2f}B"
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:,.2f}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:,.2f}K"
    return number(value)


@st.cache_data(show_spinner=False)
def load_metadata() -> dict:
    if not METADATA_PATH.exists():
        return {}
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def processed_file_for_metadata(metadata: dict) -> Path | None:
    if not metadata:
        return None

    year = metadata.get("year")
    quarter = metadata.get("quarter", "all")
    quarter_suffix = "all_quarters" if quarter == "all" else f"q{quarter}"
    path = PROCESSED_DIR / f"clean_sdud_{year}_{quarter_suffix}.csv"
    return path if path.exists() else None


@st.cache_data(show_spinner="Loading processed Medicaid data...")
def load_processed_data(path: str) -> pd.DataFrame:
    return pd.read_csv(
        path,
        dtype={
            "utilization_type": "category",
            "state": "category",
            "ndc": "string",
            "labeler_code": "string",
            "product_code": "string",
            "package_size": "string",
            "product_name": "string",
        },
        low_memory=False,
    )


@st.cache_data(show_spinner=False)
def load_summary_table(name: str) -> pd.DataFrame:
    path = SUMMARY_DIR / f"{name}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def run_etl(year: int, quarter: str, include_suppressed: bool) -> tuple[bool, str]:
    command = [
        sys.executable,
        "medicaid_analysis.py",
        "--year",
        str(year),
        "--quarter",
        quarter,
    ]
    if include_suppressed:
        command.append("--include-suppressed")

    result = subprocess.run(
        command,
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )
    output = "\n".join(part for part in [result.stdout, result.stderr] if part)
    return result.returncode == 0, output


def apply_filters(
    data: pd.DataFrame,
    states: list[str],
    quarters: list[int],
    utilization_types: list[str],
    product_search: str,
) -> pd.DataFrame:
    filtered = data
    if states:
        filtered = filtered[filtered["state"].isin(states)]
    if quarters:
        filtered = filtered[filtered["quarter"].isin(quarters)]
    if utilization_types:
        filtered = filtered[filtered["utilization_type"].isin(utilization_types)]
    if product_search.strip():
        search = product_search.strip()
        filtered = filtered[
            filtered["product_name"].str.contains(search, case=False, na=False)
        ]
    return filtered


def aggregate_for_current_filters(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    drug_summary = (
        data.groupby("product_name", as_index=False, observed=True)
        .agg(
            total_units=("units_reimbursed", "sum"),
            total_prescriptions=("number_of_prescriptions", "sum"),
            total_medicaid_spending=("medicaid_amount_reimbursed", "sum"),
            total_non_medicaid_spending=("non_medicaid_amount_reimbursed", "sum"),
            ndc_count=("ndc", "nunique"),
            labeler_count=("labeler_code", "nunique"),
        )
        .sort_values("total_medicaid_spending", ascending=False)
    )

    state_summary = (
        data.groupby("state", as_index=False, observed=True)
        .agg(
            total_medicaid_spending=("medicaid_amount_reimbursed", "sum"),
            total_non_medicaid_spending=("non_medicaid_amount_reimbursed", "sum"),
            total_prescriptions=("number_of_prescriptions", "sum"),
            total_units=("units_reimbursed", "sum"),
        )
        .sort_values("total_medicaid_spending", ascending=False)
    )

    quarter_summary = (
        data.groupby("quarter", as_index=False, observed=True)
        .agg(
            total_medicaid_spending=("medicaid_amount_reimbursed", "sum"),
            total_prescriptions=("number_of_prescriptions", "sum"),
            total_units=("units_reimbursed", "sum"),
        )
        .sort_values("quarter")
    )

    labeler_summary = (
        data.groupby("labeler_code", as_index=False, observed=True)
        .agg(
            total_medicaid_spending=("medicaid_amount_reimbursed", "sum"),
            total_prescriptions=("number_of_prescriptions", "sum"),
            product_count=("product_name", "nunique"),
        )
        .sort_values("total_medicaid_spending", ascending=False)
    )

    utilization_summary = (
        data.groupby("utilization_type", as_index=False, observed=True)
        .agg(
            total_medicaid_spending=("medicaid_amount_reimbursed", "sum"),
            total_prescriptions=("number_of_prescriptions", "sum"),
        )
        .sort_values("total_medicaid_spending", ascending=False)
    )

    return {
        "drug_summary": drug_summary,
        "state_summary": state_summary,
        "quarter_summary": quarter_summary,
        "labeler_summary": labeler_summary,
        "utilization_summary": utilization_summary,
    }


def style_chart(fig):
    fig.update_layout(
        template="plotly_white",
        colorway=["#3f5f8f", "#7c4d79", "#2f7d67", "#a45b3f", "#597353"],
        margin={"l": 20, "r": 20, "t": 58, "b": 36},
        height=460,
    )
    return fig


def download_csv_button(data: pd.DataFrame, filename: str, label: str) -> None:
    st.download_button(
        label,
        data=data.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
    )


with st.sidebar:
    st.title(PROJECT_NAME)
    st.caption("Latest CMS Medicaid SDUD ETL")
    st.link_button("CMS Source", SOURCE_URL, use_container_width=True)

    st.divider()
    st.subheader("Refresh Data")
    refresh_year = st.number_input("Dataset year", min_value=2024, max_value=2030, value=2025)
    refresh_quarter = st.selectbox("Quarter", ["all", "1", "2", "3", "4"], index=0)
    refresh_suppressed = st.checkbox("Include suppressed rows", value=False)

    if st.button("Run Latest CMS ETL", type="primary", use_container_width=True):
        with st.spinner("Running CMS ETL. This can take about a minute for the 2025 file..."):
            ok, run_output = run_etl(
                year=int(refresh_year),
                quarter=refresh_quarter,
                include_suppressed=refresh_suppressed,
            )
        if ok:
            st.cache_data.clear()
            st.success("ETL complete. Reloading app data.")
            st.code(run_output)
            st.rerun()
        else:
            st.error("ETL failed.")
            st.code(run_output)

metadata = load_metadata()
processed_path = processed_file_for_metadata(metadata)

st.title(PROJECT_NAME)

if not metadata or processed_path is None:
    st.warning("No processed CMS output was found. Use the sidebar button to run the latest CMS ETL.")
    st.stop()

st.caption(f"Source: {SOURCE_URL}")
st.caption(f"Data file: {metadata.get('data_file', 'Unknown')}")

data = load_processed_data(str(processed_path))

with st.sidebar:
    st.divider()
    st.subheader("Explore")
    all_states = sorted(data["state"].dropna().astype(str).unique().tolist())
    all_quarters = sorted(data["quarter"].dropna().astype(int).unique().tolist())
    all_utilization_types = sorted(
        data["utilization_type"].dropna().astype(str).unique().tolist()
    )

    selected_states = st.multiselect("States", all_states, default=[])
    selected_quarters = st.multiselect("Quarters", all_quarters, default=all_quarters)
    selected_utilization = st.multiselect(
        "Utilization types", all_utilization_types, default=all_utilization_types
    )
    product_search = st.text_input("Drug name contains")

filtered_data = apply_filters(
    data,
    selected_states,
    selected_quarters,
    selected_utilization,
    product_search,
)

if filtered_data.empty:
    st.warning("No rows match the current filters.")
    st.stop()

analysis = aggregate_for_current_filters(filtered_data)
drug_summary = analysis["drug_summary"]
state_summary = analysis["state_summary"]
quarter_summary = analysis["quarter_summary"]
labeler_summary = analysis["labeler_summary"]
utilization_summary = analysis["utilization_summary"]

total_medicaid = filtered_data["medicaid_amount_reimbursed"].sum()
total_non_medicaid = filtered_data["non_medicaid_amount_reimbursed"].sum()
total_prescriptions = filtered_data["number_of_prescriptions"].sum()
total_units = filtered_data["units_reimbursed"].sum()
top_drug = drug_summary.iloc[0]["product_name"] if not drug_summary.empty else "N/A"

metric_cols = st.columns(6)
metric_cols[0].metric("Rows", compact_number(len(filtered_data)))
metric_cols[1].metric("States", number(filtered_data["state"].nunique()))
metric_cols[2].metric("Medicaid", compact_money(total_medicaid))
metric_cols[3].metric("Non-Medicaid", compact_money(total_non_medicaid))
metric_cols[4].metric("Prescriptions", compact_number(total_prescriptions))
metric_cols[5].metric("Top Drug", str(top_drug))

overview_tab, state_tab, drug_tab, reimbursement_tab, etl_tab = st.tabs(
    ["Overview", "State Analysis", "Drug Analysis", "Reimbursement", "ETL & Downloads"]
)

with overview_tab:
    left, right = st.columns([1.15, 1])
    with left:
        top_state = state_summary.head(20).sort_values("total_medicaid_spending")
        fig = px.bar(
            top_state,
            x="total_medicaid_spending",
            y="state",
            orientation="h",
            title="Top States by Medicaid Reimbursement",
            labels={"total_medicaid_spending": "Medicaid reimbursement", "state": ""},
        )
        st.plotly_chart(style_chart(fig), use_container_width=True)
    with right:
        fig = px.line(
            quarter_summary,
            x="quarter",
            y="total_medicaid_spending",
            markers=True,
            title="Quarterly Medicaid Reimbursement",
            labels={"quarter": "Quarter", "total_medicaid_spending": "Medicaid reimbursement"},
        )
        st.plotly_chart(style_chart(fig), use_container_width=True)

    top_drugs = drug_summary.head(15).sort_values("total_medicaid_spending")
    fig = px.bar(
        top_drugs,
        x="total_medicaid_spending",
        y="product_name",
        orientation="h",
        title="Top Drugs by Medicaid Reimbursement",
        labels={"total_medicaid_spending": "Medicaid reimbursement", "product_name": ""},
    )
    st.plotly_chart(style_chart(fig), use_container_width=True)

with state_tab:
    fig = px.scatter(
        state_summary,
        x="total_prescriptions",
        y="total_medicaid_spending",
        size="total_units",
        color="state",
        hover_name="state",
        title="State Spending vs Prescription Volume",
        labels={
            "total_prescriptions": "Prescriptions",
            "total_medicaid_spending": "Medicaid reimbursement",
            "total_units": "Units reimbursed",
        },
    )
    st.plotly_chart(style_chart(fig), use_container_width=True)

    display_state = state_summary.copy()
    st.dataframe(
        display_state,
        use_container_width=True,
        hide_index=True,
    )
    download_csv_button(display_state, "state_analysis.csv", "Download State Analysis")

with drug_tab:
    left, right = st.columns(2)
    with left:
        top_units = drug_summary.sort_values("total_units", ascending=False).head(15)
        fig = px.bar(
            top_units.sort_values("total_units"),
            x="total_units",
            y="product_name",
            orientation="h",
            title="Top Drugs by Units Reimbursed",
            labels={"total_units": "Units reimbursed", "product_name": ""},
        )
        st.plotly_chart(style_chart(fig), use_container_width=True)
    with right:
        top_scripts = drug_summary.sort_values(
            "total_prescriptions", ascending=False
        ).head(15)
        fig = px.bar(
            top_scripts.sort_values("total_prescriptions"),
            x="total_prescriptions",
            y="product_name",
            orientation="h",
            title="Top Drugs by Prescriptions",
            labels={"total_prescriptions": "Prescriptions", "product_name": ""},
        )
        st.plotly_chart(style_chart(fig), use_container_width=True)

    st.dataframe(drug_summary.head(250), use_container_width=True, hide_index=True)
    download_csv_button(drug_summary, "drug_analysis.csv", "Download Drug Analysis")

with reimbursement_tab:
    fig = px.treemap(
        drug_summary.head(50),
        path=["product_name"],
        values="total_medicaid_spending",
        title="Medicaid Reimbursement Concentration by Drug",
    )
    st.plotly_chart(style_chart(fig), use_container_width=True)

    left, right = st.columns(2)
    with left:
        fig = px.bar(
            utilization_summary,
            x="utilization_type",
            y="total_medicaid_spending",
            title="Medicaid Reimbursement by Utilization Type",
            labels={
                "utilization_type": "Utilization type",
                "total_medicaid_spending": "Medicaid reimbursement",
            },
        )
        st.plotly_chart(style_chart(fig), use_container_width=True)
    with right:
        top_labelers = labeler_summary.head(20).sort_values("total_medicaid_spending")
        fig = px.bar(
            top_labelers,
            x="total_medicaid_spending",
            y="labeler_code",
            orientation="h",
            title="Top Labeler Codes by Medicaid Reimbursement",
            labels={"total_medicaid_spending": "Medicaid reimbursement", "labeler_code": ""},
        )
        st.plotly_chart(style_chart(fig), use_container_width=True)

    download_csv_button(labeler_summary, "labeler_analysis.csv", "Download Labeler Analysis")

with etl_tab:
    st.subheader("Current ETL Run")
    st.json(metadata)

    summary_files = sorted(SUMMARY_DIR.glob("*.csv"))
    normalized_files = sorted((OUTPUT_DIR / "normalized_tables").glob("*.csv"))
    processed_files = sorted(PROCESSED_DIR.glob("*.csv"))

    file_rows = []
    for path in [*processed_files, *normalized_files, *summary_files]:
        file_rows.append(
            {
                "file": str(path),
                "size_mb": round(path.stat().st_size / 1_000_000, 2),
            }
        )
    st.dataframe(pd.DataFrame(file_rows), use_container_width=True, hide_index=True)

    st.download_button(
        "Download Filtered Rows",
        data=filtered_data.to_csv(index=False).encode("utf-8"),
        file_name="filtered_medicaid_rows.csv",
        mime="text/csv",
        use_container_width=True,
    )
