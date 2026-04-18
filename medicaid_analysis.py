from __future__ import annotations

import argparse
import json
import html
from pathlib import Path
from typing import Iterable

import pandas as pd
import plotly.express as px


PROJECT_NAME = "Medicaid Drug Utilization Analysis"
SOURCE_PAGE_URL = (
    "https://www.medicaid.gov/medicaid/prescription-drugs/state-drug-utilization-data"
)
LATEST_YEAR = 2025
DEFAULT_OUTPUT_PATH = Path("outputs/medicaid_report.html")
DEFAULT_CHUNKSIZE = 200_000

CMS_SDUD_URLS_BY_YEAR = {
    2025: ("https://download.medicaid.gov/data/StateDrugUtilizationData-2025.csv",),
    2024: (
        "https://download.medicaid.gov/data/StateDrugUtilizationData-2024.csv",
        "https://download.medicaid.gov/data/sdud-2024-updated-dec2025.csv",
    ),
}

REQUIRED_COLUMNS = [
    "Utilization Type",
    "State",
    "NDC",
    "Labeler Code",
    "Product Code",
    "Package Size",
    "Year",
    "Quarter",
    "Suppression Used",
    "Product Name",
    "Units Reimbursed",
    "Number of Prescriptions",
    "Total Amount Reimbursed",
    "Medicaid Amount Reimbursed",
    "Non Medicaid Amount Reimbursed",
]

CODE_COLUMNS = {
    "NDC": "string",
    "Labeler Code": "string",
    "Product Code": "string",
    "Package Size": "string",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ETL and generate an HTML report from CMS Medicaid SDUD data."
    )
    parser.add_argument(
        "--year",
        type=int,
        default=LATEST_YEAR,
        help="Dataset year to analyze. Defaults to the latest configured year.",
    )
    parser.add_argument(
        "--quarter",
        default="all",
        help="Quarter to analyze: 1, 2, 3, 4, or all. Defaults to all.",
    )
    parser.add_argument(
        "--include-suppressed",
        action="store_true",
        help="Keep suppressed rows. By default they are excluded.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=DEFAULT_CHUNKSIZE,
        help="Rows per ETL chunk while reading the CMS CSV.",
    )
    parser.add_argument(
        "--skip-processed-output",
        action="store_true",
        help="Do not write the cleaned processed CSV and normalized table CSVs.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="HTML report output path.",
    )
    return parser.parse_args()


def parse_quarter(value: str) -> int | None:
    normalized = str(value).strip().lower()
    if normalized in {"all", "*", ""}:
        return None

    try:
        quarter = int(normalized)
    except ValueError as exc:
        raise ValueError("--quarter must be 1, 2, 3, 4, or all.") from exc

    if quarter not in {1, 2, 3, 4}:
        raise ValueError("--quarter must be 1, 2, 3, 4, or all.")

    return quarter


def period_label(year: int, quarter: int | None) -> str:
    if quarter is None:
        return f"{year}, all available quarters"
    return f"Q{quarter} {year}"


def cms_urls_for_year(year: int) -> tuple[str, ...]:
    if year in CMS_SDUD_URLS_BY_YEAR:
        return CMS_SDUD_URLS_BY_YEAR[year]
    return (f"https://download.medicaid.gov/data/StateDrugUtilizationData-{year}.csv",)


def require_columns(data: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def clean_code(series: pd.Series, width: int) -> pd.Series:
    return (
        series.astype("string")
        .str.replace(r"[^0-9]", "", regex=True)
        .str.zfill(width)
    )


def clean_money_or_count(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0)


def clean_suppression_flag(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .isin(("true", "t", "1", "yes", "y"))
    )


def clean_sdud_chunk(
    raw: pd.DataFrame,
    *,
    year: int,
    quarter: int | None,
    include_suppressed: bool,
) -> tuple[pd.DataFrame, dict[str, int]]:
    require_columns(raw, REQUIRED_COLUMNS)

    clean = pd.DataFrame(
        {
            "utilization_type": raw["Utilization Type"].astype("string").str.strip(),
            "state": raw["State"].astype("string").str.strip(),
            "ndc": clean_code(raw["NDC"], 11),
            "labeler_code": clean_code(raw["Labeler Code"], 5),
            "product_code": clean_code(raw["Product Code"], 4),
            "package_size": clean_code(raw["Package Size"], 2),
            "year": pd.to_numeric(raw["Year"], errors="coerce").astype("Int64"),
            "quarter": pd.to_numeric(raw["Quarter"], errors="coerce").astype("Int64"),
            "suppression_used": clean_suppression_flag(raw["Suppression Used"]),
            "product_name": raw["Product Name"].astype("string").str.strip(),
            "units_reimbursed": clean_money_or_count(raw["Units Reimbursed"]),
            "number_of_prescriptions": clean_money_or_count(raw["Number of Prescriptions"]),
            "total_amount_reimbursed": clean_money_or_count(raw["Total Amount Reimbursed"]),
            "medicaid_amount_reimbursed": clean_money_or_count(
                raw["Medicaid Amount Reimbursed"]
            ),
            "non_medicaid_amount_reimbursed": clean_money_or_count(
                raw["Non Medicaid Amount Reimbursed"]
            ),
        }
    )

    filters = (
        (clean["year"] == year)
        & (clean["state"] != "XX")
        & clean["product_name"].notna()
        & (clean["product_name"] != "")
    )
    if quarter is not None:
        filters &= clean["quarter"] == quarter

    filtered = clean[filters].copy()
    suppressed_rows_removed = int(filtered["suppression_used"].sum())

    if not include_suppressed:
        filtered = filtered[~filtered["suppression_used"]].copy()

    stats = {
        "raw_rows": len(raw),
        "rows_after_year_quarter_state_product_filter": int(filters.sum()),
        "suppressed_rows_removed": 0 if include_suppressed else suppressed_rows_removed,
        "rows_after_all_filters": len(filtered),
    }

    return filtered.reset_index(drop=True), stats


def read_and_clean_sdud(
    source: str | Path,
    *,
    year: int,
    quarter: int | None,
    include_suppressed: bool,
    chunksize: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    chunks: list[pd.DataFrame] = []
    stats = {
        "raw_rows": 0,
        "rows_after_year_quarter_state_product_filter": 0,
        "suppressed_rows_removed": 0,
        "rows_after_all_filters": 0,
    }

    reader = pd.read_csv(
        source,
        dtype=CODE_COLUMNS,
        low_memory=False,
        chunksize=chunksize,
    )

    for raw_chunk in reader:
        clean_chunk, chunk_stats = clean_sdud_chunk(
            raw_chunk,
            year=year,
            quarter=quarter,
            include_suppressed=include_suppressed,
        )
        for key, value in chunk_stats.items():
            stats[key] += int(value)
        if not clean_chunk.empty:
            chunks.append(clean_chunk)

    if chunks:
        clean_data = pd.concat(chunks, ignore_index=True)
    else:
        clean_data = pd.DataFrame(columns=clean_sdud_chunk.__annotations__)

    return clean_data, stats


def load_clean_data(
    *,
    year: int,
    quarter: int | None,
    include_suppressed: bool,
    chunksize: int,
) -> tuple[pd.DataFrame, str, dict[str, int]]:
    errors: list[str] = []

    for source in cms_urls_for_year(year):
        try:
            clean_data, stats = read_and_clean_sdud(
                source,
                year=year,
                quarter=quarter,
                include_suppressed=include_suppressed,
                chunksize=chunksize,
            )
            return clean_data, str(source), stats
        except Exception as exc:  # noqa: BLE001 - keep trying configured sources
            errors.append(f"{source}: {exc}")

    joined_errors = "\n".join(errors)
    raise RuntimeError(f"Could not read any configured data source:\n{joined_errors}")


def normalize_tables(sdud: pd.DataFrame) -> dict[str, pd.DataFrame]:
    drug_info = (
        sdud[
            [
                "ndc",
                "labeler_code",
                "product_code",
                "package_size",
                "product_name",
            ]
        ]
        .drop_duplicates()
        .sort_values(["ndc", "product_name"])
        .reset_index(drop=True)
    )
    drug_info.insert(0, "drug_id", range(1, len(drug_info) + 1))

    base = sdud.merge(
        drug_info,
        on=["ndc", "labeler_code", "product_code", "package_size", "product_name"],
        how="inner",
    ).reset_index(drop=True)
    base.insert(0, "prescription_id", range(1, len(base) + 1))

    prescription_state = base[
        ["prescription_id", "drug_id", "state", "utilization_type", "year", "quarter"]
    ].copy()

    volume = base[
        ["prescription_id", "units_reimbursed", "number_of_prescriptions"]
    ].copy()

    reimbursement = base[
        [
            "prescription_id",
            "total_amount_reimbursed",
            "medicaid_amount_reimbursed",
            "non_medicaid_amount_reimbursed",
        ]
    ].copy()

    return {
        "drug_info": drug_info,
        "prescription_state": prescription_state,
        "volume": volume,
        "reimbursement": reimbursement,
        "analysis_base": base,
    }


def build_analysis(base: pd.DataFrame) -> dict[str, pd.DataFrame]:
    drug_summary = (
        base.groupby("product_name", as_index=False)
        .agg(
            total_units=("units_reimbursed", "sum"),
            total_prescriptions=("number_of_prescriptions", "sum"),
            total_medicaid_spending=("medicaid_amount_reimbursed", "sum"),
            total_non_medicaid_spending=("non_medicaid_amount_reimbursed", "sum"),
            total_reimbursed=("total_amount_reimbursed", "sum"),
            ndc_count=("ndc", "nunique"),
            labeler_count=("labeler_code", "nunique"),
        )
        .sort_values(["total_medicaid_spending", "product_name"], ascending=[False, True])
    )

    top_by_units = (
        drug_summary.sort_values(["total_units", "product_name"], ascending=[False, True])
        .head(10)
        .reset_index(drop=True)
    )

    top_by_prescriptions = (
        drug_summary.sort_values(
            ["total_prescriptions", "product_name"], ascending=[False, True]
        )
        .head(10)
        .reset_index(drop=True)
    )

    state_spending = (
        base.groupby("state", as_index=False)
        .agg(
            total_medicaid_spending=("medicaid_amount_reimbursed", "sum"),
            total_non_medicaid_spending=("non_medicaid_amount_reimbursed", "sum"),
            total_prescriptions=("number_of_prescriptions", "sum"),
            total_units=("units_reimbursed", "sum"),
        )
        .sort_values(["total_medicaid_spending", "state"], ascending=[False, True])
        .reset_index(drop=True)
    )

    quarter_summary = (
        base.groupby("quarter", as_index=False)
        .agg(
            total_medicaid_spending=("medicaid_amount_reimbursed", "sum"),
            total_non_medicaid_spending=("non_medicaid_amount_reimbursed", "sum"),
            total_prescriptions=("number_of_prescriptions", "sum"),
            total_units=("units_reimbursed", "sum"),
        )
        .sort_values("quarter")
        .reset_index(drop=True)
    )

    utilization_summary = (
        base.groupby("utilization_type", as_index=False)
        .agg(
            total_medicaid_spending=("medicaid_amount_reimbursed", "sum"),
            total_non_medicaid_spending=("non_medicaid_amount_reimbursed", "sum"),
            total_prescriptions=("number_of_prescriptions", "sum"),
            total_units=("units_reimbursed", "sum"),
        )
        .sort_values("total_medicaid_spending", ascending=False)
        .reset_index(drop=True)
    )

    top_medicaid_drugs = (
        drug_summary.sort_values(
            ["total_medicaid_spending", "product_name"], ascending=[False, True]
        )
        .head(30)
        .reset_index(drop=True)
    )

    if top_by_units.empty:
        state_units_for_top_drug = pd.DataFrame()
    else:
        top_unit_drug = top_by_units.loc[0, "product_name"]
        state_units_for_top_drug = (
            base[base["product_name"] == top_unit_drug]
            .groupby("state", as_index=False)
            .agg(
                total_units=("units_reimbursed", "sum"),
                total_prescriptions=("number_of_prescriptions", "sum"),
            )
            .sort_values(["total_units", "state"], ascending=[False, True])
            .reset_index(drop=True)
        )
        state_units_for_top_drug.insert(0, "product_name", top_unit_drug)

    if top_medicaid_drugs.empty:
        state_contribution_top_spend_drug = pd.DataFrame()
    else:
        top_spend_drug = top_medicaid_drugs.loc[0, "product_name"]
        state_contribution_top_spend_drug = (
            base[base["product_name"] == top_spend_drug]
            .groupby("state", as_index=False)
            .agg(
                total_medicaid_spending=("medicaid_amount_reimbursed", "sum"),
                total_prescriptions=("number_of_prescriptions", "sum"),
                total_units=("units_reimbursed", "sum"),
            )
            .sort_values(["total_medicaid_spending", "state"], ascending=[False, True])
            .reset_index(drop=True)
        )
        state_contribution_top_spend_drug.insert(0, "product_name", top_spend_drug)

    labeler_summary = (
        base.groupby(["state", "labeler_code"], as_index=False)
        .agg(
            total_prescriptions=("number_of_prescriptions", "sum"),
            total_medicaid_spending=("medicaid_amount_reimbursed", "sum"),
            drug_count=("product_name", "nunique"),
        )
        .sort_values(
            ["state", "total_prescriptions", "total_medicaid_spending"],
            ascending=[True, False, False],
        )
    )

    leading_labelers = labeler_summary.drop_duplicates("state")[
        [
            "state",
            "labeler_code",
            "total_prescriptions",
            "total_medicaid_spending",
            "drug_count",
        ]
    ].rename(
        columns={
            "labeler_code": "leading_labeler_code",
            "total_prescriptions": "leading_labeler_prescriptions",
            "total_medicaid_spending": "leading_labeler_medicaid_spending",
            "drug_count": "leading_labeler_drug_count",
        }
    )

    state_competition = (
        labeler_summary.groupby("state", as_index=False)
        .agg(
            competitor_count=("labeler_code", "nunique"),
            state_labeler_prescriptions=("total_prescriptions", "sum"),
        )
        .merge(leading_labelers, on="state", how="left")
        .sort_values(["competitor_count", "state"], ascending=[False, True])
        .reset_index(drop=True)
    )

    non_medicaid_spending = (
        state_spending[["state", "total_non_medicaid_spending"]]
        .sort_values(["total_non_medicaid_spending", "state"], ascending=[False, True])
        .reset_index(drop=True)
    )

    return {
        "drug_summary": drug_summary.reset_index(drop=True),
        "top_by_units": top_by_units,
        "top_by_prescriptions": top_by_prescriptions,
        "state_spending": state_spending,
        "quarter_summary": quarter_summary,
        "utilization_summary": utilization_summary,
        "top_medicaid_drugs": top_medicaid_drugs,
        "state_units_for_top_drug": state_units_for_top_drug,
        "state_contribution_top_spend_drug": state_contribution_top_spend_drug,
        "labeler_summary": labeler_summary.reset_index(drop=True),
        "state_competition": state_competition,
        "non_medicaid_spending": non_medicaid_spending,
    }


def format_dollars(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"${value:,.0f}"


def format_number(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{value:,.0f}"


def table_html(data: pd.DataFrame, money_columns: Iterable[str] = ()) -> str:
    display = data.copy()
    for column in display.columns:
        if column in money_columns:
            display[column] = display[column].map(format_dollars)
        elif pd.api.types.is_numeric_dtype(display[column]):
            display[column] = display[column].map(format_number)

    return display.to_html(index=False, classes="data-table", border=0, escape=False)


def figure_html(fig, include_plotlyjs: bool | str = False) -> str:
    fig.update_layout(
        colorway=["#3f5f8f", "#7c4d79", "#2f7d67", "#a45b3f", "#597353"],
        margin={"l": 20, "r": 20, "t": 70, "b": 40},
    )
    return fig.to_html(
        full_html=False,
        include_plotlyjs=include_plotlyjs,
        config={"displaylogo": False, "responsive": True},
    )


def build_figures(analysis: dict[str, pd.DataFrame]) -> list[str]:
    figures: list[str] = []

    top_by_units = analysis["top_by_units"].sort_values("total_units")
    fig = px.bar(
        top_by_units,
        x="total_units",
        y="product_name",
        orientation="h",
        title="Top 10 Drugs by Units Reimbursed",
        labels={"total_units": "Units Reimbursed", "product_name": "Drug"},
    )
    figures.append(figure_html(fig, include_plotlyjs=True))

    top_by_prescriptions = analysis["top_by_prescriptions"].sort_values(
        "total_prescriptions"
    )
    fig = px.bar(
        top_by_prescriptions,
        x="total_prescriptions",
        y="product_name",
        orientation="h",
        title="Top 10 Drugs by Number of Prescriptions",
        labels={"total_prescriptions": "Prescriptions", "product_name": "Drug"},
    )
    figures.append(figure_html(fig))

    fig = px.bar(
        analysis["state_spending"].sort_values("total_medicaid_spending"),
        x="total_medicaid_spending",
        y="state",
        orientation="h",
        title="Medicaid Spending by State",
        labels={"total_medicaid_spending": "Medicaid Spending", "state": "State"},
    )
    figures.append(figure_html(fig))

    fig = px.treemap(
        analysis["top_medicaid_drugs"],
        path=["product_name"],
        values="total_medicaid_spending",
        title="Top Medicaid Drug Spending by Product",
    )
    figures.append(figure_html(fig))

    top_spending_bar = analysis["top_medicaid_drugs"].sort_values(
        "total_medicaid_spending"
    )
    fig = px.bar(
        top_spending_bar,
        x="total_medicaid_spending",
        y="product_name",
        orientation="h",
        title="Top 30 Drugs by Medicaid Spending",
        labels={
            "total_medicaid_spending": "Medicaid Spending",
            "product_name": "Drug",
        },
    )
    figures.append(figure_html(fig))

    fig = px.bar(
        analysis["non_medicaid_spending"].sort_values("total_non_medicaid_spending"),
        x="total_non_medicaid_spending",
        y="state",
        orientation="h",
        title="Non-Medicaid Reimbursement by State",
        labels={
            "total_non_medicaid_spending": "Non-Medicaid Reimbursement",
            "state": "State",
        },
    )
    figures.append(figure_html(fig))

    return figures


def write_summary_csvs(output_path: Path, analysis: dict[str, pd.DataFrame]) -> Path:
    summary_dir = output_path.parent / "summary_tables"
    summary_dir.mkdir(parents=True, exist_ok=True)

    for name, data in analysis.items():
        data.to_csv(summary_dir / f"{name}.csv", index=False)

    return summary_dir


def write_etl_outputs(
    output_path: Path,
    clean_data: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    *,
    year: int,
    quarter: int | None,
) -> tuple[Path, Path]:
    processed_dir = output_path.parent / "processed"
    normalized_dir = output_path.parent / "normalized_tables"
    processed_dir.mkdir(parents=True, exist_ok=True)
    normalized_dir.mkdir(parents=True, exist_ok=True)

    quarter_suffix = "all_quarters" if quarter is None else f"q{quarter}"
    clean_data.to_csv(
        processed_dir / f"clean_sdud_{year}_{quarter_suffix}.csv",
        index=False,
    )

    for table_name in ("drug_info", "prescription_state", "volume", "reimbursement"):
        tables[table_name].to_csv(normalized_dir / f"{table_name}.csv", index=False)

    return processed_dir, normalized_dir


def write_run_metadata(
    output_path: Path,
    *,
    source_page: str,
    data_file: str,
    year: int,
    quarter: int | None,
    include_suppressed: bool,
    clean_data: pd.DataFrame,
    etl_stats: dict[str, int],
) -> Path:
    metadata_path = output_path.parent / "run_metadata.json"
    metadata = {
        "project_name": PROJECT_NAME,
        "source": source_page,
        "data_file": data_file,
        "period": period_label(year, quarter),
        "year": year,
        "quarter": "all" if quarter is None else quarter,
        "include_suppressed": include_suppressed,
        "rows_read": etl_stats["raw_rows"],
        "rows_analyzed": len(clean_data),
        "states": int(clean_data["state"].nunique()),
        "drug_products": int(clean_data["product_name"].nunique()),
        "ndcs": int(clean_data["ndc"].nunique()),
        "total_prescriptions": float(clean_data["number_of_prescriptions"].sum()),
        "total_medicaid_spending": float(
            clean_data["medicaid_amount_reimbursed"].sum()
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path


def etl_stats_table(stats: dict[str, int]) -> pd.DataFrame:
    labels = {
        "raw_rows": "Raw rows read",
        "rows_after_year_quarter_state_product_filter": (
            "Rows after period/state/product filters"
        ),
        "suppressed_rows_removed": "Suppressed rows removed",
        "rows_after_all_filters": "Rows used in analysis",
    }
    return pd.DataFrame(
        [{"metric": labels[key], "value": value} for key, value in stats.items()]
    )


def build_report_html(
    *,
    source_page: str,
    data_file: str,
    year: int,
    quarter: int | None,
    clean_data: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    analysis: dict[str, pd.DataFrame],
    etl_stats: dict[str, int],
) -> str:
    figures = build_figures(analysis)
    period = period_label(year, quarter)

    state_count = clean_data["state"].nunique()
    source_note = (
        "This run uses a single-state sample, so state comparison charts are limited."
        if state_count == 1
        else "This run includes multiple states, so state comparison charts are multi-state."
    )

    metrics = {
        "Rows analyzed": format_number(len(clean_data)),
        "States": format_number(state_count),
        "Drug products": format_number(clean_data["product_name"].nunique()),
        "NDCs": format_number(clean_data["ndc"].nunique()),
        "Total prescriptions": format_number(clean_data["number_of_prescriptions"].sum()),
        "Total Medicaid spending": format_dollars(
            clean_data["medicaid_amount_reimbursed"].sum()
        ),
    }

    normalized_counts = pd.DataFrame(
        {
            "table": [
                "drug_info",
                "prescription_state",
                "volume",
                "reimbursement",
            ],
            "rows": [
                len(tables["drug_info"]),
                len(tables["prescription_state"]),
                len(tables["volume"]),
                len(tables["reimbursement"]),
            ],
        }
    )

    html_sections = {
        "etl_stats": table_html(etl_stats_table(etl_stats)),
        "normalized_counts": table_html(normalized_counts),
        "top_by_units": table_html(analysis["top_by_units"]),
        "top_by_prescriptions": table_html(analysis["top_by_prescriptions"]),
        "state_spending": table_html(
            analysis["state_spending"],
            money_columns=(
                "total_medicaid_spending",
                "total_non_medicaid_spending",
            ),
        ),
        "top_medicaid_drugs": table_html(
            analysis["top_medicaid_drugs"].head(10),
            money_columns=(
                "total_medicaid_spending",
                "total_non_medicaid_spending",
                "total_reimbursed",
            ),
        ),
        "state_units_for_top_drug": table_html(analysis["state_units_for_top_drug"]),
        "state_contribution_top_spend_drug": table_html(
            analysis["state_contribution_top_spend_drug"],
            money_columns=("total_medicaid_spending",),
        ),
        "state_competition": table_html(
            analysis["state_competition"],
            money_columns=("leading_labeler_medicaid_spending",),
        ),
        "non_medicaid_spending": table_html(
            analysis["non_medicaid_spending"],
            money_columns=("total_non_medicaid_spending",),
        ),
    }

    metric_cards = "\n".join(
        f"""
        <article class="metric-card">
          <span>{html.escape(label)}</span>
          <strong>{html.escape(value)}</strong>
        </article>
        """
        for label, value in metrics.items()
    )

    figure_sections = "\n".join(
        f'<section class="chart-block">{figure}</section>' for figure in figures
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(PROJECT_NAME)}</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #1e1f24;
      --muted: #5f6570;
      --line: #d7dce3;
      --surface: #ffffff;
      --band: #f5f7fa;
      --accent: #3f5f8f;
      --accent-2: #7c4d79;
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      color: var(--ink);
      background: var(--band);
      font-family: "Segoe UI", Arial, sans-serif;
      line-height: 1.55;
    }}

    header {{
      background: linear-gradient(135deg, #253858 0%, #566f8f 55%, #7c4d79 100%);
      color: white;
      padding: 56px 24px 46px;
    }}

    header .inner,
    main {{
      max-width: 1180px;
      margin: 0 auto;
    }}

    h1 {{
      margin: 0 0 12px;
      font-size: clamp(2rem, 4vw, 4rem);
      line-height: 1.05;
      letter-spacing: 0;
    }}

    h2 {{
      margin: 0 0 18px;
      font-size: 1.6rem;
      letter-spacing: 0;
    }}

    p {{
      max-width: 900px;
      margin: 0 0 14px;
    }}

    main {{
      padding: 28px 24px 56px;
    }}

    section {{
      margin: 0 0 28px;
    }}

    .summary-band,
    .table-section,
    .chart-block {{
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 20px;
    }}

    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-top: 20px;
    }}

    .metric-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      background: #fbfcfe;
      min-height: 94px;
    }}

    .metric-card span {{
      display: block;
      color: var(--muted);
      font-size: 0.9rem;
      margin-bottom: 8px;
    }}

    .metric-card strong {{
      display: block;
      font-size: 1.25rem;
      overflow-wrap: anywhere;
    }}

    .note {{
      border-left: 4px solid var(--accent);
      padding: 12px 14px;
      background: #eef3f8;
      color: #26374f;
      border-radius: 6px;
      margin-top: 16px;
    }}

    .table-wrap {{
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
    }}

    table.data-table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 700px;
      background: white;
    }}

    .data-table th,
    .data-table td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
      font-size: 0.94rem;
    }}

    .data-table th {{
      background: #edf1f6;
      color: #28364c;
      font-weight: 700;
    }}

    .data-table tr:last-child td {{
      border-bottom: 0;
    }}

    code {{
      background: #e9edf3;
      border-radius: 4px;
      padding: 2px 5px;
      font-family: Consolas, "Courier New", monospace;
    }}

    footer {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 0 24px 42px;
      color: var(--muted);
      font-size: 0.92rem;
    }}
  </style>
</head>
<body>
  <header>
    <div class="inner">
      <h1>{html.escape(PROJECT_NAME)}</h1>
      <p>Python ETL and analytics report for CMS State Drug Utilization Data. The analysis covers {html.escape(period)}, excludes suppressed records by default, and summarizes drug volume, prescriptions, Medicaid reimbursement, and non-Medicaid reimbursement.</p>
    </div>
  </header>

  <main>
    <section class="summary-band">
      <h2>Run Summary</h2>
      <p><strong>Source:</strong> <code>{html.escape(source_page)}</code></p>
      <p><strong>Data file:</strong> <code>{html.escape(data_file)}</code></p>
      <p><strong>Period:</strong> {html.escape(period)}</p>
      <p>{html.escape(source_note)}</p>
      <div class="metric-grid">
        {metric_cards}
      </div>
      <p class="note">Non-Medicaid reimbursement can include private insurance, copays, or other federal coverage. It should not be interpreted as private insurance only.</p>
    </section>

    <section class="table-section">
      <h2>ETL Preprocessing</h2>
      <p>The pipeline reads the source in chunks, preserves NDC and product codes as strings, filters the requested year/quarter, removes national total rows, removes suppressed rows by default, fills suppressed numeric blanks with zero before filtering, and writes clean/normalized CSV outputs.</p>
      <div class="table-wrap">{html_sections["etl_stats"]}</div>
    </section>

    <section class="table-section">
      <h2>Normalized Tables</h2>
      <p>The Python pipeline rebuilds the four project tables from the source CSV.</p>
      <div class="table-wrap">{html_sections["normalized_counts"]}</div>
    </section>

    {figure_sections}

    <section class="table-section">
      <h2>Top Drugs by Units</h2>
      <div class="table-wrap">{html_sections["top_by_units"]}</div>
    </section>

    <section class="table-section">
      <h2>Top Drugs by Prescriptions</h2>
      <div class="table-wrap">{html_sections["top_by_prescriptions"]}</div>
    </section>

    <section class="table-section">
      <h2>Medicaid Spending by State</h2>
      <div class="table-wrap">{html_sections["state_spending"]}</div>
    </section>

    <section class="table-section">
      <h2>Top Medicaid Spending Drugs</h2>
      <div class="table-wrap">{html_sections["top_medicaid_drugs"]}</div>
    </section>

    <section class="table-section">
      <h2>State Units for Highest-Volume Drug</h2>
      <div class="table-wrap">{html_sections["state_units_for_top_drug"]}</div>
    </section>

    <section class="table-section">
      <h2>State Contribution to Highest-Spending Drug</h2>
      <div class="table-wrap">{html_sections["state_contribution_top_spend_drug"]}</div>
    </section>

    <section class="table-section">
      <h2>Labeler Competition by State</h2>
      <div class="table-wrap">{html_sections["state_competition"]}</div>
    </section>

    <section class="table-section">
      <h2>Non-Medicaid Reimbursement by State</h2>
      <div class="table-wrap">{html_sections["non_medicaid_spending"]}</div>
    </section>
  </main>

  <footer>
    Generated by <code>medicaid_analysis.py</code>.
  </footer>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    quarter = parse_quarter(args.quarter)

    clean_data, data_file_location, etl_stats = load_clean_data(
        year=args.year,
        quarter=quarter,
        include_suppressed=args.include_suppressed,
        chunksize=args.chunksize,
    )

    if clean_data.empty:
        raise ValueError(
            f"No rows available after filtering to year={args.year}, "
            f"quarter={args.quarter}, include_suppressed={args.include_suppressed}."
        )

    tables = normalize_tables(clean_data)
    analysis = build_analysis(tables["analysis_base"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_dir = write_summary_csvs(output_path, analysis)

    processed_dir = None
    normalized_dir = None
    if not args.skip_processed_output:
        processed_dir, normalized_dir = write_etl_outputs(
            output_path,
            clean_data,
            tables,
            year=args.year,
            quarter=quarter,
        )

    metadata_path = write_run_metadata(
        output_path,
        source_page=SOURCE_PAGE_URL,
        data_file=data_file_location,
        year=args.year,
        quarter=quarter,
        include_suppressed=args.include_suppressed,
        clean_data=clean_data,
        etl_stats=etl_stats,
    )

    report = build_report_html(
        source_page=SOURCE_PAGE_URL,
        data_file=data_file_location,
        year=args.year,
        quarter=quarter,
        clean_data=clean_data,
        tables=tables,
        analysis=analysis,
        etl_stats=etl_stats,
    )
    output_path.write_text(report, encoding="utf-8")

    print(f"Source: {SOURCE_PAGE_URL}")
    print(f"Data file: {data_file_location}")
    print(f"Period: {period_label(args.year, quarter)}")
    print(f"Rows read: {etl_stats['raw_rows']:,}")
    print(f"Rows analyzed: {len(clean_data):,}")
    print(f"States: {clean_data['state'].nunique():,}")
    print(f"Report written: {output_path}")
    print(f"Summary tables written: {summary_dir}")
    print(f"Run metadata written: {metadata_path}")
    if processed_dir and normalized_dir:
        print(f"Processed ETL data written: {processed_dir}")
        print(f"Normalized tables written: {normalized_dir}")


if __name__ == "__main__":
    main()
