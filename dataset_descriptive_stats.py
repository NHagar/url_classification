#!/usr/bin/env python3
"""
Dataset Descriptive Statistics Script

Calculates comprehensive descriptive statistics for each dataset, with special focus on:
- Domain/publisher distribution and skew
- Category distribution per domain
- Cross-dataset comparisons

Usage:
    python dataset_descriptive_stats.py --datasets uci huffpo recognasumm --output stats_report.txt
"""

import argparse
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import tldextract
from scipy import stats

from url_classification.dataset_loading import load_data


class DatasetDescriptiveStats:
    """Calculate comprehensive descriptive statistics for news classification datasets"""

    def __init__(self):
        self.stats = {}
        self._domain_cache = {}

    def analyze_dataset(self, dataset_name: str) -> Dict:
        """Analyze a single dataset and return comprehensive statistics"""
        print(f"Analyzing {dataset_name} dataset...")

        # Load dataset
        df, _ = load_data(dataset_name)

        # Extract domains from URLs if this is the recognasumm dataset
        if dataset_name == "recognasumm" and "URL" in df.columns:
            df = self._extract_domains_from_urls(df)

        stats_dict = {
            "dataset": dataset_name,
            "total_records": len(df),
            "basic_stats": self._calculate_basic_stats(df),
            "category_distribution": self._analyze_category_distribution(df),
        }

        # Add domain-specific analysis if domain/publisher info available
        if self._has_domain_info(df, dataset_name):
            domain_col = self._get_domain_column(df, dataset_name)
            stats_dict.update(
                {
                    "domain_column": domain_col,
                    "domain_distribution": self._analyze_domain_distribution(
                        df, domain_col
                    ),
                    "domain_skew": self._calculate_domain_skew(df, domain_col),
                    "category_per_domain": self._analyze_category_per_domain(
                        df, domain_col
                    ),
                    "domain_category_cross_table": self._create_domain_category_crosstab(
                        df, domain_col
                    ),
                }
            )
        else:
            stats_dict.update(
                {
                    "domain_column": None,
                    "domain_distribution": None,
                    "domain_skew": None,
                    "category_per_domain": None,
                    "domain_category_cross_table": None,
                }
            )

        return stats_dict

    def _has_domain_info(self, df: pd.DataFrame, dataset_name: str) -> bool:
        """Check if dataset has domain/publisher information"""
        if dataset_name == "uci":
            return "PUBLISHER" in df.columns
        elif dataset_name == "huffpo":
            # HuffPost dataset might have domain info extracted from URLs
            return any(col in df.columns for col in ["domain", "publisher", "source"])
        elif dataset_name == "recognasumm":
            # Check for domain-related columns or URL field
            return any(
                col in df.columns
                for col in ["domain", "publisher", "source", "website", "URL"]
            )
        return False

    def _get_domain_column(self, df: pd.DataFrame, dataset_name: str) -> Optional[str]:
        """Get the appropriate domain column for the dataset"""
        if dataset_name == "uci" and "PUBLISHER" in df.columns:
            return "PUBLISHER"

        # Check for common domain column names
        for col in ["domain", "publisher", "source", "website"]:
            if col in df.columns:
                return col

        # For recognasumm, if we extracted domains from URLs, use that column
        if dataset_name == "recognasumm" and "extracted_domain" in df.columns:
            return "extracted_domain"

        return None

    def _extract_domains_from_urls(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract domains from URL column using tldextract"""
        print("Extracting domains from URLs...")

        def extract_domain(url):
            if pd.isna(url) or not isinstance(url, str):
                return None

            # Use cache to avoid re-extracting same URLs
            if url in self._domain_cache:
                return self._domain_cache[url]

            try:
                extracted = tldextract.extract(url)
                # Combine subdomain + domain + suffix (e.g., 'www.example.com')
                if extracted.domain and extracted.suffix:
                    if extracted.subdomain:
                        domain = f"{extracted.subdomain}.{extracted.domain}.{extracted.suffix}"
                    else:
                        domain = f"{extracted.domain}.{extracted.suffix}"

                    self._domain_cache[url] = domain
                    return domain
                else:
                    self._domain_cache[url] = None
                    return None
            except Exception:
                self._domain_cache[url] = None
                return None

        # Create a copy to avoid modifying the original
        df_copy = df.copy()
        df_copy["extracted_domain"] = df_copy["URL"].apply(extract_domain)

        # Print some statistics about domain extraction
        total_urls = len(df_copy)
        valid_domains = df_copy["extracted_domain"].notna().sum()
        unique_domains = df_copy["extracted_domain"].nunique()

        print("Domain extraction complete:")
        print(f"  Total URLs: {total_urls:,}")
        print(
            f"  Valid domains extracted: {valid_domains:,} ({valid_domains / total_urls:.1%})"
        )
        print(f"  Unique domains: {unique_domains:,}")

        return df_copy

    def _calculate_basic_stats(self, df: pd.DataFrame) -> Dict:
        """Calculate basic dataset statistics"""
        stats = {
            "n_records": len(df),
            "n_features": len(df.columns),
            "column_names": list(df.columns),
        }

        # Text length statistics if text columns exist
        text_cols = [
            col
            for col in df.columns
            if col.lower() in ["text", "title", "content", "body"]
        ]
        if text_cols:
            for col in text_cols:
                if col in df.columns and df[col].dtype == "object":
                    text_lengths = df[col].astype(str).str.len()
                    stats[f"{col}_length_stats"] = {
                        "mean": text_lengths.mean(),
                        "median": text_lengths.median(),
                        "std": text_lengths.std(),
                        "min": text_lengths.min(),
                        "max": text_lengths.max(),
                        "q25": text_lengths.quantile(0.25),
                        "q75": text_lengths.quantile(0.75),
                    }

        return stats

    def _analyze_category_distribution(self, df: pd.DataFrame) -> Dict:
        """Analyze the distribution of categories/labels"""
        if "y" not in df.columns:
            return {"error": 'No target column "y" found'}

        category_counts = df["y"].value_counts()
        category_props = df["y"].value_counts(normalize=True)

        return {
            "n_categories": len(category_counts),
            "categories": list(category_counts.index),
            "counts": category_counts.to_dict(),
            "proportions": category_props.to_dict(),
            "most_common": category_counts.index[0],
            "least_common": category_counts.index[-1],
            "balance_ratio": category_counts.min() / category_counts.max(),
            "entropy": stats.entropy(category_counts.values, base=2),
            "gini_impurity": 1 - sum((category_props.values**2)),
        }

    def _analyze_domain_distribution(self, df: pd.DataFrame, domain_col: str) -> Dict:
        """Analyze the distribution of domains/publishers"""
        domain_counts = df[domain_col].value_counts()
        domain_props = df[domain_col].value_counts(normalize=True)

        return {
            "n_domains": len(domain_counts),
            "top_10_domains": domain_counts.head(10).to_dict(),
            "top_10_proportions": domain_props.head(10).to_dict(),
            "domain_counts_stats": {
                "mean": domain_counts.mean(),
                "median": domain_counts.median(),
                "std": domain_counts.std(),
                "min": domain_counts.min(),
                "max": domain_counts.max(),
                "q25": domain_counts.quantile(0.25),
                "q75": domain_counts.quantile(0.75),
            },
        }

    def _calculate_domain_skew(self, df: pd.DataFrame, domain_col: str) -> Dict:
        """Calculate domain skew metrics"""
        domain_counts = df[domain_col].value_counts()

        # Calculate various skew metrics
        skewness = stats.skew(domain_counts.values)
        kurtosis = stats.kurtosis(domain_counts.values)

        # Gini coefficient for domain distribution
        n = len(domain_counts)
        sorted_counts = np.sort(domain_counts.values)
        gini = (2 * np.sum(np.arange(1, n + 1) * sorted_counts)) / (
            n * np.sum(sorted_counts)
        ) - (n + 1) / n

        # Concentration metrics
        total_records = len(df)
        top_1_percent = domain_counts.iloc[0] / total_records
        top_5_percent = (
            domain_counts.head(max(1, len(domain_counts) // 20)).sum() / total_records
        )
        top_10_percent = (
            domain_counts.head(max(1, len(domain_counts) // 10)).sum() / total_records
        )

        # Herfindahl-Hirschman Index (HHI) for concentration
        proportions = domain_counts / total_records
        hhi = np.sum(proportions**2)

        return {
            "skewness": skewness,
            "kurtosis": kurtosis,
            "gini_coefficient": gini,
            "top_1_domain_percent": top_1_percent,
            "top_5_domains_percent": top_5_percent,
            "top_10_domains_percent": top_10_percent,
            "herfindahl_hirschman_index": hhi,
            "effective_number_domains": 1 / hhi if hhi > 0 else len(domain_counts),
            "domains_with_single_record": (domain_counts == 1).sum(),
            "domains_with_lt_10_records": (domain_counts < 10).sum(),
            "domains_with_gt_100_records": (domain_counts > 100).sum(),
        }

    def _analyze_category_per_domain(self, df: pd.DataFrame, domain_col: str) -> Dict:
        """Analyze category distribution within each domain"""
        if "y" not in df.columns:
            return {"error": 'No target column "y" found'}

        # Calculate category proportions per domain
        domain_category_stats = {}

        for domain in df[domain_col].unique():
            domain_data = df[df[domain_col] == domain]
            category_dist = domain_data["y"].value_counts(normalize=True)
            category_counts = domain_data["y"].value_counts()

            domain_category_stats[domain] = {
                "total_records": len(domain_data),
                "n_categories": len(category_dist),
                "category_proportions": category_dist.to_dict(),
                "category_counts": category_counts.to_dict(),
                "dominant_category": category_dist.index[0]
                if len(category_dist) > 0
                else None,
                "dominant_category_percent": category_dist.iloc[0]
                if len(category_dist) > 0
                else 0,
                "entropy": stats.entropy(category_counts.values, base=2)
                if len(category_counts) > 0
                else 0,
                "is_single_category": len(category_dist) == 1,
            }

        # Summary statistics across domains
        entropies = [stats["entropy"] for stats in domain_category_stats.values()]
        dominant_percents = [
            stats["dominant_category_percent"]
            for stats in domain_category_stats.values()
        ]
        single_category_domains = sum(
            [stats["is_single_category"] for stats in domain_category_stats.values()]
        )
        
        # Calculate records within single-category domains
        records_in_single_category_domains = sum(
            [stats["total_records"] for stats in domain_category_stats.values()
             if stats["is_single_category"]]
        )
        total_records = sum(
            [stats["total_records"] for stats in domain_category_stats.values()]
        )

        return {
            "per_domain_stats": domain_category_stats,
            "summary": {
                "avg_entropy_per_domain": np.mean(entropies),
                "std_entropy_per_domain": np.std(entropies),
                "avg_dominant_category_percent": np.mean(dominant_percents),
                "std_dominant_category_percent": np.std(dominant_percents),
                "single_category_domains": single_category_domains,
                "single_category_domains_percent": single_category_domains
                / len(domain_category_stats),
                "records_in_single_category_domains": records_in_single_category_domains,
                "records_in_single_category_domains_percent": records_in_single_category_domains
                / total_records if total_records > 0 else 0,
            },
        }

    def _create_domain_category_crosstab(
        self, df: pd.DataFrame, domain_col: str
    ) -> Dict:
        """Create cross-tabulation of domains and categories"""
        if "y" not in df.columns:
            return {"error": 'No target column "y" found'}

        # Create crosstab
        crosstab = pd.crosstab(df[domain_col], df["y"], margins=True)
        crosstab_normalized = pd.crosstab(df[domain_col], df["y"], normalize="index")

        return {
            "crosstab_counts": crosstab.to_dict(),
            "crosstab_proportions": crosstab_normalized.to_dict(),
            "chi2_test": self._perform_chi2_test(df, domain_col, "y"),
        }

    def _perform_chi2_test(
        self, df: pd.DataFrame, domain_col: str, category_col: str
    ) -> Dict:
        """Perform chi-square test of independence"""
        try:
            crosstab = pd.crosstab(df[domain_col], df[category_col])
            chi2, p_value, dof, expected = stats.chi2_contingency(crosstab)

            return {
                "chi2_statistic": chi2,
                "p_value": p_value,
                "degrees_of_freedom": dof,
                "is_significant": p_value < 0.05,
                "cramers_v": np.sqrt(
                    chi2 / (crosstab.sum().sum() * (min(crosstab.shape) - 1))
                ),
            }
        except Exception as e:
            return {"error": str(e)}


def format_stats_report(
    all_stats: List[Dict], output_file: Optional[str] = None
) -> str:
    """Format comprehensive statistics report"""

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("DATASET DESCRIPTIVE STATISTICS REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")

    for stats in all_stats:
        dataset_name = stats["dataset"]
        report_lines.append(f"{'=' * 50}")
        report_lines.append(f"DATASET: {dataset_name.upper()}")
        report_lines.append(f"{'=' * 50}")
        report_lines.append("")

        # Basic stats
        basic = stats["basic_stats"]
        report_lines.append("BASIC STATISTICS:")
        report_lines.append(f"  Total Records: {basic['n_records']:,}")
        report_lines.append(f"  Features: {basic['n_features']}")
        report_lines.append(f"  Columns: {', '.join(basic['column_names'])}")
        report_lines.append("")

        # Text length stats
        for key, value in basic.items():
            if key.endswith("_length_stats"):
                col_name = key.replace("_length_stats", "")
                report_lines.append(f"TEXT LENGTH STATISTICS ({col_name.upper()}):")
                report_lines.append(f"  Mean: {value['mean']:.1f} characters")
                report_lines.append(f"  Median: {value['median']:.1f} characters")
                report_lines.append(f"  Std Dev: {value['std']:.1f}")
                report_lines.append(f"  Range: {value['min']:.0f} - {value['max']:.0f}")
                report_lines.append(
                    f"  Q25-Q75: {value['q25']:.1f} - {value['q75']:.1f}"
                )
                report_lines.append("")

        # Category distribution
        cat_dist = stats["category_distribution"]
        if "error" not in cat_dist:
            report_lines.append("CATEGORY DISTRIBUTION:")
            report_lines.append(f"  Number of Categories: {cat_dist['n_categories']}")
            report_lines.append(
                f"  Categories: {', '.join(map(str, cat_dist['categories']))}"
            )
            report_lines.append(
                f"  Most Common: {cat_dist['most_common']} ({cat_dist['proportions'][cat_dist['most_common']]:.1%})"
            )
            report_lines.append(
                f"  Least Common: {cat_dist['least_common']} ({cat_dist['proportions'][cat_dist['least_common']]:.1%})"
            )
            report_lines.append(f"  Balance Ratio: {cat_dist['balance_ratio']:.3f}")
            report_lines.append(f"  Entropy: {cat_dist['entropy']:.3f}")
            report_lines.append(f"  Gini Impurity: {cat_dist['gini_impurity']:.3f}")

            report_lines.append("  Category Breakdown:")
            for cat, count in cat_dist["counts"].items():
                prop = cat_dist["proportions"][cat]
                report_lines.append(f"    {cat}: {count:,} ({prop:.1%})")
            report_lines.append("")

        # Domain analysis (if available)
        if stats["domain_column"]:
            domain_col = stats["domain_column"]
            report_lines.append(f"DOMAIN ANALYSIS (using {domain_col}):")

            # Domain distribution
            domain_dist = stats["domain_distribution"]
            report_lines.append(f"  Total Domains: {domain_dist['n_domains']:,}")
            report_lines.append("  Domain Record Statistics:")
            domain_stats = domain_dist["domain_counts_stats"]
            report_lines.append(f"    Mean: {domain_stats['mean']:.1f} records/domain")
            report_lines.append(
                f"    Median: {domain_stats['median']:.1f} records/domain"
            )
            report_lines.append(f"    Std Dev: {domain_stats['std']:.1f}")
            report_lines.append(
                f"    Range: {domain_stats['min']:.0f} - {domain_stats['max']:.0f}"
            )

            report_lines.append("  Top 10 Domains:")
            for domain, count in domain_dist["top_10_domains"].items():
                prop = domain_dist["top_10_proportions"][domain]
                report_lines.append(f"    {domain}: {count:,} ({prop:.1%})")
            report_lines.append("")

            # Domain skew
            skew = stats["domain_skew"]
            report_lines.append("DOMAIN DISTRIBUTION SKEW:")
            report_lines.append(f"  Skewness: {skew['skewness']:.3f}")
            report_lines.append(f"  Kurtosis: {skew['kurtosis']:.3f}")
            report_lines.append(f"  Gini Coefficient: {skew['gini_coefficient']:.3f}")
            report_lines.append(
                f"  HHI (Concentration): {skew['herfindahl_hirschman_index']:.3f}"
            )
            report_lines.append(
                f"  Effective # of domains: {skew['effective_number_domains']:.1f}"
            )
            report_lines.append(
                f"  Top domain share: {skew['top_1_domain_percent']:.1%}"
            )
            report_lines.append(
                f"  Top 5 domains share: {skew['top_5_domains_percent']:.1%}"
            )
            report_lines.append(
                f"  Top 10 domains share: {skew['top_10_domains_percent']:.1%}"
            )
            report_lines.append(
                f"  Single-record domains: {skew['domains_with_single_record']} ({skew['domains_with_single_record'] / domain_dist['n_domains']:.1%})"
            )
            report_lines.append(
                f"  Domains with <10 records: {skew['domains_with_lt_10_records']} ({skew['domains_with_lt_10_records'] / domain_dist['n_domains']:.1%})"
            )
            report_lines.append(
                f"  Domains with >100 records: {skew['domains_with_gt_100_records']} ({skew['domains_with_gt_100_records'] / domain_dist['n_domains']:.1%})"
            )
            report_lines.append("")

            # Category per domain analysis
            cat_per_domain = stats["category_per_domain"]
            if "error" not in cat_per_domain:
                summary = cat_per_domain["summary"]
                report_lines.append("CATEGORY DISTRIBUTION PER DOMAIN:")
                report_lines.append(
                    f"  Average entropy per domain: {summary['avg_entropy_per_domain']:.3f} ± {summary['std_entropy_per_domain']:.3f}"
                )
                report_lines.append(
                    f"  Average dominant category %: {summary['avg_dominant_category_percent']:.1%} ± {summary['std_dominant_category_percent']:.1%}"
                )
                report_lines.append(
                    f"  Single-category domains: {summary['single_category_domains']} ({summary['single_category_domains_percent']:.1%})"
                )
                report_lines.append(
                    f"  Records in single-category domains: {summary['records_in_single_category_domains']:,} ({summary['records_in_single_category_domains_percent']:.1%})"
                )

                # Show examples of highly skewed domains
                report_lines.append("  Most category-skewed domains (top 10):")
                domain_skews = [
                    (domain, stats_dict["dominant_category_percent"])
                    for domain, stats_dict in cat_per_domain["per_domain_stats"].items()
                ]
                domain_skews.sort(key=lambda x: x[1], reverse=True)

                for domain, skew_pct in domain_skews[:10]:
                    domain_stats = cat_per_domain["per_domain_stats"][domain]
                    dominant_cat = domain_stats["dominant_category"]
                    total_records = domain_stats["total_records"]
                    report_lines.append(
                        f"    {domain}: {skew_pct:.1%} {dominant_cat} ({total_records} total records)"
                    )
                report_lines.append("")

                # Chi-square test results
                crosstab = stats["domain_category_cross_table"]
                if "chi2_test" in crosstab and "error" not in crosstab["chi2_test"]:
                    chi2_results = crosstab["chi2_test"]
                    report_lines.append("DOMAIN-CATEGORY INDEPENDENCE TEST:")
                    report_lines.append(
                        f"  Chi-square statistic: {chi2_results['chi2_statistic']:.2f}"
                    )
                    report_lines.append(f"  p-value: {chi2_results['p_value']:.2e}")
                    report_lines.append(
                        f"  Degrees of freedom: {chi2_results['degrees_of_freedom']}"
                    )
                    report_lines.append(
                        f"  Significant association: {'Yes' if chi2_results['is_significant'] else 'No'}"
                    )
                    report_lines.append(
                        f"  Cramer's V (effect size): {chi2_results['cramers_v']:.3f}"
                    )
                    report_lines.append("")

        report_lines.append("")

    # Cross-dataset comparison
    if len(all_stats) > 1:
        report_lines.append("=" * 50)
        report_lines.append("CROSS-DATASET COMPARISON")
        report_lines.append("=" * 50)
        report_lines.append("")

        comparison_table = []
        headers = [
            "Dataset",
            "Records",
            "Categories",
            "Domains",
            "Gini Coeff",
            "Top Domain %",
            "Single-Cat Domains %",
            "Records in Single-Cat %",
        ]

        for stats in all_stats:
            row = [stats["dataset"]]
            row.append(f"{stats['basic_stats']['n_records']:,}")
            row.append(str(stats["category_distribution"]["n_categories"]))

            if stats["domain_column"]:
                row.append(f"{stats['domain_distribution']['n_domains']:,}")
                row.append(f"{stats['domain_skew']['gini_coefficient']:.3f}")
                row.append(f"{stats['domain_skew']['top_1_domain_percent']:.1%}")
                if "error" not in stats["category_per_domain"]:
                    row.append(
                        f"{stats['category_per_domain']['summary']['single_category_domains_percent']:.1%}"
                    )
                    row.append(
                        f"{stats['category_per_domain']['summary']['records_in_single_category_domains_percent']:.1%}"
                    )
                else:
                    row.append("N/A")
                    row.append("N/A")
            else:
                row.extend(["N/A", "N/A", "N/A", "N/A", "N/A"])

            comparison_table.append(row)

        # Format table
        col_widths = [
            max(len(str(row[i])) for row in [headers] + comparison_table)
            for i in range(len(headers))
        ]

        # Header
        header_line = " | ".join(
            headers[i].ljust(col_widths[i]) for i in range(len(headers))
        )
        report_lines.append(header_line)
        report_lines.append("-" * len(header_line))

        # Data rows
        for row in comparison_table:
            data_line = " | ".join(
                str(row[i]).ljust(col_widths[i]) for i in range(len(row))
            )
            report_lines.append(data_line)

        report_lines.append("")

    report_text = "\n".join(report_lines)

    # Save to file if specified
    if output_file:
        with open(output_file, "w") as f:
            f.write(report_text)
        print(f"Report saved to {output_file}")

    return report_text


def main():
    parser = argparse.ArgumentParser(
        description="Calculate descriptive statistics for news classification datasets"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["uci", "huffpo", "recognasumm"],
        help="Datasets to analyze",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/dataset_descriptive_stats.txt",
        help="Output file for the report",
    )
    parser.add_argument(
        "--print-report", action="store_true", help="Print report to console"
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Analyze each dataset
    analyzer = DatasetDescriptiveStats()
    all_stats = []

    for dataset in args.datasets:
        try:
            stats = analyzer.analyze_dataset(dataset)
            all_stats.append(stats)
        except Exception as e:
            print(f"Error analyzing {dataset}: {e}")
            continue

    # Generate report
    if all_stats:
        report_text = format_stats_report(all_stats, args.output)

        if args.print_report:
            print(report_text)
    else:
        print("No datasets were successfully analyzed.")


if __name__ == "__main__":
    main()
