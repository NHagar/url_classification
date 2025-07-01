import re
from typing import Any, Dict, List
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from url_classification.dataset_loading import load_data


class URLStructureAnalyzer:
    """Analyzes URL structure patterns and their relationship to model performance"""

    def __init__(self):
        self.url_patterns = {}
        self.structure_categories = {}

    def categorize_url_structure(self, url: str) -> Dict[str, Any]:
        """Categorize a URL by its structural characteristics"""
        parsed = urlparse(url)
        path = parsed.path

        # Basic structure analysis
        path_segments = [seg for seg in path.split("/") if seg]
        num_segments = len(path_segments)

        # Pattern matching
        patterns = {
            "has_numbers": bool(re.search(r"\d", path)),
            "has_dates": bool(
                re.search(r"\d{4}[/-]\d{2}[/-]\d{2}|\d{2}[/-]\d{2}[/-]\d{4}", path)
            ),
            "has_year": bool(re.search(r"\b20\d{2}\b|\b19\d{2}\b", path)),
            "has_month": bool(
                re.search(r"\b(01|02|03|04|05|06|07|08|09|10|11|12)\b", path)
            ),
            "has_underscores": "_" in path,
            "has_hyphens": "-" in path,
            "has_file_extension": bool(re.search(r"\.[a-z]{2,4}$", path)),
            "is_id_heavy": self._is_id_heavy(path),
            "is_semantic": self._is_semantic(path),
            "has_category_indicators": self._has_category_indicators(path),
            "path_length": len(path),
            "num_segments": num_segments,
            "avg_segment_length": np.mean([len(seg) for seg in path_segments])
            if path_segments
            else 0,
        }

        # Classify URL type
        url_type = self._classify_url_type(patterns, path_segments)
        patterns["url_type"] = url_type

        return patterns

    def _is_id_heavy(self, path: str) -> bool:
        """Check if URL is heavily ID-based (more than 50% numeric characters)"""
        alpha_chars = len(re.findall(r"[a-zA-Z]", path))
        numeric_chars = len(re.findall(r"\d", path))
        total_meaningful = alpha_chars + numeric_chars

        if total_meaningful == 0:
            return False
        return numeric_chars / total_meaningful > 0.5

    def _is_semantic(self, path: str) -> bool:
        """Check if URL contains semantic/meaningful words"""
        # Common semantic indicators
        semantic_patterns = [
            r"\b(article|story|news|post|blog|page)\b",
            r"\b(politics|business|sports|tech|entertainment|world|health|science)\b",
            r"\b(breaking|latest|update|analysis|opinion|review)\b",
        ]

        path_lower = path.lower()
        return any(re.search(pattern, path_lower) for pattern in semantic_patterns)

    def _has_category_indicators(self, path: str) -> bool:
        """Check if URL contains category-like structure"""
        segments = [seg for seg in path.split("/") if seg]
        if len(segments) < 2:
            return False

        # Look for category-like first segments
        category_patterns = [
            r"^(news|sports|politics|business|tech|entertainment|world|health|science)$",
            r"^(breaking|latest|featured|trending)$",
            r"^(articles|stories|posts|blogs)$",
        ]

        if segments:
            first_segment = segments[0].lower()
            return any(
                re.match(pattern, first_segment) for pattern in category_patterns
            )

        return False

    def _classify_url_type(self, patterns: Dict, segments: List[str]) -> str:
        """Classify URL into high-level types"""
        if patterns["is_id_heavy"] and not patterns["is_semantic"]:
            return "id_based"
        elif patterns["is_semantic"] and patterns["has_category_indicators"]:
            return "semantic_structured"
        elif patterns["is_semantic"]:
            return "semantic_unstructured"
        elif patterns["has_dates"] or patterns["has_year"]:
            return "date_based"
        elif patterns["num_segments"] <= 2 and patterns["path_length"] < 50:
            return "simple"
        else:
            return "mixed"

    def analyze_dataset_structure(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """Analyze URL structure patterns across the dataset"""
        print("Analyzing URL structures...")

        # Analyze each URL
        structure_data = []
        for idx, row in df.iterrows():
            url = row["URL"]
            structure = self.categorize_url_structure(url)
            structure["url"] = url
            structure["domain"] = urlparse(url).netloc
            structure["category"] = row["y"]
            structure["publisher"] = row["PUBLISHER"]
            structure_data.append(structure)

        structure_df = pd.DataFrame(structure_data)

        # Calculate summary statistics
        analysis_results = {
            "total_urls": len(structure_df),
            "unique_domains": structure_df["domain"].nunique(),
            "url_type_distribution": structure_df["url_type"].value_counts().to_dict(),
            "structure_stats": {
                "avg_path_length": structure_df["path_length"].mean(),
                "avg_segments": structure_df["num_segments"].mean(),
                "pct_with_numbers": (
                    structure_df["has_numbers"].sum() / len(structure_df)
                )
                * 100,
                "pct_semantic": (structure_df["is_semantic"].sum() / len(structure_df))
                * 100,
                "pct_id_heavy": (structure_df["is_id_heavy"].sum() / len(structure_df))
                * 100,
                "pct_with_categories": (
                    structure_df["has_category_indicators"].sum() / len(structure_df)
                )
                * 100,
            },
            "variance_metrics": {
                "path_length_std": structure_df["path_length"].std(),
                "segments_std": structure_df["num_segments"].std(),
                "avg_segment_length_std": structure_df["avg_segment_length"].std(),
            },
        }

        return structure_df, analysis_results

    def link_performance_data(
        self, structure_df: pd.DataFrame, performance_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Link URL structure data with model performance results"""
        print("Linking structure data with performance results...")

        # The performance CSV 'domain' field corresponds to PUBLISHER in UCI data
        print("Note: Performance 'domain' field corresponds to PUBLISHER in UCI data")

        # Use performance data directly (already filtered to specific model/feature)
        perf_by_publisher = performance_df[
            ["domain", "accuracy", "precision", "recall", "f1", "support"]
        ].copy()
        perf_by_publisher = perf_by_publisher.rename(columns={"domain": "publisher"})

        # Calculate structure statistics by publisher
        structure_by_publisher = (
            structure_df.groupby("publisher")
            .agg(
                {
                    "path_length": ["mean", "std"],
                    "num_segments": ["mean", "std"],
                    "avg_segment_length": ["mean", "std"],
                    "has_numbers": "mean",
                    "is_semantic": "mean",
                    "is_id_heavy": "mean",
                    "has_category_indicators": "mean",
                }
            )
            .reset_index()
        )

        # Flatten column names
        structure_by_publisher.columns = [
            "publisher" if col[0] == "publisher" else f"{col[0]}_{col[1]}"
            for col in structure_by_publisher.columns
        ]

        # Get dominant URL type per publisher
        url_type_by_publisher = (
            structure_df.groupby("publisher")["url_type"]
            .apply(lambda x: x.value_counts().index[0])
            .reset_index()
        )
        url_type_by_publisher.columns = ["publisher", "dominant_url_type"]

        # Debug the join
        print(
            f"Performance data has {len(perf_by_publisher)} records for {perf_by_publisher['publisher'].nunique()} unique publishers"
        )
        print(
            f"Structure data has {len(structure_by_publisher)} records for {structure_by_publisher['publisher'].nunique()} unique publishers"
        )

        # Check publisher overlap
        perf_publishers = set(perf_by_publisher["publisher"].unique())
        struct_publishers = set(structure_by_publisher["publisher"].unique())
        overlap = perf_publishers.intersection(struct_publishers)
        print(
            f"Publisher overlap: {len(overlap)} out of {len(perf_publishers)} performance publishers and {len(struct_publishers)} structure publishers"
        )

        # Merge all data
        combined_df = perf_by_publisher.merge(
            structure_by_publisher, on="publisher", how="inner"
        )
        combined_df = combined_df.merge(
            url_type_by_publisher, on="publisher", how="inner"
        )

        return combined_df

    def analyze_performance_by_structure(
        self, combined_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyze how performance varies by URL structure characteristics"""
        print("Analyzing performance by URL structure...")

        results = {}

        # Performance by URL type
        perf_by_type = (
            combined_df.groupby("dominant_url_type")
            .agg(
                {
                    "accuracy": ["mean", "std", "count"],
                    "f1": ["mean", "std", "count"],
                    "support": "sum",
                }
            )
            .round(4)
        )
        results["performance_by_url_type"] = perf_by_type

        # Correlations between structure metrics and performance
        structure_cols = [
            col
            for col in combined_df.columns
            if any(
                metric in col
                for metric in [
                    "path_length",
                    "num_segments",
                    "avg_segment_length",
                    "has_",
                    "is_",
                ]
            )
        ]
        performance_cols = ["accuracy", "precision", "recall", "f1"]

        correlations = {}
        for perf_col in performance_cols:
            corr_data = {}
            for struct_col in structure_cols:
                if combined_df[struct_col].dtype in ["float64", "int64"]:
                    corr = combined_df[perf_col].corr(combined_df[struct_col])
                    if not np.isnan(corr):
                        corr_data[struct_col] = corr
            correlations[perf_col] = corr_data

        results["structure_performance_correlations"] = correlations

        # High vs low performing domains analysis
        median_f1 = combined_df["f1"].median()
        high_perf = combined_df[combined_df["f1"] >= median_f1]
        low_perf = combined_df[combined_df["f1"] < median_f1]

        comparison = {}
        for col in structure_cols:
            if combined_df[col].dtype in ["float64", "int64"]:
                comparison[col] = {
                    "high_performance_mean": high_perf[col].mean(),
                    "low_performance_mean": low_perf[col].mean(),
                    "difference": high_perf[col].mean() - low_perf[col].mean(),
                }

        results["high_vs_low_performance_comparison"] = comparison

        return results

    def generate_visualizations(
        self,
        structure_df: pd.DataFrame,
        combined_df: pd.DataFrame,
        analysis_results: Dict,
    ) -> None:
        """Generate visualizations for the analysis"""
        print("Generating visualizations...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("URL Structure Analysis", fontsize=16)

        # 1. URL Type Distribution
        url_type_counts = structure_df["url_type"].value_counts()
        axes[0, 0].pie(
            url_type_counts.values, labels=url_type_counts.index, autopct="%1.1f%%"
        )
        axes[0, 0].set_title("URL Type Distribution")

        # 2. Path Length Distribution
        axes[0, 1].hist(structure_df["path_length"], bins=30, alpha=0.7)
        axes[0, 1].set_xlabel("Path Length")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].set_title("URL Path Length Distribution")

        # 3. Performance by URL Type
        perf_by_type = (
            combined_df.groupby("dominant_url_type")["f1"]
            .mean()
            .sort_values(ascending=True)
        )
        axes[0, 2].barh(range(len(perf_by_type)), perf_by_type.values)
        axes[0, 2].set_yticks(range(len(perf_by_type)))
        axes[0, 2].set_yticklabels(perf_by_type.index)
        axes[0, 2].set_xlabel("Mean F1 Score")
        axes[0, 2].set_title("Performance by URL Type")

        # 4. Structure vs Performance Correlation Heatmap
        struct_cols = [
            "path_length_mean",
            "num_segments_mean",
            "has_numbers_mean",
            "is_semantic_mean",
            "is_id_heavy_mean",
        ]
        perf_cols = ["accuracy", "f1"]

        # Filter to only existing columns
        existing_struct_cols = [
            col for col in struct_cols if col in combined_df.columns
        ]
        existing_perf_cols = [col for col in perf_cols if col in combined_df.columns]

        if existing_struct_cols and existing_perf_cols:
            corr_matrix = combined_df[existing_struct_cols + existing_perf_cols].corr()
            im = axes[1, 0].imshow(
                corr_matrix.loc[existing_struct_cols, existing_perf_cols],
                cmap="RdBu_r",
                vmin=-1,
                vmax=1,
            )
            axes[1, 0].set_xticks(range(len(existing_perf_cols)))
            axes[1, 0].set_xticklabels(existing_perf_cols, rotation=45)
            axes[1, 0].set_yticks(range(len(existing_struct_cols)))
            axes[1, 0].set_yticklabels(
                [col.replace("_mean", "") for col in existing_struct_cols]
            )
            axes[1, 0].set_title("Structure-Performance Correlations")
            plt.colorbar(im, ax=axes[1, 0])
        else:
            axes[1, 0].text(
                0.5, 0.5, "No correlation data available", ha="center", va="center"
            )
            axes[1, 0].set_title("Structure-Performance Correlations")

        # 5. Semantic vs Non-semantic Performance
        semantic_perf = combined_df[combined_df["is_semantic_mean"] > 0.5]["f1"]
        non_semantic_perf = combined_df[combined_df["is_semantic_mean"] <= 0.5]["f1"]

        axes[1, 1].boxplot(
            [semantic_perf, non_semantic_perf], labels=["Semantic", "Non-semantic"]
        )
        axes[1, 1].set_ylabel("F1 Score")
        axes[1, 1].set_title("Performance: Semantic vs Non-semantic URLs")

        # 6. Support vs Performance
        axes[1, 2].scatter(combined_df["support"], combined_df["f1"], alpha=0.6)
        axes[1, 2].set_xlabel("Support (Number of Examples)")
        axes[1, 2].set_ylabel("F1 Score")
        axes[1, 2].set_title("Support vs Performance")

        plt.tight_layout()
        plt.savefig(
            "data/processed/url_structure_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.show()

    def generate_report(
        self,
        structure_df: pd.DataFrame,
        analysis_results: Dict,
        performance_analysis: Dict,
    ) -> str:
        """Generate a comprehensive analysis report"""
        report = []
        report.append("# URL Structure Analysis Report")
        report.append("=" * 50)
        report.append("")

        # Dataset Overview
        report.append("## Dataset Overview")
        report.append(f"- Total URLs analyzed: {analysis_results['total_urls']:,}")
        report.append(f"- Unique domains: {analysis_results['unique_domains']:,}")
        report.append("")

        # URL Structure Variance
        report.append("## URL Structure Variance")
        variance = analysis_results["variance_metrics"]
        report.append(
            f"- Path length standard deviation: {variance['path_length_std']:.2f}"
        )
        report.append(
            f"- Number of segments standard deviation: {variance['segments_std']:.2f}"
        )
        report.append(
            f"- Average segment length standard deviation: {variance['avg_segment_length_std']:.2f}"
        )
        report.append("")

        # URL Type Distribution
        report.append("## URL Type Distribution")
        for url_type, count in analysis_results["url_type_distribution"].items():
            pct = (count / analysis_results["total_urls"]) * 100
            report.append(f"- {url_type}: {count:,} ({pct:.1f}%)")
        report.append("")

        # Structure Characteristics
        report.append("## Structure Characteristics")
        stats = analysis_results["structure_stats"]
        report.append(
            f"- Average path length: {stats['avg_path_length']:.1f} characters"
        )
        report.append(f"- Average number of segments: {stats['avg_segments']:.1f}")
        report.append(f"- URLs with numbers: {stats['pct_with_numbers']:.1f}%")
        report.append(f"- Semantic URLs: {stats['pct_semantic']:.1f}%")
        report.append(f"- ID-heavy URLs: {stats['pct_id_heavy']:.1f}%")
        report.append(
            f"- URLs with category indicators: {stats['pct_with_categories']:.1f}%"
        )
        report.append("")

        # Performance by URL Type
        report.append("## Performance by URL Type")
        perf_by_type = performance_analysis["performance_by_url_type"]
        for url_type in perf_by_type.index:
            f1_mean = perf_by_type.loc[url_type, ("f1", "mean")]
            f1_std = perf_by_type.loc[url_type, ("f1", "std")]
            count = perf_by_type.loc[url_type, ("f1", "count")]
            report.append(
                f"- {url_type}: F1 = {f1_mean:.3f} ± {f1_std:.3f} (n={count})"
            )
        report.append("")

        # Key Correlations
        report.append("## Key Structure-Performance Correlations")
        f1_corrs = performance_analysis["structure_performance_correlations"]["f1"]
        sorted_corrs = sorted(f1_corrs.items(), key=lambda x: abs(x[1]), reverse=True)[
            :5
        ]

        for feature, corr in sorted_corrs:
            report.append(f"- {feature}: {corr:.3f}")
        report.append("")

        # High vs Low Performance Comparison
        report.append("## High vs Low Performance Domain Comparison")
        comparison = performance_analysis["high_vs_low_performance_comparison"]
        for feature, data in list(comparison.items())[:5]:
            diff = data["difference"]
            report.append(f"- {feature}: {diff:.3f} difference (high - low performing)")
        report.append("")

        # Conclusions
        report.append("## Key Findings")
        report.append("1. URL Structure Variance:")
        report.append(
            f"   - High variance in path lengths (std: {variance['path_length_std']:.1f})"
        )
        report.append(
            f"   - Moderate variance in segment counts (std: {variance['segments_std']:.1f})"
        )
        report.append("")

        # Determine most common URL type
        most_common_type = max(
            analysis_results["url_type_distribution"].items(), key=lambda x: x[1]
        )
        report.append(
            f"2. Dominant URL Pattern: {most_common_type[0]} ({(most_common_type[1] / analysis_results['total_urls'] * 100):.1f}%)"
        )
        report.append("")

        report.append("3. Performance Insights:")
        # Find best performing URL type
        best_type_idx = perf_by_type[("f1", "mean")].idxmax()
        best_f1 = perf_by_type.loc[best_type_idx, ("f1", "mean")]
        report.append(
            f"   - Best performing URL type: {best_type_idx} (F1: {best_f1:.3f})"
        )

        # Check semantic vs non-semantic
        semantic_pct = stats["pct_semantic"]
        report.append(f"   - {semantic_pct:.1f}% of URLs contain semantic information")
        report.append("")

        return "\n".join(report)


def main():
    """Main analysis function"""
    print("Starting URL Structure Analysis...")

    # Initialize analyzer
    analyzer = URLStructureAnalyzer()

    # Load UCI dataset
    print("Loading UCI dataset...")
    df, _ = load_data("uci")
    print(f"Loaded {len(df)} URLs from UCI dataset")

    # Analyze URL structures
    structure_df, analysis_results = analyzer.analyze_dataset_structure(df)

    # Load performance data
    print("Loading performance data...")
    performance_df = pd.read_csv(
        "data/processed/per_domain_evaluation_results_20250628_153349.csv"
    )

    # Filter to DistilBERT with url_path_raw features only
    performance_df = performance_df[
        (performance_df["model"] == "distilbert")
        & (performance_df["feature"] == "url_path_raw")
    ]
    print(f"Filtered to {len(performance_df)} DistilBERT url_path_raw results")

    # Link structure and performance data
    combined_df = analyzer.link_performance_data(structure_df, performance_df)
    print(f"Successfully linked data for {len(combined_df)} publishers")

    # Analyze performance by structure
    performance_analysis = analyzer.analyze_performance_by_structure(combined_df)

    # Generate visualizations
    analyzer.generate_visualizations(structure_df, combined_df, analysis_results)

    # Generate report
    report = analyzer.generate_report(
        structure_df, analysis_results, performance_analysis
    )

    # Save report
    with open("data/processed/url_structure_analysis_report.txt", "w") as f:
        f.write(report)

    # Save detailed data
    structure_df.to_csv("data/processed/url_structure_details.csv", index=False)
    combined_df.to_csv("data/processed/structure_performance_combined.csv", index=False)

    print("\nAnalysis complete!")
    print("Files generated:")
    print("- url_structure_analysis.png (visualizations)")
    print("- url_structure_analysis_report.txt (summary report)")
    print("- url_structure_details.csv (detailed URL analysis)")
    print("- structure_performance_combined.csv (combined data)")
    print("\n" + "=" * 50)
    print("SUMMARY REPORT")
    print("=" * 50)
    print(report)


if __name__ == "__main__":
    main()
