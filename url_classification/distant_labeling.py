"""
Distant labeling implementation for automatic weak supervision using URL patterns.

The algorithm:
1. Check if news site uses URL-based category tagging (by looking at URL patterns)
2. If YES: Classify articles based on URL sections using provided category mappings
3. Pool all labeled articles into single database with distant labels
4. Use labeled data to train supervised ML classifier
5. Apply classifier to unclassified articles (including those from non-URL-tagged sites)
6. All articles classified for analysis
"""

import os
import re
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import pandas as pd


class DistantLabelingClassifier:
    """
    Implements distant labeling algorithm for news article classification using URL patterns
    """

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.url_mapping = self._load_url_mapping()
        self.valid_categories = self._get_valid_categories()

    def _load_url_mapping(self) -> Optional[Dict[str, str]]:
        """Load URL slug to category mapping from CSV file"""
        mapping_files = {
            "uci": "data/uci_categories_attributed.csv",
            "recognasumm": "data/recognasumm_categories_attributed.csv",
            "huffpo": None,  # HuffPost doesn't use URL-based classification
        }

        mapping_file = mapping_files.get(self.dataset_name)
        if not mapping_file or not os.path.exists(mapping_file):
            return None

        df = pd.read_csv(mapping_file)
        # Create mapping from slug to category, filtering out empty categories
        mapping = {}
        for _, row in df.iterrows():
            slug = str(row["slug"]).strip()
            category = str(row["category"]).strip()
            if slug and category and category != "nan":
                mapping[slug] = category

        return mapping

    def _get_valid_categories(self) -> List[str]:
        """Get list of valid categories that can be used for distant labeling"""
        if not self.url_mapping:
            return []

        # Get unique categories from the mapping, filtering out empty ones
        categories = set()
        for category in self.url_mapping.values():
            if category and category.strip():
                categories.add(category.strip())

        return sorted(list(categories))

    def _extract_url_slug(self, url: str) -> List[str]:
        """Extract potential category slugs from URL path"""
        try:
            parsed = urlparse(url)
            path = parsed.path.strip("/")

            # Split path into segments and clean them
            segments = []
            for segment in path.split("/"):
                segment = segment.strip()
                if segment:
                    # Remove common noise patterns
                    segment = re.sub(r"^\d{4}$", "", segment)  # Remove years
                    segment = re.sub(r"^\d+$", "", segment)  # Remove pure numbers
                    if len(segment) > 2:  # Keep meaningful segments
                        segments.append(segment)

            return segments
        except Exception:
            return []

    def uses_url_based_categorization(self, df: pd.DataFrame) -> bool:
        """
        Determine if the news site uses URL-based categorization
        by checking if we can find category patterns in URLs
        """
        if not self.url_mapping or not self.valid_categories:
            return False

        # Sample URLs to check for category patterns
        sample_size = min(1000, len(df))
        sample_urls = df["URL"].sample(n=sample_size, random_state=42)

        category_matches = 0
        total_urls_checked = 0

        for url in sample_urls:
            total_urls_checked += 1
            slugs = self._extract_url_slug(url)
            for slug in slugs:
                if slug in self.url_mapping:
                    category_matches += 1
                    break  # Only count first match per URL

        # If we find category patterns in at least 10% of URLs
        if total_urls_checked > 50:  # Need sufficient sample
            category_ratio = category_matches / total_urls_checked
            return category_ratio >= 0.10

        return False

    def classify_by_url_sections(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify articles based on URL sections using the category mapping
        """
        df = df.copy()
        df["distant_label"] = "unknown"
        df["url_category"] = None
        df["matched_slug"] = None

        if not self.url_mapping:
            return df

        for idx, row in df.iterrows():
            url = row["URL"]
            slugs = self._extract_url_slug(url)

            # Find first matching slug
            matched_category = None
            matched_slug = None

            for slug in slugs:
                if slug in self.url_mapping:
                    matched_category = self.url_mapping[slug]
                    matched_slug = slug
                    break

            if matched_category:
                df.at[idx, "url_category"] = matched_category
                df.at[idx, "matched_slug"] = matched_slug
                df.at[idx, "distant_label"] = (
                    matched_category  # Use the actual category as the label
                )

        return df

    def prepare_distant_training_data(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare training data following the distant labeling algorithm

        Returns:
            - labeled_data: Articles with category labels from URL patterns
            - unlabeled_data: Articles that couldn't be classified by URL patterns
        """
        uses_url_tagging = self.uses_url_based_categorization(df)
        print(
            f"Dataset {self.dataset_name} uses URL-based categorization: {uses_url_tagging}"
        )

        if uses_url_tagging:
            # Classify based on URL sections
            df_classified = self.classify_by_url_sections(df)

            # Split into labeled and unlabeled
            labeled_data = df_classified[
                df_classified["distant_label"] != "unknown"
            ].copy()
            unlabeled_data = df_classified[
                df_classified["distant_label"] == "unknown"
            ].copy()

            print("URL-based classification results:")
            if len(labeled_data) > 0:
                # Show distribution of categories
                category_counts = labeled_data["distant_label"].value_counts()
                for category, count in category_counts.items():
                    print(f"  - {category}: {count}")
            print(f"  - Unknown: {len(unlabeled_data)}")

        else:
            # No URL-based classification possible
            df["distant_label"] = "unknown"
            df["url_category"] = None
            df["matched_slug"] = None

            labeled_data = pd.DataFrame()  # Empty - no URL-based labels available
            unlabeled_data = df.copy()

            print(
                f"No URL-based classification available - all {len(unlabeled_data)} articles unlabeled"
            )

        return labeled_data, unlabeled_data

    def create_category_labels(self, labeled_data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare category labels for training (no transformation needed since we use actual categories)
        """
        if len(labeled_data) == 0:
            return labeled_data

        labeled_data = labeled_data.copy()

        # The distant_label already contains the actual category, so we just use it as is
        labeled_data["category_label"] = labeled_data["distant_label"]

        # Remove any rows with missing labels
        labeled_data = labeled_data.dropna(subset=["category_label"])

        return labeled_data

    def get_feature_text(self, df: pd.DataFrame, feature_name: str) -> pd.Series:
        """
        Extract feature text based on feature name
        This should match the feature extraction logic from model_config.py
        """
        feature_extractors = {
            "title_subtitle": {
                "huffpo": lambda df: df["headline"] + " " + df["short_description"],
                "uci": lambda df: df["TITLE"],
                "recognasumm": lambda df: df["Titulo"] + " " + df["Subtitulo"],
            },
            "title": {
                "huffpo": lambda df: df["headline"],
                "uci": lambda df: df["TITLE"],
                "recognasumm": lambda df: df["Titulo"],
            },
            "url_path_cleaned": {
                "huffpo": lambda df: df["x"],
                "uci": lambda df: df["x"],
                "recognasumm": lambda df: df["x"],
            },
            "url_raw": {
                "huffpo": lambda df: df["link"],
                "uci": lambda df: df["URL"],
                "recognasumm": lambda df: df["URL"],
            },
        }

        if feature_name in feature_extractors:
            extractor = feature_extractors[feature_name].get(self.dataset_name)
            if extractor:
                return extractor(df)

        # Fallback to title if available
        if self.dataset_name == "huffpo" and "headline" in df.columns:
            return df["headline"]
        elif self.dataset_name == "uci" and "TITLE" in df.columns:
            return df["TITLE"]
        elif self.dataset_name == "recognasumm" and "Titulo" in df.columns:
            return df["Titulo"]

        raise ValueError(
            f"Cannot extract feature '{feature_name}' for dataset '{self.dataset_name}'"
        )


def apply_distant_labeling_algorithm(
    df: pd.DataFrame, dataset_name: str, feature_name: str = "title"
) -> Tuple[Optional[pd.DataFrame], pd.DataFrame, DistantLabelingClassifier]:
    """
    Apply the complete distant labeling algorithm

    Returns:
        - training_data: Labeled data for training category classifier (None if no URL-based labels)
        - all_data: All data to be classified (both labeled and unlabeled)
        - classifier: The distant labeling classifier instance
    """

    classifier = DistantLabelingClassifier(dataset_name)

    # Step 1 & 2: Check for URL-based categorization and classify accordingly
    labeled_data, unlabeled_data = classifier.prepare_distant_training_data(df)

    # Step 3: Pool labeled data and prepare category labels for training
    training_data = None
    if len(labeled_data) > 0:
        training_data = classifier.create_category_labels(labeled_data)

        # Check if we have sufficient examples for training
        if len(training_data) > 0:
            category_counts = training_data["category_label"].value_counts()

            print("Training data prepared:")
            for category, count in category_counts.items():
                print(f"  - {category}: {count} examples")

            # Warn if we have very few examples for some categories
            min_examples = category_counts.min()
            if min_examples < 10:
                print(
                    f"Warning: Some categories have very few examples (min: {min_examples}) - distant labeling may not work well"
                )
        else:
            print("Warning: No valid training examples found after processing")

    # Step 4: Combine labeled and unlabeled data for final classification
    all_data = (
        pd.concat([labeled_data, unlabeled_data], ignore_index=True)
        if len(labeled_data) > 0
        else unlabeled_data
    )

    return training_data, all_data, classifier
