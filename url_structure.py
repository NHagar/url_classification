import re
from collections import Counter
from statistics import StatisticsError, mean, mode
from urllib.parse import urlparse


def skeleton(url):
    """
    Convert a URL path into a more descriptive skeleton by replacing
    variable segments with specific placeholders like {year}, {month}, {day},
    {id}, and {uuid}.
    """
    try:
        p = urlparse(url)
        path = p.path
    except (TypeError, ValueError):
        # Handle cases where the URL is not a string or is malformed
        return "/{invalid_url}"

    parts = path.split("/")
    skeleton_parts = []
    for seg in parts:
        if not seg:
            continue
        # 4-digit number, likely a year
        if re.fullmatch(r"\d{4}", seg):
            skeleton_parts.append("{year}")
        # 2-digit number, likely a month or day
        elif re.fullmatch(r"\d{1,2}", seg):
            # Basic context check: if previous was year, it's likely month/day
            if skeleton_parts and skeleton_parts[-1] == "{year}":
                if "{month}" not in skeleton_parts:
                    skeleton_parts.append("{month}")
                else:
                    skeleton_parts.append("{day}")
            else:
                skeleton_parts.append("{id_short}")
        # Numeric segment, likely an ID
        elif re.fullmatch(r"\d+", seg):
            skeleton_parts.append("{id}")
        # Hex or uuid-like segment
        elif re.fullmatch(r"[0-9a-fA-F]{8,}", seg.replace("-", "")):
            skeleton_parts.append("{uuid}")
        # Slug-like segment (letters, numbers, hyphens) ending with .html, .htm, etc.
        elif re.match(r"^[a-z0-9-]+", seg, re.IGNORECASE) and seg.endswith(
            (".html", ".htm", ".php", ".aspx", ".asp", ".story", ".page")
        ):
            # Extract the part before the extension
            base_name = seg.rsplit(".", 1)[0]
            extension = seg.rsplit(".", 1)[1]
            # Further check if the base_name is just an ID
            if re.fullmatch(r"\d+", base_name):
                skeleton_parts.append("{id}." + extension)
            else:
                skeleton_parts.append("{slug}." + extension)
        else:
            skeleton_parts.append(seg)
    return "/" + "/".join(skeleton_parts)


def get_regex_counts(url_list):
    """
    Classify URLs according to a hierarchical set of regex rules.
    Returns a dict of counts.
    """
    rules = {
        "date_based": re.compile(r"/\d{4}/\d{1,2}/\d{1,2}/"),
        "numeric_id_in_path": re.compile(r"/\d{4,}/"),  # ID with at least 4 digits
        "ends_with_id.html": re.compile(r"/\d+\.html$"),
        "ends_with_slug": re.compile(r"/[a-z0-9-]+\.html$|[a-z0-9-]+\.htm$"),
        "file_based_aspx_dll": re.compile(r"\.(aspx|dll)$"),
        "root_or_index": re.compile(r"^/(index\.(html|php|asp))?$|^/$"),
    }
    counts = {name: 0 for name in rules}
    counts["other"] = 0

    for u in url_list:
        try:
            path = urlparse(u).path
        except (TypeError, ValueError):
            counts["other"] += 1
            continue

        matched = False
        for name, pat in rules.items():
            if pat.search(path):
                counts[name] += 1
                matched = True
                break
        if not matched:
            counts["other"] += 1
    return counts


def get_depth_stats(url_list):
    """
    Return detailed statistics on URL path segments, including segment type
    analysis at each depth.
    """
    depths = []
    segment_types_by_depth = {}  # {1: [type1, type2...], 2: [...]}

    for u in url_list:
        try:
            parts = [seg for seg in urlparse(u).path.split("/") if seg]
        except (TypeError, ValueError):
            continue

        depth = len(parts)
        depths.append(depth)

        for i, seg in enumerate(parts):
            d = i + 1
            if d not in segment_types_by_depth:
                segment_types_by_depth[d] = []

            if re.fullmatch(r"\d+", seg):
                segment_types_by_depth[d].append("numeric")
            elif re.fullmatch(r"[a-zA-Z-]+", seg):
                segment_types_by_depth[d].append("alpha")
            else:
                segment_types_by_depth[d].append("alphanumeric/other")

    def stats(lst):
        if not lst:
            return {"min": 0, "mean": 0, "max": 0, "mode": 0}
        try:
            mode_val = mode(lst)
        except StatisticsError:
            mode_val = "N/A"  # No unique mode
        return {"min": min(lst), "mean": mean(lst), "max": max(lst), "mode": mode_val}

    depth_type_analysis = {}
    for depth, types in segment_types_by_depth.items():
        depth_type_analysis[depth] = dict(Counter(types))

    return {
        "path_depth_stats": stats(depths),
        "segment_type_by_depth": depth_type_analysis,
    }


def get_common_segments(url_list, top_n=5):
    """
    Identifies the most common path segments at each depth level.
    """
    segments_by_depth = {}
    for u in url_list:
        try:
            parts = [seg for seg in urlparse(u).path.split("/") if seg]
        except (TypeError, ValueError):
            continue

        for i, seg in enumerate(parts):
            depth = i + 1
            if re.fullmatch(r"\d{4,}", seg):  # Ignore long numeric IDs
                continue
            if depth not in segments_by_depth:
                segments_by_depth[depth] = []
            segments_by_depth[depth].append(seg)

    common_segments = {}
    for depth, segments in segments_by_depth.items():
        common_segments[depth] = Counter(segments).most_common(top_n)

    return common_segments


def analyze_urls(url_list, top_skeletons=10):
    """
    Perform a comprehensive URL structure audit using the enhanced functions.
    """
    # Filter out invalid or non-string URLs before processing
    valid_urls = [u for u in url_list if isinstance(u, str) and u.strip()]

    # To prevent excessive memory usage on very large datasets,
    # you might consider sampling for some analyses.
    # sample_size = 50000
    # if len(valid_urls) > sample_size:
    #     report_urls = random.sample(valid_urls, sample_size)
    # else:
    #     report_urls = valid_urls

    report_urls = valid_urls  # Using all valid URLs for this case

    skeleton_counts = Counter(skeleton(u) for u in report_urls)

    return {
        "skeletons": skeleton_counts.most_common(top_skeletons),
        "regex_counts": get_regex_counts(report_urls),
        "depth_stats": get_depth_stats(report_urls),
        "common_segments": get_common_segments(report_urls, top_n=5),
    }


# --- Example Usage ---
# Ensure you have a dataframe 'df' with a 'URL' column before running this.
# Example for the first dataset:
# report_recog = analyze_urls(df["URL"].tolist(), top_skeletons=10)
# print("--- Analysis for recognasumm.parquet ---")
# print("\nTop skeletons:", report_recog['skeletons'])
# print("\nRegex classification:", report_recog['regex_counts'])
# print("\nDepth stats:", report_recog['depth_stats'])
# print("\nCommon Segments by Depth:", report_recog['common_segments'])

# Example for the second dataset:
# (Load the second parquet file into df first)
# report_uci = analyze_urls(df["URL"].tolist(), top_skeletons=10)
# print("\n--- Analysis for uci_categories.parquet ---")
# print("\nTop skeletons:", report_uci['skeletons'])
# print("\nRegex classification:", report_uci['regex_counts'])
# print("\nDepth stats:", report_uci['depth_stats'])
# print("\nCommon Segments by Depth:", report_uci['common_segments'])
