FEATURE_GROUPS_BY_PREFIX = (
    ("color", ("hsv", "lab", "ab")),
    ("texture", ("glcm", "lbp")),
    ("shape", ("shape",)),
    ("gradient", ("grad",)),
)


def describe_feature(feature_name):
    if not isinstance(feature_name, str):
        raise ValueError(f"Unknown feature name: {feature_name!r}")
    name_parts = feature_name.split("_", 2)
    if len(name_parts) == 3:
        region, group_prefix, details = name_parts
        if region in ("lesion", "leaf") and details:
            for group_name, group_prefixes in FEATURE_GROUPS_BY_PREFIX:
                if group_prefix in group_prefixes:
                    return region, group_name
    raise ValueError(f"Unknown feature name: {feature_name!r}")
