def compute_global_score(
    lighting: float,
    sharpness: float,
    composition: float,
    color: float,
    clutter: float
) -> float:
    """
    Computes the global visual quality score by combining
    individual metric scores using a weighted aggregation.

    All input scores are expected to be normalized between 0 and 100.

    Returns:
        float: global visual quality score (0–100)
    """

    # --------------------------------------------------
    # 1. Define metric weights (Version 1)
    # --------------------------------------------------
    # These weights reflect perceived importance in
    # real estate photography.
    weights = {
        "lighting": 0.25,
        "sharpness": 0.20,
        "composition": 0.20,
        "color": 0.15,
        "clutter": 0.20
    }

    # --------------------------------------------------
    # 2. Compute weighted sum
    # --------------------------------------------------
    global_score = (
        lighting * weights["lighting"] +
        sharpness * weights["sharpness"] +
        composition * weights["composition"] +
        color * weights["color"] +
        clutter * weights["clutter"]
    )

    # --------------------------------------------------
    # 3. Ensure valid score range
    # --------------------------------------------------
    global_score = max(0.0, min(100.0, global_score))

    return global_score
