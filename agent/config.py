import os

def get_strategy_weights():
    """Retrieve cost and comfort weights from environment variables or defaults (0.5 / 0.5)."""
    cost_w = float(os.environ.get("COST_WEIGHT", "0.5"))
    comfort_w = float(os.environ.get("COMFORT_WEIGHT", "0.5"))
    return cost_w, comfort_w

def get_approval_thresholds():
    """Retrieve thresholds triggering human approval requirement."""
    kw_threshold = float(os.environ.get("APPROVAL_REQUIRED_KW_REDUCTION", "30.0"))
    temp_threshold = float(os.environ.get("APPROVAL_REQUIRED_TEMP_DEVIATION", "23.5"))
    cost_threshold = float(os.environ.get("APPROVAL_REQUIRED_COST_SAVINGS", "40.0"))
    return {
        "kw_reduction": kw_threshold,
        "max_temp": temp_threshold,
        "cost_savings": cost_threshold,
    }
