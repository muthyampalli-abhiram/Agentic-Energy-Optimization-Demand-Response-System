"""
CLI Helper for Human-in-the-Loop (HITL) Demand Response Approval.
Displays pending energy optimization proposal, top-ranked strategy, alternative options,
and processes facility manager decision.
"""

from typing import Dict, Any, Tuple, Optional


def format_approval_proposal(proposal: Dict[str, Any]) -> str:
    """Format human-readable approval proposal string."""
    top = proposal.get("top_strategy", {})
    alt = proposal.get("alternative_strategy", {})
    triggers = proposal.get("threshold_triggers", [])

    sim_top = top.get("simulation_result", {})
    sim_alt = alt.get("simulation_result", {}) if alt else {}

    out = []
    out.append("==========================================================================================")
    out.append("                  HUMAN-IN-THE-LOOP (HITL) APPROVAL REQUIRED                             ")
    out.append("==========================================================================================")
    out.append(f"Notice: High-impact demand response action triggered approval gates:")
    for t in triggers:
        out.append(f"  * {t}")

    out.append("\n------------------------------------------------------------------------------------------")
    out.append(f"[OPTION 1 (RECOMMENDED)]: Top-Ranked Strategy: {top.get('strategy_id')} - {top.get('strategy_name')}")
    out.append(f"  Score              : {top.get('score', 0.0):.3f}")
    out.append(f"  Rationale          : {top.get('rationale')}")
    out.append(f"  Proposed Actions   : {top.get('actions')}")
    out.append(f"  1-Hour Cost Savings: ${sim_top.get('projected_cost_savings_usd', 0.0):.2f} ({sim_top.get('projected_kwh_reduction', 0.0)} kWh saved)")
    out.append(f"  Peak Load Red. %   : {sim_top.get('window_energy_reduction_pct', 0.0)}% window energy ({sim_top.get('instantaneous_peak_kw_reduction_pct', 0.0)}% peak kW)")
    out.append(f"  Max Indoor Temp    : {sim_top.get('max_indoor_temp_c', 21.0)}°C ({sim_top.get('comfort_impact')})")

    if alt:
        out.append("\n------------------------------------------------------------------------------------------")
        out.append(f"[OPTION 2 (ALTERNATIVE)]: Ranked #2 Strategy: {alt.get('strategy_id')} - {alt.get('strategy_name')}")
        out.append(f"  Score              : {alt.get('score', 0.0):.3f}")
        out.append(f"  Rationale          : {alt.get('rationale')}")
        out.append(f"  Proposed Actions   : {alt.get('actions')}")
        out.append(f"  1-Hour Cost Savings: ${sim_alt.get('projected_cost_savings_usd', 0.0):.2f} ({sim_alt.get('projected_kwh_reduction', 0.0)} kWh saved)")
        out.append(f"  Peak Load Red. %   : {sim_alt.get('window_energy_reduction_pct', 0.0)}% window energy ({sim_alt.get('instantaneous_peak_kw_reduction_pct', 0.0)}% peak kW)")
        out.append(f"  Max Indoor Temp    : {sim_alt.get('max_indoor_temp_c', 21.0)}°C ({sim_alt.get('comfort_impact')})")

    out.append("\n------------------------------------------------------------------------------------------")
    out.append("[OPTION 3]: Reject All Proposed Strategies (Cancel demand-response intervention)")
    out.append("==========================================================================================")

    return "\n".join(out)


def prompt_facility_manager_decision(proposal: Dict[str, Any], choice_override: Optional[str] = None) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Process facility manager decision for pending proposal.
    Choices:
    - '1' or 'approve': Approve top-ranked strategy
    - '2' or 'alternative': Approve alternative strategy #2
    - '3' or 'reject': Reject all candidate strategies
    """
    print(format_approval_proposal(proposal))

    if choice_override is not None:
        user_choice = str(choice_override).lower().strip()
    else:
        user_choice = input("\nEnter decision choice [1 = Approve Top, 2 = Pick Alternative, 3 = Reject All]: ").strip()

    if user_choice in ["1", "approve", "top"]:
        print("  -> Operator Decision: APPROVED Top-Ranked Strategy.")
        return "APPROVED", proposal.get("top_strategy")
    elif user_choice in ["2", "alternative", "alt"]:
        alt = proposal.get("alternative_strategy")
        if alt:
            print("  -> Operator Decision: SELECTED Alternative Strategy #2.")
            return "APPROVED", alt
        else:
            print("  -> Warning: No alternative strategy available. Defaulting to Top Strategy.")
            return "APPROVED", proposal.get("top_strategy")
    else:
        print("  -> Operator Decision: REJECTED All Strategies.")
        return "REJECTED", None
