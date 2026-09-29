# Energy Agent: Agentic Energy Optimization & Demand Response System

## ⚡ Overview

**Energy Agent** is an autonomous and human-in-the-loop agentic system designed to optimize commercial building energy consumption and execute demand response (DR) strategies. Targeting critical peak-pricing windows—such as high-demand **2:00 PM – 5:00 PM peak demand scenarios** in commercial office buildings—the system proactively balances energy cost reduction, grid stability, and occupant thermal comfort.

---

## 🎯 Key Objectives & Capabilities

- **Building Telemetry & Context Monitoring**: Ingests real-time environmental and operational telemetry including zone occupancy, indoor/outdoor temperatures, baseline electrical consumption (HVAC, lighting, plug loads), and dynamic electricity tariff rates (Time-of-Use / Critical Peak Pricing).
- **Short-Term Demand Forecasting**: Predicts upcoming energy consumption profiles and identifies potential peak demand spikes using predictive time-series models.
- **Agentic Peak-Shaving Planning**: Formulates intelligent mitigation strategies (e.g., pre-cooling zones prior to peak hours, adjusting HVAC setpoints, dimming non-critical lighting, shifting flexible loads).
- **What-If Scenario Simulation**: Simulates the impact of proposed interventions on energy load, cost savings, and thermal comfort before execution.
- **Human-in-the-Loop (HITL) Approval**: Generates actionable decision proposals with clear rationale, estimated savings, and comfort metrics for facility manager review and sign-off.
- **Simulated IoT Execution**: Dispatches validated, safe control actions to simulated Building Automation Systems (BAS) / IoT device APIs.

---

## 🏗 Project Architecture

```
energy-agent/
├── data/           # Raw and processed building telemetry and tariff datasets
├── simulator/      # Building thermal, occupancy, and IoT device simulators
├── forecasting/    # Time-series energy demand and load forecasting models
├── agent/          # LangGraph-powered orchestration, decision logic, and HITL state
├── tools/          # Tooling interfaces for simulation, calculation, and IoT controls
├── api/            # FastAPI backend endpoints for agent control and monitoring
├── dashboard/      # Interactive web dashboard (Plotly / UI) for operators
├── tests/          # Unit, integration, and scenario tests
├── requirements.txt# Project dependencies
├── README.md       # Project documentation
└── .gitignore      # Git exclusion rules
```

---

## ⚙️ Target Scenario: 2:00 PM – 5:00 PM Peak Demand in Office Building

1. **Pre-Peak Monitoring & Forecast (11:00 AM – 1:30 PM)**:
   - Identifies forecasted afternoon temperature highs and upcoming $0.45/kWh critical peak pricing window between 2:00 PM and 5:00 PM.
2. **Strategy Formulation & Simulation (1:30 PM)**:
   - Proposes pre-cooling the building from 1:00 PM – 2:00 PM to 21°C (70°F), drifting setpoints to 24°C (75°F) during 2–5 PM, and reducing common-area lighting by 30%.
   - Runs what-if physics simulation confirming 28% peak load reduction while maintaining Predicted Mean Vote (PMV) comfort scores within acceptable limits.
3. **Operator Approval & Staged Execution (1:45 PM – 5:00 PM)**:
   - Obtains facility manager approval via UI/API.
   - Executes staged control signals to simulated HVAC dampers, chillers, and smart lighting controllers.
4. **Post-Event Recovery & Reporting (5:00 PM+)**:
   - Safely transitions building systems back to normal operational setpoints and compiles an energy & cost savings summary.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+ (tested on Python 3.11)
- Virtual environment (`venv` recommended)

### Installation

1. **Create and activate a virtual environment**:
   ```bash
   # Windows (PowerShell)
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Verify Environment**:
   ```bash
   pytest
   ```
