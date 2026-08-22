# Retraining Trigger and Automated Workflow Design

## 1. Trigger Strategy and Thresholds

Our system uses a multi-tier trigger strategy combining live proxy signals from API logging and environmental stress boundaries identified during drift testing.

### A. Live Monitoring Triggers (Statistical Drift)
Because ground-truth labels are not instantly available on the factory line, we monitor proxy metrics over a rolling window of **1,000 predictions** (or a 24-hour shift):
* **Confidence Degradation Trigger:** Trigger an automated alert if the rolling mean confidence falls below **0.88** (baseline is 0.908).
* **Low-Confidence Spike Trigger:** Trigger retraining if the proportion of predictions with confidence `< 0.65` exceeds **12.0%** (baseline is 5.0%).
* **Class Imbalance Drift Trigger:** Trigger an investigation if the predicted defective rate deviates by more than **15 percentage points** from the historical balance (healthy baseline is ~50.0% to 65.0%).

### B. Physical Degradation Thresholds (Data Distribution Shift)
Based on empirical stress tests on the 711-image test set, performance degrades significantly (≥5 percentage points drop in accuracy) at the following boundaries:
* **Lens Blur:** Blur radius $\ge 2.0$ (accuracy drops to 88.33%).
* **Camera Misalignment / Rotation:** Angle $\ge 5^\circ$ (accuracy drops to 89.73%).
* **Illumination Failure / Brightness:** Factor $\le 0.5$ or $\ge 1.3$ (accuracy drops to 63.57% and 69.06% respectively).
* **Sensor Noise:** Noise level $\ge 5.0$ (accuracy drops to 90.86%).

*Operational Rule:* When an alarm fires, hardware diagnostics run first to check camera alignment, lighting, and lens cleanliness before executing compute-heavy model training.

### C. Time-Based Scheduled Trigger (Cadence)
* A scheduled retraining pipeline runs every **90 days** (quarterly) to capture seasonal visual shifts and subtle casting mold wear even if anomaly thresholds are not breached.

## 2. Automated Retraining Workflow

Once a trigger threshold is breached, the following pipeline executes automatically:

```text
[ Live Production Traffic ]
           │
           ▼
[ Step 1: Anomaly Trigger Fired ] ─── (Confidence < 0.88 OR Low-Conf > 12%)
           │
           ▼
[ Step 2: Data Collection ] ───────── (Pull logged images from Week 4 monitoring)
           │
           ▼
[ Step 3: Data Validation ] ───────── (Run Week 1 validation to ensure no corrupted files)
           │
           ▼
[ Step 4: Model Retraining ] ──────── (Fine-tune ResNet18 backbone using Week 2 script)
           │
           ▼
[ Step 5: Deployment Gate ] ───────── (Does it beat the 97.48% baseline?)
           ├─ YES ──► Deploy to FastAPI (Update Week 3 API)
           └─ NO  ──► Abort and Alert ML Engineer for Review