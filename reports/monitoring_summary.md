# Monitoring Summary

Generated from 100 logged predictions
(2026-08-18 16:37:33.429215+00:00 to 2026-08-18 16:37:34.490422+00:00).

## Volume
- Total predictions: 100
- Days covered: 1
- Average per day: 100.0

## Confidence
- Mean confidence: 0.908
- Share below 65% confidence: 5.0%

## Class balance
- Overall defective rate: 65.0%
- Training set baseline: ~50%

## Suggested signals to watch (for the retraining trigger design)
- Defective rate drifting more than ~15 points from the ~50% baseline
- Mean confidence dropping notably below 0.91 (this run's baseline)
- Share of low-confidence (<65%) predictions rising past 5.0%
