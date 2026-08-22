# Monitoring Summary

Generated from 204 logged predictions
(2026-08-18 16:37:33.429215+00:00 to 2026-08-22 15:01:42.587351+00:00).

## Volume
- Total predictions: 204
- Days covered: 2
- Average per day: 102.0

## Confidence
- Mean confidence: 0.908
- Share below 65% confidence: 4.9%

## Class balance
- Overall defective rate: 64.2%
- Training set baseline: ~50%

## Suggested signals to watch (for the retraining trigger design)
- Defective rate drifting more than ~15 points from the ~50% baseline
- Mean confidence dropping notably below 0.91 (this run's baseline)
- Share of low-confidence (<65%) predictions rising past 4.9%
