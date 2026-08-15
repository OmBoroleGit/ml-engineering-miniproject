
# Week 2 Model Comparison Report

## Result
**Winning model: ResNet18 (transfer learning)**

| Metric    | CNN (from scratch) | ResNet18 (transfer learning) |
|-----------|--------------------:|------------------------------:|
| Accuracy  | 0.9720 | 0.9748 |
| Precision | 0.9977 | 1.0000 |
| Recall    | 0.9581 | 0.9603 |
| F1 Score  | 0.9775 | 0.9797 |
| ROC AUC   | 0.9972 | 0.9988 |

Both models were evaluated on the identical held-out test set (715
images), using the identical metric set, computed by the identical code -- the
only way this comparison is a fair one.

## Honest limitations (for transparency, not to discount either model)
- The ResNet18 run trained for only 3 epochs with a frozen backbone, versus
  the CNN's 12 epochs with early stopping -- still a shorter, less thorough
  training run, though it now uses the same 20% validation split and a fixed
  seed (42) as the CNN, so it is no longer trained "blind."

## Reproducibility
Both runs are logged in MLflow (experiment: `casting-defect-model-comparison`,
store: `mlflow.db`) with their full parameter set, metrics, confusion
matrices, and the exact weight file used -- open with:
`mlflow ui --backend-store-uri sqlite:///mlflow.db`
