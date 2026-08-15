# Week 2 Model Comparison Report
**Flavor B: Image-Based Defect / Quality Classifier**

## Overview
For the Week 2 experimentation phase, our team trained and tracked two distinct image classification models to automatically flag defective casting products. Both models were evaluated on the same 20% validation split and scored against the exact same unseen test dataset to ensure a fair comparison. 

All experiment parameters, training curves, and final test metrics were logged centrally via MLflow.

## Models Evaluated
1. **Custom CNN (Baseline):** A custom Convolutional Neural Network built from scratch, trained for 12 epochs with early stopping.
2. **Transfer Learning (ResNet18):** A pre-trained ResNet18 model with frozen core layers and a newly initialized classification head, trained for 3 epochs.

## Performance Metrics
Both models demonstrated excellent generalization to the test dataset. Below is the final scorecard:

| Metric | Custom CNN | Transfer Learning (ResNet18) |
| :--- | :--- | :--- |
| **Accuracy** | 0.9831 | 0.9748 |
| **Precision** | 0.9823 | 1.0000 |
| **Recall** | **0.9911** | 0.9603 |
| **F1 Score** | 0.9867 | 0.9797 |
| **ROC AUC** | 0.9991 | 0.9988 |

## Conclusion & Model Selection
**Selected Model: Custom CNN**

While the ResNet18 transfer learning model achieved a mathematically perfect Precision score (1.0000), meaning it generated zero false positives, the Custom CNN is the superior model for our specific business domain. 

In a manufacturing and quality assurance environment, **Recall** is the most critical metric. The cost of a False Negative (missing a defective product and shipping it to a customer) heavily outweighs the cost of a False Positive (flagging a healthy product for secondary manual inspection). 

Because the Custom CNN achieved a significantly higher Recall rate (99.11% vs. 96.03%), it successfully catches a higher volume of actual defects. Therefore, the Custom CNN will be packaged and deployed as our REST API inference service in Week 3.