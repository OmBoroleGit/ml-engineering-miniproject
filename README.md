Machine Learning Engineering (PCAM* ZC412)
Mini-Project Assignment Brief — EC-1 Project (Weight: 40% | Duration: 28 Days)
------------------------------------------------------------------------------
This mini-project requires you to design, build, and operate an end-to-end machine learning pipeline — from raw data to a monitored, deployed service. You will choose ONE of three problem statements (Flavor A, B, or C) based on your interest. All three flavors are of equal complexity, follow the same 4-week structure, map to the same course modules (M1 through M5), and are evaluated using the same rubric — only the input data and modeling task differ.

•	Design and implement a reliable data ingestion, validation, and feature engineering pipeline.
•	Apply engineering discipline to ML experimentation through tracking, comparison, and reproducibility.
•	Package and deploy a trained model as a production-style service with a REST API.
•	Implement monitoring, simulate drift, and design a retraining strategy for a deployed model.
•	Demonstrate end-to-end ownership of the ML system lifecycle, consistent with course outcomes LO1–LO5.

Flavor B — Image-Based Defect / Quality Classifier
Domain: Vision data | Modules: M2 → M3 → M4 → M5
Problem Statement: A manufacturing or quality-assurance team wants to automatically flag defective products from images captured on the production line. Build an end-to-end ML pipeline that ingests and preprocesses product images, trains a classifier to distinguish defective from non-defective items, deploys it as an inference service, and monitors performance as new product variants or lighting conditions appear.
Week	Module	Task
Week 1	M2	Ingest and validate image data; build a preprocessing/augmentation pipeline; version the dataset.
Week 2	M3	Train and compare CNN/transfer-learning models; track experiments and reproducibility.
Week 3	M4	Package the model; serve via a REST API accepting image uploads; handle malformed/edge-case inputs.
Week 4	M5	Log predictions and confidence scores; simulate distribution shift (lighting/angle changes); monitor and design a retraining workflow.
Suggested Dataset(s): Casting Product Image Data for Quality Inspection (Kaggle), or Surface Defect Detection datasets.

Weekly Timeline & Milestones
Milestone	Expectation
Day 1–2	Flavor selection confirmed; repository initialized.
End of Week 1	Data ingestion, validation, and feature pipeline complete; dataset version tagged.
End of Week 2	At least two tracked experiments completed; best model identified with justification.
End of Week 3	Model packaged and deployed; API endpoint tested with sample inputs.
End of Week 4 (Day 28)	Monitoring/drift simulation complete; final submission package and demo ready.
