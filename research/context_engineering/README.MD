🔬 Project Title: Context Engineering - Performance Comparison
Executive Summary: A one-sentence pitch of what this research solves and why it matters to the world.

📌 Research Overview
Status: 🏗️ In-Progress
Domain: Context Engineering
Core Question: How does system context affect the performance of AIOps solutions across anomaly detection, signal correlation, RCA and remediation
🎯 Desired Outcomes
Primary Objective: Find the highest performing permutations of different context's for an AIOps soltuion
Key Deliverables:
 Approach:
 Type of Context: 
 Blog Post: Formal documentation of findings.
 Visual Dashboard: UI for easy results visualisation.
🛠 Methodology & Framework
Describe the technical approach and the "How" behind the research. For exanple: I am using the OTEL demo app to generate data, train models, develop different feature sets and compare different algorithms and AI models (predictive) comparing the results calculating precision, recall, F1 score and Area Under the Curve (AUC).

Approach: For example: collect ground truth data to train model and algortihm, then evaluate performance with a separate data set
Core Logic: ./comparison/compare_detectors.py
Tech Stack: For Example: OTEL Demo App, Python (Pandas/NumPy), Docker/Podman, Pytorch, Make
📊 Data Management & Transparency
Source: For example: OTEL Demo App logs, metrics & traces
Processing: For example: The data cleaning happens in /chaos_engineering/collect* scripts
Ethical Considerations: OTEL demo app data allows for repeatible, ethical, copyright free data collection.

📂 Repository Structure
├──  

