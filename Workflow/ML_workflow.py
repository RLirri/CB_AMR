from graphviz import Digraph

# Create a new directed graph for the ML workflow based on the provided description
detailed_ml_workflow = Digraph("DetailedMLWorkflow", format="png")
detailed_ml_workflow.attr(rankdir="LR")  # Left-to-right layout

# Nodes for the workflow
detailed_ml_workflow.node("start", "Start: Load Dataset")
detailed_ml_workflow.node("data_process", "Data Processing:\n- Extract X and y\n- Feature Matrix (X)\n- Binary Labels (y)")
detailed_ml_workflow.node("scaling", "Feature Scaling:\n- StandardScaler\n- Normalize X for MLP")
detailed_ml_workflow.node("split", "Train-Test Split:\n- 80% Train\n- 20% Test")

# Random Forest nodes
detailed_ml_workflow.node("rf_input", "Random Forest Input:\n- Raw X_train")
detailed_ml_workflow.node("rf_train", "Train Random Forest")
detailed_ml_workflow.node("rf_output", "RF Output:\n- Binary Predictions for CIP, CTX, CTZ, GEN")

# MLP nodes
detailed_ml_workflow.node("mlp_input", "MLP Input:\n- Scaled X_train_scaled")
detailed_ml_workflow.node("mlp_train", "Train MLP")
detailed_ml_workflow.node("mlp_output", "MLP Output:\n- Binary Predictions\n- Threshold: 0.5")

# Evaluation and comparison
detailed_ml_workflow.node("eval_metrics", "Evaluation Metrics:\n- Confusion Matrix\n- ROC Curve\n- AUC")
detailed_ml_workflow.node("compare", "Compare Model Performances")

# End node
detailed_ml_workflow.node("end", "End: Save Results")

# Edges for the workflow
detailed_ml_workflow.edges([
    ("start", "data_process"),
    ("data_process", "scaling"),
    ("scaling", "split"),
    ("split", "rf_input"),
    ("split", "mlp_input"),
    ("rf_input", "rf_train"),
    ("mlp_input", "mlp_train"),
    ("rf_train", "rf_output"),
    ("mlp_train", "mlp_output"),
    ("rf_output", "eval_metrics"),
    ("mlp_output", "eval_metrics"),
    ("eval_metrics", "compare"),
    ("compare", "end")
])

# Render and save the graph
detailed_ml_workflow.render("DetailedMLWorkflow", cleanup=True)
"Detailed ML Workflow Diagram saved as 'DetailedMLWorkflow.png'"
