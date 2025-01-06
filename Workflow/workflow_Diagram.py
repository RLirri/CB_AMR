from graphviz import Digraph

# Create a new directed graph
workflow = Digraph("TestRunWorkflow", format="png")
# workflow.attr(rankdir="TB")  # Top to Bottom layout
workflow.attr(rankdir="LR")  # Left-to-Right layout


workflow.node("start", "Start: Welcome to ABM and Mechanistic Simulation Tool")
workflow.node("input", "User Input:\n- Population Size\n- Growth Rate\n- Resistance Loss\n- Mutation Rate\n- Carrying Capacity\n- Duration & Steps")

# Choose Simulation Type
workflow.node("sim_type", "Choose Simulation Type\n1. ABM\n2. Mechanistic")
workflow.node("abm", "Run ABM Simulation")
workflow.node("mech", "Run Mechanistic Simulation")

# ABM Outputs
workflow.node("abm_out", "ABM Outputs:\n- Line Graph\n- Bar & Stacked Area Plots\n- Videos")
workflow.node("abm_vid", "Save ABM Video Outputs")

# Mechanistic Outputs
workflow.node("mech_out", "Mechanistic Outputs:\n- Growth Curve\n- Bar & Stacked Area Plots\n- Videos")
workflow.node("mech_vid", "Save Mechanistic Video Outputs")

# Step 3: Run Machine Learning Models
workflow.node("ml_prompt", "Run Machine Learning Models?\n1. Yes\n2. No")
workflow.node("ml_choose", "Choose Model:\n1. Random Forest\n2. MLP\n3. Both")
workflow.node("ml_train", "Train and Evaluate:\n- Random Forest\n- MLP")
workflow.node("ml_out", "ML Outputs:\n- Confusion Matrix\n- ROC Curve\n- Performance Metrics")

# Outputs
workflow.node("end", "Process Complete: Results Saved to ./output")

# Define connections
edges = [
    ("start", "input"),
    ("input", "sim_type"),
    ("sim_type", "abm", "1. ABM"),
    ("sim_type", "mech", "2. Mechanistic"),
    ("abm", "abm_out"),
    ("abm_out", "abm_vid"),
    ("mech", "mech_out"),
    ("mech_out", "mech_vid"),
    ("abm_vid", "ml_prompt"),
    ("mech_vid", "ml_prompt"),
    ("ml_prompt", "ml_choose", "1. Yes"),
    ("ml_prompt", "end", "2. No"),
    ("ml_choose", "ml_train"),
    ("ml_train", "ml_out"),
    ("ml_out", "end"),
]

# Add edges
for edge in edges:
    if len(edge) == 2:
        workflow.edge(edge[0], edge[1])
    elif len(edge) == 3:
        workflow.edge(edge[0], edge[1], label=edge[2])

# Save and render the diagram
workflow.render("TestRunWorkflowHorizontal", cleanup=True)
print("Workflow diagram saved as 'TestRunWorkflow.png'")
