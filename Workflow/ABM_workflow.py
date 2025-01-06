from graphviz import Digraph

# Create a new directed graph for ABM model workflow
abm_workflow = Digraph("ABMModelWorkflow", format="png")
abm_workflow.attr(rankdir="LR")  # Left-to-right layout

# Nodes for the workflow
abm_workflow.node("start", "Start: Input Parameters")
abm_workflow.node("params", "User Inputs:\n- Initial Population\n- Growth Rate\n- Resistance Loss\n- Mutation Rate\n- Carrying Capacity\n- Duration & Steps")
abm_workflow.node("load", "Load Dataset")
abm_workflow.node("simulate", "Run ABM Simulation")
abm_workflow.node("snapshots", "Generate Snapshots of Population Dynamics")
abm_workflow.node("visualize", "Visualize Results:\n- Line Graphs\n- Bar Charts\n- Stacked Area Plots")
abm_workflow.node("videos", "Create Video Outputs:\n- Time Series\n- Resistance Evolution")
abm_workflow.node("end", "End: Save Outputs")

# Edges for the workflow
abm_workflow.edge("start", "params")
abm_workflow.edge("params", "load")
abm_workflow.edge("load", "simulate")
abm_workflow.edge("simulate", "snapshots")
abm_workflow.edge("snapshots", "visualize")
abm_workflow.edge("visualize", "videos")
abm_workflow.edge("videos", "end")

# Render and save the graph
abm_workflow.render("ABMModelWorkflow", cleanup=True)
"ABM Model Workflow Diagram saved as 'ABMModelWorkflow.png'"
