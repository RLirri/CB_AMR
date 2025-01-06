from graphviz import Digraph

# Create a new directed graph for the Mechanistic Model workflow
mechanistic_workflow = Digraph("MechanisticModelWorkflow", format="png")
mechanistic_workflow.attr(rankdir="LR")  # Left-to-right layout

# Nodes for the workflow
mechanistic_workflow.node("start", "Start: Input Parameters")
mechanistic_workflow.node("params", "User Inputs:\n- Initial Population\n- Growth Rate\n- Resistance Loss\n- Mutation Rate\n- Carrying Capacity\n- Duration & Steps")
mechanistic_workflow.node("load", "Load Dataset")
mechanistic_workflow.node("simulate", "Simulate Logistic Growth with Resistance")
mechanistic_workflow.node("results", "Generate Results:\n- Time-Series Data of Resistant/Non-Resistant Populations")
mechanistic_workflow.node("visualize", "Visualize Results:\n- Growth Curves\n- Bar Charts\n- Stacked Area Plots")
mechanistic_workflow.node("videos", "Create Video Outputs:\n- Population Dynamics")
mechanistic_workflow.node("end", "End: Save Outputs")

# Edges for the workflow
mechanistic_workflow.edge("start", "params")
mechanistic_workflow.edge("params", "load")
mechanistic_workflow.edge("load", "simulate")
mechanistic_workflow.edge("simulate", "results")
mechanistic_workflow.edge("results", "visualize")
mechanistic_workflow.edge("visualize", "videos")
mechanistic_workflow.edge("videos", "end")

# Render and save the graph
mechanistic_workflow.render("MechanisticModelWorkflow", cleanup=True)
"Mechanistic Model Workflow Diagram saved as 'MechanisticModelWorkflow.png'"
