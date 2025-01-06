import os
import matplotlib
matplotlib.use('Agg')
from data_preprocessing import load_data
from ml_models import train_random_forest, build_mlp_model
from mechanistic_model_updated import simulate_logistic_growth_with_resistance
from abm_updated import simulate_abm
from hyperparameter_tuning import grid_search_rf
from visualization import plot_confusion_matrix, plot_roc_curve, plot_model_comparison
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Ensure the output directory exists
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_video_from_frames(frames, video_name, fps=5):
    """
    Create a video directly from frames using OpenCV.
    """
    height, width, layers = frames[0].shape
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for frame in frames:
        video.write(frame)

    video.release()
    print(f"Video successfully saved to {video_name}")

def create_video_from_images(image_folder, video_name, fps=5):
    """
    Create a video from saved images using OpenCV.
    """
    image_files = [os.path.join(image_folder, f) for f in sorted(os.listdir(image_folder)) if f.endswith('.png')]
    if not image_files:
        print(f"No images found in {image_folder}. Cannot create video.")
        return

    frame = cv2.imread(image_files[0])
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for image_file in image_files:
        frame = cv2.imread(image_file)
        video.write(frame)

    video.release()
    print(f"Video successfully saved to {video_name}")

def main():

    print("Welcome to the ABM and Mechanistic Simulation Tool!")

    # User inputs validator
    def get_valid_input(prompt, cast_func, validation_func):
        while True:
            try:
                value = cast_func(input(prompt).strip())
                if validation_func(value):
                    return value
                else:
                    print("Invalid input. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a valid value.")

    # Inputs with validation
    initial_population = get_valid_input(
        "Enter the initial population size (default 1): ", int, lambda x: x > 0
    )
    growth_rate = get_valid_input(
        "Enter the growth rate (default 0.1): ", float, lambda x: x > 0
    )
    loss_probability = get_valid_input(
        "Enter the resistance loss probability (0.0 - 1.0, default 0.01): ",
        float, lambda x: 0 <= x <= 1
    )
    mutation_rate = get_valid_input(
        "Enter the mutation rate (probability of mutation, 0.0 - 1.0, default 0.01): ",
        float, lambda x: 0 <= x <= 1
    )
    carrying_capacity = get_valid_input(
        "Enter the carrying capacity (default 500) (must be >= initial population): ",
        int, lambda x: x >= initial_population
    )
    duration = get_valid_input(
        "Enter the simulation duration in time steps (default 100): ", int, lambda x: x > 0
    )
    steps = get_valid_input(
        "Enter the number of steps (default 50): ", int, lambda x: x > 0
    )

    # User selects simulation type
    print("Choose a simulation type:")
    print("1: ABM")
    print("2: Mechanistic")
    simulation_type = get_valid_input(
        "Enter your choice (1 or 2): ", int, lambda x: x in [1, 2]
    )

    dataset_path = "../dataset/gi_cip_ctx_ctz_gen_pheno.csv"

    if simulation_type == 1:
        print(f"Loading dataset from {dataset_path} for training data...")
        X, y = load_data(dataset_path)
        _, _, _, y_train = train_test_split(X, y, test_size=0.2, random_state=42)

        print("Running ABM simulation...")
        resistance_ratio = float(input("Enter the initial resistance ratio (0.0 - 1.0, default 0.5): ") or 0.5)
        history, snapshots = simulate_abm(
            num_bacteria=initial_population,
            steps=steps,
            resistance_ratio=resistance_ratio,
            mutation_rate=mutation_rate,
            loss_rate=loss_probability,
            growth_rate=growth_rate,
            carrying_capacity=carrying_capacity,
            return_snapshots=True
        )

        # Save ABM visualization (Time Series)
        plt.plot(history, label=f"ABM: Proportion of Resistant Bacteria", linestyle='--')
        plt.title(f"Agent-Based Modeling: Resistance Evolution")
        plt.xlabel("Time Steps")
        plt.ylabel("Proportion of Resistant Bacteria")
        plt.legend()
        abm_output_path = os.path.join(OUTPUT_DIR, "abm_simulation.png")
        plt.savefig(abm_output_path)
        print(f"ABM visualization saved to {abm_output_path}")
        plt.clf()

        # Save ABM line graph visualization
        plt.plot(range(len(history)), history, color='blue', label='Proportion Resistant')
        plt.title("ABM Line Graph: Resistance Over Time")
        plt.ylim(0, max(history) * 1.1)
        plt.xlabel("Time Steps")
        plt.ylabel("Proportion Resistant")
        plt.legend()
        line_graph_path = os.path.join(OUTPUT_DIR, "abm_line_graph.png")
        plt.savefig(line_graph_path)
        print(f"ABM line graph saved to {line_graph_path}")
        plt.clf()

        frames = []
        scatter_frames = []
        resistant_counts = []
        non_resistant_counts = []

        for i, snapshot in enumerate(snapshots):
            resistant_counts.append(snapshot[0])
            non_resistant_counts.append(snapshot[1])

            # --- Bar Plot for Each Step ---
            fig, ax = plt.subplots()
            ax.bar(["Resistant", "Non-Resistant"], snapshot, color=["red", "blue"])

            # Dynamically adjust the y-axis based on current population
            total_population = sum(snapshot)
            ax.set_ylim(0, total_population + int(0.1 * total_population))
            ax.set_title(f"ABM Population at Step {i}")

            # Convert figure to frame for video
            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            frame = frame[:, :, :3]  # Drop the alpha channel to keep only RGB
            frames.append(frame)  # Add frame to the video list

            # Save the first frame of bar plot
            if i == 1:
                first_bar_frame_path = os.path.join(OUTPUT_DIR, "first_bar_frame.png")
                plt.savefig(first_bar_frame_path)
                print(f"First bar frame saved to {first_bar_frame_path}")

            plt.close(fig)

            # --- Stacked Area Plot ---
            fig, ax = plt.subplots()
            ax.stackplot(
                range(len(resistant_counts)),
                [resistant_counts, non_resistant_counts],
                labels=["Resistant", "Non-Resistant"],
                colors=["red", "blue"],
                alpha=0.6
            )

            # Dynamically adjust the y-axis for stacked area plot
            current_population = resistant_counts[-1] + non_resistant_counts[-1]
            ax.set_ylim(0, current_population + int(0.1 * current_population))
            ax.set_xlim(0, len(resistant_counts))
            ax.set_title(f"ABM Stacked Area Plot at Step {i}")
            ax.set_xlabel("Time Steps")
            ax.set_ylabel("Bacteria Count")
            ax.legend(loc="upper left")

            # Convert stacked area plot to frame for video
            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            frame = frame[:, :, :3]  # Convert RGBA to RGB
            scatter_frames.append(frame)  # Add frame to stacked area plot video

            # Save the first frame of stacked area plot
            if i == 1:
                first_stacked_frame_path = os.path.join(OUTPUT_DIR, "first_stacked_frame.png")
                plt.savefig(first_stacked_frame_path)
                print(f"First stacked frame saved to {first_stacked_frame_path}")

            plt.close(fig)

        # Save the last frame of bar plot
        if frames:
            last_bar_frame_path = os.path.join(OUTPUT_DIR, "last_bar_frame.png")
            plt.imsave(last_bar_frame_path, frames[-1])
            print(f"Last bar frame saved to {last_bar_frame_path}")

        # Save the last frame of stacked area plot
        if scatter_frames:
            last_stacked_frame_path = os.path.join(OUTPUT_DIR, "last_stacked_frame.png")
            plt.imsave(last_stacked_frame_path, scatter_frames[-1])
            print(f"Last stacked frame saved to {last_stacked_frame_path}")

        video_path = os.path.join(OUTPUT_DIR, "abm_simulation.mp4")
        create_video_from_frames(frames, video_path)
        scatter_video_path = os.path.join(OUTPUT_DIR, "stacked_area.mp4")
        create_video_from_frames(scatter_frames, scatter_video_path)

    elif simulation_type == 2:
        print(f"Loading dataset from {dataset_path} for training data...")
        X, y = load_data(dataset_path)
        _, _, _, y_train = train_test_split(X, y, test_size=0.2, random_state=42)
        print("Running Mechanistic Model simulation with resistance dynamics...")
        solution, time_points = simulate_logistic_growth_with_resistance(
            y_train, initial_population, growth_rate, carrying_capacity, duration, steps, loss_probability,
            mutation_rate
        )
        # Save Mechanistic Model visualization (Growth Curve)
        mech_output_path = os.path.join(OUTPUT_DIR, "output1", "mechanistic_model_resistance.png")
        os.makedirs(os.path.dirname(mech_output_path), exist_ok=True)

        # Calculate the maximum total population for dynamic y-axis limits
        max_population = max(solution[:, 0] + solution[:, 1]) * 1.1  # Add 10% buffer

        # Plot resistant and non-resistant populations
        plt.plot(time_points, solution[:, 0], label="Resistant Population", color="red")
        plt.plot(time_points, solution[:, 1], label="Non-Resistant Population", color="blue")
        plt.title("Mechanistic Modeling: Logistic Growth with Resistance")
        plt.xlabel("Time")
        plt.ylabel("Population Size")
        plt.legend()
        plt.savefig(mech_output_path)
        print(f"Mechanistic visualization saved to {mech_output_path}")
        plt.clf()

        # Create video frames for bar chart and stacked area chart
        bar_frames = []
        stack_frames = []

        # Prepare lists for stacked area plot
        resistant_counts = []
        non_resistant_counts = []

        for i, (resistant, non_resistant) in enumerate(solution):
            total_population = resistant + non_resistant
            resistant_counts.append(resistant)
            non_resistant_counts.append(non_resistant)

            # --- Bar Chart ---
            fig, ax = plt.subplots()
            ax.bar(["Resistant", "Non-Resistant"], [resistant, non_resistant], color=["red", "blue"])
            ax.set_ylim(0, max_population)  # Dynamic y-axis limit
            ax.set_title(f"Population at Time Step {time_points[i]:.2f}")
            fig.canvas.draw()

            # Convert bar chart to frame
            frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            frame = frame[:, :, :3]  # Convert RGBA to RGB
            bar_frames.append(frame)

            # Save specific frames of bar chart
            if i == 1:
                bar_second_frame_path = os.path.join(OUTPUT_DIR, "output1", "bar_second_frame.png")
                plt.savefig(bar_second_frame_path)
                print(f"Second frame of bar chart saved to {bar_second_frame_path}")
            elif i == len(solution) - 1:
                bar_last_frame_path = os.path.join(OUTPUT_DIR, "output1", "bar_last_frame.png")
                plt.savefig(bar_last_frame_path)
                print(f"Last frame of bar chart saved to {bar_last_frame_path}")

            plt.close(fig)

            # --- Stacked Area Chart ---
            fig, ax = plt.subplots()
            ax.stackplot(
                time_points[: len(resistant_counts)],
                resistant_counts,
                non_resistant_counts,
                labels=["Resistant", "Non-Resistant"],
                colors=["red", "blue"],
                alpha=0.6
            )
            ax.set_xlim(0, time_points[-1])
            ax.set_ylim(0, max_population)  # Dynamic y-axis limit
            ax.set_title(f"Population Dynamics at Time Step {time_points[i]:.2f}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Population Size")
            ax.legend(loc="upper left")
            fig.canvas.draw()

            # Convert stacked area chart to frame
            frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            frame = frame[:, :, :3]  # Convert RGBA to RGB
            stack_frames.append(frame)

            # Save specific frames of stacked area chart
            if i == 1:
                stack_second_frame_path = os.path.join(OUTPUT_DIR, "output1", "stack_second_frame.png")
                plt.savefig(stack_second_frame_path)
                print(f"Second frame of stacked area chart saved to {stack_second_frame_path}")
            elif i == len(solution) - 1:
                stack_last_frame_path = os.path.join(OUTPUT_DIR, "output1", "stack_last_frame.png")
                plt.savefig(stack_last_frame_path)
                print(f"Last frame of stacked area chart saved to {stack_last_frame_path}")

            plt.close(fig)

        # Create bar chart video
        bar_video_path = os.path.join(OUTPUT_DIR, "output1", "mechanistic_bar_chart.mp4")
        create_video_from_frames(bar_frames, bar_video_path)
        print(f"Bar chart video saved to {bar_video_path}")

        # Create stacked area chart video
        stack_video_path = os.path.join(OUTPUT_DIR, "output1", "mechanistic_stacked_area.mp4")
        create_video_from_frames(stack_frames, stack_video_path)
        print(f"Stacked area chart video saved to {stack_video_path}")
    else:
        print("Invalid simulation type. Please choose 1 or 2.")
        return

    # Option to run ML models
    print("Would you like to run machine learning models?")
    print("1: Yes")
    print("2: No")
    run_ml = get_valid_input(
        "Enter your choice (1 or 2): ", int, lambda x: x in [1, 2]
    )
    if run_ml == 1:
        print("Loading and splitting dataset...")
        X, y = load_data(dataset_path)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("Choose a model to train:")
        print("1: Random Forest")
        print("2: MLP")
        print("3: Both Random Forest and MLP")
        run_ml = get_valid_input(
            "Enter your choice (1, 2, or 3): ", int, lambda x: x in [1, 2, 3]
        )

        if run_ml in [1, 2, 3]:
            print("Loading and splitting dataset...")
            X, y = load_data(dataset_path)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            if run_ml == 1 or run_ml == 3:
                # Random Forest
                print("Training Random Forest model...")
                rf_model = train_random_forest(X_train, y_train)
                rf_pred = rf_model.predict(X_test)

                # Visualizations for Random Forest
                plot_confusion_matrix(y_test.values, rf_pred, "Random Forest")
                rf_auc = plot_roc_curve(y_test, rf_pred, "Random Forest")
                rf_output_path = os.path.join(OUTPUT_DIR, "random_forest.png")
                plt.savefig(rf_output_path)
                print(f"Random Forest results visualization saved to {rf_output_path}")
                plt.clf()

            if run_ml == 2 or run_ml == 3:
                # MLP
                print("Training MLP model...")
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                mlp_model = build_mlp_model(X_train.shape[1])
                mlp_model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.1)
                mlp_pred = (mlp_model.predict(X_test_scaled) > 0.5).astype(int)

                # Visualizations for MLP
                plot_confusion_matrix(y_test.values, mlp_pred, "MLP Neural Network")
                mlp_auc = plot_roc_curve(y_test, mlp_pred, "MLP Neural Network")
                mlp_output_path = os.path.join(OUTPUT_DIR, "mlp_model.png")
                plt.savefig(mlp_output_path)
                print(f"MLP Neural Network results visualization saved to {mlp_output_path}")
                plt.clf()
        else:
            print("No machine learning model was selected.")


if __name__ == "__main__":
    main()
