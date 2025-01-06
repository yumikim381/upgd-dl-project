import os, json
import matplotlib.pyplot as plt
import numpy as np

def find_json_files(base_dir):
    """Recursively find all JSON files in a directory."""
    json_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files

def extract_folder_name(file_path, base_dir):
    """Extract the first folder name relative to the base directory."""
    relative_path = os.path.relpath(file_path, base_dir)
    first_folder = relative_path.split(os.sep)[0]  # Get the first folder in the path
    return first_folder

def read_json(file_path):
    """Read and return the content of a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    losses = data.get("losses", [])
    accuracies = data.get("accuracies", [])
    plasticity_per_task = data.get("plasticity_per_task", [])
    return {
        "accuracies": accuracies,
        "plasticity_per_task": plasticity_per_task,
        "losses": losses
    }

def metrics(json_files, base_dir):
    """Compute the average accuracy and plasticity for each folder."""
    averages = {}
    
    for json_file in json_files:
        folder_name = extract_folder_name(json_file, base_dir)
        data = read_json(json_file)
        accuracies = data.get("accuracies", [])
        plasticity_per_task = data.get("plasticity_per_task", [])
        loss = data.get("losses", [])
        
        # Compute metrics
        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
        avg_plasticity = sum(plasticity_per_task) / len(plasticity_per_task) if plasticity_per_task else 0
        lipschitz = np.max(np.abs(np.diff(loss)))
        forgetting = accuracies[0] - accuracies[-1]
        diff_plasticity = plasticity_per_task[0] - plasticity_per_task[-1]


        averages[folder_name] = {
            "average_accuracy": avg_accuracy,
            "average_plasticity": avg_plasticity,
            "lipschitz_const": lipschitz,
            "forgetting": forgetting,
            "diff_plasticity": diff_plasticity
        }
    
    return averages

def print_averages(averages):
    """Print the average accuracy and plasticity for each folder."""
    print(f"{'Approach':<20}{'Average Accuracy':<20}{'Average Plasticity':<20}{'Lipschitz Constant':<20}{'Forgetting':<20}{'Diff Plasticity':<20}")
    print("-" * 120)
    for approach, metrics in averages.items():
        avg_accuracy = metrics["average_accuracy"]
        avg_plasticity = metrics["average_plasticity"]
        lipschitz_const = metrics["lipschitz_const"]
        forgetting = metrics["forgetting"]
        diff_plasticity = metrics["diff_plasticity"]

        print(f"{approach:<20}{avg_accuracy:<20.4f}{avg_plasticity:<20.4f}{lipschitz_const:<20.4f}{forgetting:<20.4f}{diff_plasticity:<20.4f}")

def plot_aggregated_data(json_files, base_dir, metric):
    """Aggregate and plot loss values from all JSON files in one graph."""

    plt.figure(figsize=(10, 6))
    
    for json_file in json_files:
        folder_name = extract_folder_name(json_file, base_dir)
        data = read_json(json_file)
        values = data.get(metric, [])
        
        if isinstance(values, list) and values:  # Check if accuracies are valid
            plt.plot(values, label=folder_name, linewidth=1)
          
    desired_order = [
    "baseline",
    "weight_norm",
    "grad_norm",
    "ratio_norm",
    "entire_kernel",
    "column_kernel",
    "sgd",
    ]
    handles, labels = plt.gca().get_legend_handles_labels()
    label_to_handle = dict(zip(labels, handles))

    # Reorder handles and labels based on the desired order
    ordered_handles = [label_to_handle[label] for label in desired_order if label in label_to_handle]
    ordered_labels = [label for label in desired_order if label in label_to_handle]

    plt.xlabel("Tasks")
    plt.ylabel(metric)
    plt.grid()
    plt.legend(ordered_handles, ordered_labels,  title="Approaches", loc="lower right")
    plt.show()

def plot_zoomed_data(json_files, base_dir, metric, num_tasks=200, accuracy_range=(0.0, 1.0), selected_approaches=None):
    """
    Plot a zoomed-in graph for the last num_tasks and focus on a specific accuracy range.
    """
    plt.figure(figsize=(10, 6))
    
    for json_file in json_files:
        folder_name = extract_folder_name(json_file, base_dir)
        if folder_name in selected_approaches: 
            data = read_json(json_file)
            values = data.get(metric, [])
        
            if values and isinstance(values, list): 
                plt.plot(values, label=folder_name, linewidth=1)

    plt.xlabel("Tasks")
    plt.ylabel(metric)
    plt.xlim(len(values) - num_tasks, len(values))  # Focus on the last num_tasks
    plt.ylim(accuracy_range)  # Focus on the specified accuracy range
    plt.grid()
    plt.legend(title="Approaches", loc="lower left")
    plt.show()

def main(base_dir):
    json_files = find_json_files(base_dir)
    # Plot aggregated data
    plot_aggregated_data(json_files, base_dir, "accuracies")
    plot_aggregated_data(json_files, base_dir, "plasticity_per_task")

    # Plot zoomed-in graph
    selected_approaches = ["baseline", "ratio_norm", "sgd"]
    plot_zoomed_data(json_files, base_dir, "accuracies", accuracy_range=(0.51, 0.67), selected_approaches=selected_approaches)
    plot_zoomed_data(json_files, base_dir, "plasticity_per_task", accuracy_range=(0.4, 0.54), selected_approaches=selected_approaches)

    # Compute and print averages
    averages = metrics(json_files, base_dir)
    print_averages(averages)
    
base_dir = "logs/label_permuted_cifar10_stats"  # Replace with the path to your logs directory
main(base_dir)