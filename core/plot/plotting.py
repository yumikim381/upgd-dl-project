import json
import matplotlib.pyplot as plt

# Function to generate file paths based on mode
def generate_file_paths(mode):
    if mode == "loss_adativeNoisescaled.json":
        return ("adaptiveNoisescaled_loss.png",
                "adaptiveNoisescaled_accuracy.png",
                "adaptiveNoisescaled_plasticity.png")
    elif mode == "loss_grad_scaled.json":
        return ("grad_loss.png",
                "grad_accuracy.png",
                "grad_plasticity.png")
    elif mode == "loss_weight_norm.json":
        return ("weightnorm_loss.png",
                "weightnorm_accuracy.png",
                "weightnorm_plasticity.png")
    else:
        raise ValueError(f"Invalid mode: {mode}")

# Input mode
mode = "loss_adativeNoisescaled.json"  # Change this to your desired file name

# Generate save paths based on mode
save_loss_path, save_accuracy_path, save_plasticity_path = generate_file_paths(mode)

# Load data from the provided JSON files
with open("loss_original.json", "r") as file:
    loss_og = json.load(file)

with open(mode, "r") as file:
    loss_scaled = json.load(file)

# Extract data for plotting
epochs_og = range(len(loss_og["losses"]))
epochs_scaled = range(len(loss_scaled["losses"]))

"""
# Plot loss
plt.figure(figsize=(12, 8))  # Set figure size to 12x8 inches
plt.plot(epochs_og, loss_og["losses"], label="Original Model - Loss")
plt.plot(epochs_scaled, loss_scaled["losses"], label="Scaled Model - Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Comparison")
plt.legend()
plt.savefig(save_loss_path)  # Save the figure as 'loss.png'

# Plot accuracies
plt.figure(figsize=(12, 8))  # Set figure size to 12x8 inches
plt.plot(epochs_og, loss_og["accuracies"], label="Original Model - Accuracy")
plt.plot(epochs_scaled, loss_scaled["accuracies"], label="Scaled Model - Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison")
plt.legend()
plt.savefig(save_accuracy_path)  # Save the figure as 'accuracy.png'

# Plot plasticity per task
plt.figure(figsize=(12, 8))  # Set figure size to 12x8 inches
plt.plot(epochs_og, loss_og["plasticity_per_task"], label="Original Model - Plasticity per Task")
plt.plot(epochs_scaled, loss_scaled["plasticity_per_task"], label="Scaled Model - Plasticity per Task")
plt.xlabel("Epochs")
plt.ylabel("Plasticity per Task")
plt.title("Plasticity per Task Comparison")
plt.legend()
plt.savefig(save_plasticity_path)  # Save the figure as 'plasticity.png'
"""
# Plot plasticity per task
plt.figure(figsize=(12, 8))  # Set figure size to 12x8 inches
plt.plot(epochs_og, loss_og["n_dead_units_per_task"], label="Original Model - n_dead_units_per_task")
plt.plot(epochs_scaled, loss_scaled["n_dead_units_per_task"], label="Scaled Model - n_dead_units_per_task")
plt.xlabel("Epochs")
plt.ylabel("n_dead_units_per_task ")
plt.title("n_dead_units_per_task Comparison")
plt.legend()
plt.savefig("n_dead_units_per_task.png")  # Save the figure as 'plasticity.png'