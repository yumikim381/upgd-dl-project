import json
import matplotlib.pyplot as plt

# Load data from the provided JSON files
with open("loss_original.json", "r") as file:
    loss_og = json.load(file)

with open("loss_grad_scaled.json", "r") as file:
    loss_scaled = json.load(file)

# Extract data for plotting
epochs_og = range(len(loss_og["losses"]))
epochs_scaled = range(len(loss_scaled["losses"]))

# Plot loss
plt.figure(figsize=(12, 8))  # Set figure size to 12x8 inches
plt.plot(epochs_og, loss_og["losses"], label="Original Model - Loss")
plt.plot(epochs_scaled, loss_scaled["losses"], label="Scaled Model - Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Comparison")
plt.legend()
plt.savefig("grad_loss.png")  # Save the figure as 'loss.png'

# Plot accuracies
plt.figure(figsize=(12, 8))  # Set figure size to 12x8 inches
plt.plot(epochs_og, loss_og["accuracies"], label="Original Model - Accuracy")
plt.plot(epochs_scaled, loss_scaled["accuracies"], label="Scaled Model - Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison")
plt.legend()
plt.savefig("grad_accuracy.png")  # Save the figure as 'accuracy.png'

# Plot plasticity per task
plt.figure(figsize=(12, 8))  # Set figure size to 12x8 inches
plt.plot(epochs_og, loss_og["plasticity_per_task"], label="Original Model - Plasticity per Task")
plt.plot(epochs_scaled, loss_scaled["plasticity_per_task"], label="Scaled Model - Plasticity per Task")
plt.xlabel("Epochs")
plt.ylabel("Plasticity per Task")
plt.title("Plasticity per Task Comparison")
plt.legend()
plt.savefig("grad_plasticity.png")  # Save the figure as 'plasticity.png'
