import json
import matplotlib.pyplot as plt


# Load data from the provided JSON files
with open("kernel_learner.json", "r") as file:
    loss_og = json.load(file)

with open("loss_adativeNoisescaled.json", "r") as file:
    loss_scaled = json.load(file)

# Extract data for plotting
epochs_og = range(len(loss_og["losses"]))
epochs_scaled = range(len(loss_scaled["losses"]))

plt.figure(figsize=(12, 8))  # Set figure size to 12x8 inches
plt.plot(epochs_og, loss_og["grad_l0_per_task"], label="Original Model - grad_l0_per_task")
plt.plot(epochs_scaled, loss_scaled["grad_l0_per_task"], label="Scaled Model - grad_l0_per_task")
plt.xlabel("Epochs")
plt.ylabel("grad_l0_per_task ")
plt.title("grad_l0_per_task Comparison")
plt.legend()
plt.savefig("kernelavg_adaptive_comparison/grad_l0_per_task.png")  

plt.figure(figsize=(12, 8))  # Set figure size to 12x8 inches
plt.plot(epochs_og, loss_og["grad_l1_per_task"], label="Original Model - grad_l2_per_task")
plt.plot(epochs_scaled, loss_scaled["grad_l1_per_task"], label="Scaled Model - grad_l2_per_task")
plt.xlabel("Epochs")
plt.ylabel("grad_l1_per_task ")
plt.title("grad_l1_per_task Comparison")
plt.legend()
plt.savefig("kernelavg_adaptive_comparison/grad_l1_per_task.png")  

plt.figure(figsize=(12, 8))  # Set figure size to 12x8 inches
plt.plot(epochs_og, loss_og["grad_l2_per_task"], label="Original Model - grad_l2_per_task")
plt.plot(epochs_scaled, loss_scaled["grad_l2_per_task"], label="Scaled Model - grad_l2_per_task")
plt.xlabel("Epochs")
plt.ylabel("grad_l2_per_task ")
plt.title("grad_l2_per_task Comparison")
plt.legend()
plt.savefig("kernelavg_adaptive_comparison/grad_l2_per_task.png")  


