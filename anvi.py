import numpy as np
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import os


# Define the log joint probability function
def log_joint(x, theta, z, alpha = 1):
    # Shift z to compute differences (prepend 0)

    zero_tensor = torch.zeros((z.size(0), 1), device=z.device)
    # Concatenate the zero tensor with z[:-1] along the first dimension
    z_shifted = torch.cat([zero_tensor, z[:, :-1]], dim=1)

    # Compute the log joint probability
    log_prob_x = -0.5 / sigma ** 2 * torch.sum((x - theta - torch.sin(z)) ** 2, dim=1)
    log_prob_z = -0.5 / tau ** 2 * torch.sum((z - 0.5*z_shifted) ** 2, dim=1)

    return alpha * log_prob_x + log_prob_z


# Define the log variational distribution (q)
def log_q(theta, z, nu_mean_theta, nu_sd_theta, nu_mean_z, nu_sd_z):
    # Compute log_q for theta
    log_q_theta = torch.sum(
        -0.5 * torch.log(2 * torch.pi * nu_sd_theta ** 2) - 0.5 * ((theta - nu_mean_theta) ** 2) / (nu_sd_theta ** 2),
        dim=1
    )
    log_q_z = torch.sum(
        -0.5 * torch.log(2 * torch.pi * nu_sd_z ** 2) - 0.5 * ((z - nu_mean_z) ** 2) / (nu_sd_z ** 2),
        dim=1
    )

    return log_q_theta + log_q_z


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Inference_NN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Inference_NN, self).__init__()

        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean = nn.Linear(hidden_dim, 1)
        self.FC_sd_log = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h_ = F.relu(self.FC_input(x))
        h_ = F.relu(self.FC_input2(h_))
        mean = self.FC_mean(h_).reshape(N)
        sd_log = self.FC_sd_log(h_).reshape(N)

        return mean, sd_log


class Model(nn.Module):
    def __init__(self, input_dim = 1,
                 use_avi = False, hidden_dim = 0, const_z = False, N_mc = 100, alpha = 1):
        super(Model, self).__init__()
        self.use_avi = use_avi
        self.const_z = const_z
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.alpha = alpha

        # variational parameters for q(theta)
        self.nu_mean_theta = torch.nn.Parameter(torch.randn(1))
        self.nu_sd_theta_log = torch.nn.Parameter(torch.randn(1))

        # variational parameters for q(z)
        if (const_z and (not use_avi)):
            self.nu_mean_z = torch.nn.Parameter(torch.randn(1))
            self.nu_sd_z_log = torch.nn.Parameter(torch.randn(1))
        elif (use_avi):
            self.inference_nn = Inference_NN(input_dim, hidden_dim)
        else:
            self.nu_mean_z = torch.nn.Parameter(torch.randn(N))
            self.nu_sd_z_log = torch.nn.Parameter(torch.randn(N))

        self.N_mc = N_mc

    def reparam(self, nu_mean_z, nu_sd_z, nu_mean_theta, nu_sd_theta, N_mc):
        epsilon = torch.randn((N_mc, N)).to(device)
        z = nu_mean_z + nu_sd_z * epsilon

        epsilon_theta = torch.randn((N_mc, 1)).to(device)
        theta = nu_mean_theta + nu_sd_theta * epsilon_theta

        return z, theta

    def variational_z(self, x):
        if (self.use_avi and self.input_dim == 1):
            nu_mean_z, nu_sd_z_log = self.inference_nn(x.reshape(N, 1))
        elif (self.use_avi and self.input_dim == 2):
            x1 = torch.cat((torch.tensor([0]), x[:-1]),dim = 0)
            x_input = torch.transpose(torch.stack([x1, x]), 0, 1)
            nu_mean_z, nu_sd_z_log = self.inference_nn(x_input)
        elif (self.use_avi and self.input_dim == 3):
            x1 = torch.cat((torch.tensor([0]), x[:-1]),dim = 0)
            x2 = torch.cat((torch.tensor([0]), x1[:-1]),dim = 0)
            x_input = torch.transpose(torch.stack([x2, x1, x]), 0, 1)
            nu_mean_z, nu_sd_z_log = self.inference_nn(x_input)
        elif (self.const_z):
            nu_mean_z = self.nu_mean_z.repeat(N)
            nu_sd_z_log = self.nu_sd_z_log.repeat(N)
        else:
            nu_mean_z = self.nu_mean_z
            nu_sd_z_log = self.nu_sd_z_log
        return nu_mean_z, torch.exp(nu_sd_z_log)

    def compute_elbo(self, x):
        nu_mean_z, nu_sd_z = self.variational_z(x)
        z, theta = self.reparam(nu_mean_z, nu_sd_z,
                                self.nu_mean_theta, torch.exp(self.nu_sd_theta_log),
                                self.N_mc)
        log_p = log_joint(x, theta, z, alpha = self.alpha)
        log_q_ = log_q(theta, z, self.nu_mean_theta, torch.exp(self.nu_sd_theta_log), nu_mean_z, nu_sd_z)
        elbo = log_p - log_q_
        return torch.mean(elbo)



def run_vi(seed, n_iter_optimizer, use_avi=False, input_dim=1, hidden_dim=0, const_z=False,
           print_output=False,alpha=1):
    torch.manual_seed(seed)

    model = Model(input_dim, use_avi, hidden_dim, const_z,alpha=alpha).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_saved = torch.empty(n_iter_optimizer)

    start_time = time.time()
    for i in range(n_iter_optimizer):
        loss = -model.compute_elbo(x.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_saved[i] = loss.data

        if i % 1000 == 0 and print_output:
            print("Loss:", loss_saved[i].item())

    end_time = time.time()
    run_time = end_time - start_time

    return loss_saved, model, run_time


# Sample sizes to test
sample_sizes = [100, 200, 400, 500]
methods = {
    "MFVI": {"use_avi": False, "const_z": False, "input_dim": 1, "hidden_dim": 0,"alpha": 0.99},
    "const": {"use_avi": False, "const_z": True, "input_dim": 1, "hidden_dim": 0,"alpha": 0.99},
    "ANVI,k=1": {"use_avi": True, "const_z": False, "input_dim": 1, "hidden_dim": 20,"alpha": 0.99},
    "ANVI,k=2": {"use_avi": True, "const_z": False, "input_dim": 2, "hidden_dim": 20,"alpha": 0.99},
    "ANVI,k=3": {"use_avi": True, "const_z": False, "input_dim": 3, "hidden_dim": 20,"alpha": 0.99}
}

results = {}
all_elbo_values = {}

for N in sample_sizes:
    print(f"Running experiments for sample size N={N}")
    # Simulate new data for the given sample size
    tau = torch.tensor(0.5)
    sigma = torch.tensor(0.7)
    theta = torch.tensor(2)
    z = torch.zeros(N)
    x = torch.zeros(N)
    z[0] = torch.normal(mean=torch.tensor(0.0), std=torch.tensor(1.0))
    x[0] = theta + torch.normal(mean=torch.sin(z[0]), std=sigma)
    for i in range(N - 1):
        z[i + 1] = z[i] + torch.normal(mean=torch.tensor(0.0), std=tau)
        x[i + 1] = theta + torch.normal(mean=torch.sin(z[i + 1]), std=sigma)
    results[N] = {}
    all_elbo_values[N] = {}

    for method, params in methods.items():
        print(f"  Running {method}...")
        torch.manual_seed(2025)
        loss_saved, _, run_time = run_vi(seed=2025, n_iter_optimizer=int(4e4), **params)

        results[N][method] = {"loss": loss_saved[-1].item(), "time": run_time}
        all_elbo_values[N][method] = loss_saved.numpy()
        print(f"    Final loss: {loss_saved[-1].item()}, Time: {run_time:.2f}s")

# Convert results to structured format
loss_matrix = np.zeros((len(sample_sizes), len(methods)))
time_matrix = np.zeros((len(sample_sizes), len(methods)))
method_names = list(methods.keys())

for i, N in enumerate(sample_sizes):
    for j, method in enumerate(method_names):
        loss_matrix[i, j] = -results[N][method]["loss"]
        time_matrix[i, j] = results[N][method]["time"]

# Plot ELBO vs Sample Size and Running Time
os.chdir('/home/sfan211/scratch/amortization/plot')
colors = ["black", "red", "blue", "green", "orange"]
plt.figure(figsize=(8, 6))
for j, method in enumerate(method_names):
    plt.plot(sample_sizes,
             loss_matrix[:, j],
             marker="o",
             label=method,
             color=colors[j],
             linewidth=2,
             alpha=0.7 if j >= 2 else 0.9)  # more transparent for alpha≠1 methods

plt.xlabel("Sample Size (N)")
plt.ylabel("ELBO")
plt.xscale("log")
plt.legend(fontsize=8)
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("anvi_hmm_gnn_elbo_vs_sample_size.png", dpi=300)
plt.show()

plt.figure(figsize=(8, 6))
for j, method in enumerate(method_names):
    plt.plot(sample_sizes, time_matrix[:, j], marker="o", label=method, color=colors[j])

plt.xlabel("Sample Size (N)")
plt.ylabel("Running Time (s)")
plt.xscale("log")
plt.legend()
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.savefig("anvi_hmm_alpha_running_time_vs_sample_size.png", dpi=300)
plt.show()

# Plot ELBO vs Iteration for each sample size
# Define linestyles
plt.figure(figsize=(12, 8))

for i, N in enumerate(sample_sizes):
    plt.subplot(2, 2, i + 1)
    for j, method in enumerate(method_names):  # method_names = list(methods.keys())
        transparency = 0.8 if j < 2 else 0.5  # first two methods more solid, others more transparent
        plt.plot(all_elbo_values[N][method],
                 label=method,
                 color=colors[j % len(colors)],
                 linewidth=1.5,
                 alpha=transparency)

    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("-ELBO(log)")
    plt.title(f"N={N}")
    plt.legend(fontsize=8)
    plt.grid(True, which="both", ls="--", lw=0.5)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("anvi_hmm_alpha_elbo_vs_iteration_inverted_transparent.png", dpi=300)
plt.show()

output_dir = "/home/sfan211/scratch/amortization/results"
os.makedirs(output_dir, exist_ok=True)

np.savez(
    os.path.join(output_dir, "ANVI_experiment_results.npz"),
    sample_sizes=np.array(sample_sizes),
    method_names=np.array(method_names),
    loss_matrix=loss_matrix,
    time_matrix=time_matrix,
    # if you want all ELBO traces too:
    **{f"elbo_{N}": all_elbo_values[N] for N in sample_sizes}
)
