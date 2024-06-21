# one_max.py

# %%

# Step 1: Setup and imports
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mlrose_hiive import DiscreteOpt, GARunner, MIMICRunner, OneMax, RHCRunner, SARunner

# %%

# Step 2: Define the OneMax problem
length = 50  # Length of the bit string
fitness = OneMax()
problem = DiscreteOpt(length=length, fitness_fn=fitness, maximize=True, max_val=2)

# %%

# Step 3: Randomized Hill Climbing Implementation
restart_list = [0, 5, 10]
rhc_curves_list = []
rhc_times = []
for restarts in restart_list:
    start_time = time.time()
    rhc = RHCRunner(
        problem=problem,
        experiment_name="RHC",
        output_directory="./one_max",
        seed=42,
        iteration_list=2 ** np.arange(12),
        max_attempts=500,
        restart_list=[restarts],
    )
    _, rhc_curves = rhc.run()
    end_time = time.time()
    rhc_curves_list.append((restarts, rhc_curves))
    rhc_times.append(end_time - start_time)

# %%

# Step 4: Simulated Annealing Implementation
temperature_list = [10, 50, 100]
sa_curves_list = []
sa_times = []
for temperature in temperature_list:
    start_time = time.time()
    sa = SARunner(
        problem=problem,
        experiment_name="SA",
        output_directory="./one_max",
        seed=42,
        iteration_list=2 ** np.arange(12),
        max_attempts=500,
        temperature_list=[temperature],
    )
    _, sa_curves = sa.run()
    end_time = time.time()
    sa_curves_list.append((temperature, sa_curves))
    sa_times.append(end_time - start_time)

# %%

# Step 5: Genetic Algorithm Implementation
population_sizes = [10, 100, 200]
ga_curves_list = []
ga_times = []
for pop_size in population_sizes:
    start_time = time.time()
    ga = GARunner(
        problem=problem,
        experiment_name="GA",
        output_directory="./one_max",
        seed=42,
        iteration_list=2 ** np.arange(12),
        max_attempts=500,
        population_sizes=[pop_size],
        mutation_rates=[0.1],
    )
    _, ga_curves = ga.run()
    end_time = time.time()
    ga_curves_list.append((pop_size, ga_curves))
    ga_times.append(end_time - start_time)

# %%

# Step 6: MIMIC Implementation
keep_pct_list = [0.1, 0.3, 0.5]
mimic_curves_list = []
mimic_times = []
for keep_pct in keep_pct_list:
    start_time = time.time()
    mimic = MIMICRunner(
        problem=problem,
        experiment_name="MIMIC",
        output_directory="./one_max",
        seed=42,
        iteration_list=2 ** np.arange(12),
        max_attempts=500,
        keep_percent_list=[keep_pct],
        population_sizes=[200],
    )
    _, mimic_curves = mimic.run()
    end_time = time.time()
    mimic_curves_list.append((keep_pct, mimic_curves))
    mimic_times.append(end_time - start_time)

# %%

# Step 7: Visualization of Results
plt.figure(figsize=(20, 20))
sns.set_style("whitegrid")
colors = sns.color_palette(
    "husl", 12
)  # Using a colorful palette with 12 distinct colors

# Plot for RHC
for i, (restarts, rhc_curves) in enumerate(rhc_curves_list):
    plt.subplot(4, 3, i + 1)
    plt.plot(rhc_curves["Fitness"], label=f"RHC Restarts={restarts}", color=colors[i])
    plt.title(f"RHC Restarts={restarts}")
    plt.xlabel("Iterations")
    plt.ylabel("Fitness")
    plt.legend()

# Plot for SA
for i, (temperature, sa_curves) in enumerate(sa_curves_list):
    plt.subplot(4, 3, i + 4)
    plt.plot(
        sa_curves["Iteration"],
        sa_curves["Fitness"],
        label=f"SA Temp={temperature}",
        color=colors[i + 3],
    )
    plt.title(f"SA Temperature={temperature}")
    plt.xlabel("Iterations")
    plt.ylabel("Fitness")
    plt.legend()

# Plot for GA
for i, (pop_size, ga_curves) in enumerate(ga_curves_list):
    plt.subplot(4, 3, i + 7)
    plt.plot(
        ga_curves["Iteration"],
        ga_curves["Fitness"],
        label=f"GA PopSize={pop_size}",
        color=colors[i + 6],
    )
    plt.title(f"GA Population Size={pop_size}")
    plt.xlabel("Iterations")
    plt.ylabel("Fitness")
    plt.legend()

# Plot for MIMIC
for i, (keep_pct, mimic_curves) in enumerate(mimic_curves_list):
    plt.subplot(4, 3, i + 10)
    plt.plot(
        mimic_curves["Iteration"],
        mimic_curves["Fitness"],
        label=f"MIMIC KeepPct={keep_pct}",
        color=colors[i + 9],
    )
    plt.title(f"MIMIC KeepPct={keep_pct}")
    plt.xlabel("Iterations")
    plt.ylabel("Fitness")
    plt.legend()

plt.tight_layout()
plt.show()

# %%

# Step 8: Visualization of Execution Times
labels = (
    [f"RHC Restarts={r}" for r in restart_list]
    + [f"SA Temp={t}" for t in temperature_list]
    + [f"GA PopSize={p}" for p in population_sizes]
    + [f"MIMIC KeepPct={k}" for k in keep_pct_list]
)
times = rhc_times + sa_times + ga_times + mimic_times

plt.figure(figsize=(10, 6))
plt.bar(labels, times, color=colors)
plt.title("Execution Time Comparison")
plt.xlabel("Algorithm Configuration")
plt.ylabel("Time (seconds)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
