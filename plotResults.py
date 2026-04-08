import os
import pandas as pd
from matplotlib import ticker
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator


def extract_data_from_tb(log_dir, strategy_name):

    all_runs_data = []

    for run_folder in os.listdir(log_dir):
        run_path = os.path.join(log_dir, run_folder)
        if os.path.isdir(run_path):
            ea = event_accumulator.EventAccumulator(run_path)
            ea.Reload()

            if 'rollout/success_rate' in ea.Tags()['scalars']:
                events = ea.Scalars('rollout/success_rate')
                df = pd.DataFrame([(e.step, e.value) for e in events],
                                  columns=['step', 'success_rate'])
                df['Strategy'] = strategy_name
                df['Run_ID'] = run_folder
                all_runs_data.append(df)

    return pd.concat(all_runs_data) if all_runs_data else pd.DataFrame()


data_future = extract_data_from_tb(
    'logs/10x10_random/DQN+HER/future', 'Future')
data_episode = extract_data_from_tb(
    'logs/10x10_random/DQN+HER/episode', 'Episode')
data_final = extract_data_from_tb('logs/10x10_random/DQN+HER/final', 'Final')
data_novelty = extract_data_from_tb(
    'logs/10x10_random/DQN+HER/novelty', 'Novelty')

if data_future.empty and data_episode.empty and data_final.empty and data_novelty.empty:
    print("Keine Daten gefunden!")
else:
    final_df = pd.concat([data_future, data_episode, data_final, data_novelty])
    final_df['step'] = pd.to_numeric(final_df['step'])
    final_df['success_rate'] = pd.to_numeric(final_df['success_rate'])
    final_df = final_df.sort_values(['Strategy', 'Run_ID', 'step'])

bin_size = 5000
final_df['Steps'] = (final_df['step'] // bin_size) * bin_size
final_df['Success_Rate'] = final_df.groupby(['Strategy', 'Run_ID'])['success_rate'].transform(
    lambda x: x.rolling(window=10, min_periods=1).mean()
)


plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")

ax = sns.lineplot(
    data=final_df,
    x="Steps",
    y="Success_Rate",
    hue="Strategy",
    errorbar=("ci", 95)
)

ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

plt.title('Comparison of HER Strategies (5 Seeds per Strategy)', fontsize=14)
plt.xlabel('Training Steps', fontsize=12)
plt.ylabel('Success Rate', fontsize=12)
plt.ylim(0, 1.05)
plt.savefig('logs/10x10_random/DQN+HER/HER_Comparison_Plot.pdf',
            bbox_inches='tight')
print("Plot erfolgreich als PDF gespeichert.")


plt.figure(figsize=(8, 6))

RUN_ID = 'run_20260406-131918_0'
ENV_SCENARIO = '10x10_random'
ALGORITHM = 'DQN+HER'
STRATEGY = 'novelty'
STEP_COUNT = 30000
ENV_TYPE = "standard"  # "KeyGoal" oder "standard"

try:
    if ENV_TYPE == "KeyGoal":
        file_path = f"logs/{ENV_TYPE}/{ENV_SCENARIO}/{ALGORITHM}/{STRATEGY}/{RUN_ID}/heatmap_{STEP_COUNT}.npy"
    else:
        file_path = f"logs/{ENV_SCENARIO}/{ALGORITHM}/{STRATEGY}/{RUN_ID}/heatmap_{STEP_COUNT}.npy"
    counts = np.load(file_path)

    if counts.ndim == 3:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        data0 = counts[1:11, 1:11, 0].T
        sns.heatmap(np.log1p(data0), cmap="YlGnBu", ax=ax1,
                    cbar_kws={'label': 'Log(Visit Counts)'})
        ax1.set_title("Exploration OHNE Schlüssel")
        ax1.invert_yaxis()

        data1 = counts[1:11, 1:11, 1].T
        sns.heatmap(np.log1p(data1), cmap="OrRd", ax=ax2,
                    cbar_kws={'label': 'Log(Visit Counts)'})
        ax2.set_title("Exploration MIT Schlüssel")
        ax2.invert_yaxis()

        plt.suptitle(
            f"Agent's State Coverage - Step {STEP_COUNT}", fontsize=16)
        plt.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(np.log1p(counts[1:11, 1:11].T), cmap="YlGnBu", ax=ax,
                    cbar_kws={'label': 'Log(Visit Counts)'})

    save_path = f"logs/{ENV_SCENARIO}/{ALGORITHM}/{STRATEGY}/{RUN_ID}/Heatmap_3D_{STEP_COUNT}.pdf"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Heatmap für Step {STEP_COUNT} gespeichert.")
    plt.show()
except FileNotFoundError:
    print(f"Heatmap-Datei in {file_path} nicht gefunden.")


def plot_buffer_heatmap(model, save_path="novelty_heatmap.pdf"):
    counts = model.replay_buffer.visit_counts

    if counts.ndim == 3:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        sns.heatmap(np.log1p(counts[1:11, 1:11, 0].T), cmap="YlGnBu", ax=ax1)
        ax1.set_title("Ebene: Kein Schlüssel")
        ax1.invert_yaxis()

        sns.heatmap(np.log1p(counts[1:11, 1:11, 1].T), cmap="OrRd", ax=ax2)
        ax2.set_title("Ebene: Mit Schlüssel")
        ax2.invert_yaxis()
    else:
        plt.figure(figsize=(8, 6))
        sns.heatmap(np.log1p(counts[1:11, 1:11].T), cmap="YlGnBu")
        plt.gca().invert_yaxis()

    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
