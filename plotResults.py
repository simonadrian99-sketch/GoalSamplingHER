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


data_future = extract_data_from_tb('logs/10x10/DQN+HER/future', 'Future')
data_episode = extract_data_from_tb('logs/10x10/DQN+HER/episode', 'Episode')
data_final = extract_data_from_tb('logs/10x10/DQN+HER/final', 'Final')

if data_future.empty and data_episode.empty and data_final.empty:
    print("Keine Daten gefunden! Prüfe deine Pfade.")
else:
    final_df = pd.concat([data_future, data_episode, data_final])
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
plt.savefig('HER_Comparison_Plot.pdf', bbox_inches='tight')
print("Plot erfolgreich als PDF gespeichert.")


plt.figure(figsize=(8, 6))

try:
    counts = np.load("logs/novelty_heatmap_final.npy")
    sns.heatmap(np.log1p(counts.T), cmap="YlGnBu",
                cbar_kws={'label': 'Log(Visit Counts)'})
    plt.title("Agent's State Coverage (Novelty Map)")
    plt.gca().invert_yaxis()
    plt.savefig("Heatmap_Visual.pdf")
except FileNotFoundError:
    print("Heatmap-Datei nicht gefunden.")


plt.show()


def plot_buffer_heatmap(model, save_path="novelty_heatmap.pdf"):

    counts = model.replay_buffer.visit_counts.T

    plt.figure(figsize=(8, 6))

    log_counts = np.log1p(counts)

    sns.heatmap(
        log_counts,
        annot=False,
        cmap="YlGnBu",
        cbar_kws={'label': 'Log(Visit Counts + 1)'}
    )

    plt.title("Agent's State Coverage (Novelty Map)")
    plt.xlabel("Grid X")
    plt.ylabel("Grid Y")
    plt.gca().invert_yaxis()

    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
