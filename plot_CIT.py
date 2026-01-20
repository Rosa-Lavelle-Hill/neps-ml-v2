import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from Functions.CIT import diagnose_degenerate_features
import matplotlib.lines as mlines

# set to true if want to reduce computation time to test code -- will only run for a sample of features
sample_features = False
run_diagnostics = False
plot_pos_estimates_only = True

# --- prepare data ---
# start_string = '26_Nov_2024__16.45'
start_string = '12_Dec_2024__11.47_test'
save_path = "Sim-CIT/"
model_name = "HGB"
test = "CPI" # "RPT" or "CPI"

if sample_features == True:
    n_features = 10 # number of features to sample
    load_path = save_path + f"{test}/" + f'{start_string}_{model_name}_{n_features}_features.csv'
    fig_save_path = f"Sim-CIT/{test}/Plots/" + f'{start_string}_{model_name}_{n_features}feature_cumulative_CIT_estimates'
else:
    load_path = save_path + f"{test}/" + f'{start_string}_{model_name}_results.csv'
    fig_save_path = f"Sim-CIT/{test}/Plots/" + f'{start_string}_{model_name}_feature_cumulative_CIT_estimates'

# sort by most informative (largest estimate first)
df = pd.read_csv(load_path)
df_sorted = df.sort_values('estimate', ascending=False).reset_index(drop=False)

if plot_pos_estimates_only == True:
    df_sorted = df_sorted[df_sorted['estimate'] > 0]
    pos = "_pos_only_"
else:
    pos = ""

x = np.arange(1, len(df_sorted) + 1)                       # 1..N features
y = df_sorted['estimate'].cumsum().to_numpy()              # cumulative estimates
sig = (df_sorted['p_value'].to_numpy() < 0.05)             # significance mask

# --- get standard errors per feature ---
if 'std_error' in df_sorted.columns:
    se = df_sorted['std_error'].fillna(0).to_numpy()
else:
    se = np.zeros_like(y)

# --- build colored line segments (color by the variable at the left endpoint) ---
points = np.column_stack([x, y]).reshape(-1, 1, 2)         # shape (N,1,2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)  # shape (N-1,2,2)
segment_colors = np.where(sig[:-1], 'lime', 'red')

fig, ax = plt.subplots(figsize=(8, 5))
# --- add grey ±SE shading (non-cumulative) ---
ax.fill_between(
    x,
    y - se,
    y + se,
    color='gray',
    alpha=0.8,
    label='±1 SE',
)
# To dod: *2 to make more obvious?

# --- main colored line ---
lc = LineCollection(segments, colors=segment_colors, linewidths=3)
ax.add_collection(lc)

# draw last point so it’s visible even though there’s no outgoing segment
ax.plot(x[-1], y[-1], marker='o', markersize=3, linestyle='None', color=('lime' if sig[-1] else 'red'))

# baseline & cosmetics
ax.axhline(0, linewidth=1, linestyle='--', color='gray', alpha=0.6)
ax.set_xlim(1, len(x))
ax.set_ylim(1.4, 1.81)
ax.set_xlabel('Features sorted by most to least informative (by MSE difference estimate)')
ax.set_ylabel('Cumulative estimate of performance difference (MSE)')
ax.set_title(f'Cumulative {test} estimates by feature for {model_name} model')

# legend
leg_green = mlines.Line2D([], [], color='lime', linewidth=3, label='Significant (p<0.05)')
leg_red   = mlines.Line2D([], [], color='red', linewidth=3, label='Not significant')
leg_gray  = mlines.Line2D([], [], color='gray', linewidth=8, alpha=0.8, label='±1 SE band')
ax.legend(handles=[leg_green, leg_red, leg_gray], frameon=True, loc='lower right')

plt.tight_layout()
plt.savefig(fig_save_path + pos +".png", dpi=300)

# run diagnostics on 0 SE:
if run_diagnostics == True:
    print("Diagnosing features with zero standard error...")

    print("Loading data...")
    X_and_y = pd.read_csv("Data/Preprocessed/X_and_y.csv", index_col=[0])
    X = X_and_y.drop("y", axis=1)
    print("X shape: " + str(X.shape))
    y = X_and_y["y"]
    print("y shape: " + str(y.shape))

    # run diagnostics
    X.fillna(-999, inplace=True)  # simple imputation for testing
    deg = diagnose_degenerate_features(df, X)
    deg.to_csv(f"Sim-CIT/{test}/Diagnostics/" + f'{start_string}_{model_name}_{pos}degenerate_features.csv', index=False)

print("Done.")