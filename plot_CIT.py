import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from Functions.CIT import diagnose_degenerate_features
import matplotlib.lines as mlines
import os

# set to true if want to reduce computation time to test code -- will only run for a sample of features
sample_features = False
run_diagnostics = False
plot_pos_estimates_only = True
plot_group_colours = True


def normalize_group_name(group_name):
    """Map legacy/source labels to final group names used in plots."""
    if pd.isna(group_name):
        return "Unknown"
    raw = str(group_name).strip()
    rename_map = {
        "teacher/school": "Pedagogical",
        "student": "Student",
        "parents/home": "Home",
        "pCourseClass": "Pedagogical",
        "pTarget": "Student",
        "pParent": "Home",
    }
    return rename_map.get(raw, raw if raw else "Unknown")

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

# --- optional: map each feature to a variable group for group-colored plot ---
feature_groups = np.array(["Unknown"] * len(df_sorted), dtype=object)
if plot_group_colours:
    group_info_path = "Data/Meta/var_info_used_in_model_final_groups.csv"
    if os.path.exists(group_info_path):
        group_info = pd.read_csv(group_info_path)
        group_col = "Final Groups" if "Final Groups" in group_info.columns else "Variable_Group"
        if "var" in group_info.columns and group_col in group_info.columns:
            group_map = dict(
                zip(
                    group_info["var"].astype(str),
                    group_info[group_col].apply(normalize_group_name),
                )
            )
            feature_groups = df_sorted["feature"].astype(str).map(group_map).fillna("Unknown").to_numpy()

    unknown_features = df_sorted.loc[pd.Series(feature_groups).eq("Unknown"), "feature"].astype(str).tolist()
    if unknown_features:
        print(f"Features in Unknown group ({len(unknown_features)}):")
        for feat in unknown_features:
            print(f"  - {feat}")
    else:
        print("No features mapped to Unknown group.")

# --- get standard errors per feature ---
if 'std_error' in df_sorted.columns:
    se = df_sorted['std_error'].fillna(0).to_numpy()
else:
    se = np.zeros_like(y)

# --- build colored line segments (color by the variable at the left endpoint) ---
points = np.column_stack([x, y]).reshape(-1, 1, 2)         # shape (N,1,2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)  # shape (N-1,2,2)
segment_colors = np.where(sig[:-1], 'lime', 'red')

group_palette = {
    "Baseline": "#1f77b4",      # blue
    "Student": "#ff7f0e",       # orange
    "Home": "#2ca02c",          # green
    "Pedagogical": "#d62728",   # red
    "Unknown": "#7f7f7f",       # gray
}
group_segment_colors = [group_palette.get(g, group_palette["Unknown"]) for g in feature_groups[:-1]]

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


if plot_group_colours:
    fig2, ax2 = plt.subplots(figsize=(8, 5))

    # keep the same uncertainty band as reference
    ax2.fill_between(
        x,
        y - se,
        y + se,
        color='gray',
        alpha=0.35,
        label='±1 SE',
    )

    # thicker line so group colours are readable
    lc2 = LineCollection(segments, colors=group_segment_colors, linewidths=6)
    ax2.add_collection(lc2)

    # last marker in its feature group colour
    last_group_colour = group_palette.get(feature_groups[-1], group_palette["Unknown"])
    ax2.plot(x[-1], y[-1], marker='o', markersize=4, linestyle='None', color=last_group_colour)

    # coloured strip under the curve to show per-feature group membership
    for i, group_name in enumerate(feature_groups):
        strip_colour = group_palette.get(group_name, group_palette["Unknown"])
        ax2.axvspan(i + 0.5, i + 1.5, ymin=0.0, ymax=0.06, facecolor=strip_colour, alpha=0.5, linewidth=0)

    ax2.axhline(0, linewidth=1, linestyle='--', color='gray', alpha=0.6)
    ax2.set_xlim(1, len(x))
    ax2.set_ylim(1.4, 1.81)
    ax2.set_xlabel('Features sorted by most to least informative (by MSE difference estimate)')
    ax2.set_ylabel('Cumulative estimate of performance difference (MSE)')
    ax2.set_title(f'Cumulative {test} estimates by feature group for {model_name} model')

    # group legend + SE band legend
    group_handles = [
        mlines.Line2D([], [], color=group_palette[g], linewidth=6, label=g)
        for g in ["Baseline", "Student", "Home", "Pedagogical", "Unknown"]
    ]
    leg_gray_group = mlines.Line2D([], [], color='gray', linewidth=8, alpha=0.35, label='±1 SE band')
    # Lift legend above the bottom colour strip so both remain readable.
    ax2.legend(
        handles=group_handles + [leg_gray_group],
        frameon=True,
        loc='lower right',
        bbox_to_anchor=(1.0, 0.16),
    )

    plt.tight_layout()
    plt.savefig(fig_save_path + "_group_colours" + pos + ".png", dpi=300)

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