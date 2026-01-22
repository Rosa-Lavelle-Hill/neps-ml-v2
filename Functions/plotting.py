from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import base64
import seaborn as sns
import pandas as pd
import shap
from scipy.stats import pearsonr


def plot_hist(save_name, x, bins, save_path, title, xlim=None, ylim=None,
              fig_size=(4,4), xlab="", ylab='', fontsize=10, html=False, save_name_exceptions=[]):
    """Plots a histogram and saves to file. Also outputs a html image tag if html=True"""
    if isinstance(x, np.ndarray) == True:
        x = pd.Series(x.reshape((x.shape[0],)))
    plt.figure(figsize=fig_size)
    if xlim != None:
        plt.xlim(xlim)
    if ylim != None:
        plt.ylim(ylim)
    if (x.isnull().all() == False):
        plt.hist(x, bins = bins, color ="skyblue", alpha=0.5)
        plt.title(title, fontsize=fontsize)
        plt.xlabel(xlab, fontsize=fontsize)
        plt.ylabel(ylab, fontsize=fontsize)
        if save_name in save_name_exceptions:
            print('odd case')
        elif x.dtype == "str":
            print('string')
        elif (x.dtype == "float64") or (x.dtype == "int"):

                plt.axvline(x=np.mean(x) - np.std(x), ls="--", color='#2ca02c', alpha=0.7)
                plt.axvline(x=np.mean(x) + np.std(x), ls="--", color='#2ca02c', alpha=0.7)
                plt.axvline(x=np.mean(x), ls="-", color='red', alpha=0.7)
                plt.axvline(x=np.percentile(x, 5), ls="dotted", color='lightblue', alpha=0.7)
                plt.axvline(x=np.percentile(x, 95), ls="dotted", color='lightblue', alpha=0.7)
                plt.tight_layout()
                plt.savefig(save_path+save_name+".png")
                plt.clf()
                plt.cla()
                plt.close()
        else:
            print("not int or float")
    else:
            print("nas")
    if html==True:
            data_uri = base64.b64encode(open(save_path+save_name+".png", 'rb').read()).decode('utf-8')
            image_tag = '<img src="data:image/png;base64,{0}">'.format(data_uri)
            return image_tag
    else:
        return



def plot_scatt(x, y, save_path, save_name, xlab, ylab, data=None, fontsize=12, jitter=False, jitter_scale=0.1):
    if data != None:
        x = data[x]
        y = data[y]

    # computes correlations excluding NAs
    valid_indices = np.logical_and(~np.isnan(x), ~np.isnan(y))
    x_valid = x[valid_indices]
    y_valid = y[valid_indices]
    corr, _ = pearsonr(x_valid, y_valid)
    title = "Pearson r Correlation = {}".format(round(corr, 2))

    if jitter == True:
        x_jitter = x + np.random.normal(loc=0, scale=jitter_scale, size=len(x))
        y_jitter = y + np.random.normal(loc=0, scale=jitter_scale, size=len(y))
        plt.scatter(x_jitter, y_jitter, s=0.1, marker='.', alpha=0.5)

    else:
        plt.scatter(x, y)
    plt.title(title, fontsize=fontsize)
    plt.xlabel(xlab, fontsize=fontsize)
    plt.ylabel(ylab, fontsize=fontsize)
    plt.savefig(save_path + save_name + ".png")
    plt.clf()
    plt.cla()
    plt.close()



def plot_scat(x, y, x_lab, y_lab, save_path, save_name):
   
    save_path = Path(save_path)
    # Ensure directory exists
    save_path.mkdir(parents=True, exist_ok=True)
    save_name = str(save_name)
    if save_name.lower().endswith(".png"):
        filename = save_name
    else:
        filename = f"{save_name}.png"

    fig, ax = plt.subplots()
    plt.scatter(x, y)
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    fig.tight_layout() 
    
    fig.savefig(save_path / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return


def plot_label_reg_sns(x, y, x_lab, y_lab,
                    save_path, save_name, print_cor=True, same_axis=False,
                       country=None, anov_var='Country_Name',
                       cor=True, ano=True):
    plt.figure(figsize=(20, 20))
    fig, ax = plt.subplots()
    sns.regplot(x=x, y=y)
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    if same_axis == True:
        if x.min() <= y.min():
            min = x.min()
        else:
            min = y.min()
        if x.max() >= y.max():
            max = x.max()
        else:
            max = y.max()
        plt.xlim(min, max)
        plt.ylim(min, max)

    if ano == True:
        n = list(anov_var)
        for i, txt in enumerate(n):
            plt.annotate(txt, (np.array(x)[i], np.array(y)[i]))
    if cor == True:
        # computes correlations excluding NAs:
        corr = round(abs(pd.Series(x).corr(pd.Series(y), method='pearson', min_periods=None)), 2)
        if print_cor == True:
            x_str = x.name
            y_str = y.name
            print('correlation between x ({}) and y ({}) = {}'.format(x_str, y_str, str(corr)))
        if type(country) is list:
            if len(list(country.unique())) == 1:
                title = "{}: Pearson r Correlation = {}".format(np.array(country)[0], corr)
        else:
            title = "Pearson r Correlation = {}".format(corr)
        plt.title(title, fontsize=12)
    fig.tight_layout()
    filename = str(save_name) + ".png"
    plt.savefig(save_path / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
# todo: check the correlation...^^^^^


def plot_permutation(perm_imp_df, save_path, save_name, figsize=(8, 6)):
    y_ticks = np.arange(0, perm_imp_df.shape[0])
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(y_ticks, perm_imp_df["importance_mean"], color="dodgerblue")
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(perm_imp_df["Feature"])
    ax.set_title("Permutation Importance (test set)")
    ax.set_xlabel("Permutation Importance")
    fig.tight_layout()
    filename = str(save_name) + ".png"
    save_path_p = Path(save_path)
    save_path_p.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path_p / filename, dpi=300, bbox_inches="tight")
    plt.clf()
    plt.cla()
    plt.close()
    return


def plot_permutation_bars_orig(result, save_path, save_name, plot_n_features, vars):
    perm_imp_df = pd.DataFrame({
        'feature': vars,
        'importance_mean': result['importances_mean'],
        'importance_std': result['importances_std']
    })

    perm_imp_df.sort_values(by='importance_mean', ascending=False, inplace=True)
    non_zero = len(perm_imp_df[perm_imp_df['importance_mean'] > 0])
    if non_zero < plot_n_features:
        plot_n_features = non_zero
    perm_imp_df = perm_imp_df[0:plot_n_features]
    perm_imp_df.sort_values(by='importance_mean', ascending=True, inplace=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.barh(perm_imp_df['feature'], perm_imp_df['importance_mean'],
             xerr=perm_imp_df['importance_std'], align='center')
    plt.xlabel('Importance')
    plt.title('Permutation Importances (test set)')
    plt.tight_layout()
    plt.savefig(save_path + save_name + ".png")
    plt.clf()
    plt.cla()
    plt.close()


def plot_permutation_bars(perm_imp_df, save_path, save_name, plot_n_features):

    perm_imp_df.sort_values(by='importance_mean', ascending=False, inplace=True)
    non_zero = len(perm_imp_df[perm_imp_df['importance_mean'] > 0])
    if non_zero < plot_n_features:
        plot_n_features = non_zero
    perm_imp_df = perm_imp_df[0:plot_n_features]
    perm_imp_df.sort_values(by='importance_mean', ascending=True, inplace=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    perm_imp_df.dropna(inplace=True)
    # todo: ^^^^^^^^^^why na?
    plt.barh(perm_imp_df['Feature'], perm_imp_df['importance_mean'],
             xerr=perm_imp_df['importance_std'], align='center')
    plt.xlabel('Importance')
    plt.title(f'Permutation Importance (test set)')
    plt.tight_layout()
    plt.savefig(save_path + save_name + ".png")
    plt.clf()
    plt.cla()
    plt.close()


def plot_SHAP(shap_dict, col_list, plot_type, n_features, data,
              save_path, save_name, figsize=(20, 20), title=''):
    """
    Plots SHAP values from a dictionary of values
    :param shap_dict: dictionary output from the SHAP function
    :param col_list: list of column/feature names
    :param plot_type: the type of SHAP plot (e.g., summary, bar, violin)
    :param n_features: the number of features to display (plots most important first)
    :param save_path: path where plot should be saved
    :param save_name: name of plot to be saved
    :param figsize: size of figure as a tuple
    :param title: title of plot
    """
    plt.figure(figsize=figsize)
    shap.summary_plot(shap_dict, feature_names=col_list, show=False,
                      plot_type=plot_type, max_display=n_features)
    plt.title(title)
    plt.tight_layout()
    save_path_p = Path(save_path)
    save_path_p.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path_p / save_name, bbox_inches='tight')
    plt.clf()
    plt.cla()
    plt.close()


def plot_results(y, data, colour, save_path, save_name,
                               xlab, ylab, title, x_ticks, legend=False,
                               fontsize=12, legend_pos="lower right"):
    palette = ["plum", "cornflowerblue", "coral", "mediumaquamarine", "lightsteelblue", "khaki"]

    sns.set_palette(palette)
    fig, ax = plt.subplots()
    plt.figure(figsize=(7, 4))

    data = data.transpose().reset_index()
    data.rename(columns={'index': 'Model'}, inplace=True)

    g=sns.barplot(x="Model", y=y, data=data, hue=colour, dodge=False,
                  palette=sns.color_palette(palette, len(x_ticks)))

    plt.title(title, fontsize=fontsize)
    plt.xlabel(xlab, fontsize=fontsize)
    plt.ylabel(ylab, fontsize=fontsize)

    g.set_xticklabels(x_ticks, rotation=45, fontsize=fontsize)

    plt.tight_layout()
    if legend == True:
        plt.legend(loc=legend_pos)
    else:
        plt.legend([], [], frameon=False)
    plt.savefig(save_path + save_name + ".png")
    plt.clf()
    plt.cla()
    plt.close()


def plot_group_perm_importance(group_importance, save_path,
                               save_name,
                               title="Group Importance"):
    """
    Plot group importance as a bar chart.

    Parameters:
        group_importance (dict): Dictionary of group names and their importance scores.
        title (str): Title of the plot.
    """
    # Convert dictionary to sorted lists for plotting
    groups = list(group_importance.keys())
    importances = list(group_importance.values())

    # Sort groups by importance (descending)
    sorted_indices = np.argsort(importances)[::-1]
    sorted_groups = [groups[i] for i in sorted_indices]
    sorted_importances = [importances[i] for i in sorted_indices]

    # Create a bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=sorted_importances, y=sorted_groups, palette="viridis")
    plt.title(title, fontsize=16)
    plt.xlabel("Permutation Importance", fontsize=14)
    plt.ylabel("Group", fontsize=14)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    filename = str(save_name) + ".png"
    save_path_p = Path(save_path)
    save_path_p.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path_p / filename, dpi=300, bbox_inches="tight")
    plt.clf()
    plt.cla()
    plt.close()


def plot_group_SHAP_importance(group_shap_values, save_name, save_path):
    # Sort values in descending order for top-to-bottom ranking
    group_shap_values = group_shap_values.sort_values(ascending=True)
    # plot
    plt.figure(figsize=(8, 6))
    plt.barh(group_shap_values.index,
             group_shap_values.values,
             color='skyblue')
    plt.xlabel('Group Importance (Sum of Absolute SHAP Values)')
    plt.ylabel('Groups')
    plt.title('SHAP Value Importance by Group')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path + save_name + ".png")
    plt.clf()
    plt.cla()
    plt.close()