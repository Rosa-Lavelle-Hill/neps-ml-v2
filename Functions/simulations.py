import numpy as np
import seaborn as sns
import pandas as pd
import contextlib
import sys
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance
from Params.Grids import sim_dt_param_grid, sim_rf_param_grid, sim_gb_param_grid
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree

def check_corr(X_and_y, save_path, save_name, X_feature_names, decimal_places):
    """
    Plots a heatmap of Pearson r correlation coefficients
    :param X_and_y: dataframe containing X and y features
    :param save_name: name of plot to be saved
    :param X_feature_names: list of feature names
    :param decimal_places: number (int) of decimal places to round to
    """
    cor = round(X_and_y.corr(method='pearson'), decimal_places)
    cor_cols = X_feature_names + ["y"]
    cor.columns = cor_cols
    cor.index = cor_cols
    cor.to_csv(save_path + "data_correlations.csv")
    ax = sns.heatmap(cor, linewidth=0.1, cmap="Oranges")
    plt.savefig(save_path + save_name + ".png")
    plt.clf()
    plt.cla()
    plt.close()
    return


def extract_coef(X, y, X_feature_names, decimal_places,
                 file_path,
                 file_name="coefficients.txt"):
    """
    Extracts and saves the regression coefficients from a fitted model
    :param X: X dataframe
    :param y: y dataframe
    :param X_feature_names: list of strings containing feature names
    :param decimal_places: integer, number of decimal places for rounding
    :param file_path: string for file path to save to
    :param file_name: string for file save name
    """
    # Create a LinearRegression object
    lr = LinearRegression()

    # Fit the model using the training data
    lr.fit(X, y)

    # Extract the coefficient values
    coef_list = list(np.round(lr.coef_, decimal_places))

    # Print the coefficient values
    with open(file_path+file_name, "w") as txt:
        txt.write("Coefficient values:\n")
        for feature_name, coef in zip(X_feature_names, coef_list):
            txt.write(f"{feature_name} = {coef}\n")
    return


def generate_regression_data(n_samples=100, n_features=10, random_state=42):
    np.random.seed(random_state)
    X = np.random.rand(n_samples, n_features)
    y = np.random.rand(n_samples) * 100  # Continuous target variable
    return pd.DataFrame(X), y

# Introduce missing values at different percentages
def introduce_missing_values(X, missing_percentage, random_state=42):
    np.random.seed(random_state)
    X_missing = X.copy()
    mask = np.random.rand(*X.shape) < missing_percentage
    X_missing[mask] = np.nan
    return X_missing

# Replace missing values with given code
def replace_missing_values(X, missing_code):
    return X.fillna(missing_code)

# Function to train model and check R-squared and feature importance
def data_evaluate_model(X_train, X_test, y_train, y_test, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = round(r2_score(y_test, y_pred), 2)

    # Permutation importance
    perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    y_pred = y_pred.tolist()

    return r2, y_pred, perm_importance.importances_mean

# Test the effect of different missing value codes and percentages
def data_test_missing_value_effects(X, y, missing_values, pipes, params_list, var_names, dvt1, plot_Xmiss_y=False):
    results = {}
    perm_results = {}
    y_pred_results = {}

    DT = pipes[0]
    DT.set_params(**params_list[0])

    RF = pipes[1]
    RF.set_params(**params_list[1])

    GB = pipes[2]
    GB.set_params(**params_list[2])

    models = {
        "Decision Tree": DT,
        "Random Forest": RF,
        "HistGradientBoosting": GB
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for missing_value in missing_values:
        X_train_filled = replace_missing_values(X_train, missing_value)
        X_test_filled = replace_missing_values(X_test, missing_value)

        if plot_Xmiss_y==True:
            if missing_value == 999999:
                X_train_filled_df = pd.DataFrame(X_train_filled, columns=var_names)
                for var in X_train_filled_df.columns:
                    if X_train_filled_df[var].isin([missing_value]).any() == True:
                        plt.scatter(X_train_filled_df[var], y_train)
                        plt.title(f"{missing_value}: {var}")
                        plt.xlabel(var)
                        plt.ylabel("y")
                        plt.savefig(f"Sim-Data-Missing-Integer/Outputs/Distribution_plots/{missing_value}_{var}{dvt1}.png")
                        plt.clf()
                        plt.cla()
                        plt.close()
        #                 todo: need to adjust x-axis, density, and colour so more useful


        for model_name, model in models.items():
            if model_name not in results:
                results[model_name] = {}
            else:
                results[model_name][missing_value] = {}
            if model_name not in perm_results:
                perm_results[model_name] = {}
            else:
                perm_results[model_name][missing_value] = {}
            if model_name not in y_pred_results:
                y_pred_results[model_name] = {}
            else:
                y_pred_results[model_name][missing_value] = {}

            r2, y_pred, perm_importance = data_evaluate_model(X_train_filled, X_test_filled, y_train, y_test, model)

            results[model_name][missing_value] = r2
            perm_results[model_name][missing_value] = perm_importance.tolist()
            y_pred_results[model_name][missing_value] = y_pred

    # save results as a .json
    return results, y_pred_results, perm_results

def plot_results_seaborn(results, dvt1):

    # Convert to DataFrame
    df = pd.DataFrame(results).reset_index().melt(id_vars='index', var_name='Model', value_name='R2')

    # Rename the 'index' column to 'Category' for clarity
    df = df.rename(columns={'index': 'Category'})

    # Plot using seaborn catplot
    sns.catplot(
        data=df,
        kind='bar',
        x='Model',  # Outer x-axis (models)
        y='R2',  # y-axis (values of the inner dictionary)
        hue='Category',  # Inner x-axis
        palette='Set2'
    )

    plt.xlabel("Model")
    plt.ylabel("R2 Score")
    plt.grid(True)
    plt.savefig(f"Sim-Data-Missing-Integer/Results/Plots/all{dvt1}.png")
    plt.clf()
    plt.cla()
    plt.close()



def sim_generate_regression_data(n_samples, n_features=5, random_state=93):
    np.random.seed(random_state)

    # Generate normally distributed predictor values and scale them between 0 and 100
    X = np.random.normal(loc=0.5, scale=0.15, size=(n_samples, n_features))

    # Rescale X to be within the range of 0 to 100
    X = np.clip((X - np.min(X)) / (np.max(X) - np.min(X)) * 100, 0, 100)

    # Coefficients for linear model (weights)
    coef = np.random.rand(n_features)

    # Generate target variable y with noise to achieve R² = 0.5
    y_true = np.dot(X, coef)

    # Introduce noise to ensure R² is 0.5
    noise = np.random.normal(0, np.std(y_true) * np.sqrt(2), size=n_samples)

    y = y_true + noise  # This ensures R² is approximately 0.5

    return pd.DataFrame(X), y


# Introduce missing values at different percentages
def sim_introduce_missing_values(X, missing_percentage, random_state=42):
    np.random.seed(random_state)
    X_missing = X.copy()
    mask = np.random.rand(*X.shape) < missing_percentage
    X_missing[mask] = np.nan
    return X_missing


# Replace missing values with given code
def sim_replace_missing_values(X, missing_code):
    return X.fillna(missing_code)


# Function to train model and check R-squared and feature importance
def sim_evaluate_model(X_train, X_test, y_train, y_test, model, model_name, param_grid,
                       filename, feature_names, missing_percentage, missing_value):
    original_stdout = sys.stdout
    # Perform grid search with 5-fold cross-validation
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Best model from grid search
    best_model = grid_search.best_estimator_

    # Predict on the test set
    y_pred = best_model.predict(X_test)

    # Calculate R2 score
    r2 = round(r2_score(y_test, y_pred), 2)

    # plot tree:
    if model_name == "Decision Tree":
        # Plot the tree
        plt.figure(figsize=(10, 8))
        plot_tree(best_model, filled=True, feature_names=feature_names)

        # Save the plot to a file
        plt.savefig(f"Sim-Missing-Integer/Results/Decision_trees/decision_tree_{missing_value}_{missing_percentage*100}%.png", dpi=300)
        plt.clf()
        plt.cla()
        plt.close()

    # Permutation importance
    perm_importance = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=42)

    print(f"{model_name} --- R2: {r2}; Best params: {grid_search.best_params_}")
    with open(filename, 'a') as f:
        with contextlib.redirect_stdout(f):
            print(f"{model_name} --- R2: {r2}; Best params: {grid_search.best_params_}")
    sys.stdout = original_stdout

    y_pred = y_pred.tolist()

    return r2, y_pred, perm_importance.importances_mean.tolist()


# Test the effect of different missing value codes and percentages
def sim_test_missing_value_effects(X, y, missing_values, missing_percentages, filename, feature_names):
    results = {}
    original_stdout = sys.stdout

    models = {
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(),
        "HistGradientBoosting": HistGradientBoostingRegressor()
    }

    param_dict = {"Decision Tree": sim_dt_param_grid,
        "Random Forest": sim_rf_param_grid,
        "HistGradientBoosting": sim_gb_param_grid}

    for missing_percentage in missing_percentages:
        with open(filename, 'a') as f:
            sys.stdout = f
            print(f"Missing %: {missing_percentage * 100}")
        sys.stdout = original_stdout

        print(f"\nTesting for {missing_percentage * 100}% missingness...")
        X_missing = sim_introduce_missing_values(X, missing_percentage)

        results[missing_percentage] = {}
        X_train, X_test, y_train, y_test = train_test_split(X_missing, y, test_size=0.2, random_state=42)

        for missing_value in missing_values:
            print(f"missing value: {missing_value}")

            with open(filename, 'a') as f:
                sys.stdout = f
                print(f"missing value: {missing_value}")
            sys.stdout = original_stdout

            X_train_filled = sim_replace_missing_values(X_train, missing_value)
            X_test_filled = sim_replace_missing_values(X_test, missing_value)
            results[missing_percentage][missing_value] = {}

            for model_name, model in models.items():
                r2, y_pred, perm_importance = sim_evaluate_model(X_train=X_train_filled,
                                                     X_test=X_test_filled,
                                                     y_train=y_train,
                                                     y_test=y_test,
                                                     model=model,
                                                     model_name = model_name,
                                                     param_grid=param_dict[model_name],
                                                     filename=filename,
                                                     feature_names = feature_names,
                                                     missing_percentage=missing_percentage,
                                                     missing_value=missing_value)
                results[missing_percentage][missing_value][model_name] = {
                    "R2": r2,
                    "Permutation Importance": perm_importance,
                    "y_pred": y_pred,
                }

    return results


# Plotting function
def sim_plot_results(results, missing_percentages, missing_value_codes, n):
    models = ["Decision Tree", "Random Forest", "HistGradientBoosting"]

    # Generate a colormap with enough distinct colors for all missing values
    cmap = plt.get_cmap('tab10')

    for model_name in models:
        plt.figure(figsize=(10, 6))
        for i, missing_value in enumerate(missing_value_codes):
            r2_scores = [results[mp][missing_value][model_name]["R2"] for mp in missing_percentages]
            plt.plot([mp * 100 for mp in missing_percentages], r2_scores,
                     label=f'Missing Code: {missing_value}', color=cmap(i))

        plt.title(f"{model_name}")
        plt.xlabel("% Missing Data")
        plt.ylabel("R2 Score")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"Sim-Missing-Integer/Results/Plots/{model_name}_n{n}.png")
        plt.clf()
        plt.cla()
        plt.close()



def sim_plot_y_pred(results, missing_percentages, missing_value_codes, n):
    models = ["Decision Tree", "Random Forest", "HistGradientBoosting"]

    # Generate a colormap with enough distinct colors for all missing values
    cmap = plt.get_cmap('tab10')
    for perc in missing_percentages:
        for model_name in models:
            # Prepare data for plotting
            y_pred1 = np.array(results[perc][-999999][model_name]['y_pred'])
            y_pred2 = np.array(results[perc][999999][model_name]['y_pred'])

            # Calculate Pearson correlation coefficient
            pearson_corr, _ = stats.pearsonr(y_pred1, y_pred2)

            # Create scatter plot
            plt.figure(figsize=(8, 6))
            plt.scatter(y_pred1, y_pred2, color='blue', alpha=0.7, label='Predicted Values')
            plt.xlabel('y_pred for -999999')
            plt.ylabel('y_pred for 999999')
            plt.title(f'Scatter Plot of y Predicted Values {perc*100}% missing\nPearson r: {pearson_corr:.2f}')

            # Fit a line to the data
            slope, intercept, r_value, p_value, std_err = stats.linregress(y_pred1, y_pred2)
            line = slope * y_pred1 + intercept
            plt.plot(y_pred1, line, color='red', label='Line of Best Fit')

            # Display legend
            plt.legend()
            plt.grid()

            # Save
            plt.savefig(f"Sim-Missing-Integer/Results/Plots/y_pred_{model_name}_{perc*100}%miss_n{n}.png")
            plt.clf()
            plt.cla()
            plt.close()



def data_plot_y_pred(results, dvt1):
    # Prepare data for plotting
    models = ["Decision Tree", "Random Forest", "HistGradientBoosting"]
    for model_name in models:
        y_pred1 = np.array(results[model_name][-999999])  # Ensure y_pred1 is a NumPy array
        y_pred2 = np.array(results[model_name][999999])   # Ensure y_pred2 is a NumPy array

        # Calculate Pearson correlation coefficient
        pearson_corr, _ = stats.pearsonr(y_pred1, y_pred2)

        # Create scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(y_pred1, y_pred2, color='blue', alpha=0.7, label='Predicted Values')
        plt.xlabel('y_pred for -999999')
        plt.ylabel('y_pred for 999999')
        plt.title(f'Scatter Plot of Predicted Values\nPearson r: {pearson_corr:.2f}')

        # Fit a line to the data
        slope, intercept, r_value, p_value, std_err = stats.linregress(y_pred1, y_pred2)
        line = slope * y_pred1 + intercept  # This now works correctly as y_pred1 is a NumPy array
        plt.plot(y_pred1, line, color='red', label='Line of Best Fit')

        # Display legend
        plt.legend()
        plt.grid()
        # Save
        plt.savefig(f"Sim-Data-Missing-Integer/Results/Plots/y_pred_{model_name}{dvt1}.png")
        plt.clf()
        plt.cla()
        plt.close()
