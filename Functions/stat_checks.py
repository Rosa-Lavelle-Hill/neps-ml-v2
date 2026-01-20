import numpy as np
import pandas as pd

def is_binary(column):
    # Check if all values are either 0, 1, or NA
    unique_values = column.dropna().unique()
    return len(unique_values) <= 2 and all(value in [0, 1] for value in unique_values)


def phi_coefficient(data):
    # Convert data to a pandas DataFrame if not already
    df = pd.DataFrame(data)
    num_vars = df.shape[1]

    # Initialize a matrix to store Phi coefficients
    phi_matrix = np.eye(num_vars)

    # Create a DataFrame to store Phi coefficients with variable names as index and columns
    phi_df = pd.DataFrame(phi_matrix, index=df.columns, columns=df.columns)

    # Iterate through pairs of variables
    for i in range(num_vars):
        for j in range(i + 1, num_vars):
            # Check if both variables are perfectly correlated
            if df.iloc[:, i].equals(df.iloc[:, j]):
                phi_df.iloc[i, j] = 1
                phi_df.iloc[j, i] = 1  # Symmetric matrix
            else:
                # Calculate the contingency table
                contingency_table = pd.crosstab(df.iloc[:, i], df.iloc[:, j])

                # Calculate N11, N00, N10, and N01
                N11 = contingency_table.iloc[1, 1]
                N00 = contingency_table.iloc[0, 0]
                N10 = contingency_table.iloc[1, 0]
                N01 = contingency_table.iloc[0, 1]

                # Calculate N1•, N0•, N•1, and N•0
                N1dot = N11 + N10
                N0dot = N01 + N00
                Ndot1 = N11 + N01
                Ndot0 = N10 + N00

                # Calculate the Phi coefficient
                phi = (N11 * N00 - N10 * N01) / np.sqrt(N1dot * N0dot * Ndot1 * Ndot0)
                phi_df.iloc[i, j] = phi
                phi_df.iloc[j, i] = phi  # Symmetric matrix

    return phi_df

