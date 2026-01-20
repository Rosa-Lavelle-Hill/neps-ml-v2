import numpy as np
from sklearn.linear_model import LinearRegression

def add_noise(y_pred, r_squared):
    """
    Adds random noise to y to achieve the desired R squared value.
    :param y_pred: y variable noise should be added to
    :param r_squared: desired R-squared value y should be predicted from X
    :return: y_red with noise, number of iterations
    """
    n = y_pred.shape[0] #The number of observations or data points in y_pred
    v = np.sum((y_pred - np.mean(y_pred)) **2) #variance: sum of the squared differences between each value in y_pred and the mean of y_pred
    u = v * (1 - r_squared) / r_squared #scaled variance by the desired r_squared
    noise_var = u / n #The variance of the noise to be added to y_pred
    noise = np.random.normal(0, np.sqrt(noise_var), size=n) # normal distribution on 0, with SD (sqrt of noise_var)
    iter = 0
    while np.mean(noise) > 0.01 or -0.01 > np.mean(noise) or np.std(noise) > np.sqrt(noise_var) + 0.01 or np.sqrt(noise_var) - 0.01 > np.std(noise):
        noise = np.random.normal(0, np.sqrt(noise_var), size=n)
        iter = iter + 1
    y = y_pred + noise
    print(f"noise variance: {round(noise_var, 2)}; noise SD: {round(np.sqrt(noise_var), 2)}; noise mean: {round(np.mean(noise),2)}")
    return y, iter