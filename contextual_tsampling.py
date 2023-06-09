import pandas as pd
import numpy as np
import scipy as sc

"""
This module contains helper functions and 
helper classes to run Algorithm 3 from Online 
Logistic Regression by Chapelle et. al
https://proceedings.neurips.cc/paper_files/paper/2011/file/e53a0a2978c28872a4505bdb51db06dc-Paper.pdf. 
"""


def logistic_regression_objective(m_new, m, q, x, y):
    """

    :param m_new: The weights the objective function is being minimized
                  with respect to. Initialized with the current estimates
                  for the means.
    :param m: Current estimates for the means of the model weights.
    :param q: 1/q is the current estimate for the variance of the parameters
    :param x: Matrix of context variable.
    :param y: Vector of reward signals
    :return: Value of the objective function.
    """
    term1 = .5 * np.sum(q * ((m_new - m) ** 2))
    term2 = np.sum(np.log(1 + np.exp(-y * (x @ m_new))))
    return term1 + term2


class Arm:
    def __init__(self, n_features, q_initial):
        """

        :param n_features: Number of context variables.
        :param q_initial: Initial guess for the variance
                          of the model parameters.
        """
        self.n_features = n_features
        self.m = np.zeros(n_features + 1)
        self.q = np.array([q_initial] * (n_features + 1))

    def update(self, x, r):
        """
        Updates the estimates for the means and
        variances of the model parameters.
        :param x: matrix of context variables.
        :param r: vector of reward signals
        :return: None
        """
        # Concatenate column of ones to the context variables
        # to account for the intercept term.
        x = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)

        # new estimate for the means is the minimizer of the objective
        # function.
        self.m = sc.optimize.minimize(logistic_regression_objective, self.m, (self.m, self.q, x, r))['x']

        # update estimates for the variances.
        p = 1 / (1 + np.exp(-x @ self.m))
        self.q = np.array([self.q[j] + np.sum((x[:, j] ** 2) * p * (1 - p)) for j in range(self.n_features + 1)])
        return None

    def sample_coef(self):
        """
        sample coefficients from the current
        posterior distributions.
        :return: Vector of model coefficients drawn
                 their posterior distributions.
        """
        coef = np.array([np.random.normal(m, 1 / q) for m, q in zip(self.m, self.q)])
        return coef

    def make_prediction(self, context):
        """

        :param context: vector of context variables
        :return: Estimate of the probability of conversion
                 for this arm.
        """
        coef = self.sample_coef()
        logit = coef[0] + context @ coef[1:]
        return 1 / (1 + np.exp(-logit))


class MultiArm:
    def __init__(self, n_arms, n_features, q_initial=1):
        """

        :param n_arms: Number of available arms
        :param n_features: Number of context variables
        :param q_initial: Initialization of parameter variance.
        """
        self.n_arms = n_arms
        self.arms = [Arm(n_features, q_initial) for _ in range(n_arms)]

    def select_arm(self, context):
        """
        Selects the arm with the highest expected
        reward. The reward is calculated by sampling
        from each coefficient's posterior distribution.
        :param context: Matrix of context variables
        :return: Selected Arm
        """
        predictions = np.zeros((context.shape[0], self.n_arms))
        for i in range(self.n_arms):
            predictions[:, i] = [self.arms[i].make_prediction(context[k]) for k in range(context.shape[0])]
        return np.argmax(predictions, axis=1)

    def update_arms(self, results):
        """
        Updates the posterior distribution
        for each arm's coefficients based of
        off new data.
        :param results:
        :return: None
        """
        results_dict = {int(path.split('_')[1]): results[results.path == path] for path in results.path.unique()}
        for arm, result in results_dict.items():
            self.arms[arm - 1].update(result.drop(['id', 'path', 'reward'], axis=1).to_numpy(),
                                      result.reward.to_numpy())
        return None
