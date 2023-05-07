import pandas as pd
import numpy as np

# Parameters
n_samples = 10_000
n_features = 10
n_arms = 3
num_paths = 3


def get_reward(row):
    arm = int(row.path.split('_')[1]) - 1
    logit = true_coef[arm][0] + np.sum([c * x for c, x in zip(true_coef[arm][1:], [row[f"feature_{n + 1}"]
                                                                                   for n in range(n_features)])])
    prob = 1 / (1 + np.exp(-logit))
    return np.random.choice([-1, 1], p=[1 - prob, prob])


# Crate context variables
x = np.random.normal(size=(n_samples, n_features))
columns = ['id'] + [f"feature_{n + 1}" for n in range(n_features)]
df = pd.DataFrame(np.concatenate([np.arange(n_samples).reshape(n_samples, 1), x], axis=1), columns=columns)
df['id'] = df['id'].astype('int')
df.to_csv('data/context_data.csv', index=False)

# True coefficients to estimate
true_coef = np.random.uniform(-1, 1, size=(3, n_features + 1))
pd.DataFrame(true_coef, columns=[f"b_{n}" for n in range(n_features + 1)]).to_csv("data/true_coef", index=False)

# Create Batch Data with reward signal
results = df.copy()
results['path'] = np.random.choice([f'path_{n + 1}' for n in range(num_paths)], size=n_samples)
results['reward'] = results.apply(get_reward, axis=1)
results.to_csv("data/batch_update.csv", index=False)
