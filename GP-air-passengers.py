import matplotlib.pyplot as plt
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import kaggle
import pandas as pd

def plot_gpr_samples(gpr_model, n_samples):
    x = np.linspace(0, 142, 142)
    X = x.reshape(-1, 1)

    y_mean, y_std = gpr_model.predict(X, return_std=True)
    y_samples = gpr_model.sample_y(X, n_samples, random_state=42)

    for idx, single_prior in enumerate(y_samples.T):
        plt.plot(
            x,
            single_prior,
            linestyle="--",
            alpha=0.7,
            label=f"Sampled function #{idx + 1}",
        )
    plt.plot(x, y_mean, color="black", label="Mean")
    plt.fill_between(
        x,
        y_mean - y_std,
        y_mean + y_std,
        alpha=0.50,
        color="black",
        label=r"$\pm$ 1 std. dev.",
    )
    plt.xlabel("x")
    plt.ylabel("y")


download_dataset = False
if download_dataset:
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files('chirag19/air-passengers', path='air_passengers_dataset', unzip=True)

df = pd.read_csv("air_passengers_dataset\AirPassengers.csv")
data = df.to_numpy()
X = data[:, 0].reshape(-1, 1)
y = data[:, 1]
label_encoder = LabelEncoder()  # This just encodes date to numerical values, i.e 10-1947 will just get encoded to a number, say 2. 11-1947 will be 3 and so on
# Fit and transform the data
X = label_encoder.fit_transform(X)
X = X.reshape(-1, 1)

plt.scatter(X, y)
plt.show()

kernel = C(constant_value=78961) + (9158.49 * RBF(length_scale=1.36, length_scale_bounds=(1e-1, 100.0))) #optimal kernel
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, random_state=42)

X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# plot prior
plot_gpr_samples(gpr, n_samples=5)
plt.title("Samples from prior distribution")
plt.show()

# plot posterior
gpr.fit(X_train, y_train)
plot_gpr_samples(gpr, n_samples=5)
plt.scatter(X_train, y_train, color="red", zorder=10, label="Observations")
plt.scatter(x_test, y_test, color="blue", zorder=10, label="Actual")
plt.legend(bbox_to_anchor=(1.05, 1.5), loc="upper left")
plt.title("Samples from posterior distribution")

plt.show()

print(
    f"Kernel parameters after fit: \n{gpr.kernel_} \n"
)
