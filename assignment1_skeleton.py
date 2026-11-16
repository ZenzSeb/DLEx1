"""Starter skeleton for Deep Learning for AIE Assignment 1.

Fill in the TODOs, run the experiments, and discuss the results in your report.
"""

from copy import deepcopy
from itertools import product
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# Reproducibility
SEED = 42  # TODO: choose and report a seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEATURE_NAMES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]


def load_data():
    """Fetch the provided dataset and create the specified train/test split."""
    X, y = fetch_california_housing(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=302
    )
    return X_train, X_test, y_train, y_test


def inspect_features(X_train):
    """Plot feature distributions using training data only."""
    frame = pd.DataFrame(X_train, columns=FEATURE_NAMES)
    axes = frame.hist(figsize=(12, 9), bins=30)
    plt.suptitle("Training feature distributions")
    plt.tight_layout()
    plt.show()

    # TODO: describe skew, scale differences, outliers, and notable features
    return frame.describe()


def make_loader(X, y, batch_size, shuffle):
    x_tensor = torch.as_tensor(X, dtype=torch.float32)
    y_tensor = torch.as_tensor(y, dtype=torch.float32).reshape(-1, 1)
    return DataLoader(
        TensorDataset(x_tensor, y_tensor),
        batch_size=batch_size,
        shuffle=shuffle,
    )


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_units):
        super().__init__()
        layers = []
        input_units = len(FEATURE_NAMES)
        for units in hidden_units:
            layers.extend([nn.Linear(input_units, units), nn.ReLU()])
            input_units = units

        # Regression output: one unconstrained scalar prediction.
        layers.append(nn.Linear(input_units, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def train_model(
    model,
    train_loader,
    val_loader,
    *,
    epochs,
    learning_rate,
    loss_fn,
):
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    history = {"train_loss": []}
    if val_loader is not None:
        history["val_loss"] = []

    for epoch in range(epochs):
        model.train()
        train_total = 0.0
        train_count = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            predictions = model(xb)
            loss = loss_fn(predictions, yb)
            loss.backward()
            optimizer.step()
            train_total += loss.item() * len(xb)
            train_count += len(xb)

        history["train_loss"].append(train_total / train_count)
        if val_loader is not None:
            model.eval()
            val_total = 0.0
            val_count = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    loss = loss_fn(model(xb), yb)
                    val_total += loss.item() * len(xb)
                    val_count += len(xb)
            history["val_loss"].append(val_total / val_count)

        # TODO: consider logging progress at a suitable interval

    return model, history


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(X, dtype=torch.float32, device=DEVICE)).cpu().numpy()


def run_regression():
    X_train, X_test, y_train, y_test = load_data()
    print(inspect_features(X_train))

    # Keep test data out of model selection. Fit the scaler only on the train split.
    X_subtrain, X_vld, y_subtrain, y_vld = train_test_split(
        X_train, y_train, test_size=0.2, random_state=SEED
    )
    scaler = StandardScaler()
    X_subtrain_scaled = scaler.fit_transform(X_subtrain)
    X_vld_scaled = scaler.transform(X_vld)

    # At least 10 architecture / optimizer configurations.
    configurations = [
        {
            "hidden_units": hidden_units,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
        }
        for hidden_units, learning_rate, epochs, batch_size in product(
            [(32,), (64,), (128,), (64, 32), (128, 64), (64, 64)],
            [0.01, 0.001],
            [100],
            [64],
        )
    ]
    # TODO: narrow or expand this list based on sensible compute and convergence.
    results = []
    best_config = None
    best_val_loss = float("inf")
    best_history = None

    for config in configurations:
        train_loader = make_loader(
            X_subtrain_scaled,
            y_subtrain,
            config["batch_size"],
            shuffle=True,
        )
        val_loader = make_loader(
            X_vld_scaled, y_vld, config["batch_size"], shuffle=False
        )
        model = FeedForwardNetwork(config["hidden_units"]).to(DEVICE)
        model, history = train_model(
            model,
            train_loader,
            val_loader,
            epochs=config["epochs"],
            learning_rate=config["learning_rate"],
            loss_fn=nn.MSELoss(),
        )
        train_loss = history["train_loss"][-1]
        val_loss = history["val_loss"][-1]
        results.append({**config, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_config = config
            best_history = deepcopy(history)

    results_frame = pd.DataFrame(results).sort_values("val_loss")
    print(results_frame.to_string(index=False))
    # TODO: discuss the comparison and selection criteria in the report.

    # Plot the selected model's learning curves from model selection.
    plt.plot(best_history["train_loss"], label="Training loss")
    plt.plot(best_history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.legend()
    plt.title("Regression learning curves")
    plt.tight_layout()
    plt.show()

    # Retrain the selected architecture on the full training set.
    final_scaler = StandardScaler()
    X_train_scaled = final_scaler.fit_transform(X_train)
    X_test_scaled = final_scaler.transform(X_test)
    final_loader = make_loader(
        X_train_scaled, y_train, best_config["batch_size"], shuffle=True
    )
    final_model = FeedForwardNetwork(best_config["hidden_units"]).to(DEVICE)
    final_model, _ = train_model(
        final_model,
        final_loader,
        None,
        epochs=best_config["epochs"],
        learning_rate=best_config["learning_rate"],
        loss_fn=nn.MSELoss(),
    )

    y_pred = predict(final_model, X_test_scaled).reshape(-1)
    test_mse = mean_squared_error(y_test, y_pred)
    print(f"Final test MSE: {test_mse:.4f}")

    plt.scatter(y_pred, y_test, alpha=0.4)
    limits = [
        min(float(y_pred.min()), float(y_test.min())),
        max(float(y_pred.max()), float(y_test.max())),
    ]
    plt.plot(limits, limits, "--", color="gray")
    plt.xlabel("Predicted house value ($100,000)")
    plt.ylabel("Ground-truth house value ($100,000)")
    plt.title("Test-set predictions")
    plt.tight_layout()
    plt.show()

    return best_config, final_scaler, final_model


def run_classification(best_config, X_train, X_test, y_train, y_test):
    """Train and evaluate the binary below/at-least $200,000 task."""
    y_train_binary = (y_train >= 2.0).astype(np.float32)
    y_test_binary = (y_test >= 2.0).astype(np.int64)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    train_loader = make_loader(
        X_train_scaled,
        y_train_binary,
        best_config["batch_size"],
        shuffle=True,
    )

    # TODO: implement a classifier with one logit output (no sigmoid in the model).
    # Use BCEWithLogitsLoss; its target tensor must have the same shape as logits.
    classifier = FeedForwardNetwork(best_config["hidden_units"]).to(DEVICE)
    classification_loss = nn.BCEWithLogitsLoss()
    classifier, _ = train_model(
        classifier,
        train_loader,
        None,
        epochs=best_config["epochs"],
        learning_rate=best_config["learning_rate"],
        loss_fn=classification_loss,
    )

    logits = predict(classifier, X_test_scaled).reshape(-1)
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    y_pred = (probabilities >= 0.5).astype(np.int64)  # TODO: justify threshold

    print("Confusion matrix:\n", confusion_matrix(y_test_binary, y_pred))
    print(f"Accuracy:  {accuracy_score(y_test_binary, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test_binary, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test_binary, y_pred):.4f}")
    print(f"F1:        {f1_score(y_test_binary, y_pred):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test_binary, probabilities):.4f}")
    # TODO: discuss class balance, metric interpretation, and model performance.


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    best_config, _, _ = run_regression()
    run_classification(best_config, X_train, X_test, y_train, y_test)
