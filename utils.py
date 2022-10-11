import pandas as pd
import matplotlib.pyplot as plt
import math
import random

from sklearn.metrics import auc, precision_recall_curve, roc_curve


TARGET = "class"
TARGET_MAPPING = {"e": "eatable", "p": "poisonous"}
POS_LABEL = "e"


def get_data() -> tuple[pd.DataFrame, list]:
    header = open("./data/agaricus-lepiota.header").read()[:-1].split(",")
    df = pd.read_table(
        "./data/agaricus-lepiota.data",
        delimiter=",",
        names=[TARGET] + header,
        index_col=False,
    )
    header = [f for f in header if df[f].unique().shape[0] > 1]
    features = random.sample(header, k=math.ceil(math.sqrt(len(header))))
    df = df.loc[:, features + [TARGET]]
    return df, features


def draw_roc_auc(y_test, y_proba, target):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr[target], tpr[target], _ = roc_curve(
        y_test.loc[:, target], y_proba.loc[:, target]
    )
    roc_auc[target] = auc(fpr[target], tpr[target])

    plt.figure(figsize=(8, 6), dpi=80)
    plt.plot(
        fpr[target],
        tpr[target],
        color="darkorange",
        lw=2,
        label=f"{TARGET_MAPPING[target]} ROC (area = {roc_auc[target] : 0.4f})",
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-AUC")
    plt.legend(loc="lower right")
    plt.show()


def draw_pr_auc(y_test, y_pred, target):
    precision, recall, _ = precision_recall_curve(y_test == target, y_pred == target)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(8, 6), dpi=80)
    plt.plot(
        recall,
        precision,
        color="darkorange",
        lw=2,
        label=f"{TARGET_MAPPING[target]} PR (area = {pr_auc : 0.4f})",
    )
    plt.plot([0, 1], [0.5, 0.5], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR-AUC")
    plt.legend(loc="lower right")
    plt.show()
