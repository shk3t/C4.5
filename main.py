import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
)

from c45_model import C45
from utils import POS_LABEL, TARGET, draw_pr_auc, draw_roc_auc, get_data


if __name__ == "__main__":
    df, features = get_data()
    print(df)

    X, y = df.loc[:, features], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = C45().fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    print("accuracy:", accuracy_score(y_test, y_pred))
    print("precision:", precision_score(y_test, y_pred, pos_label=POS_LABEL))
    print("recall:", recall_score(y_test, y_pred, pos_label=POS_LABEL))

    draw_roc_auc(pd.get_dummies(y_test), y_proba, POS_LABEL)
    draw_pr_auc(y_test, y_pred, POS_LABEL)
