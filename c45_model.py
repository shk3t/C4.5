import numpy as np

from c45_math import probability, gain_ratio


class C45:
    def __init__(self):
        self.root = None

    def fit(self, X_train, y_train):
        self._targets = y_train.unique()
        self.root = self._make_node(X_train, y_train)
        return self

    def predict(self, X):
        probas = self._predict_proba(X)
        return probas.applymap(np.round).apply(
            lambda row: row[row.astype(bool)].index[0], axis=1
        )

    def predict_proba(self, X):
        return self._predict_proba(X)

    def _predict_proba(self, X):
        def to_probas(row):
            node = self.root
            while isinstance(node, Node):
                node = node.childs[row[node.feature]]
            return node

        return X.apply(to_probas, axis=1, result_type="expand")

    def _make_node(self, X, y):
        selected_feature = self._get_max_gain_feature(X, y)
        unique_values = X[selected_feature].unique()

        if unique_values.shape[0] == 1:
            return self._get_target_probas(y)

        parent = Node(selected_feature)
        for value in unique_values:
            mask = X[selected_feature] == value
            sX, sy = X[mask].drop(columns=selected_feature), y[mask]
            len_u_targets = sy.unique().shape[0]

            if sX.shape[1] > 0 and len_u_targets > 1:
                parent.childs[value] = self._make_node(sX, sy)
            elif len_u_targets > 0:
                parent.childs[value] = self._get_target_probas(sy)

        return parent

    def _get_target_probas(self, y):
        return {target: probability(target, y) for target in self._targets}

    def _get_max_gain_feature(self, X, y):
        selected_feature = max(
            [(f, gain_ratio(X[f], y)) for f in X],
            key=(lambda element: element[1]),
        )[0]
        return selected_feature


class Node:
    def __init__(self, feature=None):
        self.feature = feature
        self.childs = {}

    def __repr__(self):
        return f"{self.feature} -> {self.childs}\n"
