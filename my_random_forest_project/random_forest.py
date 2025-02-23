import numpy as np
from decision_tree import SimpleDecisionTree

class SimpleRandomForest:
    def __init__(self, n_estimators=10, max_depth=None, min_samples_split=2):
        """
        ランダムフォレストの初期化
        
        Parameters:
        -----------
        n_estimators : int
            決定木の数
        max_depth : int or None
            各決定木の最大深さ
        min_samples_split : int
            分割に必要な最小サンプル数
        
        Note:
        -----
        将来の拡張のために以下のパラメータを追加することができます：
        - bootstrap : bool
            ブートストラップサンプリングを行うかどうか
        - max_features : int or float
            各決定木で使用する特徴量の数や割合
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def fit(self, X, y):
        """
        ランダムフォレストのトレーニングを行う
        
        Parameters:
        -----------
        X : array-like
            トレーニングデータの特徴量
        y : array-like
            トレーニングデータのターゲット値
        """
        n_samples, n_features = X.shape
        
        # クラスごとのサンプル数を計算
        unique_classes, class_counts = np.unique(y, return_counts=True)
        min_class_count = np.min(class_counts)
        
        # クラスごとの重みを設定（少数クラスは2倍の重み）
        class_weights = {}
        for class_label, count in zip(unique_classes, class_counts):
            if count == min_class_count:
                class_weights[class_label] = 2.0
            else:
                class_weights[class_label] = 1.0
        
        # 各サンプルの重みを初期化
        base_weights = np.array([class_weights[label] for label in y])
        
        # 決定木を作成し、学習を行う
        for _ in range(self.n_estimators):
            tree = SimpleDecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            
            # ブートストラップサンプリングのための重みを正規化
            sample_weights = base_weights / np.sum(base_weights)
            
            # 重み付きブートストラップサンプリング
            bootstrap_indices = np.random.choice(
                n_samples,
                size=n_samples,
                replace=True,
                p=sample_weights
            )
            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]
            bootstrap_weights = base_weights[bootstrap_indices]
            
            # 特徴量のサンプリング
            n_features_subset = max(1, int(np.sqrt(n_features)))
            feature_indices = np.random.choice(n_features, size=n_features_subset, replace=False)
            X_subset = X_bootstrap[:, feature_indices]
            
            # 重み付きで学習を実行
            tree.fit(X_subset, y_bootstrap, sample_weight=bootstrap_weights)
            # 特徴量インデックスを保存
            tree.feature_indices = feature_indices
            self.trees.append(tree)

    def predict(self, X):
        """
        新しいデータの予測を行う
        
        Parameters:
        -----------
        X : array-like
            予測したいデータの特徴量
        
        Returns:
        --------
        array-like
            予測されたクラス（多数決による）
        """
        predictions = []
        for tree in self.trees:
            # 各決定木で使用した特徴量のみを使用
            X_subset = X[:, tree.feature_indices]
            predictions.append(tree.predict(X_subset))
        
        predictions = np.array(predictions)
        # 各サンプルごとに多数決を取る
        return np.array([
            np.argmax(np.bincount(predictions[:, i]))
            for i in range(X.shape[0])
        ])
