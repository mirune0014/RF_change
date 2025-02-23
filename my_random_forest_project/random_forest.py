import numpy as np
from sklearn.metrics import f1_score
from decision_tree import SimpleDecisionTree

class SimpleRandomForest:
    def __init__(self, n_estimators=10, max_depth=None, min_samples_split=2, epsilon=1e-10):
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
        self.epsilon = epsilon  # 0除算回避用の微小値
        self.trees = []
        self.tree_weights = []  # 各決定木の重み
        self.feature_indices_list = []  # 各決定木の特徴量インデックス

    def _calculate_oob_score(self, tree, X, y, bootstrap_indices, feature_indices):
        """
        Out-of-Bag (OOB)スコアを計算する

        Parameters:
        -----------
        tree : SimpleDecisionTree
            評価対象の決定木
        X : array-like
            元の特徴量データ
        y : array-like
            元のターゲットデータ
        bootstrap_indices : array-like
            ブートストラップサンプルのインデックス
        feature_indices : array-like
            使用した特徴量のインデックス

        Returns:
        --------
        float
            F1スコア（少数クラスの識別性能を評価）
        """
        # OOBサンプルのインデックスを取得
        n_samples = X.shape[0]
        oob_indices = np.array([i for i in range(n_samples) if i not in bootstrap_indices])
        
        if len(oob_indices) == 0:  # OOBサンプルがない場合
            return 0.0
            
        # OOBサンプルで予測
        X_oob = X[oob_indices]
        X_oob_subset = X_oob[:, feature_indices]
        y_oob = y[oob_indices]
        y_pred = tree.predict(X_oob_subset)
        
        # F1スコアを計算（マクロ平均を使用して全クラスを均等に評価）
        return f1_score(y_oob, y_pred, average='macro')

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
            
            # OOBスコアを計算
            oob_score = self._calculate_oob_score(tree, X, y, bootstrap_indices, feature_indices)
            tree_weight = oob_score + self.epsilon  # 0除算回避
            
            # 木と関連情報を保存
            self.trees.append(tree)
            self.tree_weights.append(tree_weight)
            self.feature_indices_list.append(feature_indices)
            
        # 重みを正規化
        self.tree_weights = np.array(self.tree_weights)
        self.tree_weights = self.tree_weights / np.sum(self.tree_weights)

    def predict(self, X):
        """
        新しいデータの予測を行う（加重多数決による）
        
        Parameters:
        -----------
        X : array-like
            予測したいデータの特徴量
        
        Returns:
        --------
        array-like
            予測されたクラス
        """
        n_samples = X.shape[0]
        predictions = np.zeros((n_samples, len(self.trees)))
        
        # 各決定木での予測を取得
        for i, (tree, feature_indices) in enumerate(zip(self.trees, self.feature_indices_list)):
            X_subset = X[:, feature_indices]
            predictions[:, i] = tree.predict(X_subset)
        
        # 加重多数決による最終予測
        final_predictions = []
        for i in range(n_samples):
            # 各サンプルについて、クラスごとの重み付き投票を集計
            sample_predictions = predictions[i]
            unique_classes = np.unique(sample_predictions)
            class_votes = {cls: 0.0 for cls in unique_classes}
            
            for tree_idx, pred_class in enumerate(sample_predictions):
                class_votes[pred_class] += self.tree_weights[tree_idx]
            
            # 最大の重み合計を持つクラスを選択
            final_predictions.append(max(class_votes.items(), key=lambda x: x[1])[0])
            
        return np.array(final_predictions)
