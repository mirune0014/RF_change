import numpy as np

def _weighted_gini_index(y, sample_weight=None):
    """
    重み付きGini不純度を計算する

    Parameters:
    -----------
    y : array-like
        ノードに含まれるサンプルのクラスラベル
    sample_weight : array-like or None
        各サンプルの重み。Noneの場合は全て1とする

    Returns:
    --------
    float
        重み付きGini不純度
    """
    if sample_weight is None:
        sample_weight = np.ones_like(y, dtype=float)

    total_weight = np.sum(sample_weight)
    # クラス毎の重み合計を計算
    unique_classes, class_indices = np.unique(y, return_inverse=True)
    sum_weights_per_class = np.zeros(len(unique_classes))
    for i in range(len(y)):
        sum_weights_per_class[class_indices[i]] += sample_weight[i]

    # Gini = Σ (p_i * (1 - p_i)) 
    # p_i = (クラスiの総重量) / (全体の総重量)
    p = sum_weights_per_class / total_weight
    gini = np.sum(p * (1 - p))

    return gini

class SimpleDecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        """
        決定木の初期化
        
        Parameters:
        -----------
        max_depth : int or None
            木の最大深さ。Noneの場合は制限なし
        min_samples_split : int
            分割に必要な最小サンプル数
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def _find_best_split(self, X, y, sample_weight):
        """
        最適な分割点を見つける

        Parameters:
        -----------
        X : array-like
            特徴量
        y : array-like
            ターゲット値
        sample_weight : array-like
            サンプルの重み

        Returns:
        --------
        dict
            最適な分割情報を含む辞書
        """
        n_samples, n_features = X.shape
        best_gini = float('inf')
        best_split = {}

        for feature_idx in range(n_features):
            feature_values = np.unique(X[:, feature_idx])
            
            for threshold in feature_values:
                # データを分割
                left_mask = X[:, feature_idx:feature_idx+1].flatten() <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                # 重み付きGini不純度を計算
                gini_left = _weighted_gini_index(y[left_mask], sample_weight[left_mask])
                gini_right = _weighted_gini_index(y[right_mask], sample_weight[right_mask])
                
                # 全体のGini不純度を重み付き平均で計算
                w_left = np.sum(sample_weight[left_mask])
                w_right = np.sum(sample_weight[right_mask])
                w_total = w_left + w_right
                gini = (w_left * gini_left + w_right * gini_right) / w_total
                
                if gini < best_gini:
                    best_gini = gini
                    best_split = {
                        'feature_idx': feature_idx,
                        'threshold': threshold,
                        'gini': gini
                    }
        
        return best_split

    def _split_node(self, X, y, depth, sample_weight=None):
        """
        ノードの分割を再帰的に行う
        
        Parameters:
        -----------
        X : array-like
            現在のノードのデータの特徴量
        y : array-like
            現在のノードのデータのターゲット値
        depth : int
            現在の深さ
        sample_weight : array-like or None
            サンプルの重み
        
        Returns:
        --------
        dict
            ノードの情報を含む辞書
        """
        n_samples = len(y)
        
        if sample_weight is None:
            sample_weight = np.ones(n_samples)
        
        node = {}

        # 停止条件のチェック
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split:
            # 重み付き多数決で葉ノードの値を決定
            unique_classes = np.unique(y)
            class_weights = np.array([np.sum(sample_weight[y == c]) for c in unique_classes])
            node['value'] = unique_classes[np.argmax(class_weights)]
            return node

        # 最適な分割点を見つける
        best_split = self._find_best_split(X, y, sample_weight)
        
        if not best_split:  # 分割点が見つからない場合
            node['value'] = np.argmax([np.sum(sample_weight[y == c]) for c in np.unique(y)])
            return node
            
        # データを分割
        feature_idx = best_split['feature_idx']
        threshold = best_split['threshold']
        
        left_mask = X[:, feature_idx:feature_idx+1].flatten() <= threshold
        right_mask = ~left_mask
        
        # 子ノードを再帰的に構築
        node['feature_idx'] = feature_idx
        node['threshold'] = threshold
        node['left'] = self._split_node(
            X[left_mask], y[left_mask], depth + 1, sample_weight[left_mask]
        )
        node['right'] = self._split_node(
            X[right_mask], y[right_mask], depth + 1, sample_weight[right_mask]
        )
        
        return node

    def fit(self, X, y, sample_weight=None):
        """
        決定木のトレーニングを行う
        
        Parameters:
        -----------
        X : array-like
            トレーニングデータの特徴量
        y : array-like
            トレーニングデータのターゲット値
        sample_weight : array-like or None
            サンプルの重み
        """
        self.n_classes = len(np.unique(y))
        self.n_features = X.shape[1]
        self.tree = self._split_node(X, y, depth=0, sample_weight=sample_weight)

    def _predict_single(self, x, node):
        """
        単一サンプルの予測を行う

        Parameters:
        -----------
        x : array-like
            予測したい単一サンプルの特徴量
        node : dict
            現在のノード情報

        Returns:
        --------
        int
            予測されたクラス
        """
        if 'value' in node:
            return node['value']
            
        if x[node['feature_idx']:node['feature_idx']+1][0] <= node['threshold']:
            return self._predict_single(x, node['left'])
        else:
            return self._predict_single(x, node['right'])

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
            予測されたクラス
        """
        if self.tree is None:
            raise Exception("モデルが学習されていません。先にfitメソッドを実行してください。")
        
        # 各サンプルに対して再帰的に予測を行う
        return np.array([self._predict_single(x, self.tree) for x in X])
