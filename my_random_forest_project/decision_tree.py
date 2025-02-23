import numpy as np

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

    def fit(self, X, y):
        """
        決定木のトレーニングを行う
        
        Parameters:
        -----------
        X : array-like
            トレーニングデータの特徴量
        y : array-like
            トレーニングデータのターゲット値
        """
        self.n_classes = len(np.unique(y))
        self.n_features = X.shape[1]
        self.tree = self._split_node(X, y, depth=0)

    def _split_node(self, X, y, depth):
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
        
        Returns:
        --------
        dict
            ノードの情報を含む辞書
        """
        n_samples = len(y)
        node = {}

        # 停止条件のチェック
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split:
            node['value'] = np.argmax(np.bincount(y))
            return node

        # 最適な分割点を見つける処理をここに実装
        # （実装の簡略化のため、この部分は省略）
        node['value'] = np.argmax(np.bincount(y))
        return node

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
            
        # 予測の実装（簡略化のため、単純な実装としています）
        predictions = np.array([self.tree['value']] * len(X))
        return predictions
