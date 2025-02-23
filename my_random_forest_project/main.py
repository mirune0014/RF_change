from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

from random_forest import SimpleRandomForest

def main():
    # irisデータセットの読み込み
    iris = load_iris()
    X, y = iris.data, iris.target

    # データを学習用とテスト用に分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ランダムフォレストモデルの作成と学習
    rf = SimpleRandomForest(
        n_estimators=10,  # 決定木の数
        max_depth=5,      # 木の深さの制限
        min_samples_split=2
    )
    
    print("モデルの学習を開始します...")
    rf.fit(X_train, y_train)
    
    # テストデータで予測
    print("テストデータで予測を行います...")
    y_pred = rf.predict(X_test)
    
    # 正解率の計算と表示
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n結果:")
    print(f"正解率: {accuracy:.4f}")
    
    # クラスごとの予測数を表示
    unique, counts = np.unique(y_pred, return_counts=True)
    print("\nクラスごとの予測数:")
    for class_id, count in zip(unique, counts):
        print(f"クラス {class_id}: {count}個")

if __name__ == "__main__":
    main()
