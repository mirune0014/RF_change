import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv(r'my_random_forest_project\combined15_23_processdata_ver2_3.csv')

# データの前処理
data = data.drop(['primaryid','caseid','route','age_grp','dose_amt','dose_unit','dose_freq'], axis=1)


print(data['adverse_event'].value_counts())

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, recall_score
import numpy as np

from random_forest import SimpleRandomForest

def main():

    # DataFrameをnumpy配列に変換
    X = data.drop('adverse_event', axis=1).to_numpy()
    y = data['adverse_event'].to_numpy()
    
    # データの正規化
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 不均衡データに対処するための層化サンプリング
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ランダムフォレストモデルの作成と学習
    rf = SimpleRandomForest(
        n_estimators=100,  # 決定木の数を増やす
        max_depth=10,      # 木の深さを増やす
        min_samples_split=5  # 分割のための最小サンプル数を増やす
    )
    
    print("モデルの学習を開始します...")
    rf.fit(X_train, y_train)
    
    # テストデータで予測
    print("テストデータで予測を行います...")
    y_pred = rf.predict(X_test)
    
    # 評価指標の計算と表示
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"\n結果:")
    print(f"正解率: {accuracy:.4f}")
    print(f"F1スコア（マクロ平均）: {f1:.4f}")
    print(f"Recall: {recall_score(y_test, y_pred, average='macro'):.4f}")
    
    # 詳細な性能レポートを表示
    print("\n詳細な性能評価:")
    print(classification_report(y_test, y_pred))
    
    # クラスごとの予測数を表示
    unique, counts = np.unique(y_pred, return_counts=True)
    print("\nクラスごとの予測数:")
    for class_id, count in zip(unique, counts):
        print(f"クラス {class_id}: {count}個")

if __name__ == "__main__":
    main()
