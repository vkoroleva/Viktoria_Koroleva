"""
Основной файл с решением соревнования
Здесь должен быть весь ваш код для создания предсказаний
"""



def create_submission(predictions):
    """
    Пропишите здесь создание файла submission.csv в папку results
    !!! ВНИМАНИЕ !!! ФАЙЛ должен иметь именно такого названия
    """
    import os

    os.makedirs('results', exist_ok=True)
    submission_path = 'results/submission.csv'

    submission = predictions
    submission.to_csv(submission_path, index=False)

    print(f"Submission файл сохранен: {submission_path}")

    return submission_path

import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import ndcg_score
from catboost import CatBoostRanker, Pool

RANDOM_SEED = 993
np.random.seed(RANDOM_SEED)

def load_data(train_path, test_path):
    """Загрузка train и test данных"""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"\nTrain columns: {train_df.columns.tolist()}")

    return train_df, test_df

def preprocess_text(text):
    """Базовая очистка текста"""
    if not isinstance(text, str):
        return ""

    text = text.lower()

    text = re.sub(r'[^\w\s]', ' ', text)

    text = re.sub(r'\s+', ' ', text).strip()

    return text

def create_features(df):
    """Создание признаков из текстовых данных"""
    df = df.copy()

    text_columns = ['query', 'product_title', 'product_description',
                    'product_bullet_point', 'product_brand', 'product_color']

    for col in text_columns:
        if col in df.columns:
            df[f'{col}_processed'] = df[col].fillna('').apply(preprocess_text)

    if 'product_title_processed' in df.columns and 'product_brand_processed' in df.columns:
        df['title_brand'] = df['product_title_processed'] + ' ' + df['product_brand_processed']

    if 'product_description_processed' in df.columns and 'product_bullet_point_processed' in df.columns:
        df['desc_bullets'] = df['product_description_processed'] + ' ' + df['product_bullet_point_processed']

    for col in text_columns:
        if f'{col}_processed' in df.columns:
            df[f'{col}_len'] = df[f'{col}_processed'].apply(len)
            df[f'{col}_word_count'] = df[f'{col}_processed'].apply(lambda x: len(x.split()))

    if 'query_processed' in df.columns:
        for prod_col in ['product_title_processed', 'product_description_processed',
                         'product_bullet_point_processed']:
            if prod_col in df.columns:
                col_name = prod_col.replace('_processed', '')
                df[f'query_{col_name}_common_words'] = df.apply(
                    lambda row: len(set(row['query_processed'].split()) &
                                   set(row[prod_col].split())), axis=1
                )
                df[f'query_{col_name}_jaccard'] = df.apply(
                    lambda row: len(set(row['query_processed'].split()) &
                                   set(row[prod_col].split())) /
                    max(1, len(set(row['query_processed'].split()) |
                             set(row[prod_col].split()))), axis=1
                )

    categorical_cols = ['product_locale']
    for col in ['product_brand', 'product_color']:
        if col in df.columns:
            categorical_cols.append(col)

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna('unknown')
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col])

    return df

def select_features(df):
    """Отбор признаков для модели"""
    exclude_cols = ['id', 'query_id', 'product_id', 'relevance']
    text_columns = ['query', 'product_title', 'product_description',
                    'product_bullet_point', 'product_brand', 'product_color']

    for col in text_columns + [f'{c}_processed' for c in text_columns] + ['title_brand', 'desc_bullets']:
        if col in df.columns:
            exclude_cols.append(col)

    feature_cols = [col for col in df.columns if col not in exclude_cols]

    feature_cols = [col for col in feature_cols if df[col].notnull().all() or df[col].dtype == 'object']

    return feature_cols

def evaluate_model(model, X_test, y_test, query_groups_test, model_type='catboost'):
    """Оценка модели на тестовых данных"""
    if model_type == 'catboost':
        cat_features = []
        for col in X_test.columns:
            if X_test[col].dtype == 'object' or 'encoded' in col:
                cat_features.append(col)

        test_pool = Pool(
            X_test,
            y_test,
            group_id=query_groups_test,
            cat_features=cat_features if cat_features else None
        )
        predictions = model.predict(test_pool)
    elif model_type == 'lightgbm':
        predictions = model.predict(X_test)
    elif model_type == 'xgboost':
        predictions = model.predict(X_test)
    else:
        predictions = model.predict(X_test)

    unique_queries = np.unique(query_groups_test)
    ndcg_scores = []

    for query in unique_queries:
        mask = query_groups_test == query
        if mask.sum() == 0:
            continue

        y_true_subset = y_test[mask].reshape(1, -1)
        y_score_subset = predictions[mask].reshape(1, -1)

        if len(y_true_subset[0]) >= 2:
            try:
                ndcg = ndcg_score(y_true_subset, y_score_subset, k=min(10, len(y_true_subset[0])))
                ndcg_scores.append(ndcg)
            except:
                continue

    return np.mean(ndcg_scores) if ndcg_scores else 0

def train_ranking_model_catboost(train_df, test_df):
    """Обучение модели ранжирования CatBoost"""

    print(f"\n=== Обучение модели типа: catboost ===")

    print("1. Создание признаков...")
    train_df_features = create_features(train_df)
    test_df_features = create_features(test_df)

    print("2. Отбор признаков...")
    feature_cols = select_features(train_df_features)

    print(f"   Используется {len(feature_cols)} признаков")
    print(f"   Пример признаков: {feature_cols[:10]}...")

    X = train_df_features[feature_cols].copy()
    y = train_df_features['relevance'].values
    query_groups = train_df_features['query_id'].values

    print("3. Разделение данных...")
    unique_queries = train_df_features['query_id'].unique()

    train_queries, val_queries = train_test_split(
        unique_queries, test_size=0.2, random_state=RANDOM_SEED
    )

    train_mask = train_df_features['query_id'].isin(train_queries)
    val_mask = train_df_features['query_id'].isin(val_queries)

    X_train, X_val = X[train_mask], X[val_mask]
    y_train, y_val = y[train_mask], y[val_mask]
    query_groups_train, query_groups_val = query_groups[train_mask], query_groups[val_mask]

    print(f"   Train shape: {X_train.shape}, Val shape: {X_val.shape}")

    print("4. Обучение модели...")

    cat_features = []
    for col in X_train.columns:
        if X_train[col].dtype == 'object' or 'encoded' in col:
            X_train[col] = X_train[col].astype(str).fillna('nan')
            X_val[col] = X_val[col].astype(str).fillna('nan')
            cat_features.append(col)

    print(f"   Категориальные признаки: {len(cat_features)}")

    params = {
        'iterations': 1500,
        'depth': 8,
        'learning_rate': 0.05,
        'l2_leaf_reg': 3.0,
        'random_strength': 1.0,
        'bagging_temperature': 0.8,
        'od_type': 'Iter',
        'od_wait': 50,
        'random_seed': RANDOM_SEED,
        'verbose': 100,
        'task_type': 'CPU',
        'loss_function': 'YetiRank'
    }

    model = CatBoostRanker(**params)

    train_pool = Pool(
        X_train, y_train,
        group_id=query_groups_train,
        cat_features=cat_features if cat_features else None
    )

    val_pool = Pool(
        X_val, y_val,
        group_id=query_groups_val,
        cat_features=cat_features if cat_features else None
    )

    model.fit(
        train_pool,
        eval_set=val_pool,
        verbose=100,
        early_stopping_rounds=100,
        use_best_model=True
    )

    print("5. Оценка на валидации...")
    val_score = evaluate_model(model, X_val, y_val, query_groups_val, 'catboost')
    print(f"   nDCG@10 на валидации: {val_score:.4f}")

    print("6. Обучение на всех данных...")

    for col in cat_features:
        X[col] = X[col].astype(str).fillna('nan')

    full_pool = Pool(
        X, y,
        group_id=query_groups,
        cat_features=cat_features if cat_features else None
    )
    model.fit(full_pool, verbose=100)

    print("7. Предсказания на тестовых данных...")
    X_test = test_df_features[feature_cols].copy()

    for col in cat_features:
        if col in X_test.columns:
            X_test[col] = X_test[col].astype(str).fillna('nan')

    test_predictions = model.predict(X_test)

    submission = pd.DataFrame({
        'id': test_df['id'],
        'prediction': test_predictions
    })

    return model, submission, val_score

def main():
    """Основной пайплайн"""
    print("=" * 50)
    print("RANKING MODEL TRAINING WITH CATBOOST")
    print("=" * 50)

    train_path = 'data/train.csv'
    test_path = 'data/test.csv'

    print("\n1. Загрузка данных...")
    train_df, test_df = load_data(train_path, test_path)

    print("\n" + "=" * 50)
    catboost_model, catboost_submission, catboost_score = train_ranking_model_catboost(
        train_df, test_df
    )

    print(f"\n   Submission shape: {catboost_submission.shape}")
    print("\n   Первые 5 строк submission:")
    print(catboost_submission.head())

    create_submission(catboost_submission)

    print("\n" + "=" * 50)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 50)

    return catboost_submission

if __name__ == "__main__":
    submission = main()
