"""
Основной файл с решением соревнования
заполните его вашей загрузкой, предобработкой, процессингом и сохранением файла:

⚠️ НЕ ЗАБУДЬТЕ ЗАПИСАТЬ функцию create_submission() - она создает ответ он должен быть в формате SAMPLE SUBMISSION!
Здесь должен быть весь ваш код для создания предсказаний
"""

def create_submission(predictions):
    """
    Пропишите здесь создание файла submission.csv в папку results
    !!! ВНИМАНИЕ !!! ФАЙЛ должен иметь именно такого названия
    """
    # Создать пандас таблицу submission
    import os
    import pandas as pd
    
    os.makedirs('results', exist_ok=True)
    submission_path = 'results/submission.csv'
    predictions.to_csv(submission_path, index=False)
    
    print(f"Submission файл сохранен: {submission_path}")
    
    return submission_path


def main():
    """
    Главная функция программы
    
    Вы можете изменять эту функцию под свои нужды,
    но обязательно вызовите create_submission() в конце!
    """
    print("=" * 50)
    print("Запуск решения соревнования")
    print("=" * 50)
    
    # Ваш код начинается здесь
    import numpy as np
    import pandas as pd
    import random
    import os

    SEED = 322
    np.random.seed(SEED)
    random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)

    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.metrics import mean_squared_error
    import category_encoders as ce
    # Заменяем lightgbm на xgboost
    import xgboost as xgb
    import optuna
    from scipy import stats

    # Загрузка данных
    train = pd.read_csv("data/train.csv", parse_dates=["dt"])
    test = pd.read_csv("data/test.csv", parse_dates=["dt"])

    # ============================
    # 1. Улучшенная обработка дат
    # ============================
    def add_date_features(df):
        df = df.copy()

        # Цикличные признаки
        df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["day_sin"] = np.sin(2 * np.pi * df["day_of_month"] / 31)
        df["day_cos"] = np.cos(2 * np.pi * df["day_of_month"] / 31)

        # Квартал и сезонность
        df["quarter"] = df["month"] // 4
        df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)

        # Признаки времени (дни от начала)
        if 'dt' in df.columns:
            df["days_from_start"] = (df["dt"] - df["dt"].min()).dt.days

        # Взаимодействие день недели + месяц
        df["dow_month"] = df["dow"] * 100 + df["month"]

        return df

    train = add_date_features(train)
    test = add_date_features(test)

    # ============================
    # 2. Детекция аномалий (улучшенная)
    # ============================
    def detect_anomalies(df, target_cols):
        """Комбинированный детектор аномалий"""

        # Isolation Forest
        iso = IsolationForest(
            n_estimators=200,
            contamination=0.01,
            random_state=SEED,
            n_jobs=-1
        )

        # Local Outlier Factor
        lof = LocalOutlierFactor(
            n_neighbors=50,
            contamination=0.01,
            novelty=False,
            n_jobs=-1
        )

        # Статистические границы (IQR метод)
        q1 = df[target_cols].quantile(0.25)
        q3 = df[target_cols].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 3 * iqr
        upper_bound = q3 + 3 * iqr

        # Совмещаем методы
        iso_scores = iso.fit_predict(df[target_cols])
        lof_scores = lof.fit_predict(df[target_cols])

        # Статистические выбросы
        stat_outliers = ((df[target_cols] < lower_bound) |
                         (df[target_cols] > upper_bound)).any(axis=1)

        # Финальная маска (сохраняем если все методы согласны)
        mask = (iso_scores == 1) & (lof_scores == 1) & (~stat_outliers)

        print(f"Удалено аномалий: {len(df) - mask.sum()} ({100*(len(df)-mask.sum())/len(df):.1f}%)")
        return mask

    anomaly_mask = detect_anomalies(train[['price_p05', 'price_p95']].copy(),
                                    ['price_p05', 'price_p95'])
    train = train[anomaly_mask].reset_index(drop=True)

    # ============================
    # 3. Кластеризация товаров (улучшенная)
    # ============================
    def create_product_features(df, train_only=False):
        """Создание признаков на уровне товара"""

        product_stats = df.groupby("product_id").agg({
            'price_p05': ['mean', 'std', 'median', 'min', 'max'],
            'price_p95': ['mean', 'std', 'median', 'min', 'max'],
            'n_stores': ['mean', 'std'],
            'holiday_flag': 'mean',
            'activity_flag': 'mean'
        }).fillna(0)

        product_stats.columns = ['_'.join(col).strip() for col in product_stats.columns.values]
        product_stats = product_stats.reset_index()

        # Разница между p95 и p05
        product_stats['price_range_mean'] = product_stats['price_p95_mean'] - product_stats['price_p05_mean']
        product_stats['price_range_std'] = product_stats['price_p95_std'] - product_stats['price_p05_std']

        # Стабильность цены
        product_stats['price_stability'] = product_stats['price_p05_std'] / (product_stats['price_p05_mean'] + 1e-6)

        return product_stats

    product_stats_train = create_product_features(train)

    # Кластеризация с оптимизацией количества кластеров
    from sklearn.metrics import silhouette_score

    features_for_clustering = ['price_p05_mean', 'price_p95_mean', 'price_range_mean',
                              'price_stability', 'n_stores_mean']

    X_cluster = StandardScaler().fit_transform(product_stats_train[features_for_clustering])

    # Оптимальное число кластеров (силуэтный метод)
    best_score = -1
    best_n = 5
    for n in range(5, 15):
        kmeans = KMeans(n_clusters=n, random_state=SEED, n_init=10)
        labels = kmeans.fit_predict(X_cluster)
        score = silhouette_score(X_cluster, labels)
        if score > best_score:
            best_score = score
            best_n = n

    print(f"Оптимальное число кластеров: {best_n} (score={best_score:.3f})")

    kmeans = KMeans(n_clusters=best_n, random_state=SEED, n_init=20)
    product_stats_train["product_cluster"] = kmeans.fit_predict(X_cluster)

    # Присоединяем кластеры к данным
    train = train.merge(product_stats_train[["product_id", "product_cluster"]],
                        on="product_id", how="left")
    test = test.merge(product_stats_train[["product_id", "product_cluster"]],
                      on="product_id", how="left")

    # ============================
    # 4. Расширенные признаки (ОБНОВЛЕНО!)
    # ============================
    def create_lag_features(df, group_cols, value_cols, lags=[1, 7, 14]):
        """Создание лаговых признаков - теперь создаем в обеих выборках"""
        df = df.sort_values(['product_id', 'dt']).copy()

        for group in group_cols:
            for value in value_cols:
                for lag in lags:
                    # Для лагов используем shift, но только в пределах группы
                    df[f'{value}_lag_{lag}_{group}'] = df.groupby(group)[value].shift(lag)

        return df

    # Для train создаем лаги
    train = create_lag_features(
        train,
        group_cols=['product_id'],
        value_cols=['price_p05', 'price_p95', 'n_stores']
    )

    # Заполняем пропуски медианами
    lag_cols = [col for col in train.columns if 'lag' in col]
    for col in lag_cols:
        train[col] = train[col].fillna(train[col].median())

    # Для test НЕ создаем лаги на будущих значениях
    # Вместо этого используем последние известные значения для каждого продукта
    def create_test_lag_features(test_df, train_df, group_cols, value_cols, lags=[1, 7, 14]):
        """Создание лаговых признаков для test на основе последних значений из train"""
        test_df = test_df.copy()

        # Берем последние значения для каждого продукта из train
        last_values = {}
        for group in group_cols:
            for value in value_cols:
                for lag in lags:
                    # Для каждого продукта берем последнее значение
                    last_vals = train_df.groupby('product_id')[value].last()
                    test_df = test_df.merge(
                        last_vals.rename(f'{value}_lag_{lag}_{group}'),
                        on='product_id',
                        how='left'
                    )

        return test_df

    # Создаем лаги для test из последних значений train
    test = create_test_lag_features(
        test,
        train,
        group_cols=['product_id'],
        value_cols=['price_p05', 'price_p95', 'n_stores'],
        lags=[1, 7, 14]
    )

    # ============================
    # 5. Кодирование категорий (улучшенное)
    # ============================
    cat_cols = [
        "product_id",
        "management_group_id",
        "first_category_id",
        "second_category_id",
        "third_category_id",
        "product_cluster",
        "dow_month"
    ]

    # Комбинируем разные методы кодирования
    # Target Encoding
    te_encoder = ce.TargetEncoder(cols=cat_cols, smoothing=0.5)
    train_te = te_encoder.fit_transform(train[cat_cols], train['price_p95'])
    test_te = te_encoder.transform(test[cat_cols])

    # LeaveOneOut Encoding (более стабильное)
    loo_encoder = ce.LeaveOneOutEncoder(cols=cat_cols, sigma=0.1)
    train_loo = loo_encoder.fit_transform(train[cat_cols], train['price_p95'])
    test_loo = loo_encoder.transform(test[cat_cols])

    # Объединяем кодировки
    for col in cat_cols:
        train[f'{col}_te'] = train_te[col]
        test[f'{col}_te'] = test_te[col]
        train[f'{col}_loo'] = train_loo[col]
        test[f'{col}_loo'] = test_loo[col]

    # ============================
    # 6. Снижение размерности для категорий
    # ============================
    # PCA для категориальных признаков
    pca_features = [f'{col}_te' for col in cat_cols] + [f'{col}_loo' for col in cat_cols]

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train[pca_features].fillna(0))
    test_scaled = scaler.transform(test[pca_features].fillna(0))

    # Выбираем количество компонент объясняющих 95% дисперсии
    pca = PCA(n_components=0.95, random_state=SEED)
    train_pca = pca.fit_transform(train_scaled)
    test_pca = pca.transform(test_scaled)

    print(f"PCA: сохранили {train_pca.shape[1]} компонент из {train_scaled.shape[1]}")

    for i in range(train_pca.shape[1]):
        train[f'pca_cat_{i}'] = train_pca[:, i]
        test[f'pca_cat_{i}'] = test_pca[:, i]

    # ============================
    # 7. Признаки взаимодействий
    # ============================
    def create_interaction_features(df):
        df = df.copy()

        # Взаимодействие погоды с сезонностью
        df['temp_month'] = df['avg_temperature'] * df['month']
        df['humidity_month'] = df['avg_humidity'] * df['month']

        # Взаимодействие n_stores с категориями
        for cat in ['first_category_id', 'second_category_id', 'third_category_id']:
            df[f'n_stores_{cat}'] = df['n_stores'] * df[cat]

        return df

    train = create_interaction_features(train)
    test = create_interaction_features(test)

    # ============================
    # 8. Дополнительные статистические признаки
    # ============================
    def add_statistical_features(df, product_stats):
        """Добавляем статистические признаки по продукту"""
        # Медианные цены и диапазоны
        df = df.merge(
            product_stats[['product_id', 'price_p05_median', 'price_p95_median']],
            on='product_id',
            how='left'
        )

        # Отношение текущей цены к медианной
        df['p05_to_median'] = df['price_p05'] / (df['price_p05_median'] + 1e-6)
        df['p95_to_median'] = df['price_p95'] / (df['price_p95_median'] + 1e-6)

        return df

    # Только для train добавляем отношение цен
    train = add_statistical_features(train, product_stats_train)

    # Для test добавляем только медианы (отношения будут рассчитаны позже)
    test = test.merge(
        product_stats_train[['product_id', 'price_p05_median', 'price_p95_median']],
        on='product_id',
        how='left'
    )

    # ============================
    # 9. Собираем финальные признаки
    # ============================
    # Создаем список уникальных признаков, которые есть в ОБЕИХ выборках
    base_features = [
        "n_stores", "holiday_flag", "activity_flag",
        "precpt", "avg_temperature", "avg_humidity", "avg_wind_level",
        "dow_sin", "dow_cos", "month_sin", "month_cos",
        "day_sin", "day_cos", "day_of_month", "week_of_year",
        "quarter", "is_weekend", "days_from_start", "dow_month"
    ]

    # Лаговые признаки (они теперь есть в обеих выборках)
    lag_features = [col for col in train.columns if 'lag' in col]

    # PCA компоненты
    pca_features = [f'pca_cat_{i}' for i in range(train_pca.shape[1])]

    # Признаки взаимодействий
    interaction_features = [col for col in train.columns if 'temp_month' in col or
                           'humidity_month' in col or 'n_stores_' in col]

    # Медианные цены
    stat_features = ['price_p05_median', 'price_p95_median']

    # Для train добавляем дополнительные признаки
    train_only_features = ['p05_to_median', 'p95_to_median'] if 'p05_to_median' in train.columns else []

    # Собираем все признаки для train
    all_train_features = base_features + lag_features + pca_features + interaction_features + stat_features + train_only_features
    all_train_features = list(set([f for f in all_train_features if f in train.columns]))

    # Собираем все признаки для test
    all_test_features = base_features + lag_features + pca_features + interaction_features + stat_features
    all_test_features = list(set([f for f in all_test_features if f in test.columns]))

    # Находим пересечение признаков
    common_features = list(set(all_train_features) & set(all_test_features))
    print(f"Общих признаков: {len(common_features)}")
    print(f"Признаков только в train: {len(all_train_features) - len(common_features)}")
    print(f"Признаков только в test: {len(all_test_features) - len(common_features)}")

    # Используем только общие признаки
    existing_features = common_features
    print(f"Используем {len(existing_features)} общих признаков")

    # ============================
    # 10. Обучение ансамбля моделей (ИЗМЕНЕНО для XGBoost)
    # ============================
    def train_quantile_ensemble(X_train, y_train, alpha, n_models=3):
        """Обучает ансамбль моделей XGBoost для квантиля"""
        models = []
        
        # Преобразуем alpha для XGBoost (0.05 → 5, 0.95 → 95)
        quantile = int(alpha * 100)

        for i in range(n_models):
            # Разные подвыборки для каждой модели
            sample_idx = np.random.choice(
                len(X_train),
                size=int(len(X_train) * 0.9),
                replace=False
            )

            # Параметры XGBoost для квантильной регрессии
            params = {
                'objective': 'reg:quantileerror',
                'quantile_alpha': alpha,  # Alpha для квантиля
                'eval_metric': 'quantile',
                'learning_rate': 0.05 * (0.8 + 0.4 * np.random.random()),
                'max_depth': int(6 * (0.8 + 0.4 * np.random.random())),
                'n_estimators': 800,
                'subsample': 0.8 * (0.9 + 0.2 * np.random.random()),
                'colsample_bytree': 0.8 * (0.9 + 0.2 * np.random.random()),
                'reg_alpha': 1.0 * (0.5 + np.random.random()),  # L1 регуляризация
                'reg_lambda': 1.0 * (0.5 + np.random.random()),  # L2 регуляризация
                'min_child_weight': int(40 * (0.8 + 0.4 * np.random.random())),
                'random_state': SEED + i,
                'verbosity': 0
            }

            model = xgb.XGBRegressor(**params)
            
            model.fit(
                X_train.iloc[sample_idx],
                y_train.iloc[sample_idx],
                eval_set=[(X_train.iloc[sample_idx], y_train.iloc[sample_idx])],
                verbose=False
            )

            models.append(model)

        return models

    # Обучаем ансамбли для p05 и p95
    print("Обучаем модель для price_p05...")
    models_p05 = train_quantile_ensemble(train[existing_features], train['price_p05'], 0.05, n_models=4)
    print("Обучаем модель для price_p95...")
    models_p95 = train_quantile_ensemble(train[existing_features], train['price_p95'], 0.95, n_models=4)

    # ============================
    # 11. Предсказание
    # ============================
    def predict_ensemble(models, X):
        """Предсказание ансамблем моделей"""
        predictions = np.zeros((len(X), len(models)))

        for i, model in enumerate(models):
            predictions[:, i] = model.predict(X)

        # Используем взвешенное усреднение (первые модели имеют больший вес)
        weights = np.exp(np.arange(len(models))) / np.exp(np.arange(len(models))).sum()
        return np.average(predictions, axis=1, weights=weights)

    test['price_p05_pred'] = predict_ensemble(models_p05, test[existing_features])
    test['price_p95_pred'] = predict_ensemble(models_p95, test[existing_features])

    # ============================
    # 12. Постобработка
    # ============================
    def postprocess_predictions(df, product_stats):
        """Постобработка предсказаний"""

        # Убедимся, что у нас есть необходимые колонки
        if 'price_p05_median' not in df.columns:
            df = df.merge(product_stats[['product_id', 'price_p05_median', 'price_p95_median']],
                          on='product_id', how='left')

        # Рассчитываем ширину диапазона
        df['width_med'] = df['price_p95_median'] - df['price_p05_median']

        # Блендинг с историческими значениями
        alpha = 0.2  # Вес исторических данных

        df['price_p05'] = (1 - alpha) * df['price_p05_pred'] + alpha * df['price_p05_median']
        df['price_p95'] = (1 - alpha) * df['price_p95_pred'] + alpha * df['price_p95_median']

        # Гарантируем что p05 <= p95
        mask = df['price_p05'] > df['price_p95']
        # Если верхняя граница меньше нижней, расширяем верхнюю
        df.loc[mask, 'price_p95'] = df.loc[mask, 'price_p05'] + df.loc[mask, 'width_med'].fillna(1)

        # Ограничиваем ширину диапазона (не более 3 * исторической медианы)
        max_width_multiplier = 3.0
        current_width = df['price_p95'] - df['price_p05']
        max_allowed_width = max_width_multiplier * df['width_med'].fillna(1)

        mask = current_width > max_allowed_width
        center = (df.loc[mask, 'price_p05'] + df.loc[mask, 'price_p95']) / 2
        half_width = max_allowed_width[mask] / 2

        df.loc[mask, 'price_p05'] = center - half_width
        df.loc[mask, 'price_p95'] = center + half_width

        # Минимальная ширина (для избежания нулевых интервалов)
        min_width = 0.1
        mask = (df['price_p95'] - df['price_p05']) < min_width
        center = (df.loc[mask, 'price_p05'] + df.loc[mask, 'price_p95']) / 2
        df.loc[mask, 'price_p05'] = center - min_width / 2
        df.loc[mask, 'price_p95'] = center + min_width / 2

        return df

    test = postprocess_predictions(test, product_stats_train)

    # ============================
    # 13. Финализация и сохранение
    # ============================
    submission = test[["row_id", "price_p05", "price_p95"]].copy()

    # Обработка особых случаев
    eps = 1e-3
    lower = submission["price_p05"].values
    upper = submission["price_p95"].values

    # Гарантируем корректные границы
    lower = np.minimum(lower, upper - eps)
    upper = np.maximum(upper, lower + eps)

    # Замена бесконечностей
    lower = np.where(np.isinf(lower), np.nan, lower)
    upper = np.where(np.isinf(upper), np.nan, upper)

    # Заполнение пропусков
    lower_median = np.nanmedian(lower)
    upper_median = np.nanmedian(upper)

    lower = np.where(np.isnan(lower), lower_median, lower)
    upper = np.where(np.isnan(upper), upper_median, upper)

    submission["price_p05"] = lower
    submission["price_p95"] = upper

    # Проверка на корректность
    assert (submission["price_p05"] < submission["price_p95"]).all(), "Некорректные границы!"
    assert not submission.isnull().any().any(), "Есть пропуски!"

    # Сохранение
    print(f"\nSubmission saved. Shape: {submission.shape}")
    print(f"Price p05: min={submission['price_p05'].min():.2f}, max={submission['price_p05'].max():.2f}")
    print(f"Price p95: min={submission['price_p95'].min():.2f}, max={submission['price_p95'].max():.2f}")
    print(f"Average width: {(submission['price_p95'] - submission['price_p05']).mean():.2f}")
    print(f"\nSample submission:")
    print(submission.head())
    
    # Создание submission файла (ОБЯЗАТЕЛЬНО!)
    create_submission(submission)
    
    print("=" * 50)
    print("Выполнение завершено успешно!")
    print("=" * 50)


if __name__ == "__main__":
    main()
