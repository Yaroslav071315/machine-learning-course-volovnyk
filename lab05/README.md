# lab05 — PCA, кластеризація та виявлення аномалій

Відтворюваний pipeline: preprocessing без leakage → PCA-діагностика →
KMeans у двох просторах → калібрування та оцінювання детекторів аномалій.

## Мета
Реалізувати повний ML-пайплайн із **коректним розділенням train/test**,
зафіксованими seeds і порогами, що калібруються **тільки на train**.
Отримати числові відповіді на три питання:
1. Який простір кращий для кластеризації — scaled чи PCA?
2. Скільки PCA-компонент достатньо для 90% дисперсії?
3. Який детектор аномалій найкращий для ін'єктованих спотворень?

## Структура репозиторію
lab05/
├── README.md
├── AI_USAGE.md
├── requirements.txt
├── configs/
│ └── config.json
├── notebook/
│ └── lab05.ipynb
├── src/
│ └── (опційно) модулі з функціями
├── results/
│ ├── pca_table.csv
│ ├── pca_dim_selection.csv
│ ├── clust_scaled.csv
│ ├── clust_pca.csv
│ ├── detector_eval.csv
│ └── anomaly_meta.csv
├── illustrations/
│ ├── pca_scree.png
│ ├── pca_2d.png
│ └── recon_error.png
├── artifacts/
│ ├── config.json
│ └── pipeline.pkl
└── tests/
└── (автоматичні тести)

text

## Дані
Синтетика через `sklearn.datasets.make_classification`:
- 600 об'єктів, 10 числових ознак
- 3 класи, 7 інформативних + 2 надлишкові ознаки
- Train/test = 450/150, split за MD5(seed + id), перевірка неперетину

## Швидкий старт
```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute notebook/lab05.ipynb \
    --output notebook/lab05_executed.ipynb
Основні результати
PCA
Компонента	Explained	Cumulative
PC1	0.248	0.248
PC2	0.214	0.462
PC3	0.167	0.628
PC4	0.128	0.757
PC5	0.099	0.856
PC6	0.070	0.926
PC7	0.057	0.983
PC8–10	~0.017	1.000
Робоча розмірність: d = 6 (поріг 0.90, 92.6% дисперсії).

Кластеризація (K=4, n_seeds=5)
Метрика	Scaled	PCA (d=6)
Silhouette	0.163	0.176
Davies–Bouldin	1.726	1.614
Calinski–Harabasz	83.6	93.8
ARI (між seeds)	0.978	0.995
Висновок: PCA-простір кращий, K=4 — стабільне (обидва простори дали 4).

Детектори аномалій (95-й процентиль train, 30 аномалій на test)
detector	ROC-AUC	AP	Precision	Recall	F1
IF_raw	0.691	0.301	0.286	0.067	0.108
IF_pca	0.651	0.412	0.625	0.167	0.263
Recon	0.850	0.705	0.750	0.500	0.600
Переміг PCA-reconstruction error. Причина: ін'єктовані аномалії
(spike ×5, shift +5, noise) погано відновлюються в d=6 компонентах.

Зовнішній аудит
ARI = 0.313, NMI = 0.346, Homogeneity = 0.390, Completeness = 0.311

Кластери 2 і 3 майже ідеально відповідають класам 1 і 2.

Клас 0 розпадається на кластери 0 і 1 (додаткова підструктура).

Реалізація
Основні функції (3.20):

load_and_split(config) -> DataBundle

fit_preprocessor(X_train, config) -> Transformer

select_pca_dimension(X_train_scaled, config) -> DataFrame

evaluate_clustering(X, k_values, seeds) -> DataFrame

measure_cluster_stability(label_runs) -> StabilityReport

inject_controlled_anomalies(X_test, ids, config) -> AnomalyBatch

fit_anomaly_detectors(X_train, Z_train, config) -> DetectorBundle

score_anomalies(bundle, X, Z) -> dict

evaluate_anomalies(scores, recon, y, thr) -> DataFrame

save_artifacts(run) -> None

Тести
tests/ містить перевірки:

Форми transform / inverse_transform.

Неперетин train/test ids.

ARI-інваріантність до перестановки міток.

Детермінованість генератора аномалій.

Діапазони метрик ∈ [0, 1].

Наявність усіх артефактів.

Завантаження pipeline.pkl і прогноз для нового batch.

Валідація конфігу → ValueError на помилкових входах.

Залежності
Див. requirements.txt. Основні: numpy, pandas, scikit-learn,
matplotlib, seaborn, joblib.

Обмеження
Синтетичні дані (гауссоподібні кластери).

w=0.5 у комбінованій формулі — без оптимізації.

Пороги 95-го перцентиля чутливі до частки аномалій.

text

---

## 4. `requirements.txt`

```txt
numpy>=1.26
pandas>=2.1
scikit-learn>=1.4
matplotlib>=3.8
seaborn>=0.13
joblib>=1.3
jupyter>=1.0
pytest>=8.0