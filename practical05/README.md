# ML Model Selection — Hill-Valley Classification

Ноутбук для **інженерного вибору ML-моделі** за 16 розділами методики: від
формулювання вимірюваних вимог — до одноразової перевірки на test set і
збереження відтворюваних артефактів.

---

## 📌 Призначення

Порівняти декілька ML-кандидатів на **єдиному CV-розбитті**, оцінити їх за
якістю **та** інженерними метриками (latency, розмір артефакту, час навчання),
відфільтрувати за жорсткими SLA-обмеженнями і обрати переможця за зваженою
нормалізованою метрикою.

Задача: бінарна класифікація форми сигналу (пагорб vs долина),
датасет **Hill-Valley without noise** (UCI).

---

## 🗂️ Структура проєкту

├── Hill_Valley_without_noise_Training.data # вхідні дані
├── Hill_Valley_without_noise_Testing.data # опційно (для об'єднання)
├── Practical05.ipynb # основний ноутбук
├── artifacts/ # автогенеровані результати
│ ├── benchmark_results.csv
│ ├── quality_vs_latency.png
│ ├── latency_distribution.png
│ ├── scenario_fast.json
│ ├── scenario_quality.json
│ ├── ranking_fast.csv
│ ├── ranking_quality.csv
│ ├── rejected_fast.csv
│ ├── rejected_quality.csv
│ ├── summary_fast.json
│ ├── summary_quality.json
│ ├── report_fast.md
│ ├── report_quality.md
│ ├── pipeline_fast_main.pkl
│ ├── pipeline_fast_backup.pkl
│ └── sensitivity_fast_roc_auc.csv
│ └── sensitivity_fast_roc_auc.png
├── README.md
├── USAGE.md
└── AIUsage.md

---

## 🧩 Компоненти

| Компонент | Призначення |
|---|---|
| `CANDIDATES` | Реєстр 5 моделей: `dummy`, `logreg`, `knn`, `svc_rbf`, `rf` |
| `evaluate_candidate` | Єдине CV-оцінювання (`StratifiedKFold`, 5 folds) |
| `profile_candidate` | 5 прогрівів + 30 вимірювань latency / size / throughput |
| `check_constraints` | Повертає `(ok, violations)` — усі порушення одразу |
| `normalize` | Підтримує `maximize` / `minimize` з явним `direction` |
| `rank_models` | Повертає `(valid_df, rejected_df)` — детерміноване сортування |
| `persist_scenario` | Генерує CSV / JSON / MD / `.pkl` |

---

## ⚙️ Сценарії

| Сценарій | Пріоритет | Головна вага | Ліміт p95 |
|---|---|---|---|
| `fast_inference` | Швидкодія інференсу | `p95_predict_ms` = 0.4 | 50 мс |
| `quality_first`  | Якість           | `roc_auc` = 0.6       | 200 мс |

Обидва — у `artifacts/scenario_*.json`. Зміна конфігу **не вимагає**
редагування Python-коду.

---

## ⚠️ Відомий стан (поточний запуск)

**Обидва сценарії завершились без переможця.**

Причина: `accuracy_min` у конфігах (0.80 / 0.85) вищий за фактичну точність
найкращої моделі (`logreg`, `accuracy_mean = 0.6812`).

**Наслідки:**
- `valid_df` порожній → `ranking_*.csv` порожній, `pipeline_*.pkl` не створені;
- `test_load_saved_pipeline` фейлиться — очікувано;
- графік чутливості порожній — очікувано;
- test set **не використано** (лишився чистим).

**Fix для наступної версії (`config_version: 1.1`):**
1. Знизити `accuracy_min` до реалістичного (0.65 для FAST, 0.70 для QUALITY).
2. Розслідувати аномалію `ROC-AUC = 0.9585` vs `accuracy = 0.6812` у `logreg`
   (поріг 0.5, дисбаланс класів або прихований третій стан).

---

## 🔬 Вимоги до середовища

- Python ≥ 3.9
- `pandas`, `numpy`, `matplotlib`
- `scikit-learn ≥ 1.3`
- `unittest` (стандартна бібліотека)

```bash
pip install pandas numpy matplotlib scikit-learn