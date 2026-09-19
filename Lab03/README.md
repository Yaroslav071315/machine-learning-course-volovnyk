# ML Lab: Geometric Experiment, Scaling, Metric Classifiers, SVM & Dimensionality Study

## Опис проєкту
Цей репозиторій містить повне виконання лабораторної/практичної роботи з машинного навчання. Робота розділена на наступні етапи:

1. **Контрольний геометричний експеримент**: Візуалізація та аналіз впливу масштабування ознак на синтетичному датасеті `make_moons` для KNN, Linear SVM та RBF SVM.
2. **Підготовка індивідуальних даних**: Побудова препроцесингових пайплайнів (`Pipeline`, `ColumnTransformer`) із застосуванням `StandardScaler`, `MinMaxScaler` та `OneHotEncoder`.
3. **Дослідження масштабування**: Порівняльний аналіз моделей без scaler, зі `StandardScaler` та з `MinMaxScaler` на основі 5-fold cross-validation з урахуванням одиниць вимірювання та діапазонів ознак.
4. **Метричні класифікатори**: Налаштування та оцінка `NearestCentroid` та `KNeighborsClassifier` за сіткою гіперпараметрів ($k \in \{1, 3, 5, 7, 11\}$, $p \in \{1, 2\}$, `weights \in {'uniform', 'distance'}`).
5. **Машини опорних векторів (SVM)**: 
   - Налаштування $C$ для Linear SVM.
   - Спільна підбірка $C$ та $\gamma$ за сіткою для RBF SVM.
   - Аналіз кількості опорних векторів та побудова теплової карти (Heatmap).
6. **Експеримент із розмірністю (Curse of Dimensionality)**: 
   - Додавання неінформативного шуму ($N_{noise} \in [0, 500]$) для ізоляції впливу розмірності.
   - Обчислення медіани $d_{min} / d_{max}$ та оцінка деградації метрики KNN.

## Структура файлів
* `notebook.ipynb` — Jupyter Notebook із повним виконаним кодом, графіками та теоретичними роз'ясненнями.
* `AI_USAGE.md` — декларація про використання інструментів ШІ.
* `README.md` — поточний файл опису.

## Вимоги до середовища
* Python 3.9+
* `numpy`
* `pandas`
* `scipy`
* `matplotlib`
* `seaborn`
* `scikit-learn` >= 1.2

Встановлення залежностей:
```bash
pip install numpy pandas scipy matplotlib seaborn scikit-learn