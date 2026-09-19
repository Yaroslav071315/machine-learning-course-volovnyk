# data/

Ця робота **не потребує локального зберігання файлу даних**.

- **Набір даних:** Iris Data Set
- **Джерело:** UCI Machine Learning Repository — https://archive.ics.uci.edu/dataset/166/hill+valley
- **Спосіб отримання:**  завантажується безпосередньо в коді зі звернення до мережі:

  ```python
	url = "https://archive.ics.uci.edu/ml/machine-learning-databases/hill-valley/Hill_Valley_without_noise_Training.data"
	df = pd.read_csv(url)
  ```

- **Розмір:** 606 об'єктів, 101 числові ознаки.
- **Ліцензія / умови використання:** відкритий навчальний набір даних, вільно розповсюджується для дослідницьких і навчальних цілей.


