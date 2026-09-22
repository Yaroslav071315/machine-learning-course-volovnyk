# Дані

## Джерело

Набір даних **Hill-Valley** отримано з UCI Machine Learning Repository:
[https://archive.ics.uci.edu/dataset/166/hill+valley](https://archive.ics.uci.edu/dataset/166/hill+valley)

## Спосіб отримання

Дані завантажуються програмно через пакет `ucimlrepo`:

```python
from ucimlrepo import fetch_ucirepo
hill_valley = fetch_ucirepo(id=166)
X = hill_valley.data.features
y = hill_valley.data.targets