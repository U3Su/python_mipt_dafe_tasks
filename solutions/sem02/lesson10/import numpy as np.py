import numpy as np
import seaborn as sns

# 1. загрузка
df = sns.load_dataset("titanic")

# 2. очистка столбцов и строк (Часть 2)
df = df.loc[:, df.isnull().sum() <= len(df) / 2]
df = df.dropna(thresh=int(np.ceil(df.shape[1] / 2)))

# 3. заполнение возраста (Часть 3)
who, age = df["who"], df["age"]
medians = {c: round(age[who == c].median()) for c in ["man", "woman", "child"]}
df["age"] = age.fillna(who.map(medians))

# 4. удаление строк с >1 пропуском (Часть 4)
df = df.dropna(thresh=df.shape[1] - 1)

# теперь df готов — можно делать любые проверки
print("пассажиров с fare=0:", (df["fare"] == 0).sum())
print("из них выжило:", df.loc[df["fare"] == 0, "survived"].sum())
