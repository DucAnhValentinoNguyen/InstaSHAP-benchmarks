import pandas as pd

cols = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "class"
]

# Load train
df_train = pd.read_csv("data/adult.data", header=None, names=cols, skipinitialspace=True)

# Load test (skip first header row)
df_test = pd.read_csv("data/adult.test", header=0, names=cols, skipinitialspace=True)

# Fix trailing '.' in labels
df_test["class"] = df_test["class"].str.replace(".", "", regex=False)

# Combine
df = pd.concat([df_train, df_test], ignore_index=True)

df.to_csv("data/adult.csv", index=False)
print("Saved data/adult.csv")
