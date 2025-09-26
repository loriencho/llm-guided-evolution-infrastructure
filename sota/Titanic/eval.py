import pandas as pd
from model import Model
import sklearn.metrics
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    train_df = pd.read_csv('data/processed_train.csv')
    X = train_df.drop("Survived", axis=1)
    y = train_df["Survived"]

    X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=0)

    model = Model()
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)


    matrix = sklearn.metrics.confusion_matrix(y_val, predictions)
    # Print out FP, FN
    print(matrix[0,1], matrix[1,0])

