import numpy as np

def walk_forward(X, y, model, window=200):
    results = []

    for i in range(window, len(X)-1):
        X_train, y_train = X[:i], y[:i]
        X_test, y_test = X[i:i+1], y[i:i+1]

        model.fit(X_train, y_train)
        pred = model.predict_proba(X_test)[0]

        results.append(pred)

    return np.array(results)