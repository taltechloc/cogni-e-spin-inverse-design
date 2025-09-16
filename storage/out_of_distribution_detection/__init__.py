class NoveltyDetection:
    def __init__(self, definition, model):
        self._model = model

    def run(self, new_data):
        y_pred_train = self._model.predict(X_train)
        y_pred_test = self._model.predict(X_test)
        y_pred_outliers = self._model.predict(X_outliers)
        n_error_train = y_pred_train[y_pred_train == -1].size
        n_error_test = y_pred_test[y_pred_test == -1].size
        n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size
