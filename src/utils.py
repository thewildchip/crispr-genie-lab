from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def print_validation(val_y, pred_y):
    print("MAE: ", mean_absolute_error(val_y, pred_y))
    print("MSE: ", mean_squared_error(val_y, pred_y))
    print("R2:", r2_score(val_y, pred_y))
