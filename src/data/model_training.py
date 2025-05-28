import yaml
import mlflow
from sklearn.ensemble import RandomForestRegressor
from src.utils.io import save_model, save_predictions
from src.utils.metrics import rmse
import pandas as pd



def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def train_and_evaluate(X_train, y_train, X_val, y_val, config_path, params_path, suffix=""):
    config = load_yaml(config_path)
    params = load_yaml(params_path)
    rf_params = params["random_forest"]
    model = RandomForestRegressor(**rf_params)

    mlflow.set_tracking_uri("file:../outputs/mlruns")
    mlflow.set_tracking_uri("file:../outputs/mlruns")
    mlflow.set_experiment("RandomForestRegression")
    with mlflow.start_run():
        model = RandomForestRegressor(**rf_params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        score = rmse(y_val, preds)

        # Log with MLflow
        mlflow.log_params(rf_params)
        mlflow.log_metric("rmse", score)
        mlflow.sklearn.log_model(model, "model")

        # Save model
        model_path = config["output"]["model_dir"] + f"/random_forest_{suffix}.joblib"
        save_model(model, model_path)

        print(f"Validation RMSE: {score:.4f}")
        return model, score

def predict_and_save(model, X_test, config_path, suffix=""):
    config = load_yaml(config_path)
    preds = model.predict(X_test)
    target_col = config["model"]["target_column"]
    preds_df = pd.DataFrame({
        "id": X_test["id"].astype(int).values,
        target_col: preds
    })
    # Add suffix to the filename if provided
    filename = f"random_forest_predictions{suffix}.csv"
    pred_path = f"{config['output']['predictions_dir']}/{filename}"
    save_predictions(preds_df, pred_path)