import yaml
import mlflow
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from src.utils.io import save_model, save_predictions
from src.utils.metrics import rmse
import pandas as pd
import optuna



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

def optimize_xgboost_with_optuna(X_train, y_train, X_val, y_val, config_path, suffix="", n_trials=50):
    config = load_yaml(config_path)

    mlflow.set_tracking_uri("file:../outputs/mlruns")
    mlflow.set_experiment("XGBoost_Optuna")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
            "random_state": 42,
            "n_jobs": -1,
        }

        model = XGBRegressor(**params)
        model.fit(X_train, y_train, verbose=False)

        preds = model.predict(X_val)
        score = rmse(y_val, preds)
        trial.set_user_attr("best_model", model)
        return score

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_model = study.best_trial.user_attrs["best_model"]
    best_score = study.best_value
    best_params = study.best_params

    with mlflow.start_run():
        mlflow.log_params(best_params)
        mlflow.log_metric("rmse", best_score)
        mlflow.sklearn.log_model(best_model, "model")

        model_path = config["output"]["model_dir"] + f"/xgboost_optuna_{suffix}.joblib"
        save_model(best_model, model_path)
        print(f"Best Validation RMSE: {best_score:.4f}")

    return best_model, best_score, best_params


def predict_and_save(model, X_test, config_path, suffix=""):
    config = load_yaml(config_path)
    preds = model.predict(X_test)
    target_col = config["model"]["target_column"]
    preds_df = pd.DataFrame({
        "id": X_test["id"].astype(int).values,
        target_col: preds
    })
    # Add suffix to the filename if provided
    filename = f"predictions_{suffix}.csv"
    pred_path = f"{config['output']['predictions_dir']}/{filename}"
    save_predictions(preds_df, pred_path)