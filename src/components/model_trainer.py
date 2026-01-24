import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import evaluate_models, save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing data")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "K-Neighbors": KNeighborsRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor()
            }

            model_report = evaluate_models(X_train, y_train, X_test, y_test, models)

            for model_name, model in models.items():
                logging.info(f"Training {model_name}")
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                r2 = r2_score(y_test, y_pred)
                model_report[model_name] = r2
                logging.info(f"{model_name} R2 Score: {r2}")

            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]
            best_accuracy = model_report[best_model_name]

            if best_accuracy < 0.5:
                raise CustomException("No suitable model found with accuracy above threshold", sys)
            
            logging.info(f"Best Model: {best_model_name} with R2 Score: {best_accuracy}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logging.info("Model saved successfully")

            return best_accuracy

        except Exception as e:
            logging.error("Error occurred during model training")
            raise CustomException(e, sys)


