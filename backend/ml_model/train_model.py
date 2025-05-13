import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

class DiseasePredictionModel:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.scaler = None
        self.trained_feature_columns = []
        self.categorical_modes = {}
        self.target_column = 'Disease'

    def preprocess_data(self, df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        df_processed = df.copy()
        logger.debug(f"PREPROCESS (is_training={is_training}) - Initial df shape: {df_processed.shape}, Columns: {df_processed.columns.tolist()}")
        if not is_training:
             logger.debug(f"PREPROCESS (is_training={is_training}) - Input data head for prediction:\n{df_processed.head()}")


        try:
            feature_cols_to_lower = ['Animal', 'Symptom 1', 'Symptom 2', 'Symptom 3']
            for col in feature_cols_to_lower:
                if col in df_processed.columns:
                    df_processed[col] = df_processed[col].astype(str).str.lower().str.strip()
                else:
                    if not is_training: logger.warning(f"PREPROCESS: Input column '{col}' missing for preprocessing. Filling with 'unknown'.")
                    df_processed[col] = 'unknown'
            
            numerical_cols_raw = ['Age', 'Temperature']
            for col in numerical_cols_raw:
                if col in df_processed.columns:
                    df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                    if is_training:
                        median_val = df_processed[col].median()
                        df_processed[col] = df_processed[col].fillna(median_val if not pd.isna(median_val) else 0)
                    else: 
                        if df_processed[col].isnull().any():
                            logger.warning(f"PREPROCESS: NaN found in numerical column '{col}' during prediction. Filling with 0.")
                            df_processed[col] = df_processed[col].fillna(0)
                else:
                    if not is_training: logger.warning(f"PREPROCESS: Numerical column '{col}' missing. Filling with 0.")
                    df_processed[col] = 0

            # Symptom_Combination feature REMOVED
            # symptom_cols_for_combination = ['Symptom 1', 'Symptom 2', 'Symptom 3']
            # df_processed['Symptom_Combination'] = df_processed.apply(
            #     lambda x: '_'.join(sorted([str(x[s_col]) for s_col in symptom_cols_for_combination])),
            #     axis=1
            # )

            df_processed['Temperature_Category'] = pd.cut(
                df_processed['Temperature'],
                bins=[0, 100, 101, 102, 103, 104, float('inf')],
                labels=['very_low', 'low', 'normal', 'high', 'very_high', 'extreme'],
                include_lowest=True, right=False
            ).astype(str)

            df_processed['Age_Category'] = pd.cut(
                df_processed['Age'],
                bins=[0, 1, 2, 3, 5, 10, float('inf')],
                labels=['very_young', 'young', 'adolescent', 'adult', 'mature', 'senior'],
                include_lowest=True, right=False
            ).astype(str)

            df_processed['Age_Temperature'] = df_processed['Age_Category'] + '_' + df_processed['Temperature_Category']
            df_processed['Animal_Age'] = df_processed['Animal'] + '_' + df_processed['Age_Category']
            
            if not is_training: logger.debug(f"PREPROCESS: After feature engineering:\n{df_processed.head()}")

            categorical_features_to_encode = [
                'Animal', 'Symptom 1', 'Symptom 2', 'Symptom 3',
                # 'Symptom_Combination', # REMOVED
                'Temperature_Category',
                'Age_Category', 'Age_Temperature', 'Animal_Age'
            ]

            for col in categorical_features_to_encode:
                df_processed[col] = df_processed[col].astype(str)
                if is_training:
                    self.categorical_modes[col] = df_processed[col].mode()[0] if not df_processed[col].mode().empty else 'unknown'
                    le = LabelEncoder()
                    unique_values = list(df_processed[col].unique())
                    if 'unknown_label_placeholder' not in unique_values: 
                        unique_values.append('unknown_label_placeholder')
                    le.fit(unique_values)
                    self.label_encoders[col] = le
                    df_processed[col] = le.transform(df_processed[col])
                else: 
                    if col in self.label_encoders:
                        le = self.label_encoders[col]
                        transformed_col_values = []
                        for label_idx, label in enumerate(df_processed[col]):
                            if label in le.classes_:
                                transformed_col_values.append(le.transform([label])[0])
                            else:
                                logger.warning(f"PREPROCESS: Unseen label '{label}' in column '{col}' (row {label_idx}) during prediction. Mapping to 'unknown_label_placeholder'.")
                                if 'unknown_label_placeholder' in le.classes_:
                                    transformed_col_values.append(le.transform(['unknown_label_placeholder'])[0])
                                else:
                                    mode_val = self.categorical_modes.get(col, 'unknown_label_placeholder') 
                                    logger.warning(f"PREPROCESS: 'unknown_label_placeholder' not in classes for '{col}'. Using mode '{mode_val}'.")
                                    if mode_val in le.classes_:
                                        transformed_col_values.append(le.transform([mode_val])[0])
                                    else:
                                        logger.error(f"PREPROCESS: Mode '{mode_val}' for '{col}' also unseen. Defaulting to 0 for '{label}'. This is problematic.")
                                        transformed_col_values.append(0) 
                        df_processed[col] = transformed_col_values
                    else:
                        logger.error(f"PREPROCESS: LabelEncoder for column '{col}' not found during prediction. Filling with 0. Model performance will be poor.")
                        df_processed[col] = 0
            
            if not is_training: logger.debug(f"PREPROCESS: After categorical encoding:\n{df_processed[categorical_features_to_encode].head()}")

            numerical_features_to_scale = ['Age', 'Temperature']
            if is_training:
                self.scaler = StandardScaler()
                df_processed[numerical_features_to_scale] = self.scaler.fit_transform(df_processed[numerical_features_to_scale])
            else:
                if self.scaler and hasattr(self.scaler, 'mean_'):
                    df_processed[numerical_features_to_scale] = self.scaler.transform(df_processed[numerical_features_to_scale])
                else:
                    logger.warning("PREPROCESS: Scaler not fitted or found. Numerical features will not be scaled for this prediction.")
            
            if not is_training: logger.debug(f"PREPROCESS: After numerical scaling:\n{df_processed[numerical_features_to_scale].head()}")

            # Define the final list of features for the model
            final_model_features = categorical_features_to_encode + numerical_features_to_scale
            
            if is_training:
                self.trained_feature_columns = final_model_features
                logger.info(f"PREPROCESS: Model will be trained on columns: {self.trained_feature_columns}")

            output_df = pd.DataFrame()
            columns_to_use = self.trained_feature_columns if not is_training and self.trained_feature_columns else final_model_features
            
            missing_during_selection = []
            for col_name in columns_to_use:
                if col_name in df_processed:
                    output_df[col_name] = df_processed[col_name]
                else:
                    logger.error(f"PREPROCESS CRITICAL: Column '{col_name}' missing from processed DataFrame for model input. Filling with 0.")
                    output_df[col_name] = 0 
                    missing_during_selection.append(col_name)
            
            if missing_during_selection:
                 logger.error(f"PREPROCESS: The following columns were missing and filled with 0 before returning: {missing_during_selection}")

            logger.debug(f"PREPROCESS (is_training={is_training}) - Preprocessing complete. Output df shape: {output_df.shape}, Columns: {output_df.columns.tolist()}")
            if not is_training: logger.debug(f"PREPROCESS (is_training={is_training}) - Final preprocessed data head for prediction:\n{output_df.head()}")
            return output_df

        except Exception as e:
            logger.error(f"Error during preprocessing (is_training={is_training}): {e}", exc_info=True)
            raise

    def train_model(self, data_path: Path):
        logger.info(f"Starting model training process using data from: {data_path}")
        try:
            df_original = pd.read_csv(data_path)
            logger.info(f"Dataset loaded successfully. Shape: {df_original.shape}")
        except FileNotFoundError:
            logger.error(f"Dataset file not found at {data_path}. Please check the path.")
            raise
        except Exception as e:
            logger.error(f"Error loading dataset: {e}", exc_info=True)
            raise

        if self.target_column not in df_original.columns:
            logger.error(f"Target column '{self.target_column}' not found in the dataset.")
            raise ValueError(f"Target column '{self.target_column}' missing.")

        X_raw = df_original.drop(columns=[self.target_column], errors='ignore')
        y_raw = df_original[self.target_column].astype(str).str.lower().str.strip()

        logger.info("Preprocessing features (X)...")
        X_processed = self.preprocess_data(X_raw, is_training=True)
        
        logger.info("Encoding target variable (y)...")
        target_le = LabelEncoder()
        y_encoded = target_le.fit_transform(y_raw)
        self.label_encoders[self.target_column] = target_le
        
        logger.info(f"Class distribution in target variable (encoded):\n{pd.Series(y_encoded).value_counts(normalize=True)}")
        logger.info(f"Target classes learned by encoder: {list(target_le.classes_)}")


        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        logger.info(f"Data split into training and testing sets. Train shape: {X_train.shape}, Test shape: {X_test.shape}")

        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'class_weight': ['balanced', 'balanced_subsample'],
            'criterion': ['gini', 'entropy']
        }
        
        rf_model = RandomForestClassifier(random_state=42)
        
        logger.info("Starting GridSearchCV for hyperparameter tuning...")
        grid_search = GridSearchCV(
            estimator=rf_model,
            param_grid=param_grid,
            cv=3, 
            n_jobs=-1,
            verbose=1,
            scoring='f1_weighted'
        )
        
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        logger.info(f"GridSearchCV complete. Best parameters: {grid_search.best_params_}")
        
        y_pred_test = self.model.predict(X_test)
        
        logger.info("\n--- Test Set Evaluation ---")
        logger.info(f"Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")
        logger.info("Classification Report:")
        try:
            class_names = self.label_encoders[self.target_column].classes_
            logger.info("\n" + classification_report(y_test, y_pred_test, target_names=class_names, zero_division=0))
        except Exception as report_err:
            logger.warning(f"Could not generate classification report with target names: {report_err}")
            logger.info("\n" + classification_report(y_test, y_pred_test, zero_division=0))
        
        if hasattr(self.model, 'feature_importances_') and self.trained_feature_columns and len(self.trained_feature_columns) == len(self.model.feature_importances_):
            feature_importances_df = pd.DataFrame({
                'feature': self.trained_feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values(by='importance', ascending=False)
            logger.info("\nTop 10 Feature Importances:")
            logger.info(f"\n{feature_importances_df.head(10)}")
        else:
            logger.warning("Could not log feature importances: Mismatch in feature column length or model has no feature_importances_ attribute.")


        logger.info("Model training complete.")
        return accuracy_score(y_test, y_pred_test)

    def save_model_artifacts(self, model_dir_path: Path):
        model_dir_path.mkdir(parents=True, exist_ok=True)
        if not self.model:
            logger.error("Model is not trained. Cannot save artifacts.")
            raise ValueError("Model not trained.")
        try:
            joblib.dump(self.model, model_dir_path / 'model.joblib')
            joblib.dump(self.label_encoders, model_dir_path / 'label_encoders.joblib')
            joblib.dump(self.scaler, model_dir_path / 'scaler.joblib')
            joblib.dump(self.trained_feature_columns, model_dir_path / 'trained_feature_columns.joblib')
            joblib.dump(self.categorical_modes, model_dir_path / 'categorical_modes.joblib')
            logger.info(f"All model artifacts saved successfully to {model_dir_path}")
        except Exception as e:
            logger.error(f"Error saving model artifacts: {e}", exc_info=True)
            raise

    def load_model_artifacts(self, model_dir_path: Path):
        if not model_dir_path.exists():
            logger.error(f"Model directory not found: {model_dir_path}")
            raise FileNotFoundError(f"Model directory not found: {model_dir_path}")
        try:
            self.model = joblib.load(model_dir_path / 'model.joblib')
            self.label_encoders = joblib.load(model_dir_path / 'label_encoders.joblib')
            self.scaler = joblib.load(model_dir_path / 'scaler.joblib')
            self.trained_feature_columns = joblib.load(model_dir_path / 'trained_feature_columns.joblib')
            self.categorical_modes = joblib.load(model_dir_path / 'categorical_modes.joblib')
            logger.info(f"All model artifacts loaded successfully from {model_dir_path}")
        except FileNotFoundError as fnf_err:
            logger.error(f"Error loading model artifacts: A required file was not found in {model_dir_path}. {fnf_err}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Error loading model artifacts: {e}", exc_info=True)
            raise

    def predict(self, input_data_df: pd.DataFrame) -> tuple[str, float]:
        logger.info("PREDICT: Starting prediction process...")
        logger.debug(f"PREDICT: Raw input DataFrame to predict method:\n{input_data_df}")

        if not self.model:
            logger.error("PREDICT: Model is not loaded. Cannot make predictions.")
            raise ValueError("Model not loaded.")
        if not self.trained_feature_columns:
            logger.error("PREDICT: Trained feature columns list is empty. Model may not be loaded correctly.")
            raise ValueError("Trained feature columns not available.")

        X_processed_for_predict = self.preprocess_data(input_data_df, is_training=False)
        logger.debug(f"PREDICT: DataFrame after preprocessing (before column alignment):\n{X_processed_for_predict}")
        logger.debug(f"PREDICT: Columns in X_processed_for_predict: {X_processed_for_predict.columns.tolist()}")
        logger.debug(f"PREDICT: Expected trained_feature_columns: {self.trained_feature_columns}")


        try:
            final_X_for_model = pd.DataFrame(columns=self.trained_feature_columns)
            for col in self.trained_feature_columns:
                if col in X_processed_for_predict.columns:
                    final_X_for_model[col] = X_processed_for_predict[col]
                else:
                    logger.error(f"PREDICT CRITICAL: Column '{col}' from trained_feature_columns is MISSING in X_processed_for_predict. Filling with 0.")
                    final_X_for_model[col] = 0 
            final_X_for_model = final_X_for_model[self.trained_feature_columns]

        except KeyError as e:
            missing_cols = set(self.trained_feature_columns) - set(X_processed_for_predict.columns)
            extra_cols = set(X_processed_for_predict.columns) - set(self.trained_feature_columns)
            logger.error(f"PREDICT: Column mismatch error during prediction. Missing: {missing_cols}, Extra: {extra_cols}. Error: {e}", exc_info=True)
            raise ValueError(f"Feature mismatch during prediction. Missing: {missing_cols}, Extra: {extra_cols}")

        logger.info(f"PREDICT: Final feature vector shape for model: {final_X_for_model.shape}")
        logger.debug(f"PREDICT: Final feature vector columns for model: {final_X_for_model.columns.tolist()}")
        logger.debug(f"PREDICT: Final feature vector values (numeric array) being fed to model.predict():\n{final_X_for_model.values}")


        prediction_encoded = self.model.predict(final_X_for_model)[0]
        probabilities = self.model.predict_proba(final_X_for_model)[0]
        
        target_label_encoder = self.label_encoders.get(self.target_column)
        if not target_label_encoder:
            logger.error("PREDICT: Target label encoder not found! Cannot decode prediction.")
            raise ValueError("Target label encoder missing.")

        predicted_disease_label = target_label_encoder.inverse_transform([prediction_encoded])[0]
        
        try:
            class_names = target_label_encoder.classes_
            log_probs = {class_names[i]: f"{prob:.4f}" for i, prob in enumerate(probabilities)}
            logger.info(f"PREDICT: Prediction probabilities for all classes:\n{log_probs}")
        except Exception as e_prob:
            logger.warning(f"PREDICT: Could not log detailed probabilities with class names: {e_prob}")
            logger.info(f"PREDICT: Raw probabilities: {probabilities}")

        confidence = float(probabilities[prediction_encoded]) 
        
        logger.info(f"PREDICT: Final prediction: '{predicted_disease_label}', Confidence: {confidence:.4f}")
        return predicted_disease_label, confidence

if __name__ == "__main__":
    logger.info("Executing train_model.py script...")
    PROJECT_ROOT = Path(__file__).parent.parent.parent 
    DATASET_PATH = PROJECT_ROOT / "animal_disease_dataset - animal_disease_dataset.csv.csv"
    SAVED_MODEL_DIR = Path(__file__).parent / "saved_model"

    if not DATASET_PATH.exists():
        logger.error(f"FATAL: Dataset not found at the expected path: {DATASET_PATH}")
    else:
        model_trainer = DiseasePredictionModel()
        try:
            logger.info(f"Attempting to train model using dataset: {DATASET_PATH}")
            model_trainer.train_model(DATASET_PATH)
            logger.info("Model training finished. Saving artifacts...")
            model_trainer.save_model_artifacts(SAVED_MODEL_DIR)
            logger.info(f"Model artifacts saved to: {SAVED_MODEL_DIR}")

            logger.info("\n--- Test: Loading model and making a sample prediction ---")
            loaded_model = DiseasePredictionModel()
            loaded_model.load_model_artifacts(SAVED_MODEL_DIR)
            sample_raw_data = pd.DataFrame([{
                'Animal': 'cow', 'Age': 3, 'Temperature': 105.6, 
                'Symptom 1': 'swelling in muscle', 
                'Symptom 2': 'swelling in limb', 
                'Symptom 3': 'swelling in abdomen'
            }])
            predicted_disease, confidence = loaded_model.predict(sample_raw_data)
            logger.info(f"Sample Prediction on loaded model -> Disease: {predicted_disease}, Confidence: {confidence:.2f}")

            sample_raw_data_2 = pd.DataFrame([{ 
                'Animal': 'cow', 'Age': 4.0, 'Temperature': 103.1,
                'Symptom 1': 'fever', 
                'Symptom 2': 'lameness', 
                'Symptom 3': 'loss of appetite' 
            }])
            logger.info(f"\n--- Test 2: Predicting with potentially unseen 'fever' ---")
            predicted_disease_2, confidence_2 = loaded_model.predict(sample_raw_data_2)
            logger.info(f"Sample Prediction 2 -> Disease: {predicted_disease_2}, Confidence: {confidence_2:.2f}")

        except Exception as e:
            logger.error(f"An error occurred during the main training script execution: {e}", exc_info=True)
