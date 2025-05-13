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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DiseasePredictionModel:
    def __init__(self):
        self.model = None
        self.label_encoders = {}  # Stores LabelEncoder for each categorical column, including target
        self.scaler = None        # StandardScaler for numerical features
        self.trained_feature_columns = [] # List of feature column names model was trained on
        self.categorical_modes = {} # Stores mode for each categorical feature for handling unseen labels
        self.target_column = 'Disease' # Name of the target variable column

    def preprocess_data(self, df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        """
        Preprocesses the input DataFrame by applying cleaning, feature engineering,
        encoding, and scaling.
        """
        df_processed = df.copy()
        logger.debug(f"Preprocessing data. Is training: {is_training}. Input df shape: {df_processed.shape}")

        try:
            # --- 1. Lowercase and Basic Cleaning ---
            feature_cols_to_lower = ['Animal', 'Symptom 1', 'Symptom 2', 'Symptom 3']
            for col in feature_cols_to_lower:
                if col in df_processed.columns:
                    df_processed[col] = df_processed[col].astype(str).str.lower().str.strip()
                else: # Handle missing input columns for prediction robustness
                    logger.warning(f"Feature column '{col}' not found in input for preprocessing. Filling with 'unknown'.")
                    df_processed[col] = 'unknown'
            
            # Ensure numerical columns are numeric, fill NaNs with median (or 0 if all NaN)
            numerical_cols_raw = ['Age', 'Temperature']
            for col in numerical_cols_raw:
                if col in df_processed.columns:
                    df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                    if is_training: # Calculate median from training data only
                        median_val = df_processed[col].median()
                        # Store median for use in prediction if needed, though we usually expect non-null inputs
                        # For simplicity here, we just fill. A more robust approach might store these medians.
                        df_processed[col] = df_processed[col].fillna(median_val if not pd.isna(median_val) else 0)
                    else: # For prediction, fill with 0 or a pre-calculated median if available
                        df_processed[col] = df_processed[col].fillna(0) # Simple fill for prediction
                else:
                    logger.warning(f"Numerical column '{col}' not found. Filling with 0.")
                    df_processed[col] = 0


            # --- 2. Feature Engineering ---
            symptom_cols_for_combination = ['Symptom 1', 'Symptom 2', 'Symptom 3']
            # Ensure these columns exist (they should due to the loop above)
            df_processed['Symptom_Combination'] = df_processed.apply(
                lambda x: '_'.join(sorted([str(x[s_col]) for s_col in symptom_cols_for_combination])),
                axis=1
            )

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
            
            logger.debug("Feature engineering complete.")

            # --- 3. Encoding Categorical Features ---
            categorical_features_to_encode = [
                'Animal', 'Symptom 1', 'Symptom 2', 'Symptom 3',
                'Symptom_Combination', 'Temperature_Category',
                'Age_Category', 'Age_Temperature', 'Animal_Age'
            ]

            for col in categorical_features_to_encode:
                df_processed[col] = df_processed[col].astype(str) # Ensure string type
                if is_training:
                    self.categorical_modes[col] = df_processed[col].mode()[0] if not df_processed[col].mode().empty else 'unknown'
                    le = LabelEncoder()
                    # Fit on unique values including a placeholder for unseen labels
                    unique_values = list(df_processed[col].unique())
                    if 'unknown_label_placeholder' not in unique_values:
                        unique_values.append('unknown_label_placeholder')
                    le.fit(unique_values)
                    self.label_encoders[col] = le
                    df_processed[col] = le.transform(df_processed[col])
                else: # Prediction
                    if col in self.label_encoders:
                        le = self.label_encoders[col]
                        transformed_col_values = []
                        for label in df_processed[col]:
                            if label in le.classes_:
                                transformed_col_values.append(le.transform([label])[0])
                            else:
                                logger.warning(f"Unseen label '{label}' in column '{col}' during prediction. Mapping to 'unknown_label_placeholder'.")
                                if 'unknown_label_placeholder' in le.classes_:
                                    transformed_col_values.append(le.transform(['unknown_label_placeholder'])[0])
                                else: # Fallback: use stored mode
                                    mode_val = self.categorical_modes.get(col, 'unknown')
                                    logger.warning(f"Using mode '{mode_val}' for unseen label '{label}' in '{col}'.")
                                    if mode_val in le.classes_:
                                        transformed_col_values.append(le.transform([mode_val])[0])
                                    else: # Ultimate fallback: 0
                                        logger.error(f"Mode '{mode_val}' for '{col}' also unseen. Defaulting to 0 for '{label}'.")
                                        transformed_col_values.append(0) 
                        df_processed[col] = transformed_col_values
                    else:
                        logger.error(f"LabelEncoder for column '{col}' not found during prediction. Filling with 0. This is critical.")
                        df_processed[col] = 0 # Placeholder for missing encoder
            logger.debug("Categorical encoding complete.")

            # --- 4. Scaling Numerical Features ---
            numerical_features_to_scale = ['Age', 'Temperature']
            if is_training:
                self.scaler = StandardScaler()
                df_processed[numerical_features_to_scale] = self.scaler.fit_transform(df_processed[numerical_features_to_scale])
            else:
                if self.scaler and hasattr(self.scaler, 'mean_'):
                    df_processed[numerical_features_to_scale] = self.scaler.transform(df_processed[numerical_features_to_scale])
                else:
                    logger.warning("Scaler not fitted or found. Numerical features will not be scaled for this prediction.")
                    # Ensure columns exist even if not scaled
                    for num_col in numerical_features_to_scale:
                        if num_col not in df_processed.columns:
                            df_processed[num_col] = 0
            logger.debug("Numerical scaling complete.")

            # --- 5. Define and Select Final Feature Columns ---
            # These are the columns that go into the model
            final_model_features = categorical_features_to_encode + numerical_features_to_scale
            
            if is_training:
                self.trained_feature_columns = final_model_features
                logger.info(f"Model will be trained on columns: {self.trained_feature_columns}")

            # Ensure all trained_feature_columns are present for prediction, and in the correct order
            output_df = pd.DataFrame()
            columns_to_use = self.trained_feature_columns if not is_training and self.trained_feature_columns else final_model_features
            
            for col_name in columns_to_use:
                if col_name in df_processed:
                    output_df[col_name] = df_processed[col_name]
                else:
                    logger.error(f"CRITICAL: Column '{col_name}' missing from processed DataFrame for model input. Filling with 0.")
                    output_df[col_name] = 0 # This indicates a serious problem in preprocessing logic
            
            logger.debug(f"Preprocessing complete. Output df shape: {output_df.shape}, Columns: {output_df.columns.tolist()}")
            return output_df

        except Exception as e:
            logger.error(f"Error during preprocessing: {e}", exc_info=True)
            raise

    def train_model(self, data_path: Path):
        """Trains the Random Forest model, including preprocessing and hyperparameter tuning."""
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
        self.label_encoders[self.target_column] = target_le # Store for inverse_transform later
        
        logger.info(f"Class distribution in target variable:\n{y_raw.value_counts(normalize=True)}")

        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        logger.info(f"Data split into training and testing sets. Train shape: {X_train.shape}, Test shape: {X_test.shape}")

        param_grid = {
            'n_estimators': [100, 200], # Keep it small for faster initial runs
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
            cv=3, # Use 3-5 folds. 3 for speed here.
            n_jobs=-1, # Use all available cores
            verbose=1,
            scoring='f1_weighted' # Good for potentially imbalanced classes
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
        
        if hasattr(self.model, 'feature_importances_') and self.trained_feature_columns:
            feature_importances_df = pd.DataFrame({
                'feature': self.trained_feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values(by='importance', ascending=False)
            logger.info("\nTop 10 Feature Importances:")
            logger.info(f"\n{feature_importances_df.head(10)}")

        logger.info("Model training complete.")
        return accuracy_score(y_test, y_pred_test)

    def save_model_artifacts(self, model_dir_path: Path):
        """Saves the trained model and all associated preprocessing artifacts."""
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
        """Loads the trained model and all associated preprocessing artifacts."""
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
        """
        Makes a disease prediction on preprocessed input data.
        Assumes input_data_df is a single row DataFrame.
        """
        if not self.model:
            logger.error("Model is not loaded. Cannot make predictions.")
            raise ValueError("Model not loaded.")
        if not self.trained_feature_columns:
            logger.error("Trained feature columns list is empty. Model may not be loaded correctly.")
            raise ValueError("Trained feature columns not available.")

        logger.debug(f"Predicting on input data: \n{input_data_df}")
        
        # Preprocess the input data (is_training=False)
        X_processed = self.preprocess_data(input_data_df, is_training=False)
        
        # Ensure columns are in the same order as during training
        try:
            X_processed = X_processed[self.trained_feature_columns]
        except KeyError as e:
            missing_cols = set(self.trained_feature_columns) - set(X_processed.columns)
            extra_cols = set(X_processed.columns) - set(self.trained_feature_columns)
            logger.error(f"Column mismatch error during prediction. Missing: {missing_cols}, Extra: {extra_cols}. Error: {e}", exc_info=True)
            raise ValueError(f"Feature mismatch during prediction. Missing: {missing_cols}, Extra: {extra_cols}")

        logger.debug(f"Data processed for prediction. Shape: {X_processed.shape}, Columns: {X_processed.columns.tolist()}")

        prediction_encoded = self.model.predict(X_processed)[0]
        probabilities = self.model.predict_proba(X_processed)[0]
        
        predicted_disease_label = self.label_encoders[self.target_column].inverse_transform([prediction_encoded])[0]
        confidence = float(max(probabilities)) # Max probability for the predicted class
        
        logger.info(f"Prediction: {predicted_disease_label}, Confidence: {confidence:.4f}")
        return predicted_disease_label, confidence

# Main execution block for training
if __name__ == "__main__":
    logger.info("Executing train_model.py script...")
    
    # Define paths
    # Assumes this script (train_model.py) is in backend/ml_model/
    # Dataset is expected in project_root/ (i.e., backend/../../)
    PROJECT_ROOT = Path(__file__).parent.parent.parent 
    DATASET_PATH = PROJECT_ROOT / "animal_disease_dataset - animal_disease_dataset.csv.csv"
    SAVED_MODEL_DIR = Path(__file__).parent / "saved_model"

    if not DATASET_PATH.exists():
        logger.error(f"FATAL: Dataset not found at the expected path: {DATASET_PATH}")
        logger.error("Please ensure 'animal_disease_dataset - animal_disease_dataset.csv.csv' is in the project root directory.")
    else:
        model_trainer = DiseasePredictionModel()
        try:
            logger.info(f"Attempting to train model using dataset: {DATASET_PATH}")
            model_trainer.train_model(DATASET_PATH)
            logger.info("Model training finished. Saving artifacts...")
            model_trainer.save_model_artifacts(SAVED_MODEL_DIR)
            logger.info(f"Model artifacts saved to: {SAVED_MODEL_DIR}")

            # Example of loading and making a test prediction (optional)
            logger.info("\n--- Test: Loading model and making a sample prediction ---")
            loaded_model = DiseasePredictionModel()
            loaded_model.load_model_artifacts(SAVED_MODEL_DIR)
            
            # Create a sample DataFrame for prediction (must match expected raw input structure)
            # This sample should be representative of what the API might receive
            sample_raw_data = pd.DataFrame([{
                'Animal': 'cow',
                'Age': 3,
                'Temperature': 102.5,
                'Symptom 1': 'fever',
                'Symptom 2': 'loss of appetite',
                'Symptom 3': 'lameness'
                # Add other raw features if your model expects them before preprocessing
            }])
            predicted_disease, confidence = loaded_model.predict(sample_raw_data)
            logger.info(f"Sample Prediction on loaded model -> Disease: {predicted_disease}, Confidence: {confidence:.2f}")

        except Exception as e:
            logger.error(f"An error occurred during the main training script execution: {e}", exc_info=True)
