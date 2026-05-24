import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple,Optional
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
import joblib
from .config import *

logger=logging.getLogger()

class DataPipeline:
    """
       End-to-end data pipeline for churn prediction.
       Flow:
       1. Load raw data
       2. Clean (handle missing values, type conversions)
       3. Feature engineering (derive new features)
       4. Encode categorical variables
       5. Scale numeric features
       6. Split train/test
       7. Save processed data
    """
    def __init__(self):
        self.scaler=None
        self.categorical_encoder=None
        self.feature_columns=None
        logger.info("DatePipeline initialized")

    def load_raw(self,filepath:Path)->pd.DataFrame:
        filepath=Path(filepath)
        if not filepath.exists():
            logger.error(f"File not found {filepath}")
            raise FileNotFoundError(f"Data file not found{filepath}")
        try:
            df=pd.read_csv(filepath)
            logger.info(f"Loaded{len(df)} records,{len(df.columns)} columns and from {filepath}")
            if df.empty:
                raise ValueError('Loaded data is empty')
            return df
        except Exception as e:
            logger.error(f"Error loading data:{str(e)}")
            raise

    def clean(self,df:pd.DataFrame) -> pd.DataFrame:
        df=df.copy()
        logger.info("Starting data cleaning..")

        #Drop features to drop(CustomerID)
        df = df.drop(columns=FEATURES_TO_DROP, errors='ignore')

        #Type conversions
        if "TotalCharges" in df.columns:
            df["TotalCharges"]=pd.to_numeric(df["TotalCharges"],errors="coerce")
            missing_total_charges=df["TotalCharges"].isna().sum()

            if missing_total_charges>0:
                logger.warning(f"Found {missing_total_charges} non-numeric Total charges values")
                df["TotalCharges"]=df["TotalCharges"].fillna(df["TotalCharges"].median())

            #Handling Missing Values
        missing_counts=df.isna().sum()
        if missing_counts.sum()>0:
            logger.warning(f"Missing values found:\n{missing_counts[missing_counts>0]}")
            #Drop rows with missing target
            if TARGET_COLUMN  in df.columns:
                df=df.dropna(subset=[TARGET_COLUMN])
            for col in NUMERIC_FEATURES:
                if col in df.columns and df[col].isnull().sum()>0:
                    df[col]=df[col].fillna(df[col].median())

        #Removing Customers with tenure=0 because they have no real signal
        if "tenure" in df.columns:
            initial_len=len(df)
            df=df[df["tenure"]>0]
            removed=initial_len-len(df)
            if removed>0:
                logger.info(f"Removed {removed} records with tenure=0")

        #Standardize Target Variable
        if TARGET_COLUMN in df.columns:
            df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(str).str.strip()
            df[TARGET_COLUMN] = (df[TARGET_COLUMN] == CHURN_VALUE).astype(int)
            churn_rate = df[TARGET_COLUMN].mean()
            logger.info(f"Churn rate: {churn_rate:.2%}")

        logger.info(f"Data Cleaning complete.Final shape :{df.shape}")
        return df

    #Feature Engineering

    def create_features(self,df:pd.DataFrame)->pd.DataFrame:
        df=df.copy()
        logger.info("Starting feature engineering...")

        #Interaction Features
        """These features will give an idea about historical prices by monthly and 
        the gap of charges increases historically """
        if "TotalCharges" in df.columns and "tenure" in df.columns:
            df["avg_charges_per_month"]=df["TotalCharges"]/(df["tenure"]+1)#+1 to avoid division by zero
            df["charge_increase"] = df["MonthlyCharges"] - df["avg_charges_per_month"]

        if all(col in df.columns for col in["MonthlyCharges","tenure","TotalCharges"]):
            df["Charges_consistency"]=(
                    (df["TotalCharges"]-df["MonthlyCharges"]*df["tenure"]).abs()<1).astype(int)

            #Binning Features
        #Tenure groups(customer lifecycle)
        if "tenure" in df.columns:
            df["tenure_group"]=pd.cut(
                df["tenure"],
                bins=[0,12,24,48,float("inf")],
                labels=["0-1yr","1-2yr","2-4yr","4+yr"],
                include_lowest=True
            )
        logger.info(f"Feature engineering complete.Shape:{df.shape}")
        return df

                    #Encoding
    def encode_categorical(self,df:pd.DataFrame,fit:bool=True)->pd.DataFrame:
        """One-hot encode categorical feature"""
        df=df.copy()
        logger.info(f"Encoding categorical feature (fit={fit})...")

        if fit:
            cols_to_encode=[col for col in CATEGORICAL_FEATURES if col in df.columns]
            df_encoded=pd.get_dummies(df,columns=cols_to_encode,drop_first=True,dtype=int)
            self.categorical_encoder=cols_to_encode
            logger.info(f"Encoded {len(cols_to_encode)} categorical columns")
        else:
            if self.categorical_encoder is None:
                raise ValueError("Encoder not fitted.Call with fit=True first.")
            df_encoded=pd.get_dummies(df,columns=self.categorical_encoder,drop_first=True,dtype=int)
            logger.info(f"After encoding :{df_encoded.shape}")
        return df_encoded

    def scale_numeric(self,df:pd.DataFrame,fit:bool=True)->pd.DataFrame:
        #Scaling the numeric cols
        df=df.copy()
        logger.info(f"Scaling numeric features (fit={fit})...")
        cols_to_scale=[col for col in NUMERIC_FEATURES if col in df.columns]

        if not cols_to_scale:
            logger.warning("No numeric features to scale")
            return df
        if fit:
            self.scaler=StandardScaler()
            df[cols_to_scale]=self.scaler.fit_transform(df[cols_to_scale])
            logger.info(f"Fitted scaler on {len(cols_to_scale)}")

        else:
            if self.scaler is None:
                raise ValueError("Scaler to fitted.call with fit=True first")
            df[cols_to_scale]=self.scaler.transform(df[cols_to_scale])
        return df

    def split_data(self,df:pd.DataFrame)->Tuple[pd.DataFrame,pd.DataFrame,pd.Series,pd.Series]:
        #Split data into train and test sets with stratification

        logger.info("Splitting data into train/test...")
        x=df.drop(columns=[TARGET_COLUMN])
        y=df[TARGET_COLUMN]

        x_train,x_test,y_train,y_test=train_test_split(
            x,y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )
        logger.info(f"Train set : {x_train.shape}")
        logger.info(f"Test size : {x_test.shape}")
        logger.info(f"Train churn rate :{y_train.mean():.2%}")
        logger.info(f"Test churn rate :{y_test.mean():.2%}")

        return x_train,x_test,y_train,y_test

    def process(self,filepath:Path,save:bool=True)->Tuple[pd.DataFrame,pd.DataFrame,pd.Series,pd.Series]:

        """Run full pipeline:load->clean->engineer->encode->split->scale"""
        logger.info("Starting Full Data Pipeline")

        df=self.load_raw(filepath)  #First Load the data
        df=self.clean(df) #Clean the loaded raw data
        df=self.create_features(df) #Feature engineering
        df=self.encode_categorical(df,fit=True) # encoding the categorical features
        #Spliting before scaling it prevents data leakage
        x_train,x_test,y_train,y_test=self.split_data(df)
        x_train=self.scale_numeric(x_train,fit=True)#fit scaler on training data
        x_test=self.scale_numeric(x_test,fit=False)#Apply to test data(if you fit :data leakage)

        # Persist canonical feature column order for inference
        self.feature_columns = x_train.columns.tolist()

        if save: #save processed Data
            train_df=x_train.copy()
            train_df[TARGET_COLUMN]=y_train

            test_df=x_test.copy()
            test_df[TARGET_COLUMN]=y_test

            train_path=PROCESSED_DATA_DIR/"train.csv"
            test_path=PROCESSED_DATA_DIR/"test.csv"

            train_df.to_csv(train_path,index=False)
            test_df.to_csv(test_path,index=False)

            self.save_pipeline()

        return x_train,x_test,y_train,y_test

    def save_pipeline(self,filepath:Optional[Path]=None)->None:
        if filepath is None:
            filepath = PROCESSED_DATA_DIR / "pipeline_artifacts.pkl"
        artifacts = {
            "scaler": self.scaler,
            "categorical_encoder": self.categorical_encoder
        }
        # include feature column ordering when available
        if getattr(self, "feature_columns", None) is not None:
            artifacts["feature_columns"] = self.feature_columns
        joblib.dump(artifacts, filepath)
        logger.info(f"Saved pipeline artifacts to {filepath}")
    def load_pipeline(self,filepath:Optional[Path]=None)->None:
        """Load scaler and encoder from disk"""
        if filepath is None:
            filepath=PROCESSED_DATA_DIR/"pipeline_artifacts.pkl"

        if not filepath.exists():
            raise FileNotFoundError(f"File not found of path{filepath}")

        artifacts=joblib.load(filepath)
        self.scaler=artifacts["scaler"]
        self.categorical_encoder=artifacts["categorical_encoder"]
        # Load feature columns if available; otherwise try to recover from processed train.csv
        if "feature_columns" in artifacts:
            self.feature_columns = artifacts["feature_columns"]
        else:
            # fallback: attempt to read processed train.csv
            train_path = PROCESSED_DATA_DIR / "train.csv"
            if train_path.exists():
                try:
                    df_train = pd.read_csv(train_path)
                    # remove target column if present
                    if TARGET_COLUMN in df_train.columns:
                        df_train = df_train.drop(columns=[TARGET_COLUMN])
                    self.feature_columns = df_train.columns.tolist()
                    logger.info("Recovered feature_columns from processed train.csv")
                except Exception:
                    logger.warning("Could not recover feature_columns from train.csv")

    def map_api_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map API (snake_case) field names to raw dataset column names used in training."""
        mapping = {
            "tenure": "tenure",
            "monthly_charges": "MonthlyCharges",
            "total_charges": "TotalCharges",
            "contract": "Contract",
            "internet_service": "InternetService",
            "online_security": "OnlineSecurity",
            "online_backup": "OnlineBackup",
            "device_protection": "DeviceProtection",
            "tech_support": "TechSupport",
            "senior_citizen": "SeniorCitizen",
            "partner": "Partner",
            "dependents": "Dependents",
            "gender": "gender",
            "phone_service": "PhoneService",
            "streaming_tv": "StreamingTV",
            "streaming_movies": "StreamingMovies",
            "paperless_billing": "PaperlessBilling",
            "payment_method": "PaymentMethod",
        }
        # rename only keys present
        rename_map = {k: v for k, v in mapping.items() if k in df.columns}
        if rename_map:
            df = df.rename(columns=rename_map)
        return df

    def prepare_for_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare incoming API dataframe for prediction to match training features.

        Steps: map api names -> raw names, feature engineer, encode (fit=False), scale (fit=False),
        and align columns to saved `feature_columns`.
        """
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame([df])

        df = df.copy()
        df = self.map_api_features(df)

        # Fill missing categorical features with most common value from training if possible
        if self.categorical_encoder is not None:
            missing_cats = [c for c in self.categorical_encoder if c not in df.columns]
            if missing_cats:
                train_path = PROCESSED_DATA_DIR / "train.csv"
                train_modes = {}
                if train_path.exists():
                    try:
                        df_train = pd.read_csv(train_path)
                        for c in missing_cats:
                            if c in df_train.columns:
                                mode_val = df_train[c].mode()
                                train_modes[c] = mode_val[0] if not mode_val.empty else None
                    except Exception:
                        pass

                for c in missing_cats:
                    if c in train_modes and train_modes[c] is not None:
                        df[c] = train_modes[c]
                    else:
                        # sensible default: SeniorCitizen -> 0, else 'No'
                        df[c] = 0 if c.lower().startswith('senior') or c == 'SeniorCitizen' else 'No'

        # create features consistent with training
        df = self.create_features(df)

        # encode and scale using fitted artifacts
        df = self.encode_categorical(df, fit=False)
        df = self.scale_numeric(df, fit=False)

        if not getattr(self, "feature_columns", None):
            raise ValueError("feature_columns not available. Run process() to generate or ensure pipeline artifacts include feature_columns")

        # Align columns: add missing with 0, drop extras, and order
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0

        # drop columns that are not in feature_columns
        extra_cols = [c for c in df.columns if c not in self.feature_columns]
        if extra_cols:
            df = df.drop(columns=extra_cols)

        df = df[self.feature_columns]
        return df









