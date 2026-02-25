"""
Ensemble Forecaster
====================
Combines RandomForest and XGBoost into a single ensemble model.

WHY TWO MODELS?
    XGBoost: Builds trees sequentially, each fixing mistakes of the previous one.
             Great at capturing complex patterns. Can overfit on small data.
    
    RandomForest: Builds trees independently in parallel, each on random data subsets.
                  More stable, resistant to overfitting. Slightly less accurate.
    
    Ensemble: Averages both predictions. Errors tend to cancel out.
              XGBoost predicts $54, RF predicts $47, actual is $50
              → Ensemble predicts $50.50 (much closer)

HOW IT'S USED:
    We create 3 instances of this class:
        - EnsembleForecaster("energy_output")   → predicts MW generation
        - EnsembleForecaster("price")           → predicts $/MWh
        - EnsembleForecaster("demand")          → predicts MW grid demand
    
    Each trains on the same weather features but different targets.

WHAT THIS FILE CONTAINS:
    1. Training (fit RF + XGBoost + VotingRegressor)
    2. Prediction with confidence intervals (bootstrap sampling)
    3. Evaluation metrics (MAPE, RMSE, MAE + naive baseline comparison)
    4. Feature importance (permutation-based)
    5. Save/Load models to disk
"""

import os
import logging
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import (
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_absolute_error,
)
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor

logger = logging.getLogger("ensemble_forecaster")


class EnsembleForecaster:

    def __init__(self, target_name, n_estimators=100):
        """
        Initialize the ensemble with both models.

        Args:
            target_name: What this model predicts ("energy_output", "price", or "demand")
            n_estimators: Number of trees in each model (100 is a good default)

        RandomForest params explained:
            max_depth=12:       Each tree can be at most 12 levels deep.
                                Deeper = more complex patterns but risk of overfitting.
            min_samples_split=5: A node must have at least 5 samples to split further.
                                Prevents the tree from memorizing individual data points.
            n_jobs=-1:          Use all CPU cores for parallel training.
            random_state=42:    Fixed seed so results are reproducible.

        XGBoost params explained:
            max_depth=8:        Shallower than RF because XGBoost builds sequentially
                                (each tree corrects errors), so it needs less depth.
            learning_rate=0.1:  How much each new tree contributes. Lower = more trees
                                needed but more stable. 0.1 is a standard default.
            subsample=0.8:      Each tree only sees 80% of the training data.
                                Prevents overfitting (similar to RF's random subsets).
            colsample_bytree=0.8: Each tree only sees 80% of features.
                                Forces trees to not rely on just one dominant feature.
        """
        self.target_name = target_name
        self.n_estimators = n_estimators

        # Model 1: RandomForest
        self.rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=12,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
        )

        # Model 2: XGBoost
        self.xgb = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )

        # Ensemble: averages predictions from both models
        # VotingRegressor with default weights means simple average:
        # prediction = (rf_prediction + xgb_prediction) / 2
        self.ensemble = VotingRegressor(
            estimators=[("rf", self.rf), ("xgb", self.xgb)]
        )

        self.is_trained = False
        self.metrics = {}
        self.feature_names = []
        self.feature_importances = {}

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train all three models (RF, XGBoost, and the ensemble).

        Args:
            X_train: Training features (DataFrame or numpy array)
                     e.g., columns: [temp, humidity, wind_speed, cloud_coverage,
                                     irradiance, direct_radiation, dni]
            y_train: Training targets (Series or array)
                     e.g., actual prices: [52.3, 48.1, 65.7, ...]
            X_val:   Optional validation features (for computing metrics)
            y_val:   Optional validation targets

        WHY TRAIN INDIVIDUAL MODELS SEPARATELY?
            VotingRegressor trains both internally, but we also train
            RF and XGBoost individually so we can:
            1. Compare their predictions (which model is better for this target?)
            2. Get feature importances from each
            3. Use them independently for confidence intervals
        """
        # Save feature names for later (used in feature importance)
        if hasattr(X_train, "columns"):
            self.feature_names = list(X_train.columns)
        else:
            self.feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

        logger.info(
            f"Training {self.target_name} ensemble on {len(X_train)} samples "
            f"with {len(self.feature_names)} features: {self.feature_names}"
        )

        # Train the ensemble (this trains both RF and XGBoost internally)
        self.ensemble.fit(X_train, y_train)

        # Also train individually for comparison and confidence intervals
        self.rf.fit(X_train, y_train)
        self.xgb.fit(X_train, y_train)

        self.is_trained = True

        # If validation data is provided, compute metrics
        if X_val is not None and y_val is not None:
            self.metrics = self.evaluate(X_val, y_val)
            self.feature_importances = self.get_feature_importance(X_val, y_val)
            logger.info(f"{self.target_name} metrics: {self.metrics}")

        return self.metrics

    def predict(self, X):
        """
        Make predictions using the ensemble (average of RF and XGBoost).

        Args:
            X: Features to predict on (same columns as training data)

        Returns:
            numpy array of predictions
        """
        if not self.is_trained:
            raise ValueError(
                f"Model '{self.target_name}' not trained. Call train() first."
            )
        return self.ensemble.predict(X)

    def predict_with_confidence(self, X, n_bootstrap=30):
        """
        Predict with confidence intervals using bootstrap sampling.

        WHAT IS BOOTSTRAP SAMPLING?
            1. Get predictions from both RF and XGBoost
            2. Randomly resample the input data 30 times (with replacement)
            3. Predict on each resampled set
            4. Calculate standard deviation of all predictions
            5. Use std to compute 95% confidence interval

        WHY THIS WORKS:
            If all 30 bootstrap predictions are similar (low std),
            the model is confident. If they vary a lot (high std),
            the model is uncertain about this prediction.

        Args:
            X: Features to predict on
            n_bootstrap: Number of bootstrap rounds (30 is standard)

        Returns dict with:
            ensemble:          The main prediction (average of RF + XGBoost)
            rf:                RandomForest's prediction alone
            xgb:               XGBoost's prediction alone
            confidence_lower:  Lower bound of 95% confidence interval
            confidence_upper:  Upper bound of 95% confidence interval
            std:               Standard deviation (measure of uncertainty)

        EXAMPLE:
            {
                "ensemble": [50.5],         ← "We predict $50.50/MWh"
                "rf": [47.0],               ← "RF thinks $47"
                "xgb": [54.0],              ← "XGBoost thinks $54"
                "confidence_lower": [42.3],  ← "Could be as low as $42.30"
                "confidence_upper": [58.7],  ← "Could be as high as $58.70"
                "std": [4.2]                ← "Uncertainty is ±$4.20"
            }
        """
        if not self.is_trained:
            raise ValueError(f"Model '{self.target_name}' not trained.")

        # Get predictions from both models
        rf_pred = self.rf.predict(X)
        xgb_pred = self.xgb.predict(X)
        ensemble_pred = self.ensemble.predict(X)

        # Bootstrap: predict many times on resampled data
        bootstrap_preds = []
        n_samples = len(X)
        for i in range(n_bootstrap):
            # Randomly pick n_samples indices WITH replacement
            # "With replacement" means the same row can be picked multiple times
            # This creates a slightly different dataset each time
            idx = np.random.choice(n_samples, n_samples, replace=True)

            if hasattr(X, "iloc"):
                # X is a pandas DataFrame
                X_boot = X.iloc[idx]
            else:
                # X is a numpy array
                X_boot = X[idx]

            boot_pred = self.ensemble.predict(X_boot)
            bootstrap_preds.append(boot_pred)

        # Calculate standard deviation across all bootstrap predictions
        boot_array = np.array(bootstrap_preds)
        std = np.std(boot_array, axis=0)

        # 95% confidence interval: prediction ± 1.96 * standard_deviation
        # 1.96 comes from the normal distribution:
        #   68% of values fall within ±1 std
        #   95% of values fall within ±1.96 std
        #   99% of values fall within ±2.58 std
        return {
            "ensemble": ensemble_pred,
            "rf": rf_pred,
            "xgb": xgb_pred,
            "confidence_lower": ensemble_pred - 1.96 * std,
            "confidence_upper": ensemble_pred + 1.96 * std,
            "std": std,
        }

    def evaluate(self, X_val, y_val):
        """
        Compute comprehensive evaluation metrics.

        METRICS EXPLAINED:
            MAPE (Mean Absolute Percentage Error):
                Average of |actual - predicted| / actual * 100
                "On average, predictions are off by X%"
                12% MAPE means if actual price is $50, prediction is typically $44-$56

            RMSE (Root Mean Squared Error):
                Square root of average squared errors
                Penalizes large errors more than MAPE
                Useful for detecting occasional big misses

            MAE (Mean Absolute Error):
                Average of |actual - predicted|
                "On average, predictions are off by $X"
                MAE of $6 means typical prediction misses by $6

        NAIVE BASELINE:
            "What if we just predicted yesterday's value for today?"
            This is called persistence forecasting.
            If our ML model can't beat this, it's useless.
            
            Example:
                Naive MAPE:    18%  (just repeating last value)
                Ensemble MAPE: 12%  (our model)
                Improvement:   6 percentage points (33% better than naive)
        """
        # Get predictions from all models
        ensemble_preds = self.ensemble.predict(X_val)
        rf_preds = self.rf.predict(X_val)
        xgb_preds = self.xgb.predict(X_val)

        # Convert to numpy array if needed
        y_actual = np.array(y_val)

        # Avoid division by zero in MAPE (skip rows where actual = 0)
        mask = y_actual != 0
        y_masked = y_actual[mask]
        ensemble_masked = ensemble_preds[mask]
        rf_masked = rf_preds[mask]
        xgb_masked = xgb_preds[mask]

        metrics = {
            # Ensemble metrics
            "ensemble_mape": round(
                mean_absolute_percentage_error(y_masked, ensemble_masked) * 100, 2
            ),
            "ensemble_rmse": round(
                np.sqrt(mean_squared_error(y_actual, ensemble_preds)), 2
            ),
            "ensemble_mae": round(
                mean_absolute_error(y_actual, ensemble_preds), 2
            ),
            # Individual model metrics (for comparison)
            "rf_mape": round(
                mean_absolute_percentage_error(y_masked, rf_masked) * 100, 2
            ),
            "xgb_mape": round(
                mean_absolute_percentage_error(y_masked, xgb_masked) * 100, 2
            ),
        }

        # Naive baseline: predict previous value (persistence model)
        # If we have time-series data, the simplest prediction is:
        # "tomorrow's price = today's price"
        if len(y_actual) > 1:
            # Shift values by 1: [10, 20, 30, 40] → [40, 10, 20, 30]
            # Compare each value with its predecessor
            naive_preds = np.roll(y_actual, 1)[1:]  # Remove first (it wraps around)
            naive_actual = y_actual[1:]
            naive_mask = naive_actual != 0

            if naive_mask.sum() > 0:
                metrics["naive_mape"] = round(
                    mean_absolute_percentage_error(
                        naive_actual[naive_mask], naive_preds[naive_mask]
                    ) * 100, 2
                )
                # How much better is our model vs naive?
                metrics["improvement_over_naive_pct"] = round(
                    metrics["naive_mape"] - metrics["ensemble_mape"], 2
                )

        return metrics

    def get_feature_importance(self, X_val, y_val):
        """
        Calculate feature importance using permutation method.

        HOW IT WORKS (recap):
            1. Calculate baseline error with original data
            2. For each feature:
                a. Shuffle that feature's values randomly
                b. Recalculate error
                c. If error increased a lot → feature is important
                d. If error barely changed → feature is not important
            3. Repeat 10 times for stability (n_repeats=10)

        Args:
            X_val: Validation features
            y_val: Validation targets

        Returns dict like:
            {
                "temp": {"rf": 0.35, "xgb": 0.33},
                "wind_speed": {"rf": 0.22, "xgb": 0.25},
                ...
            }
        """
        importance_dict = {}

        try:
            # Permutation importance for RandomForest
            rf_result = permutation_importance(
                self.rf, X_val, y_val,
                n_repeats=10,       # Shuffle 10 times per feature for stable results
                random_state=42,
                n_jobs=-1,
            )

            # Permutation importance for XGBoost
            xgb_result = permutation_importance(
                self.xgb, X_val, y_val,
                n_repeats=10,
                random_state=42,
                n_jobs=-1,
            )

            for i, feature in enumerate(self.feature_names):
                importance_dict[feature] = {
                    "rf": round(rf_result.importances_mean[i], 4),
                    "xgb": round(xgb_result.importances_mean[i], 4),
                }

            logger.info(f"{self.target_name} feature importances: {importance_dict}")

        except Exception as e:
            logger.warning(f"Could not compute feature importance: {e}")

        return importance_dict

    def save(self, model_dir):
        """
        Save all models and metadata to disk.

        We save 4 files per target:
            price_rf.joblib       → RandomForest model
            price_xgb.joblib      → XGBoost model
            price_ensemble.joblib → VotingRegressor
            price_metadata.joblib → metrics, feature names, importances
        
        WHY SAVE SEPARATELY?
            - Can load just RF or just XGBoost for debugging
            - Metadata file lets dashboard show metrics without loading models
            - If one file corrupts, others are still usable
        """
        os.makedirs(model_dir, exist_ok=True)

        joblib.dump(self.rf, os.path.join(model_dir, f"{self.target_name}_rf.joblib"))
        joblib.dump(self.xgb, os.path.join(model_dir, f"{self.target_name}_xgb.joblib"))
        joblib.dump(
            self.ensemble,
            os.path.join(model_dir, f"{self.target_name}_ensemble.joblib"),
        )

        # Save metadata (metrics + feature info)
        metadata = {
            "metrics": self.metrics,
            "feature_names": self.feature_names,
            "feature_importances": self.feature_importances,
            "n_estimators": self.n_estimators,
            "target_name": self.target_name,
        }
        joblib.dump(
            metadata,
            os.path.join(model_dir, f"{self.target_name}_metadata.joblib"),
        )

        logger.info(f"Saved {self.target_name} models to {model_dir}")

    def load(self, model_dir):
        """
        Load pre-trained ensemble models from disk.

        Expects these files in model_dir:
            {target_name}_rf.joblib       → RandomForest
            {target_name}_xgb.joblib      → XGBoost
            {target_name}_ensemble.joblib → VotingRegressor
            {target_name}_metadata.joblib → metrics, feature names, importances

        These are created by running: python src/train_ensemble.py
        """
        self.rf = joblib.load(
            os.path.join(model_dir, f"{self.target_name}_rf.joblib")
        )
        self.xgb = joblib.load(
            os.path.join(model_dir, f"{self.target_name}_xgb.joblib")
        )
        self.ensemble = joblib.load(
            os.path.join(model_dir, f"{self.target_name}_ensemble.joblib")
        )

        # Load metadata (metrics, feature names, importances)
        meta_path = os.path.join(model_dir, f"{self.target_name}_metadata.joblib")
        if os.path.exists(meta_path):
            metadata = joblib.load(meta_path)
            self.metrics = metadata.get("metrics", {})
            self.feature_names = metadata.get("feature_names", [])
            self.feature_importances = metadata.get("feature_importances", {})

        self.is_trained = True
        logger.info(f"Loaded {self.target_name} ensemble from {model_dir}")