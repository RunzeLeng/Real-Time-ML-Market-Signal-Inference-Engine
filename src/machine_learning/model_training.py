import itertools
import random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, callback
from src.machine_learning.build_training_data import TrainingDataBuilder



class ModelTrainingService:
    
    
    def get_scale_pos_weight_ratio(
        self,
        df: pd.DataFrame,
        prediction_range: str,
    ) -> float | None:
        
        matching_cols = [col for col in df.columns if prediction_range.lower() in col.lower()]
        if not matching_cols:
            raise ValueError(f"No column found containing prediction range: {prediction_range}")

        target_col = matching_cols[0]
        values = pd.to_numeric(df[target_col], errors="coerce").dropna()

        negative_count = (values < 0).sum()
        positive_count = (values > 0).sum()

        if positive_count == 0:
            return None

        return negative_count / positive_count



    def train_xgboost_classifier(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        colsample_bylevel: float = 1.0,
        colsample_bynode: float = 1.0,
        min_child_weight: float = 1.0,
        gamma: float = 0.5,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        objective: str = "multi:softprob",
        eval_metric: str = "mlogloss",
        callback_early_stopping_rounds: int = 50,
        num_classes: int = 2,
        scale_pos_weight: float = 1.0,
    ) -> tuple:
        
        feature_columns = [col for col in df.columns if col.startswith("x_")]
        target_column = [col for col in df.columns if col.startswith("y_")]

        model_df = df[feature_columns + target_column].dropna().copy()

        if num_classes == 4:
            label_order = ["strong_sell", "sell", "buy", "strong_buy"]
        elif num_classes == 3:
            label_order = ["sell", "hold", "buy"]
        elif num_classes == 2:
            label_order = ["sell", "buy"]
        else:
            raise ValueError("num_classes must be 2, 3, or 4")

        label_mapping = {label: i for i, label in enumerate(label_order)}
        inverse_label_mapping = {i: label for label, i in label_mapping.items()}


        X = model_df[feature_columns]
        y = model_df[target_column[0]].map(label_mapping)


        X_train, X_valid, y_train, y_valid = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            colsample_bylevel=colsample_bylevel,
            colsample_bynode=colsample_bynode,
            min_child_weight=min_child_weight,
            gamma=gamma,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            objective=objective,
            eval_metric=eval_metric,
            random_state=random_state,
            callbacks=[callback.EarlyStopping(rounds=callback_early_stopping_rounds, save_best=True)],
            scale_pos_weight=scale_pos_weight,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            verbose=False,
        )
        
        return model, X_train, X_valid, y_train, y_valid



    def evaluate_xgboost_classifier(
        self,
        model,
        X_train,
        X_valid,
        y_train,
        y_valid,
        num_classes: int,
        if_print_results: bool = True,
    ):
        if num_classes == 4:
            labels = ["strong_sell", "sell", "buy", "strong_buy"]
        elif num_classes == 3:
            labels = ["sell", "hold", "buy"]
        elif num_classes == 2:
            labels = ["sell", "buy"]
        else:
            raise ValueError("num_classes must be 2, 3, or 4")


        inverse_label_mapping = {i: label for i, label in enumerate(labels)}

        y_train_pred = model.predict(X_train)
        y_valid_pred = model.predict(X_valid)

        y_train_pred = pd.Series(y_train_pred).map(inverse_label_mapping)
        y_valid_pred = pd.Series(y_valid_pred).map(inverse_label_mapping)
        y_train = pd.Series(y_train).map(inverse_label_mapping)
        y_valid = pd.Series(y_valid).map(inverse_label_mapping)


        train_accuracy_50 = accuracy_score(y_train, y_train_pred)
        valid_accuracy_50 = accuracy_score(y_valid, y_valid_pred)

        train_confusion_matrix_50 = pd.DataFrame(
            confusion_matrix(y_train, y_train_pred, labels=labels),
            index=[f"true_{label}" for label in labels],
            columns=[f"pred_{label}" for label in labels],
        )

        valid_confusion_matrix_50 = pd.DataFrame(
            confusion_matrix(y_valid, y_valid_pred, labels=labels),
            index=[f"true_{label}" for label in labels],
            columns=[f"pred_{label}" for label in labels],
        )

        probs = model.predict_proba(X_train)
        probs = probs[probs[:, 1].argsort()]

        probs_train = model.predict_proba(X_train)[:, 0]
        probs_valid = model.predict_proba(X_valid)[:, 0]   
        
        if if_print_results:
            print("\nTraining Confusion Matrix at 0.500:")
            print(train_confusion_matrix_50)

            print("\nValidation Confusion Matrix at 0.500:")
            print(valid_confusion_matrix_50)

            print("\nTraining Accuracy at 0.500:")
            print(train_accuracy_50)

            print("\nValidation Accuracy at 0.500:")
            print(valid_accuracy_50)
            
            print("\nTraining Predicted Probabilities (first 20 rows):")
            print(probs[:20])
            
            print("\nTraining Predicted Probabilities (last 20 rows):")
            print(probs[-20:]) 
        
        
        threshold_results = []
        for threshold in np.arange(0.500, 0.600, 0.002):
            y_train_pred_customized = (probs_train <= threshold).astype(int)
            
            y_train_pred_customized = pd.Series(y_train_pred_customized).map(inverse_label_mapping)
            accuracy = accuracy_score(y_train, y_train_pred_customized)

            threshold_results.append({
                "threshold": round(threshold, 3),
                "accuracy": accuracy,
            })

        threshold_df = pd.DataFrame(threshold_results)
        best_row = threshold_df.loc[threshold_df["accuracy"].idxmax()]
        best_threshold = best_row["threshold"]
        best_accuracy = best_row["accuracy"]

        if if_print_results:
            print("\nThreshold Tuning Results:")
            print(threshold_df)
            
            print("\nBest threshold on training data:", best_threshold)
            print("\nBest accuracy on training data:", best_accuracy)
        
        
        ######
        y_train_pred_customized = (probs_train <= best_threshold).astype(int)
        y_train_pred_customized = pd.Series(y_train_pred_customized).map(inverse_label_mapping)

        train_accuracy_customized = accuracy_score(y_train, y_train_pred_customized)

        train_confusion_matrix_customized = pd.DataFrame(
            confusion_matrix(y_train, y_train_pred_customized, labels=labels),
            index=[f"true_{label}" for label in labels],
            columns=[f"pred_{label}" for label in labels],
        )
        
        ######
        y_valid_pred_customized = (probs_valid <= best_threshold).astype(int)
        y_valid_pred_customized = pd.Series(y_valid_pred_customized).map(inverse_label_mapping)

        valid_accuracy_customized = accuracy_score(y_valid, y_valid_pred_customized)

        valid_confusion_matrix_customized = pd.DataFrame(
            confusion_matrix(y_valid, y_valid_pred_customized, labels=labels),
            index=[f"true_{label}" for label in labels],
            columns=[f"pred_{label}" for label in labels],
        )
        
        if if_print_results:
            print("\nCustomized Training Accuracy:")
            print(train_accuracy_customized)
            print("\nCustomized Training Confusion Matrix:")
            print(train_confusion_matrix_customized)
        
            print("\nCustomized Validation Accuracy:")
            print(valid_accuracy_customized)
            print("\nCustomized Validation Confusion Matrix:")
            print(valid_confusion_matrix_customized)
            
            print("\nDifference in Accuracy:")
            print(train_accuracy_customized - valid_accuracy_customized)


        ######
        margin = 0.04
        lower_threshold = best_threshold - margin
        upper_threshold = best_threshold + margin

        high_confidence_mask = (probs_valid <= lower_threshold) | (probs_valid >= upper_threshold)
        
        save_model_to_s3 = True
        
        if high_confidence_mask.sum() == 0:
            lower_threshold = None
            upper_threshold = None
            valid_accuracy_high_confidence = None
            high_confidence_percentage = None
            save_model_to_s3 = False
        else:
            y_valid_pred_high_confidence = np.where(
                probs_valid[high_confidence_mask] <= lower_threshold,
                1,
                0
            )

            y_valid_pred_high_confidence = pd.Series(y_valid_pred_high_confidence).map(inverse_label_mapping)
            y_valid_pred_high_confidence = y_valid_pred_high_confidence.reset_index(drop=True)
            y_valid_high_confidence = pd.Series(y_valid)[high_confidence_mask].reset_index(drop=True)

            valid_accuracy_high_confidence = accuracy_score(y_valid_high_confidence, y_valid_pred_high_confidence)

            valid_confusion_matrix_high_confidence = pd.DataFrame(
                confusion_matrix(y_valid_high_confidence, y_valid_pred_high_confidence, labels=labels),
                index=[f"true_{label}" for label in labels],
                columns=[f"pred_{label}" for label in labels],
            )
            
            high_confidence_percentage = high_confidence_mask.sum() / len(y_valid)
            
            if if_print_results:
                print("\nHigh-Confidence Lower Threshold:")
                print(lower_threshold)

                print("\nHigh-Confidence Upper Threshold:")
                print(upper_threshold)

                print("\nHigh-Confidence Validation Accuracy:")
                print(valid_accuracy_high_confidence)

                print("\nHigh-Confidence Validation Confusion Matrix:")
                print(valid_confusion_matrix_high_confidence)

                print("\nNumber of Kept High-Confidence Rows:")
                print(high_confidence_mask.sum())
                
                print("\nHigh-Confidence Row Percentage:")
                print(high_confidence_percentage)
        
        return train_accuracy_customized, valid_accuracy_customized, train_accuracy_customized - valid_accuracy_customized,\
               lower_threshold, upper_threshold, valid_accuracy_high_confidence, high_confidence_percentage, save_model_to_s3



    def train_all_hyperparameter_combinations(
        self,
        df: pd.DataFrame,
        symbol: str,
        STANDARD_METRICS: dict,
        random_state_length: int = 100,
        prediction_range: str = "30m",
    ) -> pd.DataFrame:
        training_data_builder = TrainingDataBuilder()

        complexity_group = [
            {"max_depth": 4, "min_child_weight": 3, "gamma": 2},
            {"max_depth": 5, "min_child_weight": 1, "gamma": 1},
            {"max_depth": 7, "min_child_weight": 1, "gamma": 0},
        ]

        sampling_group = [
            {"subsample": 0.5, "colsample_bytree": 0.5, "colsample_bylevel": 0.7, "colsample_bynode": 0.8},
            {"subsample": 0.6, "colsample_bytree": 0.6, "colsample_bylevel": 0.8, "colsample_bynode": 1.0},
            {"subsample": 0.8, "colsample_bytree": 0.8, "colsample_bylevel": 1.0, "colsample_bynode": 1.0},
        ]

        shrinkage_group = [
            {"learning_rate": 0.03, "reg_lambda": 3.0, "reg_alpha": 0.2},
            {"learning_rate": 0.05, "reg_lambda": 2.0, "reg_alpha": 0.0},
            {"learning_rate": 0.08, "reg_lambda": 1.0, "reg_alpha": 0.0},
        ]
        
        base_scale_pos_weight = self.get_scale_pos_weight_ratio(df, prediction_range)

        class_imbalance_group = [
            {"scale_pos_weight": base_scale_pos_weight * 0.8},
            {"scale_pos_weight": base_scale_pos_weight},
            {"scale_pos_weight": base_scale_pos_weight * 1.2},
        ]

        hold_threshold_group = [0.05, 0.07, 0.09]

        results = []

        all_combinations = itertools.product(
            complexity_group,
            sampling_group,
            shrinkage_group,
            class_imbalance_group,
            hold_threshold_group,
        )

        for combo_id, (complexity, sampling, shrinkage, class_imbalance, holding_gap) in enumerate(all_combinations, start=1):
            print(f"Running combination {combo_id} with parameters and scale_pos_weight={class_imbalance['scale_pos_weight']}:")
            
            df_model = training_data_builder.add_categorical_target_columns(df.copy(), target_config=(symbol, -holding_gap, holding_gap, prediction_range, 2))
            df_model = training_data_builder.scale_input_metric_columns(df_model, STANDARD_METRICS)
            df_model = training_data_builder.keep_only_x_and_y_columns(df_model)

            train_accuracies = []
            valid_accuracies = []
            accuracy_diffs = []
            
            lower_thresholds = []
            upper_thresholds = []
            average_thresholds = []
            
            valid_accuracies_high_confidence = []
            high_confidence_percentages = []
            
            random_number = random.randint(1, 1000)
            for i in range(random_number, random_number+random_state_length):
                model, X_train, X_valid, y_train, y_valid = self.train_xgboost_classifier(
                    df=df_model,
                    test_size=0.2,
                    callback_early_stopping_rounds=50,
                    
                    max_depth=complexity["max_depth"],
                    min_child_weight=complexity["min_child_weight"],
                    gamma=complexity["gamma"],

                    subsample=sampling["subsample"],
                    colsample_bytree=sampling["colsample_bytree"],
                    colsample_bylevel=sampling["colsample_bylevel"],
                    colsample_bynode=sampling["colsample_bynode"],

                    learning_rate=shrinkage["learning_rate"],
                    reg_lambda=shrinkage["reg_lambda"],
                    reg_alpha=shrinkage["reg_alpha"],
                
                    scale_pos_weight=class_imbalance["scale_pos_weight"],

                    objective="binary:logistic",
                    eval_metric="logloss",
                    num_classes=2,
                    random_state=i,
                )

                (
                    train_accuracy_customized,
                    valid_accuracy_customized,
                    accuracy_diff,
                    lower_threshold,
                    upper_threshold,
                    valid_accuracy_high_confidence,
                    high_confidence_percentage,
                    save_model_to_s3,
                ) = self.evaluate_xgboost_classifier(
                    model,
                    X_train,
                    X_valid,
                    y_train,
                    y_valid,
                    num_classes=2,
                    if_print_results=False,
                )

                train_accuracies.append(train_accuracy_customized)
                valid_accuracies.append(valid_accuracy_customized)
                accuracy_diffs.append(accuracy_diff)

                if lower_threshold is not None:
                    lower_thresholds.append(lower_threshold)
                if upper_threshold is not None:
                    upper_thresholds.append(upper_threshold)
                if lower_threshold is not None and upper_threshold is not None:
                    average_thresholds.append((lower_threshold + upper_threshold) / 2.0)
                if valid_accuracy_high_confidence is not None:
                    valid_accuracies_high_confidence.append(valid_accuracy_high_confidence)
                if high_confidence_percentage is not None:
                    high_confidence_percentages.append(high_confidence_percentage)

            result_row = {
                "symbol": symbol,
                
                "combo_id": combo_id,

                "holding_gap": holding_gap,

                "max_depth": complexity["max_depth"],
                "min_child_weight": complexity["min_child_weight"],
                "gamma": complexity["gamma"],

                "subsample": sampling["subsample"],
                "colsample_bytree": sampling["colsample_bytree"],
                "colsample_bylevel": sampling["colsample_bylevel"],
                "colsample_bynode": sampling["colsample_bynode"],

                "learning_rate": shrinkage["learning_rate"],
                "reg_lambda": shrinkage["reg_lambda"],
                "reg_alpha": shrinkage["reg_alpha"],

                "scale_pos_weight": class_imbalance["scale_pos_weight"],

                "avg_train_accuracy": sum(train_accuracies) / len(train_accuracies),
                "avg_valid_accuracy": sum(valid_accuracies) / len(valid_accuracies),
                "avg_accuracy_gap": sum(accuracy_diffs) / len(accuracy_diffs),

                "avg_lower_threshold": sum(lower_thresholds) / len(lower_thresholds) if lower_thresholds else None,
                "avg_upper_threshold": sum(upper_thresholds) / len(upper_thresholds) if upper_thresholds else None,
                "avg_threshold": sum(average_thresholds) / len(average_thresholds) if average_thresholds else None,

                "avg_valid_accuracy_high_confidence": (
                    sum(valid_accuracies_high_confidence) / len(valid_accuracies_high_confidence)
                    if valid_accuracies_high_confidence else None
                ),
                "std_valid_accuracy_high_confidence": (
                    pd.Series(valid_accuracies_high_confidence).std()
                    if len(valid_accuracies_high_confidence) > 1 else None
                ),  
                "avg_high_confidence_coverage_percentage": (
                    sum(high_confidence_percentages) / len(high_confidence_percentages)
                    if high_confidence_percentages else None
                ),
                "pct_valid_accuracy_high_confidence": len(valid_accuracies_high_confidence)/random_state_length,
            }

            results.append(result_row)

        return pd.DataFrame(results)



    def train_selected_hyperparameter_combinations(
        self,
        df: pd.DataFrame,
        model_combos: pd.DataFrame,
        symbol: str,
        STANDARD_METRICS: dict,
        random_state_length: int = 100,
        prediction_range: str = "30m",
        prefix: str = "models",
    ) -> pd.DataFrame:
        
        results = []
        training_data_builder = TrainingDataBuilder()
        
        for _, combo_row in model_combos.iterrows():
            print(f"\nRunning {combo_row['symbol']} combination {combo_row['combo_id']} with parameters and scale_pos_weight={combo_row['scale_pos_weight']}:")
            
            holding_gap = combo_row["holding_gap"]

            df_model = training_data_builder.add_categorical_target_columns(df.copy(), target_config=(symbol, -holding_gap, holding_gap, prediction_range, 2))
            df_model = training_data_builder.scale_input_metric_columns(df_model, STANDARD_METRICS)
            df_model = training_data_builder.keep_only_x_and_y_columns(df_model)

            train_accuracies = []
            valid_accuracies = []
            accuracy_diffs = []
            
            lower_thresholds = []
            upper_thresholds = []
            average_thresholds = []
            
            valid_accuracies_high_confidence = []
            high_confidence_percentages = []
            
            random_number = random.randint(1, 1000)
            for i in range(random_number, random_number+random_state_length):
                model, X_train, X_valid, y_train, y_valid = self.train_xgboost_classifier(
                    df=df_model,
                    test_size=0.2,
                    callback_early_stopping_rounds=50,
                    
                    max_depth=combo_row["max_depth"],
                    min_child_weight=combo_row["min_child_weight"],
                    gamma=combo_row["gamma"],

                    subsample=combo_row["subsample"],
                    colsample_bytree=combo_row["colsample_bytree"],
                    colsample_bylevel=combo_row["colsample_bylevel"],
                    colsample_bynode=combo_row["colsample_bynode"],

                    learning_rate=combo_row["learning_rate"],
                    reg_lambda=combo_row["reg_lambda"],
                    reg_alpha=combo_row["reg_alpha"],
                
                    scale_pos_weight=combo_row["scale_pos_weight"],

                    objective="binary:logistic",
                    eval_metric="logloss",
                    num_classes=2,
                    random_state=i,
                )

                (
                    train_accuracy_customized,
                    valid_accuracy_customized,
                    accuracy_diff,
                    lower_threshold,
                    upper_threshold,
                    valid_accuracy_high_confidence,
                    high_confidence_percentage,
                    save_model_to_s3,
                ) = self.evaluate_xgboost_classifier(
                    model,
                    X_train,
                    X_valid,
                    y_train,
                    y_valid,
                    num_classes=2,
                    if_print_results=False,
                )
                
                from src.machine_learning.model_artifact_store import ModelArtifactStore
                model_artifact_store = ModelArtifactStore()
                model_artifact_store.save_xgboost_models(
                    model,
                    symbol=symbol,
                    combo_id=combo_row["combo_id"],
                    random_state=i,
                    if_save_model=save_model_to_s3,
                    prediction_range=prediction_range,
                    prefix=prefix,
                )

                train_accuracies.append(train_accuracy_customized)
                valid_accuracies.append(valid_accuracy_customized)
                accuracy_diffs.append(accuracy_diff)

                if lower_threshold is not None:
                    lower_thresholds.append(lower_threshold)
                if upper_threshold is not None:
                    upper_thresholds.append(upper_threshold)
                if lower_threshold is not None and upper_threshold is not None:
                    average_thresholds.append((lower_threshold + upper_threshold) / 2.0)
                if valid_accuracy_high_confidence is not None:
                    valid_accuracies_high_confidence.append(valid_accuracy_high_confidence)
                if high_confidence_percentage is not None:
                    high_confidence_percentages.append(high_confidence_percentage)

            result_row = {
                "symbol": symbol,
                
                "prediction_range": prediction_range,
                
                "combo_id": combo_row["combo_id"],

                "holding_gap": holding_gap,

                "max_depth": combo_row["max_depth"],
                "min_child_weight": combo_row["min_child_weight"],
                "gamma": combo_row["gamma"],

                "subsample": combo_row["subsample"],
                "colsample_bytree": combo_row["colsample_bytree"],
                "colsample_bylevel": combo_row["colsample_bylevel"],
                "colsample_bynode": combo_row["colsample_bynode"],

                "learning_rate": combo_row["learning_rate"],
                "reg_lambda": combo_row["reg_lambda"],
                "reg_alpha": combo_row["reg_alpha"],

                "scale_pos_weight": combo_row["scale_pos_weight"],

                "avg_train_accuracy": sum(train_accuracies) / len(train_accuracies),
                "avg_valid_accuracy": sum(valid_accuracies) / len(valid_accuracies),
                "avg_accuracy_gap": sum(accuracy_diffs) / len(accuracy_diffs),

                "avg_lower_threshold": sum(lower_thresholds) / len(lower_thresholds) if lower_thresholds else None,
                "avg_upper_threshold": sum(upper_thresholds) / len(upper_thresholds) if upper_thresholds else None,
                "avg_threshold": sum(average_thresholds) / len(average_thresholds) if average_thresholds else None,

                "avg_valid_accuracy_high_confidence": (
                    sum(valid_accuracies_high_confidence) / len(valid_accuracies_high_confidence)
                    if valid_accuracies_high_confidence else None
                ),
                "std_valid_accuracy_high_confidence": (
                    pd.Series(valid_accuracies_high_confidence).std()
                    if len(valid_accuracies_high_confidence) > 1 else None
                ),  
                "avg_high_confidence_coverage_percentage": (
                    sum(high_confidence_percentages) / len(high_confidence_percentages)
                    if high_confidence_percentages else None
                ),
                "pct_valid_accuracy_high_confidence": len(valid_accuracies_high_confidence)/random_state_length,
            }

            results.append(result_row)

        return pd.DataFrame(results)
