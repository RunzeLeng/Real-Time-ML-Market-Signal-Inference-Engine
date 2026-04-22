import pandas as pd
import numpy as np
from xgboost import XGBClassifier, callback
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


def train_xgboost_classifier(
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
           
           
           
if __name__ == "__main__":
    pass