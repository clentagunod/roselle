import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import clear_output
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

sns.set_theme(
    style="whitegrid",
    context="paper",
    font_scale=1.2
)

def y_fdistro(y, name=None): 
    sns.histplot(y, kde=True)
    plt.xlabel("pIC50")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {name}")
    plt.show()

def actual_vs_prediction(y_true, y_pred, name=None):
    sns.scatterplot(x=y_true, y=y_pred)

    plt.plot(
        [y_true.min(), y_true.max()],
        [y_true.min(), y_true.max()],
        linestyle="--"
    )

    plt.xlabel(f"Actual {name}")
    plt.ylabel(f"Predicted {name}")
    plt.title(f"Actual vs Predicted {name}")
    plt.show()

def feature_importance(model, feature_names, top_n=20):

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    })

    importance_df = (
        importance_df
        .sort_values("Importance", ascending=False)
        .head(top_n)
        .sort_values("Importance", ascending=True)
    )

    plt.figure(figsize=(10, 8))

    sns.barplot(
        data=importance_df,
        x="Importance",
        y="Feature"
    )

    plt.title(f"Top {top_n} Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")

    plt.tight_layout()
    plt.show()

    return importance_df




def classification_plots(
    model,
    X_test,
    y_test,
    feature_names=None,
    top_n=20
):
    
  

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    
    
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    plt.tight_layout()
    plt.show()

    
    
    

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    plt.figure(figsize=(6, 5))

    plt.plot(
        fpr,
        tpr,
        label=f"Random Forest (AUC = {auc:.3f})"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random classifier"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")

    plt.legend()
    plt.tight_layout()
    plt.show()

    
    
    

    precision, recall, _ = precision_recall_curve(
        y_test,
        y_prob
    )

    ap = average_precision_score(
        y_test,
        y_prob
    )

    plt.figure(figsize=(6, 5))

    plt.plot(
        recall,
        precision,
        label=f"Random Forest (AP = {ap:.3f})"
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")

    plt.legend()
    plt.tight_layout()
    plt.show()

    
    
    

    if hasattr(model, "feature_importances_"):

        importances = model.feature_importances_

        if feature_names is None:
            feature_names = [
                f"Feature {i}"
                for i in range(len(importances))
            ]

        importance = (
            np.asarray(importances)
        )

        indices = np.argsort(importance)[-top_n:]

        plt.figure(figsize=(8, 7))

        plt.barh(
            np.array(feature_names)[indices],
            importance[indices]
        )

        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.title(
            f"Top {top_n} Feature Importances"
        )

        plt.tight_layout()
        plt.show()

    
    
    

    plt.figure(figsize=(7, 5))

    sns.histplot(
        y_prob[np.array(y_test) == 0],
        bins=20,
        kde=True,
        stat="density",
        label="Class 0",
        alpha=0.5
    )

    sns.histplot(
        y_prob[np.array(y_test) == 1],
        bins=20,
        kde=True,
        stat="density",
        label="Class 1",
        alpha=0.5
    )

    plt.xlabel("Predicted Probability of Class 1")
    plt.ylabel("Density")
    plt.title("Predicted Probability Distribution")

    plt.legend()
    plt.tight_layout()
    plt.show()


def callback_graph(weights, obj_func_eval, objective_func_vals):
    clear_output(wait=True)
    objective_func_vals.append(obj_func_eval)
    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)
    plt.show()