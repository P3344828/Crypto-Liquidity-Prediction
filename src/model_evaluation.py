"""
Model Evaluation Module for Cryptocurrency Liquidity Prediction

This provides comprehensive evaluation and visualization of trained models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def load_model_and_data():
    """
    Load the saved model, scaler, and test data.
    
    Returns:
        model, scaler, feature_columns, df
    """
    model = joblib.load("models/best_model.joblib")
    scaler = joblib.load("models/scaler.joblib")
    feature_columns = joblib.load("models/feature_columns.joblib")
    df = pd.read_csv("data/processed/crypto_featured.csv")
    
    return model, scaler, feature_columns, df


def calculate_metrics(y_true, y_pred) -> dict:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R² Score': r2_score(y_true, y_pred),
        'MAPE': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100,
        'Max Error': np.max(np.abs(y_true - y_pred)),
        'Min Error': np.min(np.abs(y_true - y_pred)),
        'Std Error': np.std(y_true - y_pred)
    }
    
    return metrics


def plot_predictions_vs_actual(y_true, y_pred, save_path: str = None):
    """
    Create scatter plot of predictions vs actual values.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    
    plt.scatter(y_true, y_pred, alpha=0.5, edgecolors='none', s=50)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Liquidity Ratio', fontsize=12)
    plt.ylabel('Predicted Liquidity Ratio', fontsize=12)
    plt.title('Predictions vs Actual Values', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.tight_layout()
    plt.show()


def plot_residuals(y_true, y_pred, save_path: str = None):
    """
    Create residual plots for error analysis.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        save_path: Path to save the plot
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.5, edgecolors='none')
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Predicted Values', fontsize=12)
    axes[0].set_ylabel('Residuals', fontsize=12)
    axes[0].set_title('Residuals vs Predicted Values', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # Residual Distribution
    axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Residuals', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Distribution of Residuals', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def plot_feature_importance(model, feature_columns, save_path: str = None):
    """
    Plot feature importance from the model.
    
    Args:
        model: Trained model
        feature_columns: List of feature names
        save_path: Path to save the plot
    """
    if not hasattr(model, 'feature_importances_'):
        print("Feature importance not available for this model.")
        return
    
    importance = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=True)
    
    plt.figure(figsize=(10, 8))
    
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(importance)))
    plt.barh(importance['Feature'], importance['Importance'], color=colors)
    
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title('Feature Importance for Liquidity Prediction', fontsize=14)
    plt.grid(True, alpha=0.3, axis='x')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.tight_layout()
    plt.show()
    
    return importance


def plot_model_comparison(results_path: str = "models/model_results.csv", save_path: str = None):
    """
    Create comparison chart of different models.
    
    Args:
        results_path: Path to model results CSV
        save_path: Path to save the plot
    """
    results = pd.read_csv(results_path)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # RMSE comparison
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(results)))
    axes[0].barh(results['Model'], results['RMSE'], color=colors)
    axes[0].set_xlabel('RMSE (Lower is Better)', fontsize=11)
    axes[0].set_title('RMSE Comparison', fontsize=12)
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # MAE comparison
    axes[1].barh(results['Model'], results['MAE'], color=colors)
    axes[1].set_xlabel('MAE (Lower is Better)', fontsize=11)
    axes[1].set_title('MAE Comparison', fontsize=12)
    axes[1].grid(True, alpha=0.3, axis='x')
    
    # R² comparison
    axes[2].barh(results['Model'], results['R² Score'], color=colors)
    axes[2].set_xlabel('R² Score (Higher is Better)', fontsize=11)
    axes[2].set_title('R² Score Comparison', fontsize=12)
    axes[2].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    
    return results


def generate_evaluation_report(metrics: dict, output_path: str = "reports/evaluation_report.txt"):
    """
    Generate a text report of model evaluation.
    
    Args:
        metrics: Dictionary of evaluation metrics
        output_path: Path to save the report
    """
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report = []
    report.append("=" * 60)
    report.append("MODEL EVALUATION REPORT")
    report.append("Cryptocurrency Liquidity Prediction")
    report.append("=" * 60)
    report.append("")
    report.append("PERFORMANCE METRICS:")
    report.append("-" * 40)
    
    for metric, value in metrics.items():
        if isinstance(value, float):
            report.append(f"{metric:20s}: {value:.6f}")
        else:
            report.append(f"{metric:20s}: {value}")
    
    report.append("")
    report.append("-" * 40)
    report.append("")
    report.append("INTERPRETATION:")
    report.append(f"- RMSE of {metrics['RMSE']:.6f} indicates average prediction error magnitude")
    report.append(f"- R² of {metrics['R² Score']:.4f} means the model explains {metrics['R² Score']*100:.1f}% of variance")
    report.append(f"- MAPE of {metrics['MAPE']:.2f}% shows average percentage error")
    report.append("")
    report.append("=" * 60)
    
    report_text = "\n".join(report)
    
    with open(output_path, 'w') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\nReport saved to: {output_path}")
    
    return report_text


def main():
    """Main function to run model evaluation."""
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    # Create reports directory
    Path("reports").mkdir(exist_ok=True)
    
    # Load model and data
    print("\nLoading model and data...")
    model, scaler, feature_columns, df = load_model_and_data()
    
    # Prepare test data
    X = df[feature_columns].fillna(0)
    y = df['liquidity_ratio'].fillna(df['liquidity_ratio'].median())
    
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(y.values, y_pred)
    
    # Generate plots
    print("\nGenerating visualizations...")
    plot_predictions_vs_actual(y.values, y_pred, "reports/predictions_vs_actual.png")
    plot_residuals(y.values, y_pred, "reports/residuals.png")
    plot_feature_importance(model, feature_columns, "reports/feature_importance.png")
    plot_model_comparison(save_path="reports/model_comparison.png")
    
    # Generate report
    print("\nGenerating evaluation report...")
    generate_evaluation_report(metrics)
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
