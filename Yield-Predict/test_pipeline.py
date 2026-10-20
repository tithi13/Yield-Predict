"""
test_pipeline.py
================
Unit & Integration Tests for Crop Yield Prediction Machine Learning Pipeline.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np

def test_artifacts_exist():
    """Verify all model artifacts and metadata files are present."""
    assert os.path.exists("models/linear_regression_pipeline.joblib"), "LR pipeline missing!"
    assert os.path.exists("models/random_forest_pipeline.joblib"), "RF pipeline missing!"
    assert os.path.exists("models/model_metadata.json"), "Model metadata JSON missing!"
    assert os.path.exists("models/test_predictions_sample.csv"), "Test predictions sample CSV missing!"
    print("[PASS] All artifact files exist.")

def test_metadata_contents():
    """Verify required metadata fields and metric keys."""
    with open("models/model_metadata.json", "r") as f:
        meta = json.load(f)
    
    assert "evaluation_metrics" in meta
    assert "Linear_Regression" in meta["evaluation_metrics"]
    assert "Random_Forest" in meta["evaluation_metrics"]
    for model_key in ["Linear_Regression", "Random_Forest"]:
        for metric_name in ["MAE", "MSE", "RMSE", "R2"]:
            assert metric_name in meta["evaluation_metrics"][model_key]
            assert isinstance(meta["evaluation_metrics"][model_key][metric_name], (int, float))
    
    assert meta["yield_units"] == "Tonnes per Hectare (t/ha)"
    assert len(meta["crops"]) > 50
    print(f"[PASS] Metadata contains valid metrics for both models (Random Forest R²: {meta['evaluation_metrics']['Random_Forest']['R2']}).")

def test_model_inference():
    """Verify that both pipelines successfully predict on user inputs without shape or encoding mismatch."""
    lr_pipe = joblib.load("models/linear_regression_pipeline.joblib")
    rf_pipe = joblib.load("models/random_forest_pipeline.joblib")

    test_input = pd.DataFrame([{
        "rainfall": 1200.0,
        "temperature": 27.0,
        "humidity": 70.0,
        "pH": 6.5,
        "N": 80.0,
        "P": 40.0,
        "K": 30.0,
        "crop_type": "Rice"
    }])

    pred_lr = lr_pipe.predict(test_input)
    pred_rf = rf_pipe.predict(test_input)

    assert len(pred_lr) == 1 and not np.isnan(pred_lr[0])
    assert len(pred_rf) == 1 and not np.isnan(pred_rf[0])
    assert pred_rf[0] > 0, "Predicted yield should be positive for realistic inputs"
    print(f"[PASS] Inference test passed! Rice Yield -> LR: {pred_lr[0]:.2f} t/ha, RF: {pred_rf[0]:.2f} t/ha")

def test_unseen_category_handling():
    """Verify that OneHotEncoder handles unknown crop types gracefully without throwing exceptions."""
    rf_pipe = joblib.load("models/random_forest_pipeline.joblib")
    novel_input = pd.DataFrame([{
        "rainfall": 900.0,
        "temperature": 25.0,
        "humidity": 65.0,
        "pH": 6.8,
        "N": 70.0,
        "P": 35.0,
        "K": 25.0,
        "crop_type": "UnknownNovelCrop"
    }])
    pred_novel = rf_pipe.predict(novel_input)
    assert len(pred_novel) == 1 and not np.isnan(pred_novel[0])
    print(f"[PASS] Novel categorical handling passed! Output: {pred_novel[0]:.2f} t/ha")

if __name__ == "__main__":
    print("Running automated pipeline verification tests...")
    test_artifacts_exist()
    test_metadata_contents()
    test_model_inference()
    test_unseen_category_handling()
    print("\nAll automated tests PASSED successfully!")
