# from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from typing import Dict, List, Tuple, Optional, Union
import numpy as np

class MetricsCalculator:
    """Utility class for calculating various metrics."""
    
    @staticmethod
    def calculate_metrics(y_true: List[int], y_pred: List[int], y_prob: Optional[List[float]] = None) -> Dict[str, float]:
        """Calculate comprehensive metrics for binary classification."""
        metrics = {
            'accuracy': float(sum(a == b for a, b in zip(y_true, y_pred))) / len(y_true),
            'balanced_accuracy': MetricsCalculator._balanced_accuracy_score(y_true, y_pred),
            'F1': MetricsCalculator._f1_score(y_true, y_pred),
            'macroF1': MetricsCalculator._macro_f1_score(y_true, y_pred),
            'recall': MetricsCalculator._recall(y_true, y_pred),
            'precision': MetricsCalculator._precision(y_true, y_pred),
        }
        
        if y_prob is not None:
            try:
                metrics['auc_roc'] = float(roc_auc_score(y_true, y_prob))
            except ValueError:
                metrics['auc_roc'] = 0.0

            # Add pF1 when y_prob is available
            metrics['pF1'] = MetricsCalculator._pf1_score(y_true, y_prob) # Call the new pF1 method
            metrics['pr_auc'] = MetricsCalculator._pr_auc_score(y_true, y_prob) # Add PR AUC here
                
        return metrics
    
    def _balanced_accuracy_score(y_true: List[int], y_pred: List[int]) -> float:
        """Calculate balanced accuracy."""
        TP = sum(1 for a, b in zip(y_true, y_pred) if a == b == 1)
        TN = sum(1 for a, b in zip(y_true, y_pred) if a == b == 0)
        FP = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 1)
        FN = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 0)
        bal_acc = (TP / (TP + FN) + TN / (TN + FP)) / 2 if (TP + FN) > 0 and (TN + FP) > 0 else 0.0

        return bal_acc
    
    # probabilistic F1 (no threshold)
    def _pf1_score(y_true: List[int], y_prob: List[float]) -> float:
        """
        Calculate probabilistic F1 (pF1) score.
        pF1 = 2 * pPrecision * pRecall / (pPrecision + pRecall)
        """
        p_precision = MetricsCalculator._p_precision(y_true, y_prob)
        p_recall = MetricsCalculator._p_recall(y_true, y_prob)

        if (p_precision + p_recall) == 0:
            return 0.0
        return 2 * (p_precision * p_recall) / (p_precision + p_recall)

    def _p_precision(y_true: List[int], y_prob: List[float]) -> float:
        """
        Calculate probabilistic precision (pPrecision).
        pPrecision = Sum(P(positive|true) * P(positive)) / Sum(P(positive))
        This simplifies to Sum(y_prob[i] for i where y_true[i] == 1) / Sum(y_prob)
        """
        numerator = sum(y_prob[i] for i in range(len(y_true)) if y_true[i] == 1)
        denominator = sum(y_prob)
        return numerator / denominator if denominator > 0 else 0.0

    def _p_recall(y_true: List[int], y_prob: List[float]) -> float:
        """
        Calculate probabilistic recall (pRecall).
        pRecall = Sum(P(positive|true) * P(positive)) / Sum(P(true))
        This simplifies to Sum(y_prob[i] for i where y_true[i] == 1) / Sum(1 for i where y_true[i] == 1)
        """
        numerator = sum(y_prob[i] for i in range(len(y_true)) if y_true[i] == 1)
        denominator = sum(1 for val in y_true if val == 1) # Count of actual positive instances
        return numerator / denominator if denominator > 0 else 0.0

    # binary F1
    def _f1_score(y_true: List[int], y_pred: List[int]) -> float:
        """Calculate F1 score."""
        precision = MetricsCalculator._precision(y_true, y_pred)
        recall = MetricsCalculator._recall(y_true, y_pred)
        
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    def _macro_f1_score(y_true: List[int], y_pred: List[int]) -> float:
        """Calculate macro-averaged F1 score for binary classification."""
        f1_scores = []
        for cls in [0, 1]:
            # Create binary vectors for this class
            true_cls = [1 if y == cls else 0 for y in y_true]
            pred_cls = [1 if p == cls else 0 for p in y_pred]
            
            precision = MetricsCalculator._precision(true_cls, pred_cls)
            recall = MetricsCalculator._recall(true_cls, pred_cls)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            f1_scores.append(f1)
        
        return sum(f1_scores) / len(f1_scores)

    def _recall(y_true: List[int], y_pred: List[int]) -> float:
        """Calculate recall."""
        TP = sum(1 for a, b in zip(y_true, y_pred) if a == b == 1)
        FN = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 0)
        
        return TP / (TP + FN) if (TP + FN) > 0 else 0.0
    
    def _precision(y_true: List[int], y_pred: List[int]) -> float:
        """Calculate precision."""
        TP = sum(1 for a, b in zip(y_true, y_pred) if a == b == 1)
        FP = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 1)
        
        return TP / (TP + FP) if (TP + FP) > 0 else 0.0
    
    def _pr_auc_score(y_true: List[int], y_prob: List[float]) -> float:
        """
        Calculate the Area Under the Precision-Recall Curve (PR AUC).
        """
        try:
            # Use sklearn's precision_recall_curve and auc to compute the score
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            return float(auc(recall, precision))
        except ValueError:
            # This can happen if there's only one class in y_true
            return 0.0