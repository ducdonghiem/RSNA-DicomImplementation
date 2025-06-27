# from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.metrics import roc_auc_score
from typing import Dict, List, Tuple, Optional, Union

class MetricsCalculator:
    """Utility class for calculating various metrics."""
    
    @staticmethod
    def calculate_metrics(y_true: List[int], y_pred: List[int], y_prob: Optional[List[float]] = None) -> Dict[str, float]:
        """Calculate comprehensive metrics for binary classification."""
        metrics = {
            'accuracy': sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true),
            'balanced_accuracy': MetricsCalculator._balanced_accuracy_score(y_true, y_pred),
            'pF1': MetricsCalculator._f1_score(y_true, y_pred),
            'macroF1': MetricsCalculator._macro_f1_score(y_true, y_pred),
            'recall': MetricsCalculator._recall(y_true, y_pred),
            'precision': MetricsCalculator._precision(y_true, y_pred),
        }
        
        if y_prob is not None:
            try:
                metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
            except ValueError:
                metrics['auc_roc'] = 0.0
                
        return metrics
    
    def _balanced_accuracy_score(y_true: List[int], y_pred: List[int]) -> float:
        """Calculate balanced accuracy."""
        TP = sum(1 for a, b in zip(y_true, y_pred) if a == b == 1)
        TN = sum(1 for a, b in zip(y_true, y_pred) if a == b == 0)
        FP = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 1)
        FN = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 0)
        bal_acc = (TP / (TP + FN) + TN / (TN + FP)) / 2 if (TP + FN) > 0 and (TN + FP) > 0 else 0.0

        return bal_acc
    
    # F1 is pF1 (F1 for positive class)
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