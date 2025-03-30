import os
import json
import logging
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from prometheus_client import start_http_server
from prometheus_client import Counter, Gauge, Histogram
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
EVALUATION_RUNS = Counter(
    'transaction_evaluation_runs_total',
    'Total number of evaluation runs',
    ['model_version']
)

EVALUATION_ERRORS = Counter(
    'transaction_evaluation_errors_total',
    'Total number of evaluation errors',
    ['model_version', 'error_type']
)

EVALUATION_METRICS = Gauge(
    'transaction_evaluation_metrics',
    'Model evaluation metrics',
    ['model_version', 'metric_name']
)

EVALUATION_LATENCY = Histogram(
    'transaction_evaluation_latency_seconds',
    'Evaluation latency in seconds',
    ['model_version']
)

class ModelEvaluator:
    """Evaluator for transaction classification model."""
    def __init__(
        self,
        model_url: str,
        label_mapping_path: str,
        model_version: str = '1.0.0',
        metrics_port: int = 9090
    ):
        self.model_url = model_url
        self.model_version = model_version
        
        # Load label mappings
        with open(label_mapping_path, 'r') as f:
            self.label_mappings = json.load(f)
        
        # Start Prometheus metrics server
        start_http_server(metrics_port)
        
        # Create results directory
        self.results_dir = 'evaluation_results'
        os.makedirs(self.results_dir, exist_ok=True)
    
    def evaluate_batch(
        self,
        transactions: List[Dict],
        ground_truth: Optional[Dict[str, List[int]]] = None
    ) -> Dict:
        """Evaluate model on a batch of transactions."""
        try:
            # Prepare requests
            requests_data = [
                {
                    'transaction_id': tx['transaction_id'],
                    'amount': tx['amount'],
                    'description': tx['description'],
                    'merchant_id': tx.get('merchant_id'),
                    'user_id': tx['user_id'],
                    'timestamp': tx['timestamp'],
                    'historical_transactions': tx.get('historical_transactions')
                }
                for tx in transactions
            ]
            
            # Make predictions
            start_time = datetime.now()
            response = requests.post(
                f"{self.model_url}/predict",
                json=requests_data[0]  # For now, evaluate one at a time
            )
            end_time = datetime.now()
            
            if response.status_code != 200:
                raise Exception(f"Prediction failed: {response.text}")
            
            # Update metrics
            EVALUATION_RUNS.labels(model_version=self.model_version).inc()
            EVALUATION_LATENCY.labels(model_version=self.model_version).observe(
                (end_time - start_time).total_seconds()
            )
            
            # Process predictions
            prediction = response.json()
            
            # Calculate metrics if ground truth is provided
            if ground_truth:
                metrics = self._calculate_metrics(
                    prediction,
                    ground_truth,
                    len(transactions)
                )
                
                # Update Prometheus metrics
                for metric_name, value in metrics.items():
                    EVALUATION_METRICS.labels(
                        model_version=self.model_version,
                        metric_name=metric_name
                    ).set(value)
                
                return {
                    'prediction': prediction,
                    'metrics': metrics
                }
            
            return {'prediction': prediction}
        
        except Exception as e:
            logger.error(f"Evaluation error: {str(e)}")
            EVALUATION_ERRORS.labels(
                model_version=self.model_version,
                error_type=type(e).__name__
            ).inc()
            raise
    
    def _calculate_metrics(
        self,
        prediction: Dict,
        ground_truth: Dict[str, List[int]],
        batch_size: int
    ) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        metrics = {}
        
        # Global category metrics
        global_pred = prediction['global_category_id']
        global_true = ground_truth['global_category_id'][0]
        global_report = classification_report(
            [global_true],
            [global_pred],
            output_dict=True,
            labels=list(self.label_mappings['global'].keys())
        )
        
        metrics.update({
            'global_accuracy': global_report['accuracy'],
            'global_precision': global_report['weighted avg']['precision'],
            'global_recall': global_report['weighted avg']['recall'],
            'global_f1': global_report['weighted avg']['f1-score']
        })
        
        # User category metrics
        user_pred = prediction['user_category_id']
        user_true = ground_truth['user_category_id'][0]
        user_report = classification_report(
            [user_true],
            [user_pred],
            output_dict=True,
            labels=list(self.label_mappings['user'].keys())
        )
        
        metrics.update({
            'user_accuracy': user_report['accuracy'],
            'user_precision': user_report['weighted avg']['precision'],
            'user_recall': user_report['weighted avg']['recall'],
            'user_f1': user_report['weighted avg']['f1-score']
        })
        
        return metrics
    
    def plot_confusion_matrices(
        self,
        predictions: List[Dict],
        ground_truth: Dict[str, List[int]]
    ):
        """Plot confusion matrices for global and user categories."""
        # Prepare data
        global_preds = [p['global_category_id'] for p in predictions]
        global_trues = ground_truth['global_category_id']
        user_preds = [p['user_category_id'] for p in predictions]
        user_trues = ground_truth['user_category_id']
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Global Categories', 'User Categories')
        )
        
        # Global categories confusion matrix
        global_cm = confusion_matrix(global_trues, global_preds)
        fig.add_trace(
            go.Heatmap(
                z=global_cm,
                x=list(self.label_mappings['global'].values()),
                y=list(self.label_mappings['global'].values()),
                colorscale='Viridis',
                name='Global'
            ),
            row=1, col=1
        )
        
        # User categories confusion matrix
        user_cm = confusion_matrix(user_trues, user_preds)
        fig.add_trace(
            go.Heatmap(
                z=user_cm,
                x=list(self.label_mappings['user'].values()),
                y=list(self.label_mappings['user'].values()),
                colorscale='Viridis',
                name='User'
            ),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            title='Confusion Matrices',
            height=600,
            width=1200
        )
        
        # Save plot
        fig.write_html(os.path.join(self.results_dir, 'confusion_matrices.html'))
    
    def plot_confidence_distribution(
        self,
        predictions: List[Dict]
    ):
        """Plot distribution of model confidence scores."""
        # Prepare data
        global_confidences = [p['global_category_confidence'] for p in predictions]
        user_confidences = [p['user_category_confidence'] for p in predictions]
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        # Plot distributions
        sns.kdeplot(global_confidences, label='Global Categories')
        sns.kdeplot(user_confidences, label='User Categories')
        
        plt.title('Model Confidence Distribution')
        plt.xlabel('Confidence Score')
        plt.ylabel('Density')
        plt.legend()
        
        # Save plot
        plt.savefig(os.path.join(self.results_dir, 'confidence_distribution.png'))
        plt.close()
    
    def generate_report(
        self,
        predictions: List[Dict],
        ground_truth: Optional[Dict[str, List[int]]] = None,
        save: bool = True
    ) -> Dict:
        """Generate evaluation report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_version': self.model_version,
            'num_samples': len(predictions)
        }
        
        if ground_truth:
            # Calculate metrics
            metrics = self._calculate_metrics(
                predictions[0],
                ground_truth,
                len(predictions)
            )
            report['metrics'] = metrics
            
            # Generate plots
            self.plot_confusion_matrices(predictions, ground_truth)
            self.plot_confidence_distribution(predictions)
        
        # Save report if requested
        if save:
            report_path = os.path.join(
                self.results_dir,
                f'evaluation_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            )
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
        
        return report

def main(
    model_url: str,
    label_mapping_path: str,
    model_version: str = '1.0.0',
    metrics_port: int = 9090,
    test_data_path: Optional[str] = None
):
    """Main function for model evaluation."""
    evaluator = ModelEvaluator(
        model_url=model_url,
        label_mapping_path=label_mapping_path,
        model_version=model_version,
        metrics_port=metrics_port
    )
    
    # Load test data if provided
    if test_data_path:
        test_data = pd.read_csv(test_data_path)
        ground_truth = {
            'global_category_id': test_data['global_category_id'].tolist(),
            'user_category_id': test_data['user_category_id'].tolist()
        }
        
        # Evaluate on test data
        predictions = []
        for _, row in test_data.iterrows():
            result = evaluator.evaluate_batch(
                [row.to_dict()],
                ground_truth
            )
            predictions.append(result['prediction'])
        
        # Generate report
        report = evaluator.generate_report(
            predictions,
            ground_truth,
            save=True
        )
        logger.info(f"Evaluation report saved: {report}")
    else:
        # Example evaluation
        example_transaction = {
            'transaction_id': 'test_001',
            'amount': 100.0,
            'description': 'Grocery shopping',
            'merchant_id': 'merchant_001',
            'user_id': 'user_001',
            'timestamp': datetime.now().isoformat()
        }
        
        result = evaluator.evaluate_batch([example_transaction])
        logger.info(f"Example prediction: {result}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate transaction classification model')
    parser.add_argument('--model_url', type=str, required=True,
                      help='URL of the model API')
    parser.add_argument('--label_mapping_path', type=str, required=True,
                      help='Path to label mapping JSON file')
    parser.add_argument('--model_version', type=str, default='1.0.0',
                      help='Version of the model')
    parser.add_argument('--metrics_port', type=int, default=9090,
                      help='Port for Prometheus metrics')
    parser.add_argument('--test_data_path', type=str, default=None,
                      help='Path to test data CSV file')
    
    args = parser.parse_args()
    
    main(
        model_url=args.model_url,
        label_mapping_path=args.label_mapping_path,
        model_version=args.model_version,
        metrics_port=args.metrics_port,
        test_data_path=args.test_data_path
    ) 