import os
import time
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from prometheus_client import start_http_server, Counter, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from fastapi import FastAPI
import uvicorn
from models.transaction_classifier import TransactionClassifier
from data.data_module import TransactionDataModule
import torch
import pytorch_lightning as pl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
PREDICTION_COUNT = Counter(
    'transaction_predictions_total',
    'Total number of transaction predictions',
    ['model_version', 'prediction_type']
)

PREDICTION_LATENCY = Histogram(
    'transaction_prediction_latency_seconds',
    'Latency of transaction predictions',
    ['model_version', 'prediction_type'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
)

PREDICTION_ERRORS = Counter(
    'transaction_prediction_errors_total',
    'Total number of prediction errors',
    ['model_version', 'error_type']
)

MODEL_CONFIDENCE = Gauge(
    'transaction_model_confidence',
    'Average model confidence score',
    ['model_version']
)

CATEGORY_ACCURACY = Gauge(
    'transaction_category_accuracy',
    'Accuracy of category predictions',
    ['model_version', 'category_type']
)

class ModelMonitor:
    """Monitor for transaction classification model."""
    def __init__(
        self,
        model_path: str,
        label_mapping_path: str,
        device: str = 'auto',
        model_version: str = '1.0.0'
    ):
        # Load label mappings
        with open(label_mapping_path, 'r') as f:
            self.label_mappings = json.load(f)
        
        # Create model
        self.model = self._load_model(model_path, device)
        
        # Set device
        self.device = device if device != 'auto' else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.model.eval()
        
        # Set model version
        self.model_version = model_version
        
        # Initialize metrics storage
        self.metrics_history = []
        
        # Create metrics directory
        os.makedirs('metrics', exist_ok=True)
    
    def _load_model(
        self,
        model_path: str,
        device: str
    ) -> TransactionClassifier:
        """Load trained model from checkpoint."""
        # Create model instance
        model = TransactionClassifier(
            num_global_classes=len(self.label_mappings['category_id']),
            num_user_classes=len(self.label_mappings['user_category_id']),
            gnn_hidden_dim=256,
            gnn_num_layers=3,
            gnn_dropout=0.1,
            seq_hidden_dim=256,
            seq_num_layers=2,
            seq_dropout=0.1,
            text_model_name='bert-base-uncased',
            text_max_length=128,
            text_field_weights={
                'description': 1.0,
                'memo': 0.5,
                'merchant': 0.3
            },
            fusion_hidden_dim=512,
            fusion_dropout=0.1,
            learning_rate=1e-4,
            weight_decay=1e-5,
            warmup_steps=1000
        )
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        
        return model
    
    def update_metrics(
        self,
        predictions: List[Dict],
        true_labels: Optional[List[Dict]] = None,
        latency: float = None
    ):
        """Update monitoring metrics."""
        # Update prediction count
        PREDICTION_COUNT.labels(
            model_version=self.model_version,
            prediction_type='single'
        ).inc(len(predictions))
        
        # Update prediction latency
        if latency is not None:
            PREDICTION_LATENCY.labels(
                model_version=self.model_version,
                prediction_type='single'
            ).observe(latency)
        
        # Calculate confidence scores
        confidences = [
            (p['global_category_probabilities'][p['predicted_global_category']] +
             p['user_category_probabilities'][p['predicted_user_category']]) / 2
            for p in predictions
        ]
        
        # Update model confidence
        MODEL_CONFIDENCE.labels(
            model_version=self.model_version
        ).set(np.mean(confidences))
        
        # Update accuracy if true labels are provided
        if true_labels is not None:
            global_correct = sum(
                1 for p, t in zip(predictions, true_labels)
                if p['predicted_global_category'] == t['category_id']
            )
            user_correct = sum(
                1 for p, t in zip(predictions, true_labels)
                if p['predicted_user_category'] == t['user_category_id']
            )
            
            global_accuracy = global_correct / len(predictions)
            user_accuracy = user_correct / len(predictions)
            
            CATEGORY_ACCURACY.labels(
                model_version=self.model_version,
                category_type='global'
            ).set(global_accuracy)
            
            CATEGORY_ACCURACY.labels(
                model_version=self.model_version,
                category_type='user'
            ).set(user_accuracy)
        
        # Store metrics in history
        self.metrics_history.append({
            'timestamp': datetime.now().isoformat(),
            'model_version': self.model_version,
            'num_predictions': len(predictions),
            'avg_confidence': np.mean(confidences),
            'latency': latency
        })
        
        # Save metrics history
        pd.DataFrame(self.metrics_history).to_csv(
            'metrics/metrics_history.csv',
            index=False
        )
    
    def log_error(self, error_type: str):
        """Log prediction error."""
        PREDICTION_ERRORS.labels(
            model_version=self.model_version,
            error_type=error_type
        ).inc()
        
        logger.error(f"Prediction error: {error_type}")
    
    def generate_report(self) -> Dict:
        """Generate monitoring report."""
        # Calculate metrics
        metrics = {
            'total_predictions': PREDICTION_COUNT.labels(
                model_version=self.model_version,
                prediction_type='single'
            )._value.get(),
            'total_errors': PREDICTION_ERRORS.labels(
                model_version=self.model_version,
                error_type='all'
            )._value.get(),
            'avg_confidence': MODEL_CONFIDENCE.labels(
                model_version=self.model_version
            )._value.get(),
            'global_accuracy': CATEGORY_ACCURACY.labels(
                model_version=self.model_version,
                category_type='global'
            )._value.get(),
            'user_accuracy': CATEGORY_ACCURACY.labels(
                model_version=self.model_version,
                category_type='user'
            )._value.get()
        }
        
        # Add latency statistics
        latency_values = PREDICTION_LATENCY.labels(
            model_version=self.model_version,
            prediction_type='single'
        )._sum.get() / PREDICTION_LATENCY.labels(
            model_version=self.model_version,
            prediction_type='single'
        )._count.get()
        
        metrics['avg_latency'] = latency_values
        
        # Save report
        with open('metrics/monitoring_report.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics

def create_monitoring_app(
    model_path: str,
    label_mapping_path: str,
    device: str = 'auto',
    model_version: str = '1.0.0'
) -> FastAPI:
    """Create FastAPI application for monitoring."""
    app = FastAPI(
        title="Transaction Classification Monitor",
        description="Monitoring API for transaction classification model",
        version="1.0.0"
    )
    
    # Initialize model monitor
    monitor = ModelMonitor(
        model_path=model_path,
        label_mapping_path=label_mapping_path,
        device=device,
        model_version=model_version
    )
    
    # Add Prometheus instrumentation
    Instrumentator().instrument(app).expose(app)
    
    @app.get("/metrics/report")
    async def get_report():
        """Get monitoring report."""
        return monitor.generate_report()
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy"}
    
    return app

def main(
    model_path: str,
    label_mapping_path: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    metrics_port: int = 9090,
    device: str = 'auto',
    model_version: str = '1.0.0'
):
    """Main function for running the monitoring server."""
    # Start Prometheus metrics server
    start_http_server(metrics_port)
    
    # Create FastAPI app
    app = create_monitoring_app(
        model_path=model_path,
        label_mapping_path=label_mapping_path,
        device=device,
        model_version=model_version
    )
    
    # Run server
    uvicorn.run(app, host=host, port=port)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor transaction classification model')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--label_mapping_path', type=str, required=True,
                      help='Path to label mapping JSON file')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                      help='Host to run server on')
    parser.add_argument('--port', type=int, default=8000,
                      help='Port to run server on')
    parser.add_argument('--metrics_port', type=int, default=9090,
                      help='Port to run metrics server on')
    parser.add_argument('--device', type=str, default='auto',
                      help='Device to run model on')
    parser.add_argument('--model_version', type=str, default='1.0.0',
                      help='Version of the model')
    
    args = parser.parse_args()
    
    main(
        model_path=args.model_path,
        label_mapping_path=args.label_mapping_path,
        host=args.host,
        port=args.port,
        metrics_port=args.metrics_port,
        device=args.device,
        model_version=args.model_version
    ) 