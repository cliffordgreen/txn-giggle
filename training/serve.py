import os
import json
import logging
from typing import Dict, List, Optional
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from prometheus_client import Counter, Histogram, Gauge
from prometheus_fastapi_instrumentator import Instrumentator

from models.transaction_classifier import TransactionClassifier
from data.transaction_dataset import TransactionDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
PREDICTIONS = Counter(
    'transaction_predictions_total',
    'Total number of predictions made',
    ['model_version', 'category_type']
)

LATENCY = Histogram(
    'transaction_prediction_latency_seconds',
    'Prediction latency in seconds',
    ['model_version']
)

CONFIDENCE = Gauge(
    'transaction_model_confidence',
    'Model confidence for predictions',
    ['model_version', 'category_type']
)

ERRORS = Counter(
    'transaction_prediction_errors_total',
    'Total number of prediction errors',
    ['model_version', 'error_type']
)

class PredictionRequest(BaseModel):
    """Request model for prediction endpoint."""
    transaction_id: str
    amount: float
    description: str
    merchant_id: Optional[str] = None
    user_id: str
    timestamp: str
    historical_transactions: Optional[List[Dict]] = None

class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""
    transaction_id: str
    global_category: str
    global_category_id: int
    global_category_confidence: float
    user_category: str
    user_category_id: int
    user_category_confidence: float
    model_version: str

class ModelServer:
    """Server for transaction classification model."""
    def __init__(
        self,
        model_path: str,
        label_mapping_path: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.model_version = os.getenv('MODEL_VERSION', '1.0.0')
        
        # Load label mappings
        with open(label_mapping_path, 'r') as f:
            self.label_mappings = json.load(f)
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Create FastAPI app
        self.app = FastAPI(
            title='Transaction Classifier API',
            description='API for classifying financial transactions',
            version=self.model_version
        )
        
        # Add Prometheus instrumentation
        Instrumentator().instrument(self.app).expose(self.app)
        
        # Add routes
        self._add_routes()
    
    def _load_model(self, model_path: str) -> TransactionClassifier:
        """Load model from checkpoint."""
        logger.info(f"Loading model from {model_path}")
        
        # Initialize model
        model = TransactionClassifier.load_from_checkpoint(
            model_path,
            map_location=self.device
        )
        
        # Set to evaluation mode
        model.eval()
        
        return model
    
    def _add_routes(self):
        """Add API routes."""
        @self.app.get('/health')
        async def health_check():
            """Health check endpoint."""
            return {'status': 'healthy', 'version': self.model_version}
        
        @self.app.post('/predict', response_model=PredictionResponse)
        async def predict(request: PredictionRequest):
            """Prediction endpoint."""
            try:
                with torch.no_grad():
                    # Prepare input
                    batch = self._prepare_batch(request)
                    
                    # Get predictions
                    global_logits, user_logits = self.model(batch)
                    
                    # Get probabilities
                    global_probs = torch.softmax(global_logits, dim=1)
                    user_probs = torch.softmax(user_logits, dim=1)
                    
                    # Get predicted categories
                    global_category_id = torch.argmax(global_probs).item()
                    user_category_id = torch.argmax(user_probs).item()
                    
                    # Get confidence scores
                    global_confidence = global_probs[0, global_category_id].item()
                    user_confidence = user_probs[0, user_category_id].item()
                    
                    # Update metrics
                    PREDICTIONS.labels(
                        model_version=self.model_version,
                        category_type='global'
                    ).inc()
                    PREDICTIONS.labels(
                        model_version=self.model_version,
                        category_type='user'
                    ).inc()
                    
                    CONFIDENCE.labels(
                        model_version=self.model_version,
                        category_type='global'
                    ).set(global_confidence)
                    CONFIDENCE.labels(
                        model_version=self.model_version,
                        category_type='user'
                    ).set(user_confidence)
                    
                    return PredictionResponse(
                        transaction_id=request.transaction_id,
                        global_category=self.label_mappings['global'][str(global_category_id)],
                        global_category_id=global_category_id,
                        global_category_confidence=global_confidence,
                        user_category=self.label_mappings['user'][str(user_category_id)],
                        user_category_id=user_category_id,
                        user_category_confidence=user_confidence,
                        model_version=self.model_version
                    )
            
            except Exception as e:
                logger.error(f"Prediction error: {str(e)}")
                ERRORS.labels(
                    model_version=self.model_version,
                    error_type=type(e).__name__
                ).inc()
                raise HTTPException(status_code=500, detail=str(e))
    
    def _prepare_batch(self, request: PredictionRequest) -> Dict:
        """Prepare input batch for model."""
        # Convert request to model input format
        batch = {
            'transaction_id': [request.transaction_id],
            'amount': torch.tensor([[request.amount]], device=self.device),
            'description': [request.description],
            'merchant_id': [request.merchant_id] if request.merchant_id else None,
            'user_id': [request.user_id],
            'timestamp': [request.timestamp]
        }
        
        # Add historical transactions if provided
        if request.historical_transactions:
            batch['historical_transactions'] = request.historical_transactions
        
        return batch
    
    def run(self, host: str = '0.0.0.0', port: int = 8000):
        """Run the server."""
        logger.info(f"Starting server on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)

def main(
    model_path: str,
    label_mapping_path: str,
    host: str = '0.0.0.0',
    port: int = 8000,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """Main function for model serving."""
    server = ModelServer(
        model_path=model_path,
        label_mapping_path=label_mapping_path,
        device=device
    )
    
    server.run(host=host, port=port)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Serve transaction classification model')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--label_mapping_path', type=str, required=True,
                      help='Path to label mapping JSON file')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                      help='Host to serve on')
    parser.add_argument('--port', type=int, default=8000,
                      help='Port to serve on')
    parser.add_argument('--device', type=str, default=None,
                      help='Device to run on (cuda/cpu)')
    
    args = parser.parse_args()
    
    main(
        model_path=args.model_path,
        label_mapping_path=args.label_mapping_path,
        host=args.host,
        port=args.port,
        device=args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    ) 