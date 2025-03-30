import os
import shutil
import subprocess
import yaml
from typing import Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelDeployer:
    """Deployer for transaction classification model."""
    def __init__(
        self,
        model_path: str,
        label_mapping_path: str,
        model_version: str = '1.0.0',
        docker_registry: Optional[str] = None
    ):
        self.model_path = model_path
        self.label_mapping_path = label_mapping_path
        self.model_version = model_version
        self.docker_registry = docker_registry
        
        # Create deployment directory
        self.deploy_dir = 'deployment'
        os.makedirs(self.deploy_dir, exist_ok=True)
    
    def create_dockerfile(self):
        """Create Dockerfile for model deployment."""
        dockerfile_content = f"""
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files
COPY models/ models/
COPY data/ data/
COPY training/ training/

# Copy model checkpoint and label mappings
COPY {self.model_path} /app/model.ckpt
COPY {self.label_mapping_path} /app/label_mappings.json

# Set environment variables
ENV MODEL_PATH=/app/model.ckpt
ENV LABEL_MAPPING_PATH=/app/label_mappings.json
ENV MODEL_VERSION={self.model_version}

# Expose ports
EXPOSE 8000 9090

# Run the model server
CMD ["python", "-m", "training.serve", "--model_path", "/app/model.ckpt", "--label_mapping_path", "/app/label_mappings.json"]
"""
        
        with open(os.path.join(self.deploy_dir, 'Dockerfile'), 'w') as f:
            f.write(dockerfile_content)
    
    def create_kubernetes_manifests(self):
        """Create Kubernetes manifests for model deployment."""
        # Create deployment manifest
        deployment = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': 'transaction-classifier',
                'labels': {
                    'app': 'transaction-classifier'
                }
            },
            'spec': {
                'replicas': 3,
                'selector': {
                    'matchLabels': {
                        'app': 'transaction-classifier'
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': 'transaction-classifier'
                        }
                    },
                    'spec': {
                        'containers': [{
                            'name': 'transaction-classifier',
                            'image': f"{self.docker_registry}/transaction-classifier:{self.model_version}" if self.docker_registry else f"transaction-classifier:{self.model_version}",
                            'ports': [
                                {'containerPort': 8000, 'name': 'api'},
                                {'containerPort': 9090, 'name': 'metrics'}
                            ],
                            'resources': {
                                'requests': {
                                    'cpu': '500m',
                                    'memory': '1Gi'
                                },
                                'limits': {
                                    'cpu': '2',
                                    'memory': '4Gi'
                                }
                            },
                            'livenessProbe': {
                                'httpGet': {
                                    'path': '/health',
                                    'port': 'api'
                                },
                                'initialDelaySeconds': 5,
                                'periodSeconds': 10
                            },
                            'readinessProbe': {
                                'httpGet': {
                                    'path': '/health',
                                    'port': 'api'
                                },
                                'initialDelaySeconds': 5,
                                'periodSeconds': 10
                            }
                        }]
                    }
                }
            }
        }
        
        # Create service manifest
        service = {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': 'transaction-classifier'
            },
            'spec': {
                'selector': {
                    'app': 'transaction-classifier'
                },
                'ports': [
                    {
                        'port': 80,
                        'targetPort': 'api',
                        'name': 'api'
                    },
                    {
                        'port': 9090,
                        'targetPort': 'metrics',
                        'name': 'metrics'
                    }
                ],
                'type': 'LoadBalancer'
            }
        }
        
        # Create horizontal pod autoscaler manifest
        hpa = {
            'apiVersion': 'autoscaling/v2',
            'kind': 'HorizontalPodAutoscaler',
            'metadata': {
                'name': 'transaction-classifier'
            },
            'spec': {
                'scaleTargetRef': {
                    'apiVersion': 'apps/v1',
                    'kind': 'Deployment',
                    'name': 'transaction-classifier'
                },
                'minReplicas': 2,
                'maxReplicas': 10,
                'metrics': [
                    {
                        'type': 'Resource',
                        'resource': {
                            'name': 'cpu',
                            'target': {
                                'type': 'Utilization',
                                'averageUtilization': 70
                            }
                        }
                    }
                ]
            }
        }
        
        # Save manifests
        with open(os.path.join(self.deploy_dir, 'deployment.yaml'), 'w') as f:
            yaml.dump(deployment, f)
        
        with open(os.path.join(self.deploy_dir, 'service.yaml'), 'w') as f:
            yaml.dump(service, f)
        
        with open(os.path.join(self.deploy_dir, 'hpa.yaml'), 'w') as f:
            yaml.dump(hpa, f)
    
    def create_monitoring_config(self):
        """Create Prometheus and Grafana configurations."""
        # Create Prometheus service monitor
        service_monitor = {
            'apiVersion': 'monitoring.coreos.com/v1',
            'kind': 'ServiceMonitor',
            'metadata': {
                'name': 'transaction-classifier',
                'labels': {
                    'release': 'prometheus'
                }
            },
            'spec': {
                'selector': {
                    'matchLabels': {
                        'app': 'transaction-classifier'
                    }
                },
                'endpoints': [{
                    'port': 'metrics',
                    'interval': '15s'
                }]
            }
        }
        
        # Create Grafana dashboard
        dashboard = {
            'apiVersion': 'integreatly.org/v1alpha1',
            'kind': 'GrafanaDashboard',
            'metadata': {
                'name': 'transaction-classifier'
            },
            'spec': {
                'json': {
                    'dashboard': {
                        'id': None,
                        'title': 'Transaction Classifier Dashboard',
                        'panels': [
                            {
                                'title': 'Prediction Rate',
                                'type': 'graph',
                                'datasource': 'Prometheus',
                                'targets': [{
                                    'expr': 'rate(transaction_predictions_total[5m])',
                                    'legendFormat': '{{model_version}}'
                                }]
                            },
                            {
                                'title': 'Model Confidence',
                                'type': 'graph',
                                'datasource': 'Prometheus',
                                'targets': [{
                                    'expr': 'transaction_model_confidence',
                                    'legendFormat': '{{model_version}}'
                                }]
                            },
                            {
                                'title': 'Category Accuracy',
                                'type': 'graph',
                                'datasource': 'Prometheus',
                                'targets': [{
                                    'expr': 'transaction_category_accuracy',
                                    'legendFormat': '{{model_version}} - {{category_type}}'
                                }]
                            },
                            {
                                'title': 'Prediction Latency',
                                'type': 'graph',
                                'datasource': 'Prometheus',
                                'targets': [{
                                    'expr': 'rate(transaction_prediction_latency_seconds_sum[5m]) / rate(transaction_prediction_latency_seconds_count[5m])',
                                    'legendFormat': '{{model_version}}'
                                }]
                            }
                        ]
                    }
                }
            }
        }
        
        # Save configurations
        with open(os.path.join(self.deploy_dir, 'service-monitor.yaml'), 'w') as f:
            yaml.dump(service_monitor, f)
        
        with open(os.path.join(self.deploy_dir, 'dashboard.yaml'), 'w') as f:
            yaml.dump(dashboard, f)
    
    def build_and_push_docker_image(self):
        """Build and push Docker image."""
        # Build image
        image_name = f"transaction-classifier:{self.model_version}"
        if self.docker_registry:
            image_name = f"{self.docker_registry}/{image_name}"
        
        subprocess.run([
            'docker', 'build',
            '-t', image_name,
            '-f', os.path.join(self.deploy_dir, 'Dockerfile'),
            '.'
        ], check=True)
        
        # Push image if registry is specified
        if self.docker_registry:
            subprocess.run([
                'docker', 'push', image_name
            ], check=True)
    
    def deploy_to_kubernetes(self):
        """Deploy model to Kubernetes cluster."""
        # Apply Kubernetes manifests
        manifests = [
            'deployment.yaml',
            'service.yaml',
            'hpa.yaml',
            'service-monitor.yaml',
            'dashboard.yaml'
        ]
        
        for manifest in manifests:
            subprocess.run([
                'kubectl', 'apply',
                '-f', os.path.join(self.deploy_dir, manifest)
            ], check=True)
    
    def deploy(self):
        """Deploy model."""
        logger.info("Creating deployment files...")
        self.create_dockerfile()
        self.create_kubernetes_manifests()
        self.create_monitoring_config()
        
        logger.info("Building and pushing Docker image...")
        self.build_and_push_docker_image()
        
        logger.info("Deploying to Kubernetes...")
        self.deploy_to_kubernetes()
        
        logger.info("Deployment completed successfully!")

def main(
    model_path: str,
    label_mapping_path: str,
    model_version: str = '1.0.0',
    docker_registry: Optional[str] = None
):
    """Main function for model deployment."""
    deployer = ModelDeployer(
        model_path=model_path,
        label_mapping_path=label_mapping_path,
        model_version=model_version,
        docker_registry=docker_registry
    )
    
    deployer.deploy()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Deploy transaction classification model')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--label_mapping_path', type=str, required=True,
                      help='Path to label mapping JSON file')
    parser.add_argument('--model_version', type=str, default='1.0.0',
                      help='Version of the model')
    parser.add_argument('--docker_registry', type=str, default=None,
                      help='Docker registry for image pushing')
    
    args = parser.parse_args()
    
    main(
        model_path=args.model_path,
        label_mapping_path=args.label_mapping_path,
        model_version=args.model_version,
        docker_registry=args.docker_registry
    ) 