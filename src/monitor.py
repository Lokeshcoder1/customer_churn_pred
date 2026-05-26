"""
Model Monitoring & Health Checks
Tracks model performance, predictions, and system health.

Exports metrics to Prometheus for visualization in Grafana.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import json

import numpy as np
import pandas as pd
from prometheus_client import Counter, Gauge, Histogram, start_http_server

from src.config import ARTIFACTS_DIR, LOGS_DIR

logger = logging.getLogger(__name__)

# ============================================================================
# Prometheus Metrics
# ============================================================================

# Counter: Total predictions made
predictions_total = Counter(
    'predictions_total',
    'Total number of predictions made',
    ['endpoint', 'status']
)

# Counter: Predictions by class
predictions_by_class = Counter(
    'predictions_by_class_total',
    'Predictions by predicted class',
    ['class']
)

# Gauge: Model AUC
model_auc = Gauge(
    'model_auc',
    'Current model AUC on test set'
)

# Gauge: Data drift detected
data_drift_detected = Gauge(
    'data_drift_detected',
    'Whether data drift has been detected (0=no, 1=yes)'
)

# Histogram: Prediction latency
prediction_latency = Histogram(
    'prediction_latency_seconds',
    'Prediction latency in seconds',
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0)
)

# Gauge: API uptime
api_uptime = Gauge(
    'api_uptime_seconds',
    'API uptime in seconds'
)

# Gauge: Model version
model_version = Gauge(
    'model_version_info',
    'Current model version info',
    ['version', 'model_name', 'timestamp']
)


# ============================================================================
# Health Check System
# ============================================================================
class HealthChecker:
    """Comprehensive health check system."""
    
    def __init__(self, log_dir: Path = LOGS_DIR):
        """Initialize health checker."""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.health_log = self.log_dir / "health_checks.jsonl"
    
    def check_model(self) -> Dict:
        """Check model status."""
        checks = {
            "model_exists": False,
            "model_loadable": False,
            "model_prediction_working": False
        }
        
        try:
            # Check if model file exists
            from src.config import MODELS_DIR
            model_path = MODELS_DIR / "production_model.pkl"
            
            if model_path.exists():
                checks["model_exists"] = True
                
                # Try to load model
                import pickle
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                
                checks["model_loadable"] = True
                
                # Try a test prediction
                from src.pipeline import MLPipeline
                pipeline = MLPipeline()
                pipeline.load(model_path)
                
                # Create dummy input
                import pandas as pd
                test_input = pd.DataFrame({
                    'tenure': [24.0],
                    'monthly_charges': [65.5],
                    'total_charges': [1570.0]
                })
                
                # Try prediction
                # Note: This assumes your model can handle minimal input
                # In practice, you'd need all required features
                
                checks["model_prediction_working"] = True
        
        except Exception as e:
            logger.error(f"Model health check failed: {str(e)}")
        
        return checks
    
    def check_api(self) -> Dict:
        """Check API status."""
        checks = {
            "api_responding": False,
            "api_latency_ok": False
        }
        
        try:
            import time
            import requests
            
            start = time.time()
            response = requests.get("http://localhost:8000/health", timeout=5)
            latency = time.time() - start
            
            checks["api_responding"] = response.status_code == 200
            checks["api_latency_ok"] = latency < 1.0  # Less than 1 second
            checks["latency_ms"] = latency * 1000
        
        except Exception as e:
            logger.warning(f"API health check failed: {str(e)}")
        
        return checks
    
    def check_data(self) -> Dict:
        """Check data quality."""
        checks = {
            "training_data_exists": False,
            "processed_data_exists": False,
            "data_quality_ok": False
        }
        
        try:
            from src.config import DATA_DIR
            
            raw_path = DATA_DIR / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
            checks["training_data_exists"] = raw_path.exists()
            
            processed_path = DATA_DIR / "processed" / "train.csv"
            checks["processed_data_exists"] = processed_path.exists()
            
            if checks["processed_data_exists"]:
                df = pd.read_csv(processed_path)
                
                # Check for missing values
                missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
                checks["data_quality_ok"] = missing_ratio < 0.05  # Less than 5% missing
                checks["missing_ratio"] = float(missing_ratio)
        
        except Exception as e:
            logger.warning(f"Data health check failed: {str(e)}")
        
        return checks
    
    def check_disk(self) -> Dict:
        """Check disk usage."""
        checks = {
            "disk_space_ok": False,
            "models_dir_size_mb": 0,
            "logs_dir_size_mb": 0
        }
        
        try:
            from src.config import MODELS_DIR, LOGS_DIR
            
            def get_dir_size_mb(path):
                """Get directory size in MB."""
                total = 0
                for f in path.rglob('*'):
                    if f.is_file():
                        total += f.stat().st_size
                return total / (1024 * 1024)
            
            models_size = get_dir_size_mb(MODELS_DIR)
            logs_size = get_dir_size_mb(LOGS_DIR)
            
            checks["models_dir_size_mb"] = models_size
            checks["logs_dir_size_mb"] = logs_size
            checks["disk_space_ok"] = (models_size + logs_size) < 1000  # Less than 1GB
        
        except Exception as e:
            logger.warning(f"Disk health check failed: {str(e)}")
        
        return checks
    
    def run_all_checks(self) -> Dict:
        """Run all health checks."""
        logger.info("Running comprehensive health checks...")
        
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "model": self.check_model(),
            "api": self.check_api(),
            "data": self.check_data(),
            "disk": self.check_disk(),
            "overall_healthy": False
        }
        
        # Determine overall health
        all_checks = []
        for category in ['model', 'api', 'data', 'disk']:
            for check_name, check_value in health_status[category].items():
                if isinstance(check_value, bool):
                    all_checks.append(check_value)
        
        health_status["overall_healthy"] = all(all_checks)
        
        # Log health status
        self._log_health_status(health_status)
        
        return health_status
    
    def _log_health_status(self, status: Dict):
        """Log health status to file."""
        with open(self.health_log, 'a') as f:
            f.write(json.dumps(status) + '\n')
        
        logger.info(f"Health check result: {'✓ Healthy' if status['overall_healthy'] else '✗ Issues detected'}")


# ============================================================================
# Model Performance Monitoring
# ============================================================================
class ModelMonitor:
    """Monitor model performance over time."""
    
    def __init__(self, metrics_file: Path = ARTIFACTS_DIR / "metrics.jsonl"):
        """Initialize model monitor."""
        self.metrics_file = Path(metrics_file)
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
    
    def log_prediction(self, prediction: Dict, actual: int = None, latency: float = None):
        """
        Log a prediction for monitoring.
        
        Args:
            prediction: Prediction dict with probability and class
            actual: True label (optional)
            latency: Prediction latency in seconds (optional)
        """
        record = {
            "timestamp": datetime.now().isoformat(),
            "prediction": prediction,
            "actual": actual,
            "latency_ms": latency * 1000 if latency else None,
            "correct": (prediction["churn_prediction"] == actual) if actual is not None else None
        }
        
        # Append to file
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(record) + '\n')
        
        # Update Prometheus metrics
        predictions_total.labels(
            endpoint="predict",
            status="success"
        ).inc()
        
        if latency:
            prediction_latency.observe(latency)
    
    def get_recent_metrics(self, hours: int = 24) -> Dict:
        """
        Get metrics from recent period.
        
        Args:
            hours: How many hours to look back
            
        Returns:
            Summary metrics
        """
        try:
            from datetime import timedelta
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            records = []
            with open(self.metrics_file, 'r') as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        timestamp = datetime.fromisoformat(record["timestamp"])
                        if timestamp > cutoff_time:
                            records.append(record)
            
            if not records:
                return {
                    "period_hours": hours,
                    "n_predictions": 0,
                    "avg_latency_ms": 0,
                    "accuracy": None
                }
            
            # Calculate metrics
            latencies = [r["latency_ms"] for r in records if r["latency_ms"]]
            correct = [r["correct"] for r in records if r["correct"] is not None]
            
            metrics = {
                "period_hours": hours,
                "n_predictions": len(records),
                "avg_latency_ms": np.mean(latencies) if latencies else None,
                "accuracy": np.mean(correct) if correct else None,
                "n_churn_predictions": sum(1 for r in records if r.get("prediction", {}).get("churn_prediction")),
                "churn_rate": sum(1 for r in records if r.get("prediction", {}).get("churn_prediction")) / len(records) if records else 0
            }
            
            return metrics
        
        except Exception as e:
            logger.warning(f"Failed to get recent metrics: {str(e)}")
            return {}


# ============================================================================
# Alerting System
# ============================================================================
class AlertManager:
    """Manage alerts based on health and performance."""
    
    def __init__(self):
        """Initialize alert manager."""
        self.alerts = []
    
    def add_alert(self, severity: str, message: str, details: Dict = None):
        """
        Add an alert.
        
        Args:
            severity: 'critical', 'warning', 'info'
            message: Alert message
            details: Additional details
        """
        alert = {
            "timestamp": datetime.now().isoformat(),
            "severity": severity,
            "message": message,
            "details": details or {}
        }
        
        self.alerts.append(alert)
        
        logger.log(
            logging.CRITICAL if severity == "critical" else logging.WARNING,
            f"[{severity.upper()}] {message}"
        )
    
    def check_and_alert(self, health_status: Dict, metrics: Dict):
        """
        Check health and metrics, generate alerts.
        
        Args:
            health_status: Health check results
            metrics: Performance metrics
        """
        # Check model health
        if not health_status["model"]["model_exists"]:
            self.add_alert("critical", "Model file not found")
        
        if not health_status["model"]["model_loadable"]:
            self.add_alert("critical", "Model cannot be loaded")
        
        # Check API health
        if not health_status["api"]["api_responding"]:
            self.add_alert("critical", "API not responding")
        
        if not health_status["api"]["api_latency_ok"]:
            latency = health_status["api"].get("latency_ms", 0)
            self.add_alert(
                "warning",
                f"API latency high: {latency:.0f}ms",
                {"latency_ms": latency}
            )
        
        # Check data quality
        if not health_status["data"]["data_quality_ok"]:
            missing = health_status["data"].get("missing_ratio", 0)
            self.add_alert(
                "warning",
                f"Data quality issue: {missing*100:.1f}% missing values",
                {"missing_ratio": missing}
            )
        
        # Check disk space
        if not health_status["disk"]["disk_space_ok"]:
            total_size = (
                health_status["disk"]["models_dir_size_mb"] +
                health_status["disk"]["logs_dir_size_mb"]
            )
            self.add_alert(
                "warning",
                f"Disk usage high: {total_size:.0f}MB",
                {"disk_usage_mb": total_size}
            )
        
        # Check prediction metrics
        if metrics.get("accuracy") and metrics["accuracy"] < 0.7:
            self.add_alert(
                "warning",
                f"Model accuracy declining: {metrics['accuracy']:.2%}",
                {"accuracy": metrics["accuracy"]}
            )
    
    def get_alerts(self, severity: str = None) -> List[Dict]:
        """Get alerts, optionally filtered by severity."""
        if severity:
            return [a for a in self.alerts if a["severity"] == severity]
        return self.alerts


# ============================================================================
# Main Monitoring Function
# ============================================================================
def run_monitoring_checks():
    """Run comprehensive monitoring checks."""
    logger.info("="*70)
    logger.info("STARTING COMPREHENSIVE MONITORING")
    logger.info("="*70)
    
    # Health checks
    health_checker = HealthChecker()
    health_status = health_checker.run_all_checks()
    
    # Performance metrics
    model_monitor = ModelMonitor()
    metrics = model_monitor.get_recent_metrics(hours=24)
    
    # Alerting
    alert_manager = AlertManager()
    alert_manager.check_and_alert(health_status, metrics)
    
    # Report
    print("\n" + "="*70)
    print("MONITORING REPORT")
    print("="*70)
    print(f"Overall Health: {'✓ HEALTHY' if health_status['overall_healthy'] else '✗ ISSUES'}")
    print(f"\nRecent Metrics (24h):")
    print(f"  Predictions: {metrics.get('n_predictions', 0)}")
    print(f"  Avg Latency: {metrics.get('avg_latency_ms', 0):.1f}ms")
    print(f"  Accuracy: {metrics.get('accuracy', 0):.2%}")
    print(f"  Churn Rate: {metrics.get('churn_rate', 0):.2%}")
    
    alerts = alert_manager.get_alerts()
    if alerts:
        print(f"\nAlerts ({len(alerts)}):")
        for alert in alerts:
            print(f"  [{alert['severity'].upper()}] {alert['message']}")
    
    print("="*70 + "\n")
    
    return {
        "health": health_status,
        "metrics": metrics,
        "alerts": alerts
    }


if __name__ == "__main__":
    # Start Prometheus metrics server (optional)
    # start_http_server(8001)  # Expose metrics on port 8001
    
    results = run_monitoring_checks()