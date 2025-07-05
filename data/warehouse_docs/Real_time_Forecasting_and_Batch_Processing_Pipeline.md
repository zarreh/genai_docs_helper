# Real-time Forecasting and Batch Processing Pipeline

## Document Information
- **Document Title**: Real-time Forecasting and Batch Processing Pipeline
- **Version**: 1.6
- **Last Updated**: July 2025
- **Author**: ML Engineering Team
- **Reviewers**: Data Engineering Team, DevOps
- **Related Documents**: 
  - Retail Demand Forecasting System Architecture v1.2
  - Model Performance Evaluation and Validation Framework v1.4

---

## 1. Executive Summary

This document outlines the dual-mode processing architecture for our retail demand forecasting system, supporting both real-time predictions and large-scale batch processing. Our pipeline delivers sub-second real-time forecasts while efficiently processing millions of predictions in batch mode.

### Key Performance Metrics
- **Real-time Latency**: <180ms (p95) for single predictions
- **Batch Throughput**: 50,000 predictions/minute
- **System Availability**: 99.95% uptime
- **Processing Capacity**: 2.5M daily predictions across 52 warehouses
- **Auto-scaling Response**: <30 seconds to scale up/down

---

## 2. Pipeline Architecture Overview

### 2.1 Dual-Mode Processing Design

```
Input Layer
    ├── Real-time API Requests
    │   ├── Single Warehouse Predictions
    │   ├── Multi-warehouse Queries
    │   └── Ad-hoc Forecast Requests
    └── Batch Processing Jobs
        ├── Daily Forecast Generation
        ├── Weekly Planning Updates
        └── Monthly Reforecasting
            │
Processing Layer
    ├── Real-time Engine (FastAPI + Redis)
    └── Batch Engine (Apache Spark + Airflow)
            │
Output Layer
    ├── API Responses (JSON)
    ├── Database Updates (PostgreSQL)
    └── File Exports (CSV, Parquet)
```

### 2.2 Technology Stack

**Real-time Components**:
```python
REALTIME_STACK = {
    'api_framework': 'FastAPI 0.104.1',
    'model_serving': 'MLflow + Gunicorn',
    'caching': 'Redis 7.0',
    'load_balancer': 'NGINX',
    'monitoring': 'Prometheus + Grafana',
    'deployment': 'Docker + Kubernetes'
}
```

**Batch Components**:
```python
BATCH_STACK = {
    'processing_engine': 'Apache Spark 3.4',
    'orchestration': 'Apache Airflow 2.7',
    'storage': 'PostgreSQL + S3',
    'compute': 'AWS EMR',
    'monitoring': 'DataDog + CloudWatch',
    'scheduling': 'Cron + Airflow Scheduler'
}
```

---

## 3. Real-time Forecasting Pipeline

### 3.1 API Architecture

**Endpoint Structure**:
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio

app = FastAPI(title="Forecasting API", version="1.6")

class PredictionRequest(BaseModel):
    warehouse_id: str
    forecast_date: str
    horizon_days: int = 7
    include_confidence: bool = True
    model_version: str = "latest"

@app.post("/api/v1/predict")
async def get_prediction(request: PredictionRequest):
    try:
        # Load model and features
        model = await load_model(request.warehouse_id, request.model_version)
        features = await prepare_features(request.warehouse_id, request.forecast_date)

        # Generate prediction
        prediction = await model.predict(features, request.horizon_days)

        # Add confidence intervals if requested
        if request.include_confidence:
            prediction['confidence_intervals'] = await calculate_confidence(prediction)

        return {
            "warehouse_id": request.warehouse_id,
            "predictions": prediction,
            "model_version": request.model_version,
            "generated_at": datetime.utcnow(),
            "latency_ms": get_request_latency()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### 3.2 Caching Strategy

**Multi-level Caching**:
```python
CACHING_STRATEGY = {
    'l1_cache': {
        'type': 'in_memory',
        'size': '2GB per instance',
        'ttl': '5 minutes',
        'hit_rate': '78%',
        'use_case': 'frequently_requested_predictions'
    },
    'l2_cache': {
        'type': 'redis_cluster',
        'size': '50GB total',
        'ttl': '1 hour',
        'hit_rate': '65%',
        'use_case': 'feature_data_and_models'
    },
    'l3_cache': {
        'type': 'database_cache',
        'size': '500GB',
        'ttl': '24 hours',
        'hit_rate': '45%',
        'use_case': 'historical_predictions'
    }
}
```

**Cache Implementation**:
```python
import redis
from functools import wraps

redis_client = redis.Redis(host='redis-cluster', port=6379, decode_responses=True)

def cache_prediction(ttl=300):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"prediction:{kwargs['warehouse_id']}:{kwargs['date']}:{kwargs['horizon']}"

            # Try to get from cache
            cached_result = redis_client.get(cache_key)
            if cached_result:
                return json.loads(cached_result)

            # Generate prediction if not cached
            result = await func(*args, **kwargs)

            # Store in cache
            redis_client.setex(cache_key, ttl, json.dumps(result))

            return result
        return wrapper
    return decorator
```

### 3.3 Real-time Performance Optimization

**Performance Metrics**:
```python
REALTIME_PERFORMANCE = {
    'latency_targets': {
        'p50': '<100ms',
        'p95': '<180ms',
        'p99': '<300ms'
    },
    'current_performance': {
        'p50': '87ms',
        'p95': '165ms',
        'p99': '245ms'
    },
    'throughput': {
        'target': '1000 req/min per instance',
        'current': '1200 req/min per instance',
        'peak_capacity': '2000 req/min per instance'
    }
}
```

**Optimization Techniques**:
- **Model Preloading**: Keep models in memory across requests
- **Feature Preprocessing**: Cache preprocessed features for common requests
- **Async Processing**: Non-blocking I/O for database and cache operations
- **Connection Pooling**: Reuse database connections
- **Request Batching**: Group similar requests for efficient processing

---

## 4. Batch Processing Pipeline

### 4.1 Batch Job Architecture

**Daily Batch Processing Workflow**:
```python
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'ml-engineering',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'daily_forecasting_pipeline',
    default_args=default_args,
    description='Daily batch forecasting for all warehouses',
    schedule_interval='0 2 * * *',  # Run at 2 AM daily
    catchup=False
)

# Define tasks
data_validation = PythonOperator(
    task_id='validate_input_data',
    python_callable=validate_daily_data,
    dag=dag
)

feature_engineering = PythonOperator(
    task_id='engineer_features',
    python_callable=create_batch_features,
    dag=dag
)

model_predictions = PythonOperator(
    task_id='generate_predictions',
    python_callable=batch_predict_all_warehouses,
    dag=dag
)

quality_checks = PythonOperator(
    task_id='validate_predictions',
    python_callable=validate_prediction_quality,
    dag=dag
)

data_export = PythonOperator(
    task_id='export_results',
    python_callable=export_predictions_to_systems,
    dag=dag
)

# Define dependencies
data_validation >> feature_engineering >> model_predictions >> quality_checks >> data_export
```

### 4.2 Spark Processing Engine

**Batch Processing Configuration**:
```python
SPARK_CONFIG = {
    'cluster_setup': {
        'driver_memory': '8g',
        'driver_cores': 4,
        'executor_memory': '16g',
        'executor_cores': 8,
        'num_executors': 20,
        'max_executors': 50
    },
    'performance_tuning': {
        'spark.sql.adaptive.enabled': True,
        'spark.sql.adaptive.coalescePartitions.enabled': True,
        'spark.serializer': 'org.apache.spark.serializer.KryoSerializer',
        'spark.sql.execution.arrow.pyspark.enabled': True
    },
    'processing_capacity': {
        'records_per_minute': 50000,
        'warehouses_parallel': 10,
        'total_daily_capacity': '2.5M predictions'
    }
}
```

**Batch Processing Implementation**:
```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lag
from pyspark.sql.window import Window

def batch_forecast_all_warehouses(spark_session, forecast_date):
    # Load data for all warehouses
    warehouse_data = spark_session.read.table("warehouse_orders")         .filter(col("date") <= forecast_date)

    # Feature engineering for all warehouses
    window_spec = Window.partitionBy("warehouse_id").orderBy("date")

    featured_data = warehouse_data         .withColumn("lag_7", lag("order_volume", 7).over(window_spec))         .withColumn("lag_14", lag("order_volume", 14).over(window_spec))         .withColumn("rolling_mean_7", avg("order_volume").over(window_spec.rowsBetween(-6, 0)))

    # Load models and generate predictions
    predictions = []
    for warehouse_id in get_warehouse_list():
        warehouse_features = featured_data.filter(col("warehouse_id") == warehouse_id)
        model = load_warehouse_model(warehouse_id)

        warehouse_predictions = model.transform(warehouse_features)
        predictions.append(warehouse_predictions)

    # Combine all predictions
    all_predictions = reduce(lambda df1, df2: df1.union(df2), predictions)

    # Save results
    all_predictions.write         .mode("overwrite")         .partitionBy("warehouse_id")         .table("daily_forecasts")

    return all_predictions.count()
```

### 4.3 Batch Job Scheduling and Monitoring

**Job Schedule Matrix**:
```python
BATCH_SCHEDULE = {
    'daily_jobs': {
        'daily_forecasting': {'time': '02:00', 'duration': '45min', 'priority': 'high'},
        'model_validation': {'time': '03:00', 'duration': '30min', 'priority': 'medium'},
        'data_quality_check': {'time': '04:00', 'duration': '15min', 'priority': 'high'}
    },
    'weekly_jobs': {
        'model_retraining': {'time': 'Sun 01:00', 'duration': '3hours', 'priority': 'high'},
        'performance_analysis': {'time': 'Mon 05:00', 'duration': '1hour', 'priority': 'medium'},
        'data_archival': {'time': 'Sat 23:00', 'duration': '2hours', 'priority': 'low'}
    },
    'monthly_jobs': {
        'full_reforecasting': {'time': '1st 00:00', 'duration': '6hours', 'priority': 'high'},
        'model_comparison': {'time': '15th 06:00', 'duration': '2hours', 'priority': 'medium'}
    }
}
```

---

## 5. Data Flow and Integration

### 5.1 Data Pipeline Architecture

**Data Flow Diagram**:
```
Source Systems
    ├── Warehouse Management System
    ├── Weather API
    ├── Holiday Calendar
    └── Business Events
        │
Data Ingestion Layer
    ├── Real-time Streaming (Kafka)
    └── Batch ETL (Airflow + Spark)
        │
Data Storage Layer
    ├── Raw Data Lake (S3)
    ├── Processed Data (PostgreSQL)
    └── Feature Store (Redis + PostgreSQL)
        │
Processing Layer
    ├── Real-time Pipeline
    └── Batch Pipeline
        │
Output Layer
    ├── API Responses
    ├── Database Updates
    └── File Exports
```

### 5.2 Feature Store Integration

**Feature Store Architecture**:
```python
class FeatureStore:
    def __init__(self):
        self.online_store = redis.Redis(host='redis-cluster')
        self.offline_store = PostgreSQLConnection()

    def get_realtime_features(self, warehouse_id, timestamp):
        # Get features for real-time prediction
        cache_key = f"features:{warehouse_id}:{timestamp.date()}"

        cached_features = self.online_store.hgetall(cache_key)
        if cached_features:
            return self.deserialize_features(cached_features)

        # Fallback to offline store
        features = self.offline_store.get_features(warehouse_id, timestamp)

        # Cache for future requests
        self.online_store.hmset(cache_key, self.serialize_features(features))
        self.online_store.expire(cache_key, 3600)  # 1 hour TTL

        return features

    def get_batch_features(self, warehouse_ids, date_range):
        # Get features for batch processing
        return self.offline_store.get_batch_features(warehouse_ids, date_range)
```

---

## 6. Scalability and Performance

### 6.1 Auto-scaling Configuration

**Kubernetes Auto-scaling**:
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: forecasting-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: forecasting-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
```

### 6.2 Performance Benchmarks

**Scalability Metrics**:
```python
PERFORMANCE_BENCHMARKS = {
    'realtime_scaling': {
        'baseline_capacity': '1000 req/min per pod',
        'max_tested_load': '15000 req/min (15 pods)',
        'scaling_time': '30 seconds average',
        'cpu_utilization_target': '70%',
        'memory_utilization_target': '80%'
    },
    'batch_scaling': {
        'baseline_throughput': '10000 predictions/min',
        'max_cluster_size': '50 executors',
        'max_throughput': '50000 predictions/min',
        'cost_per_prediction': '$0.0001',
        'processing_efficiency': '85%'
    }
}
```

---

## 7. Error Handling and Resilience

### 7.1 Fault Tolerance Mechanisms

**Error Handling Strategy**:
```python
class ResilientPredictor:
    def __init__(self):
        self.primary_model = load_primary_model()
        self.fallback_model = load_fallback_model()
        self.circuit_breaker = CircuitBreaker(failure_threshold=5, timeout=30)

    async def predict_with_fallback(self, warehouse_id, features):
        try:
            # Try primary model with circuit breaker
            with self.circuit_breaker:
                return await self.primary_model.predict(features)

        except CircuitBreakerOpenException:
            # Circuit breaker is open, use fallback
            logger.warning(f"Circuit breaker open for {warehouse_id}, using fallback model")
            return await self.fallback_model.predict(features)

        except Exception as e:
            # Primary model failed, try fallback
            logger.error(f"Primary model failed for {warehouse_id}: {str(e)}")
            try:
                return await self.fallback_model.predict(features)
            except Exception as fallback_error:
                # Both models failed, return cached prediction or default
                logger.error(f"Both models failed for {warehouse_id}: {str(fallback_error)}")
                return await self.get_cached_or_default_prediction(warehouse_id)
```

### 7.2 Monitoring and Alerting

**System Health Monitoring**:
```python
MONITORING_METRICS = {
    'realtime_health': {
        'response_time_p95': {'threshold': 200, 'current': 165, 'status': 'healthy'},
        'error_rate': {'threshold': 0.01, 'current': 0.003, 'status': 'healthy'},
        'throughput': {'threshold': 800, 'current': 1200, 'status': 'healthy'},
        'cache_hit_rate': {'threshold': 0.6, 'current': 0.78, 'status': 'healthy'}
    },
    'batch_health': {
        'job_success_rate': {'threshold': 0.95, 'current': 0.98, 'status': 'healthy'},
        'processing_time': {'threshold': 3600, 'current': 2700, 'status': 'healthy'},
        'data_quality_score': {'threshold': 0.95, 'current': 0.97, 'status': 'healthy'},
        'resource_utilization': {'threshold': 0.8, 'current': 0.65, 'status': 'healthy'}
    }
}
```

---

## 8. Security and Compliance

### 8.1 API Security

**Security Implementation**:
```python
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends, HTTPException, status
import jwt

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=["HS256"])
        username = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )
        return username
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )

@app.post("/api/v1/predict")
async def secure_prediction(
    request: PredictionRequest,
    current_user: str = Depends(verify_token)
):
    # Rate limiting
    if not check_rate_limit(current_user):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    # Authorization check
    if not has_warehouse_access(current_user, request.warehouse_id):
        raise HTTPException(status_code=403, detail="Access denied")

    return await get_prediction(request)
```

### 8.2 Data Privacy and Compliance

**Privacy Controls**:
```python
PRIVACY_CONTROLS = {
    'data_encryption': {
        'at_rest': 'AES-256 encryption for all stored data',
        'in_transit': 'TLS 1.3 for all API communications',
        'key_management': 'AWS KMS for encryption key rotation'
    },
    'access_controls': {
        'authentication': 'JWT tokens with 1-hour expiration',
        'authorization': 'Role-based access control (RBAC)',
        'audit_logging': 'All API calls logged with user context'
    },
    'data_retention': {
        'prediction_data': '2 years retention policy',
        'audit_logs': '7 years retention for compliance',
        'personal_data': 'Automatic deletion after retention period'
    }
}
```

---

## 9. Deployment and DevOps

### 9.1 CI/CD Pipeline

**Deployment Pipeline**:
```yaml
# .github/workflows/deploy.yml
name: Deploy Forecasting Pipeline

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run Tests
        run: |
          python -m pytest tests/
          python -m pytest tests/integration/

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Build Docker Images
        run: |
          docker build -t forecasting-api:${{ github.sha }} .
          docker build -t batch-processor:${{ github.sha }} ./batch/

  deploy-staging:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to Staging
        run: |
          kubectl set image deployment/forecasting-api forecasting-api=forecasting-api:${{ github.sha }}
          kubectl rollout status deployment/forecasting-api

  deploy-production:
    needs: deploy-staging
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Deploy to Production
        run: |
          kubectl set image deployment/forecasting-api forecasting-api=forecasting-api:${{ github.sha }}
          kubectl rollout status deployment/forecasting-api
```

### 9.2 Infrastructure as Code

**Terraform Configuration**:
```hcl
# infrastructure/main.tf
resource "aws_eks_cluster" "forecasting_cluster" {
  name     = "forecasting-cluster"
  role_arn = aws_iam_role.cluster_role.arn
  version  = "1.27"

  vpc_config {
    subnet_ids = var.subnet_ids
  }
}

resource "aws_emr_cluster" "batch_processing" {
  name          = "forecasting-batch-cluster"
  release_label = "emr-6.15.0"
  applications  = ["Spark", "Hadoop"]

  ec2_attributes {
    instance_profile = aws_iam_instance_profile.emr_profile.arn
  }

  master_instance_group {
    instance_type = "m5.xlarge"
  }

  core_instance_group {
    instance_type  = "m5.2xlarge"
    instance_count = 5

    ebs_config {
      size = 100
      type = "gp2"
    }
  }
}
```

---

## 10. Performance Optimization

### 10.1 Optimization Strategies

**Real-time Optimizations**:
- **Model Quantization**: Reduced model size by 40% with <1% accuracy loss
- **Feature Caching**: 78% cache hit rate reducing feature computation time
- **Connection Pooling**: 60% reduction in database connection overhead
- **Async Processing**: 35% improvement in concurrent request handling

**Batch Optimizations**:
- **Data Partitioning**: Optimized Spark partitions for 25% faster processing
- **Columnar Storage**: Parquet format reducing I/O by 50%
- **Predicate Pushdown**: SQL optimization reducing data scanning by 70%
- **Resource Tuning**: Dynamic executor allocation improving cluster utilization

### 10.2 Cost Optimization

**Cost Management**:
```python
COST_OPTIMIZATION = {
    'realtime_costs': {
        'compute_cost_per_prediction': '$0.00008',
        'monthly_api_costs': '$2,400',
        'optimization_savings': '35% through caching and efficiency'
    },
    'batch_costs': {
        'compute_cost_per_prediction': '$0.00003',
        'monthly_batch_costs': '$1,800',
        'optimization_savings': '45% through spot instances and scheduling'
    },
    'total_monthly_savings': '$1,470 (35% reduction from baseline)'
}
```

---

## 11. Future Enhancements

### 11.1 Short-term Roadmap (Q3-Q4 2025)

- [ ] **Stream Processing**: Apache Kafka for real-time data ingestion
- [ ] **Edge Computing**: Deploy lightweight models at warehouse locations
- [ ] **Advanced Caching**: Intelligent cache warming and eviction policies
- [ ] **GPU Acceleration**: CUDA-enabled inference for complex models

### 11.2 Long-term Vision (2026)

- [ ] **Serverless Architecture**: AWS Lambda for cost-effective scaling
- [ ] **Multi-region Deployment**: Global load balancing and data replication
- [ ] **AI-powered Optimization**: Automated performance tuning
- [ ] **Quantum-ready Infrastructure**: Preparation for quantum computing integration

---

## 12. Conclusion

### 12.1 Pipeline Benefits

**Achieved Outcomes**:
- **Dual-mode Processing**: Seamless real-time and batch prediction capabilities
- **High Performance**: <180ms real-time latency, 50K predictions/minute batch
- **Scalability**: Auto-scaling from 3 to 20 pods based on demand
- **Reliability**: 99.95% uptime with comprehensive error handling

### 12.2 Business Impact

**Operational Excellence**:
- **Response Time**: 85% faster than previous system
- **Processing Capacity**: 300% increase in daily prediction volume
- **Cost Efficiency**: 35% reduction in infrastructure costs
- **System Reliability**: 99.95% availability enabling business continuity

---

## 13. Appendices

### 13.1 Configuration Files

**API Config**: config/api_settings.yaml
**Batch Config**: config/spark_settings.conf
**Monitoring**: config/prometheus_rules.yml

### 13.2 Performance Reports

**Load Testing Results**: reports/performance_testing_Q2_2025.pdf
**Cost Analysis**: reports/infrastructure_costs_2025.xlsx

---

**Document Control**
- Next Review Date: October 2025
- Distribution: ML Engineering, Data Engineering, DevOps
- Classification: Internal Use Only
- Related Training: Pipeline Architecture Workshop
