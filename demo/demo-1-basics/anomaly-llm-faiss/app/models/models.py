from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Dict

class AnomalyData(BaseModel):
    cluster_name: str
    pod_name: str
    app_name: str
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    is_anomaly: str
    anomaly_type: Optional[str] = None  # Make optional
    description: Optional[str] = None   # Make optional
    resolution: Optional[str] = None    # Make optional
