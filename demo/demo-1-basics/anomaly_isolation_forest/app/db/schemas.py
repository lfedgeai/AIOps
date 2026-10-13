from pydantic import BaseModel
from typing import List
from datetime import datetime
from typing import Optional

class AnomalyRequest(BaseModel):
    cluster_name: str
    pod_name: str
    app_name: str
    cpu_usage: float
    memory_usage: float
    timestamp: datetime
    is_anomaly: Optional[str] = None

class BulkAnomalyRequest(BaseModel):
    data: List[AnomalyRequest]