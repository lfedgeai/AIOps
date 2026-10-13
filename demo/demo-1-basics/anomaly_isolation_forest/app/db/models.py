from sqlalchemy import Column, String, Float, TIMESTAMP
from .db import Base

class Anomaly(Base):
    __tablename__ = "anomalies"

    id = Column(String, primary_key=True)
    cluster_name = Column(String, index=True)
    pod_name = Column(String)
    app_name = Column(String)
    timestamp = Column(TIMESTAMP)
    cpu_usage = Column(Float)
    memory_usage = Column(Float)
    is_anomaly = Column(String)
