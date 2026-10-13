from sqlalchemy.orm import Session
from .models import Anomaly
from .schemas import AnomalyRequest
import uuid
from datetime import datetime
from sqlalchemy import and_
from typing import Optional


def create_anomaly(db: Session, anomaly: AnomalyRequest):
    db_anomaly = Anomaly(
        id=str(uuid.uuid4()),
        cluster_name=anomaly.cluster_name,
        pod_name=anomaly.pod_name,
        app_name=anomaly.app_name,
        timestamp=anomaly.timestamp,
        cpu_usage=anomaly.cpu_usage,
        memory_usage=anomaly.memory_usage,
        is_anomaly=anomaly.is_anomaly
    )
    db.add(db_anomaly)
    db.commit()
    db.refresh(db_anomaly)
    return db_anomaly

def get_all_anomalies(db: Session):
    return db.query(Anomaly).order_by(Anomaly.timestamp.desc()).all()

def get_all_anomalies_sorted(db: Session, skip: int = 0, limit: int = 100):
    return (
        db.query(Anomaly)
        .order_by(Anomaly.timestamp.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )

def get_anomalies_filtered(
    db: Session,
    start_dt: datetime,
    end_dt: Optional[datetime],
    cluster_name: Optional[str],
    pod_name: Optional[str],
    app_name: Optional[str],
    skip: int,
    limit: int
):
    query = db.query(Anomaly).filter(Anomaly.timestamp >= start_dt)

    if end_dt:
        query = query.filter(Anomaly.timestamp <= end_dt)
    if cluster_name:
        query = query.filter(Anomaly.cluster_name == cluster_name)
    if pod_name:
        query = query.filter(Anomaly.pod_name == pod_name)
    if app_name:
        query = query.filter(Anomaly.app_name == app_name)

    return query.offset(skip).limit(limit).all()



def get_anomalies_dates_range(
    db: Session,
    start_date: datetime,
    end_date: datetime,
    skip: int = 0,
    limit: int = 100
):
    return (
        db.query(Anomaly)
        .filter(
            and_(
                Anomaly.timestamp >= start_date,
                Anomaly.timestamp <= end_date
            )
        )
        .order_by(Anomaly.timestamp.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )

def get_anomalies_filtered(
    db: Session,
    start_date: datetime,
    end_date: datetime,
    cluster_name: str = None,
    pod_name: str = None,
    app_name: str = None,
    skip: int = 0,
    limit: int = 100
):
    filters = [
        Anomaly.timestamp >= start_date,
        Anomaly.timestamp <= end_date,
    ]
    if cluster_name:
        filters.append(Anomaly.cluster_name == cluster_name)
    if pod_name:
        filters.append(Anomaly.pod_name == pod_name)
    if app_name:
        filters.append(Anomaly.app_name == app_name)

    return (
        db.query(Anomaly)
        .filter(and_(*filters))
        .order_by(Anomaly.timestamp.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
