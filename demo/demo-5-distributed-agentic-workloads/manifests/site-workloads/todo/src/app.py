"""Minimal todo API with OpenTelemetry for demo-5 site workloads."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import Boolean, Integer, String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from telemetry import instrument_app, setup_telemetry

_log = logging.getLogger("todo")


class Base(DeclarativeBase):
    pass


class TodoRow(Base):
    __tablename__ = "todos"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(256), nullable=False)
    completed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    order: Mapped[int] = mapped_column(Integer, default=0, nullable=False)


class TodoCreate(BaseModel):
    title: str = Field(min_length=1, max_length=256)
    completed: bool = False
    order: int = 0


class TodoUpdate(BaseModel):
    title: Optional[str] = Field(default=None, min_length=1, max_length=256)
    completed: Optional[bool] = None
    order: Optional[int] = None


class TodoOut(BaseModel):
    id: int
    title: str
    completed: bool
    order: int

    model_config = {"from_attributes": True}


def _database_url() -> str:
    """Prefer discrete POSTGRES_* env (from K8s Secret); optional DATABASE_URL for local runs."""
    if url := os.getenv("DATABASE_URL"):
        return url
    user = os.getenv("POSTGRES_USER", "todo")
    password = os.getenv("POSTGRES_PASSWORD", "todo")
    host = os.getenv("POSTGRES_HOST", "postgresql")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "todo")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"


engine = create_engine(_database_url(), pool_pre_ping=True, pool_size=5, max_overflow=10)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_telemetry()
    instrument_app(app, engine)
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(title="demo-5-todo", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready(db: Session = Depends(get_db)):
    try:
        db.execute(select(1))
    except Exception as exc:  # noqa: BLE001 — surface readiness failure
        raise HTTPException(status_code=503, detail=f"database unavailable: {exc}") from exc
    return {"status": "ready"}


@app.get("/api", response_model=list[TodoOut])
@app.get("/api/", response_model=list[TodoOut])
def list_todos(db: Session = Depends(get_db)):
    rows = db.scalars(select(TodoRow).order_by(TodoRow.order, TodoRow.id)).all()
    _log.info("list_todos count=%s", len(rows))
    return rows


@app.post("/api", response_model=TodoOut, status_code=201)
@app.post("/api/", response_model=TodoOut, status_code=201)
def create_todo(body: TodoCreate, db: Session = Depends(get_db)):
    row = TodoRow(title=body.title, completed=body.completed, order=body.order)
    db.add(row)
    db.commit()
    db.refresh(row)
    _log.info("create_todo id=%s title=%s", row.id, row.title)
    return row


@app.get("/api/{todo_id}", response_model=TodoOut)
def get_todo(todo_id: int, db: Session = Depends(get_db)):
    row = db.get(TodoRow, todo_id)
    if row is None:
        raise HTTPException(status_code=404, detail="todo not found")
    return row


@app.patch("/api/{todo_id}", response_model=TodoOut)
def update_todo(todo_id: int, body: TodoUpdate, db: Session = Depends(get_db)):
    row = db.get(TodoRow, todo_id)
    if row is None:
        raise HTTPException(status_code=404, detail="todo not found")
    if body.title is not None:
        row.title = body.title
    if body.completed is not None:
        row.completed = body.completed
    if body.order is not None:
        row.order = body.order
    db.commit()
    db.refresh(row)
    return row


@app.delete("/api/{todo_id}", status_code=204)
def delete_todo(todo_id: int, db: Session = Depends(get_db)):
    row = db.get(TodoRow, todo_id)
    if row is None:
        raise HTTPException(status_code=404, detail="todo not found")
    db.delete(row)
    db.commit()
    return None


@app.get("/")
def root():
    return {
        "service": "demo-5-todo",
        "endpoints": ["/health", "/ready", "/api"],
    }
