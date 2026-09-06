"""
Persistence — stage 2 of DOCKER.md.

Stores enriched products and batch jobs in Postgres. Every entry point works
when there is no database: `Repository.available` is False and the calls become
no-ops returning None. The application must run stateless on Streamlit Cloud
exactly as it does today, so persistence is an addition, never a requirement.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional

from sqlalchemy import (
    JSON, Column, DateTime, Float, Integer, String, Text, create_engine, select,
)
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from .config import Settings, get_settings

Base = declarative_base()


def content_hash(name: str, description: str = "", attributes: Optional[Dict] = None) -> str:
    """
    Stable identity for a raw product.

    Used both as the cache key and to detect re-ingestion of the same record.
    Sorted keys so that attribute ordering cannot change the hash.
    """
    payload = json.dumps(
        {"name": name.strip(), "description": (description or "").strip(),
         "attributes": attributes or {}},
        sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ProductRecord(Base):
    """One enriched product, with enough provenance to audit it later."""

    __tablename__ = "products"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    content_hash = Column(String(64), index=True, nullable=False)
    name = Column(Text, nullable=False)
    description = Column(Text, default="")
    category = Column(String(200), index=True)

    result = Column(JSON, nullable=False)          # the full pipeline output
    spec_count = Column(Integer, default=0)
    validation_score = Column(Float, default=0.0)
    completeness = Column(Float, default=0.0)
    llm_used = Column(Integer, default=0)          # 0/1, kept int for portability

    job_id = Column(String(36), index=True, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_now, nullable=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content_hash": self.content_hash,
            "name": self.name,
            "category": self.category,
            "spec_count": self.spec_count,
            "validation_score": self.validation_score,
            "completeness": self.completeness,
            "llm_used": bool(self.llm_used),
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class JobRecord(Base):
    """A catalogue run. Created queued, moved to running, then done or failed."""

    __tablename__ = "jobs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    status = Column(String(20), default="queued", index=True, nullable=False)
    total = Column(Integer, default=0)
    processed = Column(Integer, default=0)
    error = Column(Text, nullable=True)
    summary = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_now, nullable=False)
    finished_at = Column(DateTime(timezone=True), nullable=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "total": self.total,
            "processed": self.processed,
            "error": self.error,
            "summary": self.summary,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
        }


class Repository:
    """
    Thin data-access layer.

    Construct it with no database URL and every method still works — it simply
    stores nothing and returns None. That is the whole point: the caller never
    needs to know whether persistence is configured.
    """

    def __init__(self, settings: Optional[Settings] = None, echo: bool = False) -> None:
        self.settings = settings or get_settings()
        self._engine = None
        self._session_factory = None

        if self.settings.persistence_enabled:
            try:
                self._engine = create_engine(
                    self.settings.database_url, echo=echo, pool_pre_ping=True, future=True
                )
                self._session_factory = sessionmaker(bind=self._engine, future=True)
                self._verify_connection()
            except Exception:
                # Bad URL, missing driver, or nothing listening: stay stateless
                # rather than crash. This is what makes stage 1 of the Compose
                # stack work while DATABASE_URL still points at a `db` service
                # that has not been started yet.
                self._engine = None
                self._session_factory = None

    def _verify_connection(self) -> None:
        """
        Connect once at construction.

        Without this, `available` would be True for any syntactically valid URL
        and the failure would surface later, on the first write, as a crash in
        the middle of a user's request instead of a clean stateless start.

        The trade-off is deliberate: a database that comes up *after* the
        application will not be picked up until the process restarts. Under
        Compose that is handled by `depends_on: condition: service_healthy`.
        """
        from sqlalchemy import text

        session = self._session_factory()
        try:
            session.execute(text("SELECT 1"))
        finally:
            session.close()

    @property
    def available(self) -> bool:
        return self._session_factory is not None

    def ensure_schema(self) -> bool:
        """
        Create the tables if they are missing. Idempotent and safe to call on
        every start.

        Without this, a freshly provisioned database means the first write
        fails with `relation "products" does not exist` — a 500 at runtime
        rather than a clear failure at startup. `create_all` inspects the
        database first and issues nothing when the tables already exist.

        This is deliberately not a migration system. It creates tables that are
        absent; it will not alter a table whose columns have changed. Once the
        schema starts evolving, introduce Alembic rather than extending this.
        """
        if not self.available:
            return False
        try:
            Base.metadata.create_all(self._engine)
            return True
        except Exception:
            return False

    def create_schema(self) -> bool:
        if not self.available:
            return False
        Base.metadata.create_all(self._engine)
        return True

    def drop_schema(self) -> bool:
        if not self.available:
            return False
        Base.metadata.drop_all(self._engine)
        return True

    @contextmanager
    def session(self) -> Iterator[Optional[Session]]:
        if not self.available:
            yield None
            return
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def ping(self) -> bool:
        if not self.available:
            return False
        try:
            from sqlalchemy import text
            with self.session() as s:
                s.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    # -- writes ------------------------------------------------------------

    def save_product(self, result: Dict[str, Any], job_id: Optional[str] = None) -> Optional[str]:
        """Persist one pipeline result. Returns the row id, or None if stateless."""
        if not self.available:
            return None

        name = result.get("name") or ""
        source = result.get("source") or {}
        description = source.get("description") or "" if isinstance(source, dict) else ""
        specs = result.get("specifications", {}) or {}
        validation = result.get("validation", {}) or {}

        category = result.get("category")
        if isinstance(category, dict):
            category = category.get("value")

        # The engine already computes a stable fingerprint; prefer it, and fall
        # back to hashing the raw fields when processing a result that lacks one.
        digest = result.get("fingerprint") or content_hash(name, description)

        record = ProductRecord(
            content_hash=str(digest),
            name=name,
            description=description,
            category=category,
            result=result,
            spec_count=len(specs),
            validation_score=float(validation.get("score") or 0.0),
            completeness=float(validation.get("completeness") or 0.0),
            llm_used=1 if any(
                (fv or {}).get("origin") == "llm" for fv in specs.values()
            ) else 0,
            job_id=job_id,
        )
        with self.session() as s:
            s.add(record)
            s.flush()
            return record.id

    def save_products(self, results: List[Dict[str, Any]], job_id: Optional[str] = None) -> List[str]:
        return [rid for rid in (self.save_product(r, job_id) for r in results) if rid]

    def create_job(self, total: int) -> Optional[str]:
        if not self.available:
            return None
        job = JobRecord(status="queued", total=total)
        with self.session() as s:
            s.add(job)
            s.flush()
            return job.id

    def update_job(self, job_id: str, **fields: Any) -> bool:
        if not self.available:
            return False
        with self.session() as s:
            job = s.get(JobRecord, job_id)
            if job is None:
                return False
            for key, value in fields.items():
                setattr(job, key, value)
            if fields.get("status") in ("done", "failed"):
                job.finished_at = _now()
            return True

    # -- reads -------------------------------------------------------------

    def get_product(self, product_id: str) -> Optional[Dict[str, Any]]:
        if not self.available:
            return None
        with self.session() as s:
            row = s.get(ProductRecord, product_id)
            return row.result if row else None

    def find_by_hash(self, digest: str) -> Optional[Dict[str, Any]]:
        if not self.available:
            return None
        with self.session() as s:
            row = s.execute(
                select(ProductRecord)
                .where(ProductRecord.content_hash == digest)
                .order_by(ProductRecord.created_at.desc())
                .limit(1)
            ).scalar_one_or_none()
            return row.result if row else None

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        if not self.available:
            return None
        with self.session() as s:
            row = s.get(JobRecord, job_id)
            return row.to_dict() if row else None

    def recent_products(self, limit: int = 20) -> List[Dict[str, Any]]:
        if not self.available:
            return []
        with self.session() as s:
            rows = s.execute(
                select(ProductRecord).order_by(ProductRecord.created_at.desc()).limit(limit)
            ).scalars().all()
            return [r.to_dict() for r in rows]

    def products_for_job(self, job_id: str) -> List[Dict[str, Any]]:
        if not self.available:
            return []
        with self.session() as s:
            rows = s.execute(
                select(ProductRecord).where(ProductRecord.job_id == job_id)
            ).scalars().all()
            return [r.result for r in rows]

    def count_products(self) -> int:
        if not self.available:
            return 0
        with self.session() as s:
            return len(s.execute(select(ProductRecord.id)).scalars().all())
