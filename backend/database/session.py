"""Engine creation, initialization, and session management."""

import logging
import os
import uuid

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .. import config
from .models import (
    Base,
    AudioChannel,
    EffectPreset,
    Generation,
    GenerationVersion,
    ProfileChannelMapping,
    VoiceProfile,
)
from .migrations import run_migrations
from .seed import backfill_generation_versions, seed_builtin_presets

logger = logging.getLogger(__name__)

# Initialized by init_db()
engine = None
SessionLocal = None
_db_path = None


def _create_turso_engine(turso_url: str):
    """Connect to Turso via libsql-experimental. Returns SQLAlchemy engine or None on failure."""
    try:
        import libsql_experimental as libsql  # noqa: PLC0415
        auth_token = os.environ.get("TURSO_AUTH_TOKEN", "")
        logger.info("Using Turso DB: %s", turso_url)
        sync_db_path = config.get_data_dir() / "voicebox_sync.db"
        sync_db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = libsql.connect(str(sync_db_path), sync_url=turso_url, auth_token=auth_token)
        conn.sync()
        return create_engine("sqlite://", creator=lambda: conn)
    except Exception:
        logger.exception("Turso connection failed — falling back to local SQLite")
        return None


def _create_sqlite_engine():
    """Create a local SQLite engine. Sets the module-level _db_path."""
    global _db_path
    _db_path = config.get_db_path()
    _db_path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{_db_path}", connect_args={"check_same_thread": False})


def _seed_default_channel(session_factory) -> None:
    """Create the default audio channel and map existing profiles to it if missing."""
    db = session_factory()
    try:
        if db.query(AudioChannel).filter(AudioChannel.is_default == True).first():
            return
        channel = AudioChannel(id=str(uuid.uuid4()), name="Default", is_default=True)
        db.add(channel)
        for profile in db.query(VoiceProfile).all():
            db.add(ProfileChannelMapping(profile_id=profile.id, channel_id=channel.id))
        db.commit()
    finally:
        db.close()


def init_db() -> None:
    """Initialize the database engine, run migrations, create tables, and seed data."""
    global engine, SessionLocal

    turso_url = os.environ.get("TURSO_DATABASE_URL")
    engine = (_create_turso_engine(turso_url) if turso_url else None) or _create_sqlite_engine()

    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    run_migrations(engine)
    Base.metadata.create_all(bind=engine)
    _seed_default_channel(SessionLocal)
    backfill_generation_versions(SessionLocal, Generation, GenerationVersion)
    seed_builtin_presets(SessionLocal, EffectPreset)


def get_db():
    """Yield a database session (FastAPI dependency)."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
