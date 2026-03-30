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


def init_db() -> None:
    """Initialize the database engine, run migrations, create tables, and seed data."""
    global engine, SessionLocal, _db_path

    turso_url = os.environ.get("TURSO_DATABASE_URL")

    if turso_url:
        try:
            import libsql_experimental as libsql  # noqa: PLC0415
            auth_token = os.environ.get("TURSO_AUTH_TOKEN", "")
            logger.info("Using Turso DB: %s", turso_url)
            # libsql-experimental uses its own connection, not a SQLAlchemy URL
            _libsql_conn = libsql.connect("/app/data/voicebox_sync.db", sync_url=turso_url, auth_token=auth_token)
            _libsql_conn.sync()
            engine = create_engine("sqlite://", creator=lambda: _libsql_conn)
        except Exception:
            logger.exception("Turso connection failed — falling back to local SQLite")
            turso_url = None  # fall through to SQLite below

    if not turso_url:
        _db_path = config.get_db_path()
        _db_path.parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(
            f"sqlite:///{_db_path}",
            connect_args={"check_same_thread": False},
        )

    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    run_migrations(engine)
    Base.metadata.create_all(bind=engine)

    # Create default audio channel if it doesn't exist
    db = SessionLocal()
    try:
        default_channel = db.query(AudioChannel).filter(AudioChannel.is_default == True).first()
        if not default_channel:
            default_channel = AudioChannel(
                id=str(uuid.uuid4()),
                name="Default",
                is_default=True,
            )
            db.add(default_channel)

            for profile in db.query(VoiceProfile).all():
                db.add(ProfileChannelMapping(
                    profile_id=profile.id,
                    channel_id=default_channel.id,
                ))
            db.commit()
    finally:
        db.close()

    backfill_generation_versions(SessionLocal, Generation, GenerationVersion)
    seed_builtin_presets(SessionLocal, EffectPreset)


def get_db():
    """Yield a database session (FastAPI dependency)."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
