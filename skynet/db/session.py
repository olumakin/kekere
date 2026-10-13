from contextlib import contextmanager
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from skynet.config import config
from skynet.db.schema import Base

def get_engine(url: str = None, echo: bool = None):
    db_url = url or config.db_url
    db_echo = echo if echo is not None else config.db_echo
    return create_engine(db_url, echo=db_echo, future=True)

_engine = None
_SessionFactory = None

def init_db(engine=None) -> None:
    global _engine, _SessionFactory
    if engine is None:
        _engine = get_engine()
    else:
        _engine = engine
    _SessionFactory = sessionmaker(bind=_engine, expire_on_commit=False, future=True)
    Base.metadata.create_all(bind=_engine)

def get_session_factory():
    global _SessionFactory
    if _SessionFactory is None:
        init_db()
    return _SessionFactory

@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    factory = get_session_factory()
    session: Session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
