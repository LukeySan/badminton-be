from collections.abc import Generator
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, create_engine

from app.api.deps import get_db
from app.core.config import settings
from app.main import app
from tests.crud.test_video_jobs import (  # noqa: F401  (empty_videos is a fixture)
    empty_videos,
    make_video,
)

URL = f"{settings.API_V1_STR}/health"


def test_health_without_login(client: TestClient) -> None:
    r = client.get(URL)
    assert r.status_code == 200
    assert r.json() == {
        "status": "ok",
        "database": "ok",
        "queued": 0,
        "processing": 0,
        "oldest_queued_seconds": None,
    }


def test_health_reports_the_queue(client: TestClient, db: Session) -> None:
    make_video(db, status="queued", age=timedelta(minutes=5))
    make_video(db, status="queued", age=timedelta(seconds=10))
    make_video(db, status="processing", age=timedelta(minutes=30))  # not "queued"
    make_video(db, status="done", age=timedelta(hours=2))

    body = client.get(URL).json()
    assert (body["queued"], body["processing"]) == (2, 1)
    assert body["oldest_queued_seconds"] == pytest.approx(300, abs=5)


def test_health_when_the_database_is_down(client: TestClient) -> None:
    # Nothing listens on port 1: connecting fails at once
    dead = create_engine(
        "postgresql+psycopg://postgres:x@localhost:1/app",
        connect_args={"connect_timeout": 2},
    )

    def dead_db() -> Generator[Session]:
        with Session(dead) as session:
            yield session

    working = app.dependency_overrides[get_db]
    app.dependency_overrides[get_db] = dead_db
    try:
        r = client.get(URL)
    finally:
        app.dependency_overrides[get_db] = working
    assert r.status_code == 503
    assert r.json()["status"] == "error"
    assert r.json()["database"] == "unreachable"
