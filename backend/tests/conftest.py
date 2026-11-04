from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine

from app.api.deps import get_db
from app.core.config import settings
from app.core.db import init_db
from app.main import app
from tests.utils.user import authentication_token_from_email
from tests.utils.utils import get_superuser_token_headers

# Unlike the template, tests use their own database (TEST_DATABASE_URL), never the
# development one: its tables are dropped and recreated on every run.
if settings.TEST_DATABASE_URL is None:
    raise RuntimeError("Set TEST_DATABASE_URL in .env (see .env.example)")
test_engine = create_engine(str(settings.TEST_DATABASE_URL), pool_pre_ping=True)


def override_get_db() -> Generator[Session]:
    with Session(test_engine) as session:
        yield session


@pytest.fixture(scope="session", autouse=True)
def db() -> Generator[Session]:
    SQLModel.metadata.drop_all(test_engine)
    SQLModel.metadata.create_all(test_engine)
    app.dependency_overrides[get_db] = override_get_db
    with Session(test_engine) as session:
        init_db(session)
        yield session
    app.dependency_overrides.clear()


@pytest.fixture(scope="session", autouse=True)
def storage_dir(tmp_path_factory: pytest.TempPathFactory) -> Generator[Path]:
    """Uploaded test videos go to a temporary folder, not storage/."""
    original = settings.STORAGE_DIR
    settings.STORAGE_DIR = tmp_path_factory.mktemp("storage")
    yield settings.STORAGE_DIR
    settings.STORAGE_DIR = original


@pytest.fixture(scope="module")
def client() -> Generator[TestClient]:
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def superuser_token_headers(client: TestClient) -> dict[str, str]:
    return get_superuser_token_headers(client)


@pytest.fixture(scope="module")
def normal_user_token_headers(client: TestClient, db: Session) -> dict[str, str]:
    return authentication_token_from_email(
        client=client, email=settings.EMAIL_TEST_USER, db=db
    )
