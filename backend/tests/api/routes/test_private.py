import subprocess
import uuid

from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app import crud
from app.core.config import settings
from app.models import User, Video


def test_create_user(client: TestClient, db: Session) -> None:
    r = client.post(
        f"{settings.API_V1_STR}/private/users/",
        json={
            "email": "pollo@listo.com",
            "password": "password123",
            "full_name": "Pollo Listo",
        },
    )

    assert r.status_code == 200

    data = r.json()

    user = db.exec(select(User).where(User.id == data["id"])).first()

    assert user
    assert user.email == "pollo@listo.com"
    assert user.full_name == "Pollo Listo"


def test_set_analysis_finishes_video_like_a_worker(
    client: TestClient, superuser_token_headers: dict[str, str], db: Session
) -> None:
    owner = crud.get_user_by_email(session=db, email=settings.FIRST_SUPERUSER)
    assert owner
    video = Video(
        owner_id=owner.id,
        filename="seeded.mp4",
        storage_path="x/original.mp4",
        status="uploaded",
        duration=12.0,
        fps=10.0,
    )
    db.add(video)
    db.commit()
    r = client.post(
        f"{settings.API_V1_STR}/private/videos/{video.id}/analysis",
        json={
            "rallies": [[2.0, 3.5], [8.0, 9.0]],
            "trajectory": {
                "fps": 10,
                "visible": [0, 1, 1],
                "x": [None, 0.5, 0.6],
                "y": [None, 0.4, 0.3],
            },
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "done"
    assert data["rally_count"] == 2
    assert data["play_seconds"] == 2.5

    rallies = client.get(
        f"{settings.API_V1_STR}/videos/{video.id}/rallies",
        headers=superuser_token_headers,
    ).json()
    assert [(x["start"], x["end"]) for x in rallies["data"]] == [(2.0, 3.5), (8.0, 9.0)]

    points = client.get(
        f"{settings.API_V1_STR}/videos/{video.id}/trajectory",
        params={"start": 0, "end": 1},
        headers=superuser_token_headers,
    ).json()["points"]
    assert [(p["visible"], p["x"]) for p in points] == [
        (False, None),
        (True, 0.5),
        (True, 0.6),
    ]


def test_set_analysis_makes_thumbnails(
    client: TestClient, superuser_token_headers: dict[str, str], db: Session
) -> None:
    owner = crud.get_user_by_email(session=db, email=settings.FIRST_SUPERUSER)
    assert owner
    video = Video(
        owner_id=owner.id,
        filename="thumbs.mp4",
        storage_path="pending",
        status="uploaded",
        duration=3.0,
        fps=30.0,
    )
    db.add(video)
    db.commit()
    folder = settings.STORAGE_DIR / str(video.id)
    folder.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", "testsrc=duration=3:size=320x180:rate=30", "-pix_fmt", "yuv420p",
         str(folder / "original.mp4")],
        check=True,
    )  # fmt: skip
    video.storage_path = f"{video.id}/original.mp4"
    db.add(video)
    db.commit()

    r = client.post(
        f"{settings.API_V1_STR}/private/videos/{video.id}/analysis",
        json={"rallies": [[0.5, 2.0]], "thumbnails": True},
    )
    assert r.status_code == 200
    urls = client.get(
        f"{settings.API_V1_STR}/videos/{video.id}/thumbnails-url",
        headers=superuser_token_headers,
    ).json()
    index = client.get(urls["index_url"]).json()
    assert index["count"] == 3
    assert client.get(urls["sheet_urls"][0]).status_code == 200


def test_set_analysis_unknown_video(client: TestClient) -> None:
    r = client.post(
        f"{settings.API_V1_STR}/private/videos/{uuid.uuid4()}/analysis",
        json={"rallies": []},
    )
    assert r.status_code == 404
