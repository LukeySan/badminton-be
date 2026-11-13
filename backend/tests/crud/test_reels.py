import threading
import uuid
from datetime import datetime

from sqlalchemy import Engine
from sqlmodel import Session, col, select, update

from app import crud
from app.core.config import settings
from app.models import Rally, Reel, ReelEdit, Video


def make_processing_video(
    db: Session, spans: list[tuple[float, float]]
) -> tuple[Video, uuid.UUID, list[int]]:
    """A video a worker is processing (claim_id set), with rallies from an earlier
    run. Returns (video, claim_id, rally ids)."""
    owner = crud.get_user_by_email(session=db, email=settings.FIRST_SUPERUSER)
    assert owner
    claim_id = uuid.uuid4()
    video = Video(
        owner_id=owner.id,
        filename="match.mp4",
        storage_path="x/original.mp4",
        status="processing",
        claim_id=claim_id,
        duration=120.0,
    )
    db.add(video)
    db.commit()
    rallies = [Rally(video_id=video.id, start=s, end=e) for s, e in spans]
    db.add_all(rallies)
    db.commit()
    return video, claim_id, [r.id for r in rallies if r.id is not None]


def clip(rally_id: int, *pieces: tuple[float, float]) -> dict[str, object]:
    return {
        "rally_id": rally_id,
        "pieces": [{"start": s, "end": e, "slow_mo": False} for s, e in pieces],
    }


def test_reanalysis_moves_reel_clips_to_the_new_rallies(
    db: Session, engine: Engine
) -> None:
    """Re-analyzing replaces the rallies (new ids, slightly different bounds): each
    clip moves to the new rally overlapping it most and is trimmed to its bounds;
    a clip whose rally is gone is dropped."""
    video, claim_id, old_ids = make_processing_video(
        db, [(10.0, 20.0), (30.0, 45.0), (60.0, 70.0)]
    )
    edit = ReelEdit.model_validate(
        {
            "clips": [
                clip(old_ids[1], (29.0, 35.0), (40.0, 46.0)),
                clip(old_ids[0], (9.0, 21.0)),
                clip(old_ids[2], (61.0, 65.0)),  # no rally there after re-analysis
            ]
        }
    )
    deleted = crud.create_reel(session=db, video=video, title="d", edit=edit)
    assert deleted
    crud.soft_delete_reel(session=db, reel_id=deleted.id)
    reel = crud.create_reel(session=db, video=video, title="r", edit=edit)
    assert reel
    saved_at = reel.updated_at

    with Session(engine) as session:
        assert crud.finish_video(
            session=session,
            video_id=video.id,
            claim_id=claim_id,
            rallies=[(10.5, 19.0), (31.0, 44.0)],
        )

    new = {
        (r.start, r.end): r
        for r in db.exec(select(Rally).where(col(Rally.video_id) == video.id)).all()
    }
    assert not set(old_ids) & {r.id for r in new.values()}
    # The detector's bounds are recorded with the rallies
    assert all((r.detected_start, r.detected_end) == key for key, r in new.items())
    db.expire_all()
    moved = db.get(Reel, reel.id)
    assert moved
    second, first = new[(31.0, 44.0)].id, new[(10.5, 19.0)].id
    assert (
        ReelEdit.model_validate(moved.edit).clips
        == ReelEdit.model_validate(
            {
                "clips": [
                    clip(second, (30.0, 35.0), (40.0, 45.0)),
                    clip(first, (9.5, 20.0)),
                ]
            }
        ).clips
    )
    # Moved reels count as changed (an open editor must reload)
    assert moved.updated_at > saved_at
    # Deleted reels are left as they were
    still_deleted = db.get(Reel, deleted.id)
    assert still_deleted and still_deleted.edit["clips"][0]["rally_id"] == old_ids[1]
    # ...and the moved reel passes validation again
    assert not ReelEdit.model_validate(moved.edit).problems(
        duration=120.0,
        rallies={r.id: (r.start, r.end) for r in new.values() if r.id is not None},
    )


def test_reanalysis_leaves_an_unaffected_reel_alone(
    db: Session, engine: Engine
) -> None:
    video, claim_id, _ = make_processing_video(db, [(10.0, 20.0)])
    reel = crud.create_reel(session=db, video=video, title="e", edit=ReelEdit())
    assert reel
    saved_at = reel.updated_at
    with Session(engine) as session:
        crud.finish_video(
            session=session, video_id=video.id, claim_id=claim_id, rallies=[(11, 19)]
        )
    db.expire_all()
    untouched = db.get(Reel, reel.id)
    assert untouched and untouched.updated_at == saved_at


def test_simultaneous_saves_only_one_wins(db: Session, engine: Engine) -> None:
    """Six saves from the same loaded version at once: the conditional UPDATE lets
    exactly one through, the rest get "changed since" (409 in the API)."""
    video, _, _ = make_processing_video(db, [])
    reel = crud.create_reel(session=db, video=video, title="r", edit=ReelEdit())
    assert reel
    loaded_at: datetime = reel.updated_at
    results: list[bool] = []
    barrier = threading.Barrier(6)

    def save(n: int) -> None:
        with Session(engine) as session:
            barrier.wait()
            results.append(
                crud.update_reel(
                    session=session,
                    reel_id=reel.id,
                    title=f"save {n}",
                    edit=ReelEdit(),
                    expected_updated_at=loaded_at,
                )
            )

    threads = [threading.Thread(target=save, args=(n,)) for n in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(results) == [False] * 5 + [True]


def test_simultaneous_corrections_are_serialized(db: Session, engine: Engine) -> None:
    """Four "add rally 21-25 s" at once: the video row lock makes them run one after
    the other, so only the first passes the overlap check (409 for the rest)."""
    video, _, _ = make_processing_video(db, [(10.0, 20.0)])
    db.exec(update(Video).where(col(Video.id) == video.id).values(status="done"))
    db.commit()
    owner_id = video.owner_id
    results: list[str] = []
    barrier = threading.Barrier(4)

    def add() -> None:
        with Session(engine) as session:
            barrier.wait()
            try:
                crud.add_rally(
                    session=session,
                    video_id=video.id,
                    user_id=owner_id,
                    start=21.0,
                    end=25.0,
                )
                results.append("added")
            except crud.CorrectionRefused as e:
                results.append(str(e.status_code))

    threads = [threading.Thread(target=add) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(results) == ["409", "409", "409", "added"]
    rallies = db.exec(select(Rally).where(col(Rally.video_id) == video.id)).all()
    assert len(rallies) == 2


def test_simultaneous_creates_make_one_reel(db: Session, engine: Engine) -> None:
    """Six "create the reel" saves at once (updated_at null): one creates it, the
    others are refused (409 in the API), and no request errors."""
    video_id = make_processing_video(db, [])[0].id
    results: list[bool] = []
    barrier = threading.Barrier(6)

    def create() -> None:
        with Session(engine) as session:
            video = session.get(Video, video_id)  # each thread its own session's copy
            assert video
            barrier.wait()
            reel, created = crud.save_video_reel(
                session=session,
                video=video,
                title="r",
                edit=ReelEdit(),
                expected_updated_at=None,
            )
            results.append(reel is not None and created)

    threads = [threading.Thread(target=create) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(results) == [False] * 5 + [True]
