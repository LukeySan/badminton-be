import threading
import uuid
from datetime import datetime

from sqlalchemy import Engine
from sqlmodel import Session, col, select

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


def clip(rally_id: int | None, start: float, end: float) -> dict[str, object]:
    return {
        "rally_id": rally_id,
        "trim_start": start,
        "trim_end": end,
        "slow_mo": False,
    }


def test_relink_clips_picks_the_rally_overlapping_most() -> None:
    rallies = [(1, 10.0, 20.0), (2, 30.0, 45.0)]
    edit = {
        "version": 1,
        "view": "cut",
        "clips": [
            clip(7, 9.0, 21.0),  # covers rally 1
            {**clip(8, 29.0, 46.0), "slow_mo": True},  # rally 2; slow_mo kept
            clip(9, 18.0, 31.0),  # 2 s of rally 1, 1 s of rally 2
            clip(10, 50.0, 55.0),  # no rally there any more
            clip(None, 20.0, 30.0),  # touches both, overlaps neither
        ],
    }
    relinked = crud.relink_clips(edit, rallies)
    relinked_ids = [c["rally_id"] for c in relinked["clips"]]
    assert relinked_ids == [1, 2, 1, None, None]
    assert relinked["clips"][1]["slow_mo"] is True
    assert relinked["view"] == "cut"
    assert edit["clips"][0]["rally_id"] == 7  # the input isn't changed


def test_reanalysis_relinks_reel_clips(db: Session, engine: Engine) -> None:
    """Re-analyzing replaces the rallies (new ids); reels must keep working."""
    video, claim_id, old_ids = make_processing_video(db, [(10.0, 20.0), (30.0, 45.0)])
    edit = ReelEdit.model_validate(
        {"clips": [clip(old_ids[1], 29.0, 46.0), clip(old_ids[0], 9.0, 21.0)]}
    )
    reel = crud.create_reel(session=db, video=video, title="r", edit=edit)
    deleted = crud.create_reel(session=db, video=video, title="d", edit=edit)
    crud.soft_delete_reel(session=db, reel_id=deleted.id)
    empty = crud.create_reel(session=db, video=video, title="e", edit=ReelEdit())
    saved_at = {r.id: r.updated_at for r in (reel, deleted, empty)}

    with Session(engine) as session:
        assert crud.finish_video(
            session=session,
            video_id=video.id,
            claim_id=claim_id,
            rallies=[(10.5, 19.0), (31.0, 44.0), (60.0, 70.0)],
        )

    new_ids = {
        (r.start, r.end): r.id
        for r in db.exec(select(Rally).where(col(Rally.video_id) == video.id)).all()
    }
    assert not set(old_ids) & set(new_ids.values())
    db.expire_all()
    relinked = db.get(Reel, reel.id)
    assert relinked
    assert [c["rally_id"] for c in relinked.edit["clips"]] == [
        new_ids[(31.0, 44.0)],
        new_ids[(10.5, 19.0)],
    ]
    assert [c["trim_start"] for c in relinked.edit["clips"]] == [29.0, 9.0]  # unchanged
    # The relinked reel counts as changed (an editor that had it open must reload);
    # untouched ones keep their updated_at
    assert relinked.updated_at > saved_at[reel.id]
    still_deleted, untouched = db.get(Reel, deleted.id), db.get(Reel, empty.id)
    assert still_deleted and still_deleted.edit["clips"][0]["rally_id"] == old_ids[1]
    assert untouched and untouched.updated_at == saved_at[empty.id]
    # ...and the relinked reel passes validation again
    assert not ReelEdit.model_validate(relinked.edit).problems(
        duration=120.0, rally_ids=set(new_ids.values())
    )


def test_simultaneous_saves_only_one_wins(db: Session, engine: Engine) -> None:
    """Six saves from the same loaded version at once: the conditional UPDATE lets
    exactly one through, the rest get "changed since" (409 in the API)."""
    video, _, _ = make_processing_video(db, [])
    reel = crud.create_reel(session=db, video=video, title="r", edit=ReelEdit())
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
