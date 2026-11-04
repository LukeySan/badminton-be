from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

Fraction01 = Annotated[float, Field(ge=0, le=1)]
Point = tuple[Fraction01, Fraction01]
Status = Literal["uploaded", "queued", "processing", "done", "failed"]


class Court(BaseModel):
    corners: list[Point] = Field(
        min_length=4, max_length=4,
        description="Fractions 0-1 of width/height, order: far-left, far-right, near-right, near-left",
        examples=[[[0.478, 0.347], [0.777, 0.369], [0.930, 0.943], [0.054, 0.781]]],
    )

    @model_validator(mode="after")
    def check_order(self):
        # Wrong tap order is the likely user mistake, and it gives bad rallies without
        # any error later, so catch it here. With the camera behind the baseline the far
        # baseline is higher in the image (smaller y) than the near one.
        fl, fr, nr, nl = self.corners
        if max(fl[1], fr[1]) >= min(nr[1], nl[1]):
            raise ValueError("far corners must be above the near corners; tap far-left, far-right, "
                             "near-right, near-left")
        if fl[0] >= fr[0] or nl[0] >= nr[0]:
            raise ValueError("left corners must be left of the right corners")
        return self


class VideoOut(BaseModel):
    id: str
    filename: str
    status: Status
    progress: float
    error: str | None
    duration: float | None
    fps: float | None
    width: int | None
    height: int | None
    court_corners: list[Point] | None
    created_at: str
    updated_at: str


class RallyOut(BaseModel):
    id: int
    start: float
    end: float
    duration: float
