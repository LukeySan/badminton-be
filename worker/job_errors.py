"""
Problems with the input video (not bugs). The pipeline raises these with technical
details; run_worker.py turns each type into a user-facing FailureReason
(app.models). Any other exception is treated as a bug: generic reason for the user,
full traceback in the worker log.
"""


class PipelineError(Exception):
    """Base class: a problem with the video or how it was marked."""


class VideoUnreadable(PipelineError):
    """ffmpeg/OpenCV can't decode the video."""


class CourtNotVisible(PipelineError):
    """The shuttle was (almost) never seen inside the marked court: wrong corners,
    or the court isn't really in view."""
