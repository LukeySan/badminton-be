#!/usr/bin/env bash

set -e
set -x

# macOS: Pillow loads FriBiDi (for Thai captions) from Homebrew's lib folder only
# if it's on the library path (see worker/caption_image.py)
if [ "$(uname)" = "Darwin" ] && [ -z "${DYLD_FALLBACK_LIBRARY_PATH:-}" ]; then
    export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib:/usr/local/lib:/usr/lib
fi

FASTAPI_ENV=development coverage run -m pytest tests/
coverage report
coverage html --title "${@-coverage}"
