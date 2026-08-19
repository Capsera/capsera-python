"""Single source of truth for the SDK version.

Hatchling reads this at build time (``[tool.hatch.version]`` in
pyproject.toml), and the runtime User-Agent strings import it, so the
version cannot drift between the wheel filename, package metadata, and
request headers. Bump it here and rebuild the wheel
(``python scripts/build_sdk_wheel.py``); the backend's
``_WHEEL_FILENAME_FALLBACK`` in ``instrument/runner.py`` must follow — a
test asserts it matches the staged wheel.
"""
__version__ = "0.6.1"
