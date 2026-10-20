"""
axol gst.install

Install the host-side GStreamer support the unified ZED camera pipeline needs
(``teleop --cameras`` / ``collect-data``): the GStreamer introspection typelibs
and **PyGObject**, so the in-process pipeline (:mod:`almond_axol.video.gst_zed`)
can pull headset access units from ``appsink`` and carry dataset access units
with their sensor PTS through GDP/shared memory. The NVENC elements
(``nvv4l2h264enc`` / ``nvvidconv``) come from the L4T BSP's
``nvidia-l4t-gstreamer`` package, which a minimal flash leaves out (e.g. an AGX
Thor on L4T 39.2), so this command installs it too — pinned to the running
BSP release (see :func:`_l4t_gstreamer_apt_spec`). The patched zed-gstreamer
source elements (``zedxonesrc`` / ``zedsrc``) are built separately by
``axol gst.build-zed``; this command only reports whether they are present.

PyGObject is not a usable PyPI wheel — it builds against the system
gobject-introspection and loads the system typelibs — so it can't be a normal
dependency. This command apt-installs the system packages and then builds
PyGObject into the interpreter running the CLI (the uv tool environment),
mirroring ``axol zed.install``. PyGObject is only used here to read GStreamer
``appsink`` buffers and manage the native GDP/shared-memory transport; WebRTC
transport stays on aiortc, so this does not reintroduce the
``webrtcbin``/libnice ICE path.

The hosted installer (``web/app/public/install``) runs it once; it is not run
at teleop/serve startup. Best-effort: a no-op on machines without
``apt-get``/NVENC (callers then fall back to the SDK ``ZedCamera``).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys

from ...utils.jetson import _is_jetson
from ...utils.packages import failure_detail, run_package_manager
from ...utils.sudo import prime_sudo

_logger = logging.getLogger(__name__)

# PyGObject 3.52 dropped the legacy girepository-1.0 ABI for girepository-2.0
# (GLib >= 2.80, i.e. Ubuntu 24.04+). The Jetson runs jammy (GLib 2.72), whose
# gobject-introspection is 1.x, so an unpinned install picks a version that
# cannot build here. Pin to the last 3.5x that targets girepository-1.0.
_PYGOBJECT_SPEC = "pygobject>=3.50,<3.52"

# GStreamer elements gst.install is itself responsible for verifying: the NVENC
# encode path (nvvidconv / nvv4l2h264enc, from the BSP's nvidia-l4t-gstreamer),
# plus videorate (plugins-base) and GDP payloading (plugins-bad).
# ``gdppay``/``gdpdepay`` preserve sensor-exposure PTS across the dataset
# recorder's shm boundary.
# These gate gst.install's success alongside the gi/appsink import.
_REQUIRED_ELEMENTS = (
    "nvvidconv",
    "nvv4l2h264enc",
    "videorate",
    "gdppay",
    "gdpdepay",
)

# The patched ZED source element ``axol gst.build-zed`` builds + installs
# (sensor-accurate PTS). gst.install only *reports* whether it's present — its
# absence is expected before build-zed runs, so it's a note, not a failure
# (build-zed, which the installer/provision runs right after, installs it).
_ZED_SOURCE_MARKER = "zedxonesrc"

# System packages: GStreamer introspection typelibs (Gst + GstApp/appsink) and
# the build deps PyGObject needs to compile against gobject-introspection.
_APT_PACKAGES = (
    "gstreamer1.0-tools",
    "gstreamer1.0-plugins-base",
    "gstreamer1.0-plugins-bad",
    "gir1.2-gstreamer-1.0",
    "gir1.2-gst-plugins-base-1.0",
    "libgirepository1.0-dev",
    "gobject-introspection",
    "libcairo2-dev",
    "pkg-config",
)

# The L4T BSP package carrying the NVENC GStreamer plugins (nvvidconv,
# nvv4l2h264enc, nvarguscamerasrc, ...). Present on a full JetPack image, absent
# on a minimal flash. The name is the same on every JetPack release.
_L4T_GSTREAMER_PACKAGE = "nvidia-l4t-gstreamer"
# The BSP package every flashed Jetson has; its installed version is the
# running L4T release (e.g. ``39.2.0-20260601141651``).
_L4T_CORE_PACKAGE = "nvidia-l4t-core"


def add_parser(subparsers) -> None:  # type: ignore[type-arg]
    """Register the ``gst.install`` subcommand."""
    subparsers.add_parser(
        "gst.install",
        help="Install GStreamer + PyGObject support for the ZED camera pipeline.",
    ).set_defaults(func=run)


def _run(cmd: list[str]) -> bool:
    """Run a command, logging on failure; returns True on exit code 0."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except Exception as exc:  # noqa: BLE001 - command missing / timed out
        _logger.warning("command failed (%s): %s", " ".join(cmd), exc)
        return False
    if result.returncode != 0:
        _logger.warning(
            "command failed (%s): %s",
            " ".join(cmd),
            (result.stderr or result.stdout or "").strip()[:500],
        )
        return False
    return True


def _gi_elements_ok(elements: tuple[str, ...]) -> bool:
    """True when a clean subprocess can import gi and find every element given.

    Checked in a subprocess so the result reflects a PyGObject that was just
    installed (importing it in this process would hit a stale import state).
    """
    from ...video.gst_zed import _set_typelib_path

    _set_typelib_path()  # ensure the child inherits the typelib search path
    els = ",".join(repr(e) for e in elements)
    code = (
        "import gi;"
        "gi.require_version('Gst','1.0');"
        "gi.require_version('GstApp','1.0');"
        "from gi.repository import Gst, GstApp;"  # GstApp registers AppSink
        "Gst.init(None);"
        "import sys;"
        f"els=({els},);"
        "sys.exit(0 if all(Gst.ElementFactory.find(e) for e in els) else 1)"
    )
    try:
        return (
            subprocess.run(
                [sys.executable, "-c", code], capture_output=True, timeout=60
            ).returncode
            == 0
        )
    except Exception:  # noqa: BLE001 - interpreter/import failure
        return False


def _gst_ok() -> bool:
    """True when gst.install's own deliverables work: PyGObject can import gi +
    GstApp (the appsink reader) and the Jetson NVENC elements are present.

    Deliberately excludes the zedxonesrc / zedsrc source elements: those are
    ``axol gst.build-zed``'s responsibility and aren't built until the step
    *after* this one, so gating on them here would misreport a clean install as
    failed. Their presence is reported separately by :func:`_note_zed_sources`.
    """
    return _gi_elements_ok(_REQUIRED_ELEMENTS)


def _note_zed_sources() -> None:
    """Report (note only) whether the patched ZED source elements are present.

    ``axol gst.build-zed`` builds + installs zedxonesrc / zedsrc, and the
    installer/provision runs it right after gst.install, so their absence here
    is expected on a first install — informational, never a warning.
    """
    if _gi_elements_ok((_ZED_SOURCE_MARKER,)):
        print("Patched ZED source elements (zedxonesrc / zedsrc) present.")
    else:
        print(
            "Note: the ZED source elements (zedxonesrc / zedsrc) aren't built "
            "yet — 'axol gst.build-zed' builds + installs them (the installer "
            "runs it right after this step)."
        )


def _installed_l4t_core_version() -> str | None:
    """Version of the installed ``nvidia-l4t-core`` package, if any.

    Only dpkg's *installed* state counts (a removed-but-configured package still
    answers with its old version).
    """
    dpkg_query = shutil.which("dpkg-query")
    if dpkg_query is None:
        return None
    try:
        proc = subprocess.run(
            [
                dpkg_query,
                "-W",
                "-f",
                "${db:Status-Status} ${Version}",
                _L4T_CORE_PACKAGE,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception:  # noqa: BLE001 - dpkg missing / timed out
        return None
    parts = proc.stdout.split()
    if proc.returncode != 0 or len(parts) != 2 or parts[0] != "installed":
        return None
    return parts[1]


def _l4t_gstreamer_apt_spec() -> str:
    """``nvidia-l4t-gstreamer`` pinned to the running BSP release.

    NVIDIA's apt repository for an L4T line (``r39.2``) carries later point
    releases too (39.2.1 next to the flashed 39.2.0), and every BSP package
    depends on ``nvidia-l4t-core`` of exactly its own version. An unpinned
    install therefore picks the newest plugin package and drags the whole BSP
    into an upgrade — which fails against the kernel packages the flash holds
    (``apt-mark hold``) and installs nothing. Pinning to the installed core
    version installs just the plugin package for the release that is running.
    Falls back to the bare name when the core package can't be read (a
    non-dpkg host; the apt call then fails or no-ops on its own).
    """
    version = _installed_l4t_core_version()
    if version is None:
        return _L4T_GSTREAMER_PACKAGE
    return f"{_L4T_GSTREAMER_PACKAGE}={version}"


def _apt_packages() -> tuple[str, ...]:
    """Everything ``_apt_install`` installs: the host packages plus NVENC."""
    return (*_APT_PACKAGES, _l4t_gstreamer_apt_spec())


def _apt_install() -> bool:
    apt = shutil.which("apt-get")
    if apt is None:
        _logger.info("apt-get not found; skipping system GStreamer install")
        return False
    packages = _apt_packages()
    if os.geteuid() != 0:
        # Prime sudo once (a tty prompt when run interactively); the hosted
        # installer runs this as root, so escalation is a no-op there.
        prime_sudo()
        if subprocess.run(["sudo", "-n", "true"], capture_output=True).returncode != 0:
            _logger.warning(
                "GStreamer system packages need root; run as root or install "
                "manually: sudo apt-get install -y %s",
                " ".join(packages),
            )
            return False
    run_package_manager([apt, "update"])
    installed = run_package_manager([apt, "install", "-y", *packages])
    if installed.returncode != 0:
        _logger.warning(
            "command failed (apt-get install): %s", failure_detail(installed)
        )
        return False
    return True


def _pip_install_pygobject() -> bool:
    # Target the interpreter running this CLI (the uv tool env), not whatever
    # VIRTUAL_ENV/cwd resolves to — same trick as ``axol zed.install``.
    uv = shutil.which("uv")
    pip_args = ["pip", "install", "--python", sys.executable, _PYGOBJECT_SPEC]
    if uv is not None and _run([uv, *pip_args]):
        return True
    return _run([sys.executable, "-m", "pip", "install", _PYGOBJECT_SPEC])


def run(_args: object = None) -> None:
    """Install the GStreamer framework, introspection typelibs, NVENC plugins
    (``nvidia-l4t-gstreamer``) and PyGObject.

    The ``zedxonesrc`` / ``zedsrc`` source elements come from
    ``axol gst.build-zed``; this command only reports whether they are present.
    """
    if not _is_jetson():
        print(
            "Not an NVIDIA Jetson (L4T); skipping the Jetson-only GStreamer "
            "NVENC stack. Camera capture will use the ZED SDK fallback on "
            "this host."
        )
        return
    if _gst_ok():
        print("GStreamer appsink + GDP + NVENC stack already available.")
        _note_zed_sources()
        return

    print("Installing GStreamer system packages (apt)...")
    _apt_install()
    print(f"Installing PyGObject ({_PYGOBJECT_SPEC}) into the axol environment...")
    _pip_install_pygobject()

    if _gst_ok():
        print("GStreamer appsink + GDP + NVENC stack installed.")
        _note_zed_sources()
    else:
        raise SystemExit(
            "The GStreamer appsink + GDP + NVENC stack is still unavailable "
            "after installation. Ensure PyGObject, the GDP plugins (gdppay / "
            "gdpdepay), and the Jetson NVENC elements (nvvidconv / "
            "nvv4l2h264enc) are installed, then retry."
        )
