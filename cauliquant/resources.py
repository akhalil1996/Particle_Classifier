"""Locate bundled files (models) both from source and inside the installed app."""
import os
import sys


def base_dir():
    if getattr(sys, "frozen", False):          # PyInstaller build
        return getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def resource(*parts):
    return os.path.join(base_dir(), *parts)


DEFAULT_UNET = resource("models", "cauliflower_unet.onnx")
SHAPE_MODEL = resource("models", "shape_classes_k5.json")


def user_dir():
    """Per-user folder for files the app writes (Windows: %APPDATA%\\CauliQuant)."""
    root = os.environ.get("APPDATA") or os.path.join(os.path.expanduser("~"), ".config")
    d = os.path.join(root, "CauliQuant")
    os.makedirs(d, exist_ok=True)
    return d


def local_card_path():
    """Classifier refined in the developer tab on this PC (used in place of the shipped one)."""
    return os.path.join(user_dir(), "cauliflower_unet.json")
