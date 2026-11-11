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
