"""DriveToy dataset plugin — rule-centric ego-view driving scenes
(docs/DRIVETOY.md): a traffic simulator whose agents follow a simplified
rulebook (``sim.py``), a rule checker (``rules.py``) and a Blender renderer
(``blender/render_clip.py``)."""
from .builder import build_to_disk
from .config import DRIVETOY_DEFAULT_CFG

DEFAULT_CFG = DRIVETOY_DEFAULT_CFG

__all__ = ["build_to_disk", "DRIVETOY_DEFAULT_CFG", "DEFAULT_CFG"]
