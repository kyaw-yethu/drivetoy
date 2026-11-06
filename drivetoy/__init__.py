"""DriveToy v4 (docs/drivetoy-v4-design.md): a traffic simulator whose agents follow a simplified rulebook
(``sim.py``), the decision scenes built on it (``decision.py``), a rule checker (``rules.py``) and a Blender
renderer (``blender/render_clip.py``)."""
from .config import DRIVETOY_DEFAULT_CFG

DEFAULT_CFG = DRIVETOY_DEFAULT_CFG

__all__ = ["DRIVETOY_DEFAULT_CFG", "DEFAULT_CFG"]
