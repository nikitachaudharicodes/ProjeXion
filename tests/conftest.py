# tests/conftest.py
import sys, os, types

# 1) Make sure the project root is on PYTHONPATH
#    so "import blendedMVS" or "import train" works.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# 2) Stub-out open3d so that evaluation/evaluate.py can import it
fake_o3d = types.ModuleType("open3d")
# if your code does `import open3d.io as io` or similar, stub those too:
fake_o3d.io = types.ModuleType("open3d.io")
fake_o3d.geometry = types.ModuleType("open3d.geometry")
sys.modules["open3d"] = fake_o3d
sys.modules["open3d.io"] = fake_o3d.io
sys.modules["open3d.geometry"] = fake_o3d.geometry

