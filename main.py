"""
VOX Studio — Audio Engine V1 Entrypoint
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure workspace root is first on sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app import main

if __name__ == "__main__":
    raise SystemExit(main())
