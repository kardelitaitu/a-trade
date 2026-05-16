"""pytest configuration for QuantumEdge."""

import sys
from pathlib import Path

# Ensure project root is on sys.path so imports like `research.data.loader` work
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
