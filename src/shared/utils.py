__author__ = "Pinkas Matěj"
__maintainer__ = "Pinkas Matěj"
__email__ = "pinkas.matej@gmail.com"
__created__ = "22/05/2026"
__date__ = "22/05/2026"
__status__ = "Prototype"
__version__ = "0.1.0"
__credits__ = []

"""
Project: BBAP20
Filename: utils.py
Directory: src/shared/
"""

from pathlib import Path

def project_directory(subdirectory:Path) -> Path:
    script_dir = Path(__file__).parent.parent.resolve()
    data_dir = script_dir.parent / subdirectory
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir

if __name__ == '__main__':
    pass
