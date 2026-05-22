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
Filename: 02_train_embedder.py
Directory: scripts/
"""

from src.embedder.train import Trainer


def main():
    trainer = Trainer()
    trainer.main()

if __name__ == '__main__':
    main()

