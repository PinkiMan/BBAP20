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
Filename: wrappers.py
Directory: src/shared/
"""


import sys
import time
import threading

from src.shared.utils import Colors

def monitor(func, has_tqdm=False):
    def wrapper(*args, **kwargs):
        stop_loading = False

        def loader():
            symbols = ['|', '/', '-', '\\']
            i = 0
            while not stop_loading:
                sys.stdout.write(f"{symbols[i % len(symbols)]}]")
                sys.stdout.flush()
                time.sleep(0.1)
                sys.stdout.write("\b\b")
                i += 1

        sys.stdout.write(f"{func.__name__:<20} ....... [")
        sys.stdout.flush()

        if has_tqdm:
            t = threading.Thread(target=loader)
            t.start()

        done = False
        try:
            output = func(*args, **kwargs)
            done = True
            return output
        except Exception:
            done = False
            raise
        finally:
            stop_loading = True
            if has_tqdm:
                t.join()

            if done:
                print(f"{Colors.Fg.green}✓{Colors.reset}]")
            else:
                print(f"{Colors.Fg.red}✗{Colors.reset}]")
    return wrapper


if __name__ == '__main__':
    pass
