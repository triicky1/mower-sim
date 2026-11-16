#!/usr/bin/env python3
# main.py

import sys
import matplotlib.pyplot as plt

if __name__ == "__main__":
    try:
        if "live" in sys.argv[1:]:
            from examples.live import live
            live(funmode = 'funmode' in sys.argv[1:])
        else:
            from examples.batch import main
            main()
    except (EOFError, KeyboardInterrupt):
        print()
        exit(0)

    if "plots" in sys.argv[1:]:
        plt.show()
