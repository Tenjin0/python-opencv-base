
import sys
import os

this_dir = os.path.dirname(os.path.abspath(__file__))
needed_dir = os.path.abspath(os.path.join(this_dir, '../.'))
sys.path.insert(0, needed_dir)

from helpers import generate


if __name__ == "__main__":

    print("USAGE: file.py id")
    id = None
    count = 10
    fileFormat = "pgm"

    if len(sys.argv) >= 2:

        id = int(sys.argv[1])

    if len(sys.argv) >= 3:

        count = int(sys.argv[2])

    if len(sys.argv) >= 4:

        fileFormat = sys.argv[3]

    generate(id=id, count=count, fileFormat=fileFormat)