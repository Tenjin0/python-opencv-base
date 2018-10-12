
import sys
import os

this_dir = os.path.dirname(os.path.abspath(__file__))
needed_dir = os.path.abspath(os.path.join(this_dir, '../.'))
sys.path.insert(0, needed_dir)


from helpers import generate


if __name__ == "__main__":

    print "USAGE: file.py id"
    id = None

    if len(sys.argv) > 1:

        id = int(sys.argv[1])

    generate(id)