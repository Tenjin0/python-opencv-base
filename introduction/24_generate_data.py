
import sys
import os

this_dir = os.path.split(__file__)[0]
this_dir = os.path.dirname(os.path.abspath(__file__))
needed_dir = os.path.abspath(os.path.join(this_dir, '../.'))
sys.path.insert(0, needed_dir)


from helpers import read_images


if __name__ == "__main__":
    print sys.argv
    if len(sys.argv) < 2:
        print "USAGE: facerec_demo.py </path/to/images> "
        sys.exit()
