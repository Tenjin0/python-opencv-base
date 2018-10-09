import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/' + '..'))

print sys.path

from helpers import getIdFromFolderName

if __name__ == "__main__":

    folder_id = getIdFromFolderName("s1")

    print folder_id
