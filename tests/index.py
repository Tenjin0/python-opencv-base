import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/' + '..'))

from helpers import getIdFromFolderName, searchNextId

if __name__ == "__main__":

    folder_id = getIdFromFolderName("s1")
    next_id = searchNextId()
    print folder_id
    print next_id
