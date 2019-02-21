import sys

PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range

targets = [[0], [1]]

for _ in xrange(len(targets)):
    print(_)
