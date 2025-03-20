import os
from sys import argv

path = argv[0]
try:
    os.chdir(os.path.dirname(path))
except:
    pass
