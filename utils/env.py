import os

CURR_PATH = os.path.dirname(os.path.abspath(__file__))

from dotenv import load_dotenv
load_dotenv(override=True, verbose=True)



def chdir(path):
    os.chdir(os.path.dirname(path))