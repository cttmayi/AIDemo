import re, os


checkpoint = "/outputs/QS/checkpoint-40"

PREFIX_CHECKPOINT_DIR = "checkpoint"
_re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")

def get_checkpoint_number(path):
    path = os.path.basename(path)
    if _re_checkpoint.search(path) is not None:
        return int(_re_checkpoint.search(path).groups()[0])
    return 0


max_steps = get_checkpoint_number(checkpoint)
print(max_steps) # 1000