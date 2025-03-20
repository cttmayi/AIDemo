import os
import importlib

class StreamPrint:
    def __init__(self, stream, name=None):
        self.name = name
        self.stream = stream
        self.n = 0
        pass

    def print_loop(self):
        for text in self.stream:
            print(text[self.n:], end='', flush=True)
            self.n = len(text)

    def __enter__(self):
        if self.name is not None:
            print(self.name, end=': ')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print()


def get_modules(package_path="."):
    modules = []
    files = os.listdir(package_path)
    for file in files:
        if not file.startswith("_"):
            name, _ = os.path.splitext(file)
            modules.append(name)
    return modules


def import_module(module_name, package_name):
    module = importlib.import_module("." + module_name, package_name)
    ret = {}
    for attr in dir(module):
        if not attr.startswith("_"):
            func = getattr(module, attr)
            ret[attr] = func
    return ret




if __name__ == '__main__':
    def steam_function():
        base_string = "abcdefghd_" * 200
        for i in range(1, len(base_string) + 1):
            yield base_string[:i]

    with StreamPrint(steam_function(), 'Bot') as stream:
        stream.print_loop()