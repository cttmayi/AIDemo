



def load_templates(module_name:str):
   import importlib
   module = importlib.import_module('templates.' + module_name)
   return module

