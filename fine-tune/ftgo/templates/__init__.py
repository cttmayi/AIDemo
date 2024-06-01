



def load_templates(module_name:str):
   import importlib
   module = importlib.import_module('templates.' + module_name)
   #importlib.reload(module)
   #module = __import__('templates.' + module_name)
   #print(module_name, module)
   return module

