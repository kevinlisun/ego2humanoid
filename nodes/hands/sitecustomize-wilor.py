import inspect
import numpy as np

if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec  # chumpy (MANO pickles) needs it
for _n, _t in (("bool", bool), ("int", int), ("float", float), ("complex", complex), ("object", object), ("unicode", str), ("str", str)):
    if not hasattr(np, _n):
        setattr(np, _n, _t)
