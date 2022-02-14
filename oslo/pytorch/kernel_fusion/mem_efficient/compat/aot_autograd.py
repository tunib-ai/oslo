from oslo.pytorch._C import CompilingBinder

_bindend = None

if _bindend is None:
    _bindend = CompilingBinder().bind()

CompileCache = _bindend.CompileCache
