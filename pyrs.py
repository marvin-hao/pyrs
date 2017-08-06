import _pyrs

from threading import Lock


class Context(object):
    lock = Lock()
    __ctx = _pyrs._Context()

    @staticmethod
    def n_devices() -> int:
        with Context.lock:
            del Context.__ctx
            Context.__ctx = _pyrs._Context()
            return Context.__ctx._n_devices()

