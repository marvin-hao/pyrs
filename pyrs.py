from _pyrs import _Context, _Device


class Context(_Context):
    """
    Defined as a singleton in librealsense.
    All the instances are snapshots when the context was first created.

    """
    @property
    def n_devices(self):
        return super()._n_devices()

    @property
    def all_devices(self):
        for i in range(super()._n_devices()):
            yield Device(super()._get_device(i))

    def get_device(self, dev_ind):
        if dev_ind >= super()._n_devices():
            raise IndexError('The index is out of range.')
        else:
            return Device(super()._get_device(dev_ind))

    def describe_devices(self):
        raise NotImplementedError


class Device(object):
    def __init__(self, dev):
        if isinstance(dev, _Device):
            self._dev = dev
        else:
            raise ValueError('Devices must be produced by Context objects.')

    @property
    def serial_number(self):
        return self._dev.serial_number()