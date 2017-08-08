from _pyrs import _Context, _Device


class NativeStream(object):
    DEPTH = 0
    COLOR = 1
    INFRARED = 2
    INFRARED2 = 3
    FISHEYE = 4


class SyntheticStream(object):
    POINTS = 5
    RECTIFIED_COLOR = 6
    COLOR_ALIGNED_TO_DEPTH = 7
    INFRARED2_ALIGNED_TO_DEPTH = 8
    DEPTH_ALIGNED_TO_COLOR = 9
    DEPTH_ALIGNED_TO_RECTIFIED_COLOR = 10
    DEPTH_ALIGNED_TO_INFRARED2 = 11


class Format(object):
    ANY = 0
    Z16 = 1
    DISPARITY16 = 2
    XYZ32F = 3
    YUYV = 4
    RGB8 = 5
    BGR8 = 6
    RGBA8 = 7
    BGRA8 = 8
    Y8 = 9
    Y16 = 10
    RAW10 = 11
    RAW16 = 12
    RAW8 = 13


class OutputBufferFormat(object):
    CONTINUOUS = 0
    NATIVE = 1


class Preset(object):
    BEST_QUALITY = 0
    LARGEST_IMAGE = 1
    HIGHEST_FRAMERATE = 2


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

    def enable_stream(self, stream, width, height, fmt, fps, output_format=OutputBufferFormat.CONTINUOUS):
        return self._dev._enable_stream(stream, width, height, fmt, fps, output_format)

    def enable_stream_preset(self, stream, preset):
        return self._dev._enable_stream_preset(stream, preset)

    def start(self):
        return self._dev._start()

    def stop(self):
        return self._dev._stop()

    def get_frame(self, stream):
        return self._dev._get_frame_from(stream)
