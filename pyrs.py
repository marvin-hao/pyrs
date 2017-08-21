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


class Option(object):
    COLOR_BACKLIGHT_COMPENSATION = 0
    COLOR_BRIGHTNESS = 1
    COLOR_CONTRAST = 2
    COLOR_EXPOSURE = 3
    COLOR_GAIN = 4
    COLOR_GAMMA = 5
    COLOR_HUE = 6
    COLOR_SATURATION = 7
    COLOR_SHARPNESS = 8
    COLOR_WHITE_BALANCE = 9
    COLOR_ENABLE_AUTO_EXPOSURE = 10
    COLOR_ENABLE_AUTO_WHITE_BALANCE = 11
    F200_LASER_POWER = 12
    F200_ACCURACY = 13
    F200_MOTION_RANGE = 14
    F200_FILTER_OPTION = 15
    F200_CONFIDENCE_THRESHOLD = 16
    F200_DYNAMIC_FPS = 17
    SR300_AUTO_RANGE_ENABLE_MOTION_VERSUS_RANGE = 18
    SR300_AUTO_RANGE_ENABLE_LASER = 19
    SR300_AUTO_RANGE_MIN_MOTION_VERSUS_RANGE = 20
    SR300_AUTO_RANGE_MAX_MOTION_VERSUS_RANGE = 21
    SR300_AUTO_RANGE_START_MOTION_VERSUS_RANGE = 22
    SR300_AUTO_RANGE_MIN_LASER = 23
    SR300_AUTO_RANGE_MAX_LASER = 24
    SR300_AUTO_RANGE_START_LASER = 25
    SR300_AUTO_RANGE_UPPER_THRESHOLD = 26
    SR300_AUTO_RANGE_LOWER_THRESHOLD = 27
    R200_LR_AUTO_EXPOSURE_ENABLED = 28
    R200_LR_GAIN = 29
    R200_LR_EXPOSURE = 30
    R200_EMITTER_ENABLED = 31
    R200_DEPTH_UNITS = 32
    R200_DEPTH_CLAMP_MIN = 33
    R200_DEPTH_CLAMP_MAX = 34
    R200_DISPARITY_MULTIPLIER = 35
    R200_DISPARITY_SHIFT = 36
    R200_AUTO_EXPOSURE_MEAN_INTENSITY_SET_POINT = 37
    R200_AUTO_EXPOSURE_BRIGHT_RATIO_SET_POINT = 38
    R200_AUTO_EXPOSURE_KP_GAIN = 39
    R200_AUTO_EXPOSURE_KP_EXPOSURE = 40
    R200_AUTO_EXPOSURE_KP_DARK_THRESHOLD = 41
    R200_AUTO_EXPOSURE_TOP_EDGE = 42
    R200_AUTO_EXPOSURE_BOTTOM_EDGE = 43
    R200_AUTO_EXPOSURE_LEFT_EDGE = 44
    R200_AUTO_EXPOSURE_RIGHT_EDGE = 45
    R200_DEPTH_CONTROL_ESTIMATE_MEDIAN_DECREMENT = 46
    R200_DEPTH_CONTROL_ESTIMATE_MEDIAN_INCREMENT = 47
    R200_DEPTH_CONTROL_MEDIAN_THRESHOLD = 48
    R200_DEPTH_CONTROL_SCORE_MINIMUM_THRESHOLD = 49
    R200_DEPTH_CONTROL_SCORE_MAXIMUM_THRESHOLD = 50
    R200_DEPTH_CONTROL_TEXTURE_COUNT_THRESHOLD = 51
    R200_DEPTH_CONTROL_TEXTURE_DIFFERENCE_THRESHOLD = 52
    R200_DEPTH_CONTROL_SECOND_PEAK_THRESHOLD = 53
    R200_DEPTH_CONTROL_NEIGHBOR_THRESHOLD = 54
    R200_DEPTH_CONTROL_LR_THRESHOLD = 55
    FISHEYE_EXPOSURE = 56
    FISHEYE_GAIN = 57
    FISHEYE_STROBE = 58
    FISHEYE_EXTERNAL_TRIGGER = 59
    FISHEYE_COLOR_AUTO_EXPOSURE = 60
    FISHEYE_COLOR_AUTO_EXPOSURE_MODE = 61
    FISHEYE_COLOR_AUTO_EXPOSURE_RATE = 62
    FISHEYE_COLOR_AUTO_EXPOSURE_SAMPLE_RATE = 63
    FISHEYE_COLOR_AUTO_EXPOSURE_SKIP_FRAMES = 64
    FRAMES_QUEUE_SIZE = 65
    HARDWARE_LOGGER_ENABLED = 66
    TOTAL_FRAME_DROPS = 67


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
    def serial(self):
        return self._dev.serial()

    @property
    def name(self):
        return self._dev.name()

    @property
    def usb_port_id(self):
        return self._dev.usb_port_id()

    @property
    def firmware_version(self):
        return self._dev.firmware_version()

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

    def set_options(self, options: dict):
        ops = list(options.keys())
        vals = list(options.values())
        cnt = len(ops)
        for op in ops:
            if op not in Option.__dict__.values():
                raise ValueError('The option code {} is unknown.'.format(op))
        return self._dev._set_options(ops, cnt, vals)

    def get_extrinsics(self, from_stream, to_stream):
        return self._dev._get_extrinsics(from_stream, to_stream)

    def get_masked(self, range_list=None, with_original=False):
        range_list_copy = []

        if isinstance(range_list, list):
            for depth_range in range_list:
                if (isinstance(depth_range, list) or isinstance(depth_range, tuple)) and len(depth_range) == 2:
                    depth_range_copy = []
                    if depth_range[0] is None:
                        depth_range_copy.append(-1)
                    else:
                        depth_range_copy.append(depth_range[0])

                    if depth_range[1] is None:
                        depth_range_copy.append(-1)
                    else:
                        depth_range_copy.append(depth_range[1])
                    range_list_copy.append(depth_range_copy)
            if len(range_list_copy) == 0:
                range_list_copy.append([-1, -1])
        else:
            range_list_copy.append([-1, -1])

        return self._dev._get_masked(range_list_copy, with_original)
