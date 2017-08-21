#ifndef PYRS_DEVICE_H
#define PYRS_DEVICE_H

#include <Python.h>
#include <librealsense/rs.hpp>
#include <librealsense/rsutil.h>
#include <numpy/arrayobject.h>

#include "exception.h"


typedef struct DeviceObject{
	PyObject_HEAD
	rs::device* dev;
} DeviceObject;


#define PyRsDevCharProperty(prop, rs_prop) \
static PyObject* \
Device_##prop(DeviceObject *self) \
{ \
	if (self -> dev == NULL){ \
		PyErr_SetString(PyExc_AttributeError, "dev"); \
		return NULL; \
	} \
 \
	const char* p; \
 \
	try{ \
		p = self->dev->rs_prop(); \
		return PyUnicode_FromString(p); \
	} catch (const rs::error &e) { \
		PyThrowRsErr(e) \
	} \
} \



static void Device_dealloc(DeviceObject* self);

static PyObject* Device_stop(DeviceObject *self);

static PyObject* Device_start(DeviceObject *self);

static PyObject* Device_enable_stream_preset(DeviceObject *self, PyObject *args);

static PyObject* Device_enable_stream(DeviceObject *self, PyObject *args);

static PyObject* Device_serial(DeviceObject *self);

static PyObject* Device_name(DeviceObject *self);

static PyObject* Device_usb_port_id(DeviceObject *self);

static PyObject* Device_firmware_version(DeviceObject *self);

static PyObject* Device_get_frame_from(DeviceObject *self, PyObject* args);

static PyObject* Device_set_options(DeviceObject *self, PyObject* args);

static PyObject* Device_get_extrinsics(DeviceObject *self, PyObject* args);

static PyObject* Device_get_masked(DeviceObject *self, PyObject *args);


static PyMethodDef Device_methods[] = {
		{"_stop", (PyCFunction)Device_stop, METH_NOARGS,
				"Stop the device."},
		{"_start", (PyCFunction)Device_start, METH_NOARGS,
				"Start the device."},
		{"_enable_stream_preset", (PyCFunction)Device_enable_stream_preset, METH_VARARGS,
				"Enable a preset stream."},
		{"_enable_stream", (PyCFunction)Device_enable_stream, METH_VARARGS,
				"Enable a specific stream."},
		{"serial", (PyCFunction)Device_serial, METH_NOARGS,
				"Return the serial number of the device."},
		{"name", (PyCFunction)Device_name, METH_NOARGS,
				"Return the name of the device."},
		{"usb_port_id", (PyCFunction)Device_usb_port_id, METH_NOARGS,
				"Return the device's usb port id."},
		{"firmware_version", (PyCFunction)Device_firmware_version, METH_NOARGS,
				"Return the firmware version info of the device."},
		{"_get_frame_from", (PyCFunction)Device_get_frame_from, METH_VARARGS,
				"Get a single frame from each of the enabled streams."},
		{"_set_options", (PyCFunction)Device_set_options, METH_VARARGS,
				"Set device options."},
		{"_get_extrinsics", (PyCFunction)Device_get_extrinsics, METH_VARARGS,
				"Get the extrinsics (rotation, translation) from one stream to another."},
        {"_get_masked", (PyCFunction)Device_get_masked, METH_VARARGS,
                "Get sets of masked (also aligned) color, depth and ir channel."},
		{NULL}
};


static PyTypeObject DeviceType = {
		PyVarObject_HEAD_INIT(NULL, 0)
		"_pyrs._Device",             /* tp_name */
		sizeof(DeviceObject),             /* tp_basicsize */
		0,                         /* tp_itemsize */
		(destructor)Device_dealloc, /* tp_dealloc */
		0,                         /* tp_print */
		0,                         /* tp_getattr */
		0,                         /* tp_setattr */
		0,                         /* tp_reserved */
		0,                         /* tp_repr */
		0,                         /* tp_as_number */
		0,                         /* tp_as_sequence */
		0,                         /* tp_as_mapping */
		0,                         /* tp_hash  */
		0,                         /* tp_call */
		0,                         /* tp_str */
		0,                         /* tp_getattro */
		0,                         /* tp_setattro */
		0,                         /* tp_as_buffer */
		Py_TPFLAGS_DEFAULT |
		Py_TPFLAGS_BASETYPE,   /* tp_flags */
		"_Device objects",           /* tp_doc */
		0,                         /* tp_traverse */
		0,                         /* tp_clear */
		0,                         /* tp_richcompare */
		0,                         /* tp_weaklistoffset */
		0,                         /* tp_iter */
		0,                         /* tp_iternext */
		Device_methods,             /* tp_methods */
};

PyRsDevCharProperty(serial, get_serial)
PyRsDevCharProperty(name, get_name)
PyRsDevCharProperty(usb_port_id, get_usb_port_id)
PyRsDevCharProperty(firmware_version, get_firmware_version)



static void
Device_dealloc(DeviceObject* self)
{
	if (self -> dev != NULL){
		self -> dev = NULL;
	}
	Py_TYPE(self) -> tp_free((PyObject*) self);
}

static PyObject* Device_stop(DeviceObject *self)
{
	if (self -> dev == NULL){
		PyErr_SetString(PyExc_AttributeError, "dev");
		return NULL;
	}

	self->dev->stop();

	Py_RETURN_NONE;
}


static PyObject* Device_start(DeviceObject *self)
{
	if (self -> dev == NULL){
		PyErr_SetString(PyExc_AttributeError, "dev");
		return NULL;
	}
	try {
		self->dev->start();
	} catch (const rs::error &e) {
		PyThrowRsErr(e)
	}


	Py_RETURN_NONE;
}

static PyObject* Device_enable_stream_preset(DeviceObject *self, PyObject *args)
{
	int stream_mode;
	int preset;

	if (PyArg_ParseTuple(args, "ii", &stream_mode, &preset)){

		if (self -> dev == NULL){
			PyErr_SetString(PyExc_AttributeError, "dev");
			return NULL;
		}

		rs::stream s = (rs::stream) stream_mode;
		rs::preset p = (rs::preset) preset;

		try {
			self->dev->enable_stream(s, p);
		} catch (const rs::error &e) {
			PyThrowRsErr(e)
		}
	}

	Py_RETURN_NONE;
}


static PyObject* Device_enable_stream(DeviceObject *self, PyObject *args)
{
	int stream_mode, width, height, format, fps, output;

	if (PyArg_ParseTuple(args, "iiiiii", &stream_mode, &width, &height, &format, &fps, &output)){

		if (self -> dev == NULL){
			PyErr_SetString(PyExc_AttributeError, "dev");
			return NULL;
		}

		rs::stream s = (rs::stream) stream_mode;
		rs::format f = (rs::format) format;
		rs::output_buffer_format o = (rs::output_buffer_format) output;

		try {
			self->dev->enable_stream(s, width, height, f, fps, o);
		} catch (const rs::error &e) {
			PyThrowRsErr(e)
		}
	}

	Py_RETURN_NONE;
}


static PyObject* Device_get_frame_from(DeviceObject *self, PyObject* args)
{
	int stream_mode;

	if (PyArg_ParseTuple(args, "i", &stream_mode)){

		if (self -> dev == NULL){
			PyErr_SetString(PyExc_AttributeError, "dev");
			return NULL;
		}

		rs::stream s = (rs::stream) stream_mode;

		uint16_t* dframe = NULL;
		uint8_t * frame = NULL;


		try {

			self->dev->wait_for_frames();
			if (s == rs::stream::depth
				or s == rs::stream::depth_aligned_to_color
				or s == rs::stream::depth_aligned_to_rectified_color
				or s == rs::stream::depth_aligned_to_infrared2)

				dframe = (uint16_t *)(self->dev->get_frame_data(s));
			else
				frame = (uint8_t*)self->dev->get_frame_data(s);

		} catch (const rs::error &e) {
			PyThrowRsErr(e)
		}

		PyObject* npy_cframe = NULL;
		PyObject* npy_dframe = NULL;
		PyObject* npy_irframe = NULL;

		if (s == rs::stream::color
			or s == rs::stream::color_aligned_to_depth
			or s == rs::stream::rectified_color)
		{
			npy_intp cframe_dim[3] = {self->dev->get_stream_height(s), self->dev->get_stream_width(s), 3};

			npy_cframe = PyArray_SimpleNewFromData(
					3, cframe_dim, NPY_UINT8, frame
			);
			if (npy_cframe == NULL) {
				PyErr_SetString(PyExc_ValueError, "Cannot create numpy array from the frame.");
				return NULL;
			}
			return npy_cframe;

		} else if (s == rs::stream::depth
				   or s == rs::stream::depth_aligned_to_color
				   or s == rs::stream::depth_aligned_to_rectified_color
				   or s == rs::stream::depth_aligned_to_infrared2)
		{
			npy_intp dframe_dim[2] = {self->dev->get_stream_height(s), self->dev->get_stream_width(s)};
			npy_dframe = PyArray_SimpleNewFromData(
					2, dframe_dim, NPY_UINT16, dframe
			);
			if (npy_dframe == NULL) {
				PyErr_SetString(PyExc_ValueError, "Cannot create numpy array from the frame.");
				return NULL;
			}

			return npy_dframe;
		} else if (s == rs::stream::infrared
				   or s == rs::stream::infrared2
				   or s == rs::stream::infrared2_aligned_to_depth)
		{
			npy_intp irframe_dim[2] = {self->dev->get_stream_height(s), self->dev->get_stream_width(s)};
			npy_irframe = PyArray_SimpleNewFromData(
					2, irframe_dim, NPY_UINT8, frame
			);
			if (npy_irframe == NULL) {
				PyErr_SetString(PyExc_ValueError, "Cannot create numpy array from the frame.");
				return NULL;
			}

			return npy_irframe;

		} else {
			PyErr_SetString(RsError, "The stream is not supported yet.");
			return NULL;
		}
	}
	PyErr_SetString(PyExc_ValueError, "Cannot parse the input.");
	return NULL;
}


static PyObject* Device_set_options(DeviceObject *self, PyObject* args)
{
	PyObject* py_options, * py_count, * py_values;

	if (PyArg_ParseTuple(args, "OOO", &py_options, &py_count, &py_values)){

		if (self -> dev == NULL){
			PyErr_SetString(PyExc_AttributeError, "dev");
			return NULL;
		}

		size_t count = PyLong_AsSize_t(py_count);
		rs::option* options = new rs::option[count];
		double* values = new double[count];

		for (size_t i = 0; i < count; ++i)
		{
			options[i] = (rs::option)PyLong_AsLong(PyList_GetItem(py_options, i));
			values[i] = PyFloat_AsDouble(PyList_GetItem(py_values, i));
		}

		try {
			self->dev->set_options(options, count, values);
		} catch (const rs::error &e) {
			PyThrowRsErr(e)
		}

		Py_RETURN_NONE;
	}
	PyErr_SetString(PyExc_ValueError, "Cannot parse the input.");
	return NULL;
}


static PyObject* Device_get_extrinsics(DeviceObject *self, PyObject* args)
{
	int from_stream, to_stream;

	if (PyArg_ParseTuple(args, "ii", &from_stream, &to_stream)) {

		if (self->dev == NULL) {
			PyErr_SetString(PyExc_AttributeError, "dev");
			return NULL;
		}

		rs::stream from_s = (rs::stream) from_stream;
		rs::stream to_s = (rs::stream) to_stream;

		rs::extrinsics extrin;

		try {
			extrin = self->dev->get_extrinsics(from_s, to_s);
		} catch (const rs::error &e) {
			PyThrowRsErr(e)
		}

		npy_intp r_dim[2] = {3, 3};
		npy_intp t_dim[2] = {3, 1};

		float* rotation = new float[9];

		for (int i = 0; i < 9; ++i){
			rotation[i] = extrin.rotation[i];
		}

		float* translation = new float[3];

		for (int i = 0; i < 3; ++i){
			translation[i] = extrin.translation[i];
		}

		PyArrayObject* npy_rotation_trans = (PyArrayObject*) PyArray_SimpleNewFromData(2, r_dim, NPY_FLOAT32, rotation);
		PyArrayObject* npy_translation = (PyArrayObject*) PyArray_SimpleNewFromData(2, t_dim, NPY_FLOAT32, translation);
		PyArrayObject* npy_rotation = (PyArrayObject*) PyArray_Transpose(npy_rotation_trans, NULL);
		PyArray_ENABLEFLAGS(npy_translation, NPY_ARRAY_OWNDATA);
		PyArray_ENABLEFLAGS(npy_rotation, NPY_ARRAY_OWNDATA);
		Py_DECREF(npy_rotation_trans);

		return Py_BuildValue("NN", npy_rotation, npy_translation);

	} else {
		PyErr_SetString(PyExc_ValueError, "Cannot parse the input.");
		return NULL;
	}
}


static PyObject* Device_get_masked(DeviceObject *self, PyObject *args)
{
	bool with_original;

    PyObject* range_list;

	if (!PyArg_ParseTuple(args, "O!p", &PyList_Type, &range_list, &with_original)){
		PyErr_SetString(PyExc_ValueError, "Cannot parse the input.");
		return NULL;
	}

    auto n_layer = (ulong)PyList_Size(range_list);
    auto* result = PyList_New(n_layer);

    int cheight = self->dev->get_stream_height(rs::stream::color);
    int cwidth = self->dev->get_stream_width(rs::stream::color);

    int dheight = self->dev->get_stream_height(rs::stream::depth);
    int dwidth = self->dev->get_stream_width(rs::stream::depth);

    self->dev->wait_for_frames();

    auto cframe = (uint8_t *)(self->dev->get_frame_data(rs::stream::color));
    auto dframe = (uint16_t *)(self->dev->get_frame_data(rs::stream::depth));
    auto iframe = (uint8_t *)(self->dev->get_frame_data(rs::stream::infrared));

    auto color_intrin = self->dev->get_stream_intrinsics(rs::stream::color);
    auto depth_intrin = self->dev->get_stream_intrinsics(rs::stream::depth);
    auto depth_to_color = self->dev->get_extrinsics(rs::stream::depth, rs::stream::color);
    float scale = self->dev->get_depth_scale();

    auto cframe_aligned_list = std::vector<uint8_t*>(n_layer);
    auto dframe_aligned_list = std::vector<uint16_t*>(n_layer);
    auto iframe_aligned_list = std::vector<uint8_t*>(n_layer);
    auto min_depth_list = std::vector<double>(n_layer);
    auto max_depth_list = std::vector<double>(n_layer);

    for(ulong n = 0; n < n_layer; ++n){
        cframe_aligned_list[n] = new uint8_t[cheight * cwidth * 3]();
        dframe_aligned_list[n] = new uint16_t[cheight * cwidth]();
        iframe_aligned_list[n] = new uint8_t[cheight * cwidth]();
        min_depth_list[n] = PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(range_list, n), 0));
        max_depth_list[n] = PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(range_list, n), 1));
    }

    int dx, dy;

    for(dy=0; dy<dheight; ++dy){
        for(dx=0; dx<dwidth; ++dx){
            uint16_t depth_value = dframe[dy * dwidth + dx];
            float depth_in_meters = depth_value * scale;

            for (ulong n = 0; n < n_layer; ++n){
                if (depth_in_meters <= std::fmax(0, min_depth_list[n]))
                    continue;
                else if (max_depth_list[n] > 0 && depth_in_meters >= max_depth_list[n])
                    continue;

                // Map the top-left corner of the depth pixel onto the other image
                rs::float2 depth_pix_tl = {dx-0.5f, dy-0.5f};
                auto depth_point_tl = depth_intrin.deproject(depth_pix_tl, depth_in_meters);
                auto color_point_tl = depth_to_color.transform(depth_point_tl);
                auto color_pix_tl = color_intrin.project(color_point_tl);
                auto tl_x = static_cast<int>(color_pix_tl.x+0.5f);
                auto tl_y = static_cast<int>(color_pix_tl.y+0.5f);

                // Map the bottom-right corner of the depth pixel onto the other image
                rs::float2 depth_pix_br = {dx+0.5f, dy+0.5f};
                auto depth_point_br = depth_intrin.deproject(depth_pix_br, depth_in_meters);
                auto color_point_br = depth_to_color.transform(depth_point_br);
                auto color_pix_br = color_intrin.project(color_point_br);
                auto br_x = static_cast<int>(color_pix_br.x+0.5f);
                auto br_y = static_cast<int>(color_pix_br.y+0.5f);


                if(tl_x < 0 || tl_y < 0 || br_x >= cwidth || br_y >= cheight) continue;

                // Transfer between the depth pixels and the pixels inside the rectangle on the other image
                for(int y=tl_y; y<=br_y; ++y)
                    for(int x=tl_x; x<=br_x; ++x){
                        cframe_aligned_list[n][y * cwidth * 3 + x * 3] = cframe[y * cwidth * 3 + x * 3];
                        cframe_aligned_list[n][y * cwidth * 3 + x * 3 + 1] = cframe[y * cwidth * 3 + x * 3 + 1];
                        cframe_aligned_list[n][y * cwidth * 3 + x * 3 + 2] = cframe[y * cwidth * 3 + x * 3 + 2];

                        if (dframe_aligned_list[n][y * cwidth + x] == 0)
                            dframe_aligned_list[n][y * cwidth + x] = dframe[dy * dwidth + dx];

                        if (iframe_aligned_list[n][y * cwidth + x] == 0)
                            iframe_aligned_list[n][y * cwidth + x] = iframe[dy * dwidth + dx];
                    }
            }
        }
    }

    npy_intp cframe_dim[3] = {cheight, cwidth, 3};
    npy_intp frame_dim[2] = {cheight, cwidth};

    for (ulong n = 0; n < n_layer; ++n){
        auto* npy_cframe = (PyArrayObject*) PyArray_SimpleNewFromData(
                3, cframe_dim, NPY_UINT8, cframe_aligned_list[n]
        );
        if (npy_cframe == NULL) {
            PyErr_SetString(PyExc_ValueError, "Cannot create numpy array from the frame.");
            return NULL;
        }

        auto* npy_dframe = (PyArrayObject*) PyArray_SimpleNewFromData(
                2, frame_dim, NPY_UINT16, dframe_aligned_list[n]
        );
        if (npy_dframe == NULL) {
            PyErr_SetString(PyExc_ValueError, "Cannot create numpy array from the frame.");
            return NULL;
        }

        auto* npy_iframe = (PyArrayObject*) PyArray_SimpleNewFromData(
                2, frame_dim, NPY_UINT8, iframe_aligned_list[n]
        );
        if (npy_iframe == NULL) {
            PyErr_SetString(PyExc_ValueError, "Cannot create numpy array from the frame.");
            return NULL;
        }

        PyArray_ENABLEFLAGS(npy_cframe, NPY_ARRAY_OWNDATA);
        PyArray_ENABLEFLAGS(npy_dframe, NPY_ARRAY_OWNDATA);
        PyArray_ENABLEFLAGS(npy_iframe, NPY_ARRAY_OWNDATA);

        PyObject* layer = PyList_New(3);
        PyList_SetItem(layer, 0, (PyObject*) npy_cframe);
        PyList_SetItem(layer, 1, (PyObject*) npy_dframe);
        PyList_SetItem(layer, 2, (PyObject*) npy_iframe);
        PyList_SetItem(result, n, layer);
    }

	if (with_original){
		auto* npy_cframe_origin = (PyArrayObject*) PyArray_SimpleNewFromData(
				3, cframe_dim, NPY_UINT8, cframe
		);
		if (npy_cframe_origin == NULL) {
			PyErr_SetString(PyExc_ValueError, "Cannot create numpy array from the frame.");
			return NULL;
		}

		return Py_BuildValue("NN", result, npy_cframe_origin);
	} else
    	return Py_BuildValue("N", result);
}

#endif //PYRS_DEVICE_H
