#ifndef PYRS_DEVICE_H
#define PYRS_DEVICE_H

#include <Python.h>
#include <librealsense/rs.hpp>
#include <numpy/arrayobject.h>

#include "exception.h"


typedef struct DeviceObject{
	PyObject_HEAD
	rs::device* dev;
} DeviceObject;



static void Device_dealloc(DeviceObject* self);

static PyObject* Device_serial_number(DeviceObject *self);

static PyObject* Device_stop(DeviceObject *self);

static PyObject* Device_start(DeviceObject *self);

static PyObject* Device_enable_stream_preset(DeviceObject *self, PyObject *args);

static PyObject* Device_enable_stream(DeviceObject *self, PyObject *args);

static PyObject* Device_get_frame_from(DeviceObject *self, PyObject* args);

static PyObject* Device_set_options(DeviceObject *self, PyObject* args);


static PyMethodDef Device_methods[] = {
		{"serial_number", (PyCFunction)Device_serial_number, METH_NOARGS,
				"Return the serial number of the device."},
		{"_stop", (PyCFunction)Device_stop, METH_NOARGS,
				"Stop the device."},
		{"_start", (PyCFunction)Device_start, METH_NOARGS,
				"Start the device."},
		{"_enable_stream_preset", (PyCFunction)Device_enable_stream_preset, METH_VARARGS,
				"Enable a preset stream."},
		{"_enable_stream", (PyCFunction)Device_enable_stream, METH_VARARGS,
				"Enable a specific stream."},
		{"_get_frame_from", (PyCFunction)Device_get_frame_from, METH_VARARGS,
				"Get a single frame from each of the enabled streams."},
		{"_set_options", (PyCFunction)Device_set_options, METH_VARARGS,
				"Set device options."},
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


static void
Device_dealloc(DeviceObject* self)
{
	if (self -> dev != NULL){
		self -> dev = NULL;
	}
	Py_TYPE(self) -> tp_free((PyObject*) self);
}

// todo: will cause segmentation fault if the context object is changed
static PyObject*
Device_serial_number(DeviceObject *self)
{
	if (self -> dev == NULL){
		PyErr_SetString(PyExc_AttributeError, "dev");
		return NULL;
	}

	return PyUnicode_FromString(self -> dev -> get_serial());
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
			if (s == rs::stream::depth)
				dframe = (uint16_t *)(self->dev->get_frame_data(s));
			else
				frame = (uint8_t*)self->dev->get_frame_data(s);

		} catch (const rs::error &e) {
			PyThrowRsErr(e)
		}

		PyObject* npy_cframe = NULL;
		PyObject* npy_dframe = NULL;
		PyObject* npy_irframe = NULL;

		if (s == rs::stream::color) {
			npy_intp cframe_dim[3] = {self->dev->get_stream_height(s), self->dev->get_stream_width(s), 3};

			npy_cframe = PyArray_SimpleNewFromData(
					3, cframe_dim, NPY_UINT8, frame
			);
			if (npy_cframe == NULL) {
				PyErr_SetString(PyExc_ValueError, "Cannot create numpy array from the frame.");
				return NULL;
			}
			return npy_cframe;

		} else if (s == rs::stream::depth) {
			npy_intp dframe_dim[2] = {self->dev->get_stream_height(s), self->dev->get_stream_width(s)};
			npy_dframe = PyArray_SimpleNewFromData(
					2, dframe_dim, NPY_UINT16, dframe
			);
			if (npy_dframe == NULL) {
				PyErr_SetString(PyExc_ValueError, "Cannot create numpy array from the frame.");
				return NULL;
			}

			return npy_dframe;
		} else if (s == rs::stream::infrared) {
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



#endif //PYRS_DEVICE_H
