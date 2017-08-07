#ifndef PYRS_DEVICE_H
#define PYRS_DEVICE_H

#include <Python.h>
#include <librealsense/rs.hpp>

#include "exception.h"


typedef struct DeviceObject{
	PyObject_HEAD
	rs::device* dev;
} DeviceObject;



static void Device_dealloc(DeviceObject* self);

static PyObject* Device_serial_number(DeviceObject *self);

static PyObject* Device_stop(DeviceObject *self);

static PyObject* Device_start(DeviceObject *self);


static PyMethodDef Device_methods[] = {
		{"serial_number", (PyCFunction)Device_serial_number, METH_NOARGS,
				"Return the serial number of the device."},
		{"_stop", (PyCFunction)Device_stop, METH_NOARGS,
				"Stop the device."},
		{"_start", (PyCFunction)Device_start, METH_NOARGS,
				"Start the device."},
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

#endif //PYRS_DEVICE_H
