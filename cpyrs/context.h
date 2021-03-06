#ifndef PYRS_CONTEXT_H
#define PYRS_CONTEXT_H

#include <Python.h>
#include <librealsense/rs.hpp>

#include "device.h"

typedef struct ContextObject{
	PyObject_HEAD
	rs::context* ctx;
} ContextObject;


static void Context_dealloc(ContextObject* self);

static int Context_init(ContextObject* self, PyObject* args, PyObject* kwds);

static PyObject*Context_n_devices(ContextObject *self);

static PyObject* Context_get_device(ContextObject *self, PyObject* args, PyObject* kwds);

static PyObject* Context_get_device_by_serial(ContextObject *self, PyObject* args, PyObject* kwds);


static PyMethodDef Context_methods[] = {
		{"_n_devices", (PyCFunction)Context_n_devices, METH_NOARGS,
				"Return the number of devices."},
		{"_get_device", (PyCFunction)Context_get_device, METH_VARARGS,
				"Get the first device."},
		{"_get_device_by_serial", (PyCFunction)Context_get_device_by_serial, METH_VARARGS,
				"Get the device by the serial number."},
		{NULL}
};


static PyTypeObject ContextType = {
		PyVarObject_HEAD_INIT(NULL, 0)
		"_pyrs._Context",             /* tp_name */
		sizeof(ContextObject),             /* tp_basicsize */
		0,                         /* tp_itemsize */
		(destructor)Context_dealloc, /* tp_dealloc */
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
		"_Context objects",           /* tp_doc */
		0,                         /* tp_traverse */
		0,                         /* tp_clear */
		0,                         /* tp_richcompare */
		0,                         /* tp_weaklistoffset */
		0,                         /* tp_iter */
		0,                         /* tp_iternext */
		Context_methods,             /* tp_methods */
		0,             /* tp_members */
		0,                         /* tp_getset */
		0,                         /* tp_base */
		0,                         /* tp_dict */
		0,                         /* tp_descr_get */
		0,                         /* tp_descr_set */
		0,                         /* tp_dictoffset */
		(initproc)Context_init,      /* tp_init */
};



static void
Context_dealloc(ContextObject* self)
{
	if (self -> ctx != NULL){
		delete self->ctx;
	}
	Py_TYPE(self) -> tp_free((PyObject*) self);
}


static int
Context_init(ContextObject* self, PyObject* args, PyObject* kwds)
{
	self -> ctx = new rs::context();
	return 0;
}


static PyObject*
Context_n_devices(ContextObject *self)
{
	if (self -> ctx == NULL){
		PyErr_SetString(PyExc_AttributeError, "ctx");
		return NULL;
	}

	return PyLong_FromLong(self -> ctx -> get_device_count());
}


static PyObject*
Context_get_device_by_serial(ContextObject *self, PyObject* args, PyObject* kwds)
{
	PyObject* result = NULL;
	char* serial;

	if (PyArg_ParseTuple(args, "s", &serial)){

		if (self -> ctx == NULL){
			PyErr_SetString(PyExc_AttributeError, "ctx");
			return NULL;
		}

		PyObject* arglist = Py_BuildValue("()");
		DeviceObject* device = (DeviceObject*) PyObject_CallObject((PyObject *) &DeviceType, arglist);
		Py_DECREF(arglist);

		if (device == NULL)
			return NULL;

        for( int i = 0; i < self->ctx->get_device_count(); ++i)
        {
            rs::device* dev = self->ctx->get_device(i);
            if (strcmp(dev->get_serial(), serial) == 0)
            {
                device->dev = dev;
            }
        }

		if (device -> dev == NULL)
        {
            PyErr_SetString(PyExc_ValueError, "No such device.");
            return NULL;
        }

		return (PyObject* )device;

	}
	return result;
}


static PyObject*
Context_get_device(ContextObject *self, PyObject* args, PyObject* kwds)
{
	PyObject* result = NULL;
	int dev_ind;

	if (PyArg_ParseTuple(args, "i", &dev_ind)){

		if (self -> ctx == NULL){
			PyErr_SetString(PyExc_AttributeError, "ctx");
			return NULL;
		}

		PyObject* arglist = Py_BuildValue("()");
		DeviceObject* device = (DeviceObject*) PyObject_CallObject((PyObject *) &DeviceType, arglist);
		Py_DECREF(arglist);

		if (device == NULL)
			return NULL;

		if (self->ctx->get_device_count() <= dev_ind){
			PyErr_SetString(PyExc_IndexError, "The index is out of range.");
			return NULL;
		}

		device -> dev = self -> ctx -> get_device(dev_ind);

		return (PyObject* )device;

	}
	return result;
}



#endif //PYRS_CONTEXT_H
