//
// Created by mingfei on 18/07/17.
//

#include <Python.h>
#include <librealsense/rs.hpp>
#include <librealsense/rsutil.h>


typedef struct DeviceObject{
	PyObject_HEAD
	rs::device* dev;
} DeviceObject;


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


static PyMethodDef Device_methods[] = {
		{"serial_number", (PyCFunction)Device_serial_number, METH_NOARGS,
				"Return the serial number of the device."},
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



typedef struct ContextObject{
	PyObject_HEAD
	rs::context* ctx;
} ContextObject;


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
Context_get_device(ContextObject *self)
{
	if (self -> ctx == NULL){
		PyErr_SetString(PyExc_AttributeError, "ctx");
		return NULL;
	}

	PyObject* arglist = Py_BuildValue("()");
	DeviceObject* device = (DeviceObject*) PyObject_CallObject((PyObject *) &DeviceType, arglist);
	Py_DECREF(arglist);

	if (device == NULL)
		return NULL;

	if (self->ctx->get_device_count() == 0){
		PyErr_SetString(PyExc_ValueError, "No device detected.");
		return NULL;
	}

	device -> dev = self -> ctx -> get_device(0);

	return (PyObject* )device;
}


static PyMethodDef Context_methods[] = {
		{"_n_devices", (PyCFunction)Context_n_devices, METH_NOARGS,
				"Return the number of devices."},
		{"get_device", (PyCFunction)Context_get_device, METH_NOARGS,
				"Get the first device."},
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



static PyModuleDef pyrsmodule = {
		PyModuleDef_HEAD_INIT,
		"_pyrs",
		"",
		-1,
		NULL, NULL, NULL, NULL, NULL
};


PyMODINIT_FUNC
PyInit__pyrs(void)
{
	PyObject* m;

	DeviceType.tp_new = PyType_GenericNew;
	if (PyType_Ready(&DeviceType) < 0)
		return NULL;

	ContextType.tp_new = PyType_GenericNew;
	if (PyType_Ready(&ContextType) < 0)
		return NULL;

	m = PyModule_Create(&pyrsmodule);
	if (m == NULL)
		return NULL;

	Py_INCREF(&ContextType);
	PyModule_AddObject(m, "_Context", (PyObject*)&ContextType);

	Py_INCREF(&DeviceType);
	PyModule_AddObject(m, "_Device", (PyObject*)&DeviceType);
	return m;
}