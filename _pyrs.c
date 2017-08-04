//
// Created by mingfei on 18/07/17.
//

#include <Python.h>
#include <librealsense/rs.h>
#include <librealsense/rsutil.h>

// Librealsense Error Handling
rs_error * e = 0;
int check_error(void)
{
	if(e)
	{
		printf("rs_error was raised when calling %s(%s):\n", rs_get_failed_function(e), rs_get_failed_args(e));
		printf("    %s\n", rs_get_error_message(e));
		return 1;
	}

	return 0;
}


typedef struct _pyrs__ContextObject{
	PyObject_HEAD
	rs_context* ctx;
} _pyrs__ContextObject;


static void
_Context_dealloc(_pyrs__ContextObject* self)
{
	if (self -> ctx != NULL){
		rs_delete_context(self -> ctx, &e);
		check_error();
	}
	Py_XDECREF(self -> ctx);
	Py_TYPE(self) -> tp_free((PyObject*) self);
}


static PyObject*
_Context_new(PyTypeObject* type, PyObject* args, PyObject* kwds)
{
	_pyrs__ContextObject* self;

	self = (_pyrs__ContextObject*)type -> tp_alloc(type, 0);
	if (self != NULL)
		self -> ctx = NULL;

	return (PyObject* )self;
}


static int
_Context_init(_pyrs__ContextObject* self, PyObject* args, PyObject* kwds)
{
	rs_context* ctx;
	ctx = rs_create_context(RS_API_VERSION, &e);
	check_error();
	self -> ctx = ctx;
	return 0;
}


static PyObject*
_Context_n_device(_pyrs__ContextObject* self)
{
	if (self -> ctx == NULL){
		PyErr_SetString(PyExc_AttributeError, "ctx");
		return NULL;
	}

	return PyLong_FromLong(rs_get_device_count(self -> ctx, &e));
}


static PyMethodDef _Context_methods[] = {
		{"n_device", (PyCFunction)_Context_n_device, METH_NOARGS,
		 "Return the number of devices."},
		{NULL}
};


static PyTypeObject _pyrs_ContextType = {
		PyVarObject_HEAD_INIT(NULL, 0)
		"_pyrs._Context",             /* tp_name */
		sizeof(_pyrs__ContextObject),             /* tp_basicsize */
		0,                         /* tp_itemsize */
		(destructor)_Context_dealloc, /* tp_dealloc */
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
		_Context_methods,             /* tp_methods */
		0,             /* tp_members */
		0,                         /* tp_getset */
		0,                         /* tp_base */
		0,                         /* tp_dict */
		0,                         /* tp_descr_get */
		0,                         /* tp_descr_set */
		0,                         /* tp_dictoffset */
		(initproc)_Context_init,      /* tp_init */
		0,                         /* tp_alloc */
		_Context_new,                 /* tp_new */
};


static PyModuleDef _pyrsmodule = {
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
	_pyrs_ContextType.tp_new = PyType_GenericNew;
	if (PyType_Ready(&_pyrs_ContextType) < 0)
		return NULL;

	m = PyModule_Create(&_pyrsmodule);
	if (m == NULL)
		return NULL;

	Py_INCREF(&_pyrs_ContextType);
	PyModule_AddObject(m, "_Context", (PyObject*)&_pyrs_ContextType);
	return m;
}