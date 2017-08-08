#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <librealsense/rs.hpp>
#include <librealsense/rsutil.h>

#include "context.h"
#include "device.h"
#include "exception.h"


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

	import_array();

	DeviceType.tp_new = PyType_GenericNew;
	if (PyType_Ready(&DeviceType) < 0)
		return NULL;

	ContextType.tp_new = PyType_GenericNew;
	if (PyType_Ready(&ContextType) < 0)
		return NULL;

	m = PyModule_Create(&pyrsmodule);
	if (m == NULL)
		return NULL;

	RsError = PyErr_NewException("pyrs.error", NULL, NULL);
	Py_INCREF(RsError);
	PyModule_AddObject(m, "RsError", RsError);

	Py_INCREF(&ContextType);
	PyModule_AddObject(m, "_Context", (PyObject*)&ContextType);

	Py_INCREF(&DeviceType);
	PyModule_AddObject(m, "_Device", (PyObject*)&DeviceType);
	return m;
}