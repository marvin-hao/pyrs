//
// Created by mingfei on 07/08/17.
//

#ifndef PYRS_EXCEPTION_H
#define PYRS_EXCEPTION_H

#include <Python.h>
#include <librealsense/rs.hpp>

static PyObject *RsError;

#define PyThrowRsErr(e) \
	std::string reason = std::string(e.what()); \
	std::string explain = reason + "\nFunction: " + e.get_failed_function() + "\nArgument: " + e.get_failed_args(); \
	PyErr_SetString(RsError, explain.c_str()); \
	return NULL;

#endif //PYRS_EXCEPTION_H
