//
// Created by mingfei on 18/07/17.
//

#include <Python.h>
#include <librealsense/rs.h>
#include <librealsense/rsutil.h>
#include <numpy/arrayobject.h>

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