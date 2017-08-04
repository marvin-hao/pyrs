cmake_minimum_required(VERSION 3.5)
project(pyrs)
set(CMAKE_CXX_STANDARD 11)
set(SOURCE_FILES _pyrs.c)

include_directories(/usr/local/include/python3.6m/)
link_directories(/usr/local/lib)
execute_process(COMMAND /usr/local/bin/python3.6m-config --cflags PY36_CFLAGS)


set(CMAKE_C_FLAGS ${PY36_CFLAGS})

add_library(pyrs SHARED ${SOURCE_FILES})