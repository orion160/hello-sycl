cmake_minimum_required (
    VERSION 3.27
)

project (
    hello_sycl
    LANGUAGES C CXX
)

if ("${SYCL_IMPLEMENTATION}" STREQUAL "Intel OneAPI")
    find_package (IntelSYCL REQUIRED)
endif()

add_executable (hello)
set_target_properties (hello PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

target_sources (
    hello
    PRIVATE hello.cpp
)

if ("${SYCL_IMPLEMENTATION}" STREQUAL "Intel OneAPI")

    add_sycl_to_target (
        TARGET hello
        SOURCES hello.cpp
    )
endif ()
