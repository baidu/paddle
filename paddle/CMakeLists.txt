add_subdirectory(scripts)
add_subdirectory(testing)
set(PYTHON_TESTS_DIR ${PADDLE_BINARY_DIR}/python/paddle/fluid/tests CACHE INTERNAL "python tests directory")
add_subdirectory(fluid)
# test
