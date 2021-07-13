# On Linux, Use ccache if found ccache program
# On Windows, Use sccache if found sccache program
#   sccache(Shared ccache) is a ccache-like compiler caching tool.
#   please refer to https://github.com/mozilla/sccache

if(NOT WIN32)
    find_program(CCACHE_EXECUTABLE ccache)

    if(CCACHE_EXECUTABLE)
        execute_process(COMMAND ccache -V OUTPUT_VARIABLE ccache_output)
        execute_process(COMMAND ccache -s cache directory OUTPUT_VARIABLE cache_directory)
        string(REGEX MATCH "[0-9]+.[0-9]+" ccache_version ${ccache_output})
        message(STATUS "Ccache is founded on Linux, use ccache to speed up compile.")
        # show statistics summary of ccache
        message("ccache version\t\t\t    " ${ccache_version} "\n" ${cache_directory})
        set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ${CCACHE_EXECUTABLE})
        set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ${CCACHE_EXECUTABLE})
    endif(CCACHE_EXECUTABLE)
else()
    find_program(SCCACHE_EXECUTALBE sccache)
    message("===SCCACHE_EXECUTALBE====${SCCACHE_EXECUTALBE}")

    if(SCCACHE_EXECUTALBE)
        execute_process(COMMAND sccache --version OUTPUT_VARIABLE sccache_version)
        execute_process(COMMAND sccache --show-stats OUTPUT_VARIABLE sccache_status)
        message(STATUS "Sccache is founded on Windows, use ccache to speed up compile.")
        # show statistics summary of sccache
        message("sccache version\t\t\t    " ${sccache_version} "\n" ${sccache_status})
        set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ${SCCACHE_EXECUTALBE})
        set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ${SCCACHE_EXECUTALBE})
    endif(SCCACHE_EXECUTALBE)
endif()

