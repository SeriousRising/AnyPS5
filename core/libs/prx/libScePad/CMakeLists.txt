add_library(libScePad SHARED EXCLUDE_FROM_ALL
    ${CMAKE_CURRENT_SOURCE_DIR}/Export.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/PadState.cpp
)
target_include_directories(libScePad PRIVATE ${LIBS_INCLUDE_DIR})
set_target_properties(libScePad PROPERTIES CXX_EXTENSIONS OFF)
configure_windows_unwind(libScePad)

if(BUILD_TESTING)
    add_test_executable(pad_motion_emulation_tests tests/PadMotionEmulation.cpp)
    target_include_directories(pad_motion_emulation_tests PRIVATE ${LIBS_INCLUDE_DIR})
    add_test(NAME pad_motion_emulation COMMAND pad_motion_emulation_tests)
    add_test_executable(guest_pad_tests ${LIBS_INCLUDE_DIR}/tests/GuestPad.cpp)
    target_include_directories(guest_pad_tests PRIVATE ${LIBS_INCLUDE_DIR} ${ANYPS5_SDL2_INCLUDE_DIR})
    target_link_libraries(guest_pad_tests PRIVATE libScePad libc)
    configure_windows_unwind(guest_pad_tests)
    add_test(NAME guest_pad COMMAND guest_pad_tests)
    if(WIN32)
        set_tests_properties(guest_pad PROPERTIES ENVIRONMENT_MODIFICATION
            "PATH=path_list_prepend:$<TARGET_FILE_DIR:libScePad>")
    endif()
endif()
