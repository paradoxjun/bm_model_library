message("sub CMakeLists.txt")
message("CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
message("CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")

# Add to C/C++ flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-attributes")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-attributes")

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/models
    ${CMAKE_SOURCE_DIR}/include/ppocr
    ${CMAKE_SOURCE_DIR}/include/utils
    ${CMAKE_SOURCE_DIR}/include/bytetrack
    ${CMAKE_SOURCE_DIR}/include/opencv4
)

# 获取所有子目录的头文件到HEADER变量中
file(GLOB_RECURSE HEADER 
    ${CMAKE_SOURCE_DIR}/include/*.h*
    ${CMAKE_SOURCE_DIR}/include/models/*.h*
    ${CMAKE_SOURCE_DIR}/include/ppocr/*.h*
    ${CMAKE_SOURCE_DIR}/include/utils/*.h*
    ${CMAKE_SOURCE_DIR}/include/bytetrack/*.h*
    )
file(GLOB_RECURSE SOURCE 
    ${CMAKE_SOURCE_DIR}/src/yolo_det/*.c*
    ${CMAKE_SOURCE_DIR}/src/classify/*.c*
    ${CMAKE_SOURCE_DIR}/src/utils/*.c*
    ${CMAKE_SOURCE_DIR}/src/ppocr/*.c*
    ${CMAKE_SOURCE_DIR}/src/bytetrack/*.c*
    ${CMAKE_SOURCE_DIR}/src/yolo_pose/*.c*
    )
message("HEADER: ${HEADER}")
message("SOURCE: ${SOURCE}")

add_library(
    bm_model_library
    # STATIC  # 静态库
    SHARED #动态库
    ${HEADER}  # 自己调用的头文件全部打包在静态库中，提供给外部使用的头文件放在model_params.hpp和model_func.cpp中
    ${SOURCE}
)

set(BM_LIBS bmlib bmrt bmcv yuv)
set(JPU_LIBS bmjpuapi bmjpulite)
set(OPENCV_LIBS opencv_imgproc opencv_core opencv_highgui opencv_imgcodecs opencv_videoio opencv_video)

target_link_libraries(
    bm_model_library
    ${BM_LIBS}
    ${JPU_LIBS}
    ${OPENCV_LIBS}
)
target_compile_options(bm_model_library PRIVATE -Wno-attributes)
set_target_properties(bm_model_library PROPERTIES
    LINK_FLAGS "-Wl,--allow-shlib-undefined"
)
# 设置调试标志，增加编译器-g选项以包含调试信息
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -o0")
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -o0")
# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -g -o0")
# set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -g -o0")
# set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -g -o0")
# set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Os")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -Os")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -Os")
set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} -Os")
set(CMAKE_MODULE_LINKER_FLAGS_RELEASE "${CMAKE_MODULE_LINKER_FLAGS_RELEASE} -Os")
set(CMAKE_BUILD_TYPE Release)