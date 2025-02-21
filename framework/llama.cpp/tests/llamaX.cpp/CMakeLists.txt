cmake_minimum_required(VERSION 3.0)
project(llamax-qwen2)
set(CMAKE_BUILD_TYPE Debug)
set(TARGET llamax-qwen2)

add_executable(${TARGET} main.cpp)

target_include_directories(${TARGET} PRIVATE
  ../../include 
  ../../ggml/include
  ../../common
)

# find_library(LIBLLAMA_PATH NAMES libllama.so PATHS build/src)
# find_library(LIBCOMMON_PATH NAMES libcommon.a PATHS build/common)
# find_library(LIBGGML_PATH NAMES libggml.so PATHS build/ggml/src)


set(LIBLLAMA_PATH "/home/chunyu123/github/llama.cpp/build/src/libllama.so")
set(LIBCOMMON_PATH "/home/chunyu123/github/llama.cpp/build/common/libcommon.a")
set(LIBGGML_PATH "/home/chunyu123/github/llama.cpp/build/ggml/src/libggml.so")

target_link_libraries(${TARGET} PRIVATE
  ${LIBLLAMA_PATH}
  ${LIBCOMMON_PATH}
  ${LIBGGML_PATH}
  ${CMAKE_THREAD_LIBS_INIT} # 链接线程库
)

target_compile_features(${TARGET} PRIVATE cxx_std_11)

install(TARGETS ${TARGET} RUNTIME DESTINATION bin) # 根据需要调整安装路径