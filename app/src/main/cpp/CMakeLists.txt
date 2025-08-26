cmake_minimum_required(VERSION 3.22)
project(whisper_android_bindings LANGUAGES C CXX)

# Standards
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Aggressive Optimierung – auch im Debug, sonst ist Whisper zäh auf Smartphones.
add_compile_options(-O3 -DNDEBUG -ffast-math -fstrict-aliasing -funroll-loops)
# Für C++:
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG -ffast-math -fstrict-aliasing -funroll-loops")
set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -O3 -DNDEBUG -ffast-math -fstrict-aliasing -funroll-loops")

# -----------------------------
# ggml / whisper.cpp Optionen
# (vor add_subdirectory setzen!)
# -----------------------------
set(GGML_OPENMP         OFF CACHE BOOL "" FORCE)
set(GGML_BLAS           OFF CACHE BOOL "" FORCE)
set(GGML_CUDA           OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN         OFF CACHE BOOL "" FORCE)
set(GGML_METAL          OFF CACHE BOOL "" FORCE)
set(GGML_ACCELERATE     OFF CACHE BOOL "" FORCE)   # Apple-only, hier AUS
set(GGML_USE_ACCELERATE OFF CACHE BOOL "" FORCE)   # ältere Option, sicherheitshalber AUS
set(WHISPER_FFMPEG      OFF CACHE BOOL "" FORCE)
set(WHISPER_SDL2        OFF CACHE BOOL "" FORCE)

# Pfad zu deiner whisper.cpp-Kopie
set(WHISPERCPP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/whispercpp)

# vor add_subdirectory(${WHISPERCPP_DIR} ...)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(WHISPER_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_ACCELERATE OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)

# whisper.cpp einbinden
add_subdirectory(${WHISPERCPP_DIR} ${CMAKE_BINARY_DIR}/whispercpp-build)

# WICHTIG: ggml/whisper Targets zwingen, kein fast-math zu benutzen
foreach(t ggml ggml-base ggml-cpu whisper)
    if (TARGET ${t})
        target_compile_options(${t} PRIVATE -fno-fast-math -fno-finite-math-only)
    endif()
endforeach()

# JNI-Bridge
add_library(whisper_jni SHARED
        whisper_jni.cpp
)

target_include_directories(whisper_jni PRIVATE
        ${WHISPERCPP_DIR}
        ${WHISPERCPP_DIR}/include
)

find_library(log-lib log)
target_link_libraries(whisper_jni PRIVATE whisper ${log-lib})