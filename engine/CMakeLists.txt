cmake_minimum_required(VERSION 3.20)
project(muonengine CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Spout2 SDK (DirectX 11 part), unchanged: third_party/spout/README.md
add_library(spout STATIC
    third_party/spout/SpoutCopy.cpp
    third_party/spout/SpoutDirectX.cpp
    third_party/spout/SpoutDX.cpp
    third_party/spout/SpoutFrameCount.cpp
    third_party/spout/SpoutSenderNames.cpp
    third_party/spout/SpoutSharedMemory.cpp
    third_party/spout/SpoutUtils.cpp
)
target_include_directories(spout PUBLIC third_party/spout)
target_compile_options(spout PRIVATE /W0)

add_executable(muonengine
    src/live.cpp
    src/main.cpp
    src/pool.cpp
    src/renderer.cpp
)
# the shaders are compiled at start-up, from the source folder unless a "shaders" folder sits next to the exe
target_compile_definitions(muonengine PRIVATE
    MUON_SHADER_DIR=L"${CMAKE_CURRENT_SOURCE_DIR}/shaders"
    MUON_ROOT_DIR=L"${CMAKE_CURRENT_SOURCE_DIR}/.."
    NOMINMAX UNICODE _UNICODE _CRT_SECURE_NO_WARNINGS)
target_compile_options(muonengine PRIVATE /W3 /utf-8)
target_link_libraries(muonengine PRIVATE spout d3d11 dxgi d3dcompiler dxguid winmm xaudio2 ws2_32)
