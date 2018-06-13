#-------------------------------------------------
#
# Project created by QtCreator 2017-04-10T21:54:23
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++11

TARGET = MLP_image_generator
TEMPLATE = app

SOURCES +=  main.cpp\
			mainwindow.cpp \
			mlp.cpp \
			vec2d.cpp \
			vec3d.cpp \
			solver.cpp \
    image_generation.cpp \
    FastNoise.cpp \
    layer_stuff.cpp \
    mlp_iterator.cpp \
    perlinnoise.cpp \
    saveimage.cpp \
    globals.cpp

HEADERS  += mainwindow.h \
			mlp.hpp \
			vec2d.hpp \
			vec3d.hpp \
			solver.hpp \
    image_generation.hpp \
    FastNoise.h \
    mlp_iterator.h \
    perlinnoise.h \
    saveimage.h \
    globals.hpp \
    configurations.h

FORMS    += mainwindow.ui

DISTFILES += \
    mlp.cu \
    README.md \
    .gitignore




CONFIG   += console
CONFIG   -= app_bundle

# Define output directories
DESTDIR = release
OBJECTS_DIR = release/obj
CUDA_OBJECTS_DIR = release/cuda

# This makes the .cu files appear in your project
OTHER_FILES +=  mlp.cu

# CUDA settings <-- may change depending on your system
CUDA_SOURCES += mlp.cu
#CUDA_SDK = "C:/ProgramData/NVIDIA Corporation/NVIDIA GPU Computing SDK 4.2/C"   # Path to cuda SDK install
CUDA_DIR = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v9.2"            # Path to cuda toolkit install
SYSTEM_NAME = "x64"         # Depending on your system either 'Win32', 'x64', or 'Win64'
SYSTEM_TYPE = 64            # '32' or '64', depending on your system
CUDA_ARCH = sm_30           # Type of CUDA architecture, for example 'compute_10', 'compute_11', 'sm_10'
NVCC_OPTIONS = --use_fast_math -O3 -std=c++11

# include paths
INCLUDEPATH += $$CUDA_DIR/include
			   #$$CUDA_SDK/common/inc/ \
			   #$$CUDA_SDK/../shared/inc/

# library directories
QMAKE_LIBDIR += $$CUDA_DIR/lib/$$SYSTEM_NAME
				#$$CUDA_SDK/common/lib/$$SYSTEM_NAME \
				#$$CUDA_SDK/../shared/lib/$$SYSTEM_NAME


# The following library conflicts with something in Cuda
QMAKE_LFLAGS_RELEASE = /NODEFAULTLIB:msvcrt.lib
QMAKE_LFLAGS_DEBUG   = /NODEFAULTLIB:msvcrtd.lib

# Add the necessary libraries
CUDA_LIBS = cuda cudart
# The following makes sure all path names (which often include spaces) are put between quotation marks
CUDA_INC = $$join(INCLUDEPATH,'" -I"','-I"','"')
NVCC_LIBS = $$join(CUDA_LIBS,' -l','-l', '')
LIBS += cuda.lib cudart.lib #$$join(CUDA_LIBS,'.lib ', '', '.lib')

# Configuration of the Cuda compiler
CONFIG(debug, debug|release) {
	# Debug mode
        QMAKE_CFLAGS_RELEASE += /MTd
        QMAKE_CXXFLAGS_RELEASE += /MTd
        QMAKE_CFLAGS_RELEASE -= -MDd
        QMAKE_CXXFLAGS_RELEASE -= -MDd
	cuda_d.input = CUDA_SOURCES
	cuda_d.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
	cuda_d.commands = $$CUDA_DIR/bin/nvcc.exe -D_DEBUG $$NVCC_OPTIONS $$CUDA_INC $$NVCC_LIBS --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
	cuda_d.dependency_type = TYPE_C
	QMAKE_EXTRA_COMPILERS += cuda_d
}
else {
	# Release mode
	QMAKE_CFLAGS_RELEASE += /MT
	QMAKE_CXXFLAGS_RELEASE += /MT
	QMAKE_CFLAGS_RELEASE -= -MD
	QMAKE_CXXFLAGS_RELEASE -= -MD
	cuda.input = CUDA_SOURCES
	cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
	cuda.commands = $$CUDA_DIR/bin/nvcc.exe $$NVCC_OPTIONS $$CUDA_INC $$NVCC_LIBS --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
	cuda.dependency_type = TYPE_C
	QMAKE_EXTRA_COMPILERS += cuda
}
