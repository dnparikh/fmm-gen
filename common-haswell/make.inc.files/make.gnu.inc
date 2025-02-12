CC = gcc-12
CXX = g++

ARCH = gcc-ar-12
ARCHFLAGS = cr
RANLIB = gcc-ranlib-12

ifeq ($(BLISLAB_USE_BLAS),true)
CFLAGS = -$(COMPILER_OPT_LEVEL) -fopenmp -mavx -fPIC -DUSE_BLAS
#LDLIBS = $(LIBBLISLAB) -lpthread -lm -fopenmp -lblas                                 #sudo apt-get install libblas-dev (Netlib BLAS)
#LDLIBS = $(LIBBLISLAB) ${BLAS_DIR}/lib/libblis.a -lpthread -lm -fopenmp              #blis: https://github.com/flame/blis/
#LDLIBS = $(LIBBLISLAB) ${BLAS_DIR}/lib/libopenblas.a -lpthread -lm -fopenmp          #openblas: https://github.com/xianyi/OpenBLAS
LDLIBS = $(LIBBLISLAB) ${BLAS_DIR}/lib/*.a -lpthread -lm -fopenmp                     #*/General for any static link file: .a
else
CFLAGS = -$(COMPILER_OPT_LEVEL) -fopenmp -mavx -fPIC
LDLIBS = $(LIBBLISLAB) -lpthread -lm -fopenmp
endif

LIBBLISLAB = $(BLISLAB_DIR)/lib/libblislab.a
SHAREDLIBBLISLAB = $(BLISLAB_DIR)/lib/libblislab.so

LDFLAGS = -I$(BLISLAB_DIR)/include -I$(BLISLAB_DIR)/kernels -I{BLAS_DIR}/include

OS_NAME = $(shell uname -s)
ifeq ($(OS_NAME),Linux)
LDLIBS += -lrt
endif

