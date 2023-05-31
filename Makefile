CC := nvcc
CFLAGS :=

CUDA_FLAGS := --use_fast_math
INC_DIRS :=
LIB_DIRS :=
LIBS :=

OBJ_DIR := obj
PROFILE_DIR := profile

SRCS := $(wildcard src/*.cu)
HDRS := $(wildcard src/*.cuh)
OBJS := $(patsubst src/%.cu,$(OBJ_DIR)/%.o,$(SRCS))
EXEC := sha256-password-cracker

.PHONY: all clean

all: $(EXEC)

$(EXEC): $(OBJS)
	$(CC) -O 3 $(CFLAGS) $(CUDA_FLAGS) $(INC_DIRS) $(LIB_DIRS) -o $@ $^ $(LIBS)

$(OBJ_DIR)/%.o: src/%.cu $(HDRS)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(CUDA_FLAGS) $(INC_DIRS) -c $< -o $@

# profile: CFLAGS +=-g
profile: CUDA_FLAGS +=-lineinfo
profile: $(EXEC)
	mkdir -p $(PROFILE_DIR)
	# nvprof ./$(EXEC)
	# sudo nsys profile -o $(PROFILE_DIR)/profile_`date +%Y%m%d%H%M%S` ./$(EXEC)
	ncu -o $(PROFILE_DIR)/profile_`date +%Y%m%d%H%M%S` ./$(EXEC)
	# /usr/local/NVIDIA-Nsight-Compute/nv-nsight-cu-cli -o $(PROFILE_DIR)/profile_`date +%Y%m%d%H%M%S` ./$(EXEC)

zip:
	zip -r ../sha256-password-cracker.zip . -x ".git/*" "obj/*" "sha256-password-cracker"

clean:
	rm -rf $(EXEC) $(OBJ_DIR) $(PROFILE_DIR)
