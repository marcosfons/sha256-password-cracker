CC := nvcc
CFLAGS :=--compiler-options -Wall

CUDA_FLAGS := --use_fast_math 
# -maxrregcount 110
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
	$(CC) -O 2 $(CFLAGS) $(CUDA_FLAGS) $(INC_DIRS) $(LIB_DIRS) -o $@ $^ $(LIBS)

$(OBJ_DIR)/%.o: src/%.cu $(HDRS)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(CUDA_FLAGS) $(INC_DIRS) -c $< -o $@

googlecolab: CFLAGS += -D BLOCKS_PER_ENTRY=256
googlecolab: CFLAGS += -D THREADS=1024
googlecolab: CFLAGS += -D RUNS_PER_ITERATION=8
googlecolab: CFLAGS += -D LOOPS_INSIDE_THREAD=64
googlecolab: $(EXEC)

profile: CUDA_FLAGS +=-lineinfo
profile: $(EXEC)
# mkdir -p $(PROFILE_DIR)
# sudo nsys profile -o $(PROFILE_DIR)/profile_`date +%Y%m%d%H%M%S` ./$(EXEC)
# ncu --target-processes all -o $(PROFILE_DIR)/profile_`date +%Y%m%d%H%M%S` ./$(EXEC)

zip:
	zip -r ../sha256-password-cracker.zip . -x "*.git/*" "obj/*" "sha256-password-cracker"

clean:
	rm -rf $(EXEC) $(OBJ_DIR) $(PROFILE_DIR)
