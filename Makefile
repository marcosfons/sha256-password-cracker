CC := nvcc
CFLAGS := 

# Add additional CUDA flags if needed
# -arch=sm_XX
CUDA_FLAGS := 

# Add additional include directories if needed
INC_DIRS :=

# Add additional library directories if needed
LIB_DIRS :=

# Add additional libraries if needed
LIBS :=

SRCS := $(wildcard src/*.cu)
OBJ_DIR := obj
OBJS := $(patsubst src/%.cu,$(OBJ_DIR)/%.o,$(SRCS))
EXEC := sha256-password-cracker
PROFILE_DIR := profile

.PHONY: all clean

all: $(EXEC)

$(EXEC): $(OBJS)
	$(CC) $(CFLAGS) $(CUDA_FLAGS) $(INC_DIRS) $(LIB_DIRS) -o $@ $^ $(LIBS)

$(OBJ_DIR)/%.o: src/%.cu
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(CUDA_FLAGS) $(INC_DIRS) -c $< -o $@

profile: $(EXEC)
	mkdir -p $(PROFILE_DIR)
	nsys profile -o $(PROFILE_DIR)/profile_`date +%Y%m%d%H%M%S`.qdrep ./$(EXEC)

clean:
	rm -rf $(EXEC) $(OBJ_DIR) $(PROFILE_DIR)
