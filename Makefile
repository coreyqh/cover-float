# --- Project Configuration ---
TARGET_EXEC ?= coverfloat_reference
TARGET_SHARED ?= lib_coverfloat_reference.so
BUILD_DIR   ?= ./build
SRC_DIRS    ?= ./src ./submodules/spike/softfloat/

# Compiler and Flags
CC          ?= gcc
CFLAGS      ?= -Wall -Wextra -O2
CPPFLAGS    ?= -MMD -MP $(INC_FLAGS)
LDFLAGS     ?= 

# --- Automatic File Finding ---
# Find all C source files in source directories
SRCS        := $(shell find $(SRC_DIRS) -name *.c)

# Generate list of object file paths in the build directory
OBJS        := $(patsubst $(SRC_DIRS)/%.c, $(BUILD_DIR)/%.o, $(SRCS))

# Generate list of dependency file paths in the build directory
DEPS        := $(OBJS:.o=.d)

# Find all include directories and format them with -I flag
INC_DIRS    := $(shell find $(SRC_DIRS) -type d) ./include
INC_FLAGS   := $(addprefix -I,$(INC_DIRS))

# Command for creating directories
MKDIR_P     ?= mkdir -p
RM_CMD      ?= rm -rf

# --- Targets ---

.PHONY: build clean sim B1

build: $(BUILD_DIR)/$(TARGET_EXEC) $(BUILD_DIR)/$(TARGET_SHARED)

# Rule to create the final executable
$(BUILD_DIR)/$(TARGET_EXEC): $(OBJS)
	@echo "Linking: $@"
	$(MKDIR_P) $(dir $@)
	$(CC) $(OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/$(TARGET_SHARED): $(OBJS)
	@echo "Linking Shared Library: @a"
	$(MKDIR_P) $(dir $@)
	$(CC) $(OBJS) -fPIC -shared -o $@ $(LDFLAGS)

# Rule to compile C source files into object files
# $< is the prerequisite (source file), $@ is the target (object file)
$(BUILD_DIR)/%.o: $(SRC_DIRS)/%.c
	@echo "Compiling C file: $<"
	$(MKDIR_P) $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -fPIC -c $< -o $@

sim: 
	cd sim && vsim -c -do "do run.do"

B1:
	python3 script/B1.py && ./build/coverfloat_reference ./tests/testvectors/B1_tv.txt ./tests/covervectors/B1_cv.txt
	# TODO: Add more as needed

B10:
	python3 script/B10.py && ./build/coverfloat_reference ./tests/testvectors/B10_tv.txt ./tests/covervectors/B10_cv.txt
	# TODO: Add more as needed

# Clean target to remove build artifacts
clean:
	@echo "Cleaning build directory..."
	$(RM_CMD) $(BUILD_DIR)

# --- Include Dependency Files ---
# Include auto-generated dependency files if they exist
# -include $(DEPS)
