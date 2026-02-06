# --- Targets ---

RM_CMD ?= rm -rf

.PHONY: build clean sim all-tests B1 B10

# Build target to compile the pybind11 module (if necessary) Notice that
# we pass --managed-python, we do this so that uv (scikit-build-core)
# will have a python enviornment with Python.h to build with.
build:
	@echo "Building python module"
	uv build --managed-python

sim: 
	cd sim && vsim -c -do "do run.do"

all-tests:
	uv run --managed-python cover-float-testgen

B1:
	uv run --managed-python cover-float-testgen --model B1

B10:
	uv run --managed-python cover-float-testgen --model B10

# Clean target to remove build artifacts
clean:
	@echo "Cleaning build directory..."
	$(RM_CMD) build/
	$(RM_CMD) dist/ 
	$(RM_CMD) src/cover_float/__pycache__/
	$(RM_CMD) src/cover_float/testgen/__pycache__/
	$(RM_CMD) sim/coverfloat_worklib/
	$(RM_CMD) sim/transcript
	$(RM_CMD) sim/coverfloat.ucdb

# --- Include Dependency Files ---
# Include auto-generated dependency files if they exist
# -include $(DEPS)
