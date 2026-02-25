# --- Targets ---

RM_CMD ?= rm -rf

.PHONY: build clean sim all B1 B9 B10 B12

# Notice that we pass --managed-python, we do this so that uv (scikit-build-core)
# will have a python enviornment with Python.h to build with.
all:
	uv run --managed-python cover-float-testgen

# Build target to compile the pybind11 module (if necessary)
build:
	@echo "Building python module"
	uv build --managed-python

sim:
	cd sim && vsim -c -do "do run.do"

# Helper target to parse test vectors
parse_vectors = @mkdir -p tests/readable && uv run --managed-python script/parse_testvectors.py tests/testvectors/$(1)_tv.txt tests/readable/$(1)_parsed.txt

B1:
	uv run --managed-python cover-float-testgen --model B1
	$(call parse_vectors,B1)

B9:
	uv run --managed-python cover-float-testgen --model B9
	$(call parse_vectors,B9)

B10:
	uv run --managed-python cover-float-testgen --model B10
	$(call parse_vectors,B10)

B12:
	uv run --managed-python cover-float-testgen --model B12
	$(call parse_vectors,B12)


B14:
	uv run --managed-python cover-float-testgen --model B14

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
