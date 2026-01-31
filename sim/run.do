

onbreak {resume}

vlib coverfloat_worklib

vlog -lint -sv -work coverfloat_worklib ../coverage/coverfloat.sv

vsim -lib coverfloat_worklib coverfloat

run -all

#coverage report -cvg -detail

# might neet many of these later on...
coverage save coverfloat.ucdb