generate:
	python3 gen.py

prepare_input:
	python3 input.py
	@input=$$(cat input.txt); \
	cd ../cairo-vm/cairo1-run; \
	cargo run ../../models/converter.cairo \
		--trace_file trace.bin \
		--memory_file memory.bin \
		--args "$$input" \
		--layout small \
		--print_output > ../../models/input.data; \
	rm trace.bin memory.bin; \
	cd ../../models
	python3 clean.py

install:
	pip install -r requirements.txt
