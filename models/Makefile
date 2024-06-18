generate:
	python3 gen.py

prepare_input:
	python3 input.py
	@input=$$(cat input.txt); \
	cd ../cairo-vm/cairo1-run; \
	cargo run ../../models/converter.cairo \
		--trace_file trace.json \
		--memory_file memory.json \
		--args "$$input" \
		--layout all_cairo \
		--print_output > ../../models/input.data; \
	rm trace.json memory.json; \
	cd ../../models
	python3 clean.py

install:
	pip install -r requirements.txt
