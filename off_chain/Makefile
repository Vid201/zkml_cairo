build:
	cd inference
	scarb build
	cd ..

run:
	@input=$$(cat ../models/input.data); \
	../cairo-vm/target/debug/cairo1-run inference/target/dev/inference.sierra.json \
		--air_public_input public_input.json \
		--air_private_input private_input.json \
		--trace_file trace.bin \
		--memory_file memory.bin \
		--layout small \
		--args "$$input" \
		--proof_mode \
		--print_output

gen_params:
	python3 gen_stone_params.py public_input.json

prove:
	./../cpu_air_prover \
		--out_file=proof.json \
		--private_input_file=private_input.json \
		--public_input_file=public_input.json \
		--prover_config_file=cpu_air_prover_config.json \
		--parameter_file=cpu_air_params.json

verify:
	./../cpu_air_verifier \
		--in_file=proof.json \
	&& echo "Proof is valid"
