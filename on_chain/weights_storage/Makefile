include .env

# this is done using Giza CLI
# uncomment and login to Giza platform
# transpile:
# 	giza transpile network.onnx --output-path model

prepare_input:
	python3 input.py
	@input=$$(cat input.txt); \
	cd ../../cairo-vm/cairo1-run; \
	cargo run ../../models/converter.cairo \
		--trace_file trace.bin \
		--memory_file memory.bin \
		--args "$$input" \
		--layout small \
		--print_output > ../../on_chain/default/input.data; \
	rm trace.bin memory.bin; \
	cd ../../on_chain/default
	python3 clean.py

build:
	cd model/inference; \
	scarb build

declare:
	starkli declare --account ${STARKNET_ACCOUNT} --keystore ${STARKNET_KEYSTORE} --network ${STARKNET_NETWORK} model/inference/target/dev/model_10l_fc_relu_OrionRunner.contract_class.json --compiler-version 2.6.2 

deploy:
	starkli deploy --account ${STARKNET_ACCOUNT} --keystore ${STARKNET_KEYSTORE} --network ${STARKNET_NETWORK} 

inference:
	cd model/inference; \
	snforge test --max-n-steps 10000000 --detailed-resources inference_test 
