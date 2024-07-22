use orion::numbers::{FixedTrait, FP16x16};
use orion::operators::nn::{NNTrait, FP16x16NN};
use orion::operators::tensor::{FP16x16Tensor, Tensor, TensorTrait};

#[starknet::interface]
trait IModelContract<TContractState> {
    fn add_weights(ref self: TContractState, _weights: Array<u32>, _signs: Array<bool>);
    fn get_weights(self: @TContractState, shape: Array<usize>, start_index: usize) -> Tensor<FP16x16>;
    fn inference(self: @TContractState, input: Array<felt252>) -> Array<felt252>;
}

#[starknet::contract]
mod ModelContract {
    use alexandria_storage::{List, ListTrait};
    use orion::numbers::{FixedTrait, FP16x16};
    use orion::operators::nn::{NNTrait, FP16x16NN};
    use orion::operators::tensor::{FP16x16Tensor, Tensor, TensorTrait};
    use starknet::SyscallResult;

    #[storage]
    struct Storage {
        weights: List<u32>,
        signs: List<bool>,
    }

    #[abi(embed_v0)]
    impl ModelContractImpl of super::IModelContract<ContractState> {
        fn add_weights(ref self: ContractState, _weights: Array<u32>, _signs: Array<bool>) {
            assert(_weights.len() == _signs.len(), 'Arays must have the same len');

            let mut weights = self.weights.read();
            let mut signs = self.signs.read();

            weights.append_span(_weights.span()).expect('syscallresult error');
            signs.append_span(_signs.span()).expect('syscallresult error');
        }
    
        fn get_weights(self: @ContractState, shape: Array<usize>, start_index: usize) -> Tensor<FP16x16> {
            let mut total_size: usize = 1;
            let mut index = 0;
            let mut end = shape.len();
            let shape = shape.span();
            
            loop {
                total_size *= *shape.at(index);
    
                index += 1;
                
                if index == end {
                    break;
                }
            };
    
            let mut data = array![];
            index = 0;

            let _: SyscallResult<()> = loop {
                if total_size == 0 {
                    break Result::Ok(());
                }
                let weight = match self.weights.read().get(start_index+index) {
                    Result::Ok(v) => v,
                    Result::Err(e) => { break Result::Err(e); }
                }.expect('List index out of bounds');
                let sign = match self.signs.read().get(start_index+index) {
                    Result::Ok(v) => v,
                    Result::Err(e) => { break Result::Err(e); }
                }.expect('List index out of bounds');
                data.append(FP16x16 { mag: weight, sign: sign });
                total_size -= 1;
                index += 1;
            };

            TensorTrait::new(shape, data.span())
        }
    
        fn inference(self: @ContractState, input: Array<felt252>) -> Array<felt252> {
            let mut input = input.span();
            let node_input: Tensor<FP16x16> = Serde::deserialize(ref input).unwrap();

            let mut weights_counter = 0;

            let weights1 = self.get_weights(array![90, 100], weights_counter);
            weights_counter += weights1.data.len();
            let bias1 = self.get_weights(array![1, 90], weights_counter);
            weights_counter += bias1.data.len();

            let weights2 = self.get_weights(array![80, 90], weights_counter);
            weights_counter += weights2.data.len();
            let bias2 = self.get_weights(array![1, 80], weights_counter);
            weights_counter += bias2.data.len();

            let weights3 = self.get_weights(array![70, 80], weights_counter);
            weights_counter += weights3.data.len();
            let bias3 = self.get_weights(array![1, 70], weights_counter);
            weights_counter += bias3.data.len();

            let weights4 = self.get_weights(array![60, 70], weights_counter);
            weights_counter += weights4.data.len();
            let bias4 = self.get_weights(array![1, 60], weights_counter);
            weights_counter += bias4.data.len();

            let weights5 = self.get_weights(array![50, 60], weights_counter);
            weights_counter += weights5.data.len();
            let bias5 = self.get_weights(array![1, 50], weights_counter);
            weights_counter += bias5.data.len();

            let weights6 = self.get_weights(array![40, 50], weights_counter);
            weights_counter += weights6.data.len();
            let bias6 = self.get_weights(array![1, 40], weights_counter);
            weights_counter += bias6.data.len();

            let weights7 = self.get_weights(array![30, 40], weights_counter);
            weights_counter += weights7.data.len();
            let bias7 = self.get_weights(array![1, 30], weights_counter);
            weights_counter += bias7.data.len();

            let weights8 = self.get_weights(array![20, 30], weights_counter);
            weights_counter += weights8.data.len();
            let bias8 = self.get_weights(array![1, 20], weights_counter);
            weights_counter += bias8.data.len();

            let weights9 = self.get_weights(array![10, 20], weights_counter);
            weights_counter += weights9.data.len();
            let bias9 = self.get_weights(array![1, 10], weights_counter);
            weights_counter += bias9.data.len();

            let weights10 = self.get_weights(array![10, 10], weights_counter);
            weights_counter += weights10.data.len();
            let bias10 = self.get_weights(array![1, 10], weights_counter);
            weights_counter += bias10.data.len();
    
            let node__aff1_gemm_output_0 = NNTrait::gemm(node_input, weights1, Option::Some(bias1), Option::Some(FP16x16 { mag: 65536, sign: false }), Option::Some(FP16x16 { mag: 65536, sign: false }), false, true);
            let node__aff2_gemm_output_0 = NNTrait::gemm(node__aff1_gemm_output_0, weights2, Option::Some(bias2), Option::Some(FP16x16 { mag: 65536, sign: false }), Option::Some(FP16x16 { mag: 65536, sign: false }), false, true);
            let node__aff3_gemm_output_0 = NNTrait::gemm(node__aff2_gemm_output_0, weights3, Option::Some(bias3), Option::Some(FP16x16 { mag: 65536, sign: false }), Option::Some(FP16x16 { mag: 65536, sign: false }), false, true);
            let node__aff4_gemm_output_0 = NNTrait::gemm(node__aff3_gemm_output_0, weights4, Option::Some(bias4), Option::Some(FP16x16 { mag: 65536, sign: false }), Option::Some(FP16x16 { mag: 65536, sign: false }), false, true);
            let node__aff5_gemm_output_0 = NNTrait::gemm(node__aff4_gemm_output_0, weights5, Option::Some(bias5), Option::Some(FP16x16 { mag: 65536, sign: false }), Option::Some(FP16x16 { mag: 65536, sign: false }), false, true);
            let node__aff6_gemm_output_0 = NNTrait::gemm(node__aff5_gemm_output_0, weights6, Option::Some(bias6), Option::Some(FP16x16 { mag: 65536, sign: false }), Option::Some(FP16x16 { mag: 65536, sign: false }), false, true);
            let node__aff7_gemm_output_0 = NNTrait::gemm(node__aff6_gemm_output_0, weights7, Option::Some(bias7), Option::Some(FP16x16 { mag: 65536, sign: false }), Option::Some(FP16x16 { mag: 65536, sign: false }), false, true);
            let node__aff8_gemm_output_0 = NNTrait::gemm(node__aff7_gemm_output_0, weights8, Option::Some(bias8), Option::Some(FP16x16 { mag: 65536, sign: false }), Option::Some(FP16x16 { mag: 65536, sign: false }), false, true);
            let node__aff9_gemm_output_0 = NNTrait::gemm(node__aff8_gemm_output_0, weights9, Option::Some(bias9), Option::Some(FP16x16 { mag: 65536, sign: false }), Option::Some(FP16x16 { mag: 65536, sign: false }), false, true);
            let node__aff10_gemm_output_0 = NNTrait::gemm(node__aff9_gemm_output_0, weights10, Option::Some(bias10), Option::Some(FP16x16 { mag: 65536, sign: false }), Option::Some(FP16x16 { mag: 65536, sign: false }), false, true);
            let node_output = NNTrait::relu(@node__aff10_gemm_output_0);
    
            let mut output: Array<felt252> = ArrayTrait::new();
            node_output.serialize(ref output);
            output
        }
    }
}
