use orion::numbers::{FixedTrait, FP16x16};
use orion::operators::nn::{NNTrait, FP16x16NN};
use orion::operators::tensor::{FP16x16Tensor, Tensor, TensorTrait};

#[starknet::interface]
trait IModelContract<TContractState> {
    fn add_weights(ref self: TContractState, _weights: Array<felt252>, _signs: Array<bool>);
    fn get_weights(self: @TContractState, shape: Array<usize>, start_index_weights: usize, start_index_signs: usize) -> (Tensor<FP16x16>, usize, usize);
    fn inference(self: @TContractState, input: Array<felt252>) -> Array<felt252>;
}

#[starknet::contract]
mod ModelContract {
    use alexandria_bytes::{Bytes, BytesTrait, BytesStore};
    use orion::numbers::{FixedTrait, FP16x16};
    use orion::operators::nn::{NNTrait, FP16x16NN};
    use orion::operators::tensor::{FP16x16Tensor, Tensor, TensorTrait};

    use node_aff1_weight::get_node_aff1_weight;
    use node_aff1_bias::get_node_aff1_bias;
    use node_aff2_weight::get_node_aff2_weight;
    use node_aff2_bias::get_node_aff2_bias;
    use node_aff3_weight::get_node_aff3_weight;
    use node_aff3_bias::get_node_aff3_bias;
    use node_aff4_weight::get_node_aff4_weight;
    use node_aff4_bias::get_node_aff4_bias;
    use node_aff5_weight::get_node_aff5_weight;
    use node_aff5_bias::get_node_aff5_bias;
    use node_aff6_weight::get_node_aff6_weight;
    use node_aff6_bias::get_node_aff6_bias;
    use node_aff7_weight::get_node_aff7_weight;
    use node_aff7_bias::get_node_aff7_bias;
    use node_aff8_weight::get_node_aff8_weight;
    use node_aff8_bias::get_node_aff8_bias;
    use node_aff9_weight::get_node_aff9_weight;
    use node_aff9_bias::get_node_aff9_bias;
    use node_aff10_weight::get_node_aff10_weight;
    use node_aff10_bias::get_node_aff10_bias;

    #[storage]
    struct Storage {
        weights: Bytes,
        signs: Bytes,
    }

    #[abi(embed_v0)]
    impl ModelContractImpl of super::IModelContract<ContractState> {
        fn add_weights(ref self: ContractState, _weights: Array<felt252>, _signs: Array<bool>) {
            assert(_weights.len() == _signs.len(), 'Arays must have the same len');
    
            let mut index = 0;
            let mut end = _weights.len();
    
            let mut weights = self.weights.read();
            let mut signs = self.signs.read();
    
            loop {
                weights.append_felt252(_weights.at(index).clone());
    
                if _signs.at(index).clone() {
                    signs.append_u8(1);
                } else {
                    signs.append_u8(0);
                }
    
                index += 1;

                if index == end {
                    break;
                }
            };

            self.weights.write(weights);
            self.signs.write(signs);
        }
    
        fn get_weights(self: @ContractState, shape: Array<usize>, start_index_weights: usize, start_index_signs: usize) -> (Tensor<FP16x16>, usize, usize) {
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
            let mut index_weights = start_index_weights;
            let mut index_signs = start_index_signs;
    
            loop {
                let (new_index_weights, num) = self.weights.read().read_felt252(index_weights);
                let (new_index_signs, sign) = self.signs.read().read_u8(index_signs);
    
                data.append(FP16x16 { mag: num.try_into().unwrap(), sign: sign == 1 });
                
                index_weights = new_index_weights;
                index_signs = new_index_signs;
    
                total_size -= 1;
    
                if total_size == 0 {
                    break;
                }
            };
    
            (TensorTrait::new(shape, data.span()), index_weights, index_signs)
        }
    
        fn inference(self: @ContractState, input: Array<felt252>) -> Array<felt252> {
            let mut input = input.span();
            let node_input: Tensor<FP16x16> = Serde::deserialize(ref input).unwrap();
    
            let node__aff1_gemm_output_0 = NNTrait::gemm(node_input, get_node_aff1_weight(), Option::Some(get_node_aff1_bias()), Option::Some(FP16x16 { mag: 65536, sign: false }), Option::Some(FP16x16 { mag: 65536, sign: false }), false, true);
            let node__aff2_gemm_output_0 = NNTrait::gemm(node__aff1_gemm_output_0, get_node_aff2_weight(), Option::Some(get_node_aff2_bias()), Option::Some(FP16x16 { mag: 65536, sign: false }), Option::Some(FP16x16 { mag: 65536, sign: false }), false, true);
            let node__aff3_gemm_output_0 = NNTrait::gemm(node__aff2_gemm_output_0, get_node_aff3_weight(), Option::Some(get_node_aff3_bias()), Option::Some(FP16x16 { mag: 65536, sign: false }), Option::Some(FP16x16 { mag: 65536, sign: false }), false, true);
            let node__aff4_gemm_output_0 = NNTrait::gemm(node__aff3_gemm_output_0, get_node_aff4_weight(), Option::Some(get_node_aff4_bias()), Option::Some(FP16x16 { mag: 65536, sign: false }), Option::Some(FP16x16 { mag: 65536, sign: false }), false, true);
            let node__aff5_gemm_output_0 = NNTrait::gemm(node__aff4_gemm_output_0, get_node_aff5_weight(), Option::Some(get_node_aff5_bias()), Option::Some(FP16x16 { mag: 65536, sign: false }), Option::Some(FP16x16 { mag: 65536, sign: false }), false, true);
            let node__aff6_gemm_output_0 = NNTrait::gemm(node__aff5_gemm_output_0, get_node_aff6_weight(), Option::Some(get_node_aff6_bias()), Option::Some(FP16x16 { mag: 65536, sign: false }), Option::Some(FP16x16 { mag: 65536, sign: false }), false, true);
            let node__aff7_gemm_output_0 = NNTrait::gemm(node__aff6_gemm_output_0, get_node_aff7_weight(), Option::Some(get_node_aff7_bias()), Option::Some(FP16x16 { mag: 65536, sign: false }), Option::Some(FP16x16 { mag: 65536, sign: false }), false, true);
            let node__aff8_gemm_output_0 = NNTrait::gemm(node__aff7_gemm_output_0, get_node_aff8_weight(), Option::Some(get_node_aff8_bias()), Option::Some(FP16x16 { mag: 65536, sign: false }), Option::Some(FP16x16 { mag: 65536, sign: false }), false, true);
            let node__aff9_gemm_output_0 = NNTrait::gemm(node__aff8_gemm_output_0, get_node_aff9_weight(), Option::Some(get_node_aff9_bias()), Option::Some(FP16x16 { mag: 65536, sign: false }), Option::Some(FP16x16 { mag: 65536, sign: false }), false, true);
            let node__aff10_gemm_output_0 = NNTrait::gemm(node__aff9_gemm_output_0, get_node_aff10_weight(), Option::Some(get_node_aff10_bias()), Option::Some(FP16x16 { mag: 65536, sign: false }), Option::Some(FP16x16 { mag: 65536, sign: false }), false, true);
            let node_output = NNTrait::relu(@node__aff10_gemm_output_0);
    
            let mut output: Array<felt252> = ArrayTrait::new();
            node_output.serialize(ref output);
            output
        }
    }
}
