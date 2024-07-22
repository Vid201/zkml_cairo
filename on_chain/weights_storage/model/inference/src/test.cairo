#[cfg(test)]
mod tests {
    use snforge_std::{declare, ContractClassTrait};

    use model_10l_fc_relu::contract::{IModelContractDispatcher, IModelContractDispatcherTrait};

    use node_aff1_weight::get_node_aff1_weight;
    use node_aff2_weight::get_node_aff2_weight;
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

    use core::debug::PrintTrait;

    use alexandria_bytes::{Bytes, BytesTrait, BytesStore};
    use orion::numbers::{FixedTrait, FP16x16};
    use orion::operators::nn::{NNTrait, FP16x16NN};
    use orion::operators::tensor::{FP16x16Tensor, Tensor, TensorTrait};

    #[test]
    #[available_gas(99999999999999999)]
    fn storage_test() {
        let contract = declare("ModelContract").unwrap();
        let (contract_address, _) = contract.deploy(@array![]).unwrap();
        let dispatcher = IModelContractDispatcher { contract_address };

        let weights: Array<felt252> = array![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let signs: Array<bool> = array![true, false, true, false, true, false, true, false, true, false];

        dispatcher.add_weights(weights, signs);
        let shape: Array<usize> = array![2, 5];
        let (tensor, _, _) = dispatcher.get_weights(shape, 0, 0);

        let mut output: Array<felt252> = ArrayTrait::new();
        tensor.serialize(ref output);
        
        println!("Output: {:?}", output);
    }

    #[test]
    #[available_gas(99999999999999999)]
    fn inference_test() {
        let contract = declare("ModelContract").unwrap();
        let (contract_address, _) = contract.deploy(@array![]).unwrap();
        let dispatcher = IModelContractDispatcher { contract_address };

        let input: Array<felt252> = array![2,1,100,100,3414,0,4848,0,4640,0,4649,0,1596,0,1572,0,1093,0,3980,0,1288,0,5559,0,87,0,567,0,4303,0,6409,0,323,0,3039,0,6166,0,5919,0,3719,0,4081,0,826,0,3945,0,5912,0,5808,0,4278,0,2403,0,4990,0,6091,0,2894,0,418,0,3151,0,3682,0,1693,0,801,0,4863,0,3661,0,3421,0,4704,0,148,0,2581,0,498,0,4911,0,4159,0,6061,0,4123,0,2863,0,3040,0,1769,0,5043,0,5272,0,5850,0,2646,0,1342,0,3411,0,5257,0,4493,0,5874,0,2753,0,129,0,6001,0,3000,0,838,0,3290,0,1909,0,1664,0,2056,0,2601,0,5131,0,1216,0,473,0,3356,0,4700,0,4900,0,4803,0,1903,0,4068,0,435,0,6059,0,4946,0,2365,0,5720,0,4008,0,3051,0,1657,0,4025,0,1623,0,2348,0,6469,0,4247,0,6196,0,979,0,5395,0,4612,0,4175,0,4603,0,290,0,4711,0,6171,0,116,0,2078,0];
        let mut input = input.span();
        let node_input: Tensor<FP16x16> = Serde::deserialize(ref input).unwrap();

        let bias1_num = array![143, 843, 1854, 18, 2206, 5601, 4467, 3837, 2674, 1250, 3587, 2053, 3157, 6521, 3280, 3784, 4940, 4103, 2470, 2623, 1110, 2624, 178, 5075, 168, 4535, 6306, 6281, 155, 1468, 5714, 2764, 3011, 2082, 4706, 2122, 3297, 3032, 6541, 3359, 4906, 3584, 4756, 3236, 6321, 6542, 5816, 248, 959, 4280, 3497, 297, 4247, 1532, 1829, 3308, 1955, 5481, 3646, 1821, 1031, 202, 4342, 789, 5511, 5614, 983, 2404, 5842, 2965, 1288, 2705, 2158, 5069, 107, 2958, 5950, 3859, 6095, 5914, 2695, 5494, 2427, 657, 4945, 2143, 4004, 2261, 5758, 2518];
        let bias1_sign = array![true, false, false, true, false, true, true, true, false, false, true, false, true, false, true, true, false, false, true, false, false, true, false, false, true, true, false, true, false, true, true, true, false, false, false, false, false, false, true, true, false, false, true, false, false, false, true, true, true, false, true, true, false, true, true, false, false, false, false, true, false, false, true, false, false, true, false, false, true, true, false, false, true, false, false, true, false, true, true, true, false, false, false, true, false, true, false, false, false, false];
        
        dispatcher.add_weights(bias1_num, bias1_sign);

        let (bias1, index_weights, index_signs) = dispatcher.get_weights(array![1, 90], 0, 0);

        let bias2_num = array![4495, 4923, 4176, 5129, 6092, 5179, 4690, 1948, 1710, 4751, 3591, 1007, 6884, 4336, 809, 1070, 1489, 5854, 2824, 2777, 720, 6321, 627, 4916, 851, 2170, 3175, 951, 3573, 3609, 150, 5256, 2121, 2396, 6896, 948, 4664, 550, 6110, 5309, 2177, 6080, 1622, 5721, 4824, 3991, 5932, 239, 6028, 6468, 2489, 4443, 576, 1088, 6812, 3897, 5320, 2305, 5517, 896, 4114, 6732, 372, 5317, 2745, 199, 5518, 6345, 3887, 2721, 1618, 6823, 4025, 261, 2905, 6305, 4961, 3136, 1847, 1996];
        let bias2_sign = array![true, true, false, true, false, true, false, false, true, false, false, false, false, true, true, false, false, true, true, true, true, true, false, false, false, false, true, false, true, false, false, true, true, false, false, true, false, false, true, true, true, false, true, true, true, true, false, true, true, true, true, false, false, true, true, true, false, false, true, true, true, false, true, false, true, false, true, true, false, true, false, false, true, false, false, false, false, true, true, false];

        dispatcher.add_weights(bias2_num, bias2_sign);

        let (bias2, _, _) = dispatcher.get_weights(array![1, 80], index_weights, index_signs);

        let node__aff1_gemm_output_0 = NNTrait::gemm(node_input, get_node_aff1_weight(), Option::Some(bias1), Option::Some(FP16x16 { mag: 65536, sign: false }), Option::Some(FP16x16 { mag: 65536, sign: false }), false, true);
        let node__aff2_gemm_output_0 = NNTrait::gemm(node__aff1_gemm_output_0, get_node_aff2_weight(), Option::Some(bias2), Option::Some(FP16x16 { mag: 65536, sign: false }), Option::Some(FP16x16 { mag: 65536, sign: false }), false, true);
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
        
        println!("Output: {:?}", output);
    }
}
