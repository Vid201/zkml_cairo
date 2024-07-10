#[cfg(test)]
mod tests {
    use orion::operators::tensor::{Tensor, TensorTrait};
    use orion::operators::tensor::{U32Tensor, I32Tensor, I8Tensor, FP8x23Tensor, FP16x16Tensor, FP32x32Tensor, BoolTensor};
    use orion::numbers::{FP8x23, FP16x16, FP32x32};
    use orion::operators::matrix::{MutMatrix, MutMatrixImpl};
    use orion::operators::nn::{NNTrait, FP16x16NN};

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

    use core::debug::PrintTrait;

    #[test]
    #[available_gas(99999999999999999)]
    fn test() {
        let input: Array<felt252> = array![2,1,100,100,3414,0,4848,0,4640,0,4649,0,1596,0,1572,0,1093,0,3980,0,1288,0,5559,0,87,0,567,0,4303,0,6409,0,323,0,3039,0,6166,0,5919,0,3719,0,4081,0,826,0,3945,0,5912,0,5808,0,4278,0,2403,0,4990,0,6091,0,2894,0,418,0,3151,0,3682,0,1693,0,801,0,4863,0,3661,0,3421,0,4704,0,148,0,2581,0,498,0,4911,0,4159,0,6061,0,4123,0,2863,0,3040,0,1769,0,5043,0,5272,0,5850,0,2646,0,1342,0,3411,0,5257,0,4493,0,5874,0,2753,0,129,0,6001,0,3000,0,838,0,3290,0,1909,0,1664,0,2056,0,2601,0,5131,0,1216,0,473,0,3356,0,4700,0,4900,0,4803,0,1903,0,4068,0,435,0,6059,0,4946,0,2365,0,5720,0,4008,0,3051,0,1657,0,4025,0,1623,0,2348,0,6469,0,4247,0,6196,0,979,0,5395,0,4612,0,4175,0,4603,0,290,0,4711,0,6171,0,116,0,2078,0];
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
        
        println!("Output: {:?}", output);
    }
}
