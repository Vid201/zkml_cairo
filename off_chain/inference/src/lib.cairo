use orion::operators::tensor::{Tensor, TensorTrait};
use orion::operators::tensor::{
    U32Tensor, I32Tensor, I8Tensor, FP8x23Tensor, FP16x16Tensor, FP32x32Tensor, BoolTensor
};
use orion::numbers::{FP8x23, FP16x16, FP32x32};
use orion::operators::matrix::{MutMatrix, MutMatrixImpl};
use orion::operators::nn::{NNTrait, FP16x16NN};
use orion::operators::ml;

use node_aff1_weight::get_node_aff1_weight;
use node_aff1_bias::get_node_aff1_bias;


fn main(input: Array<felt252>) -> Array<felt252> {
    let mut input = input.span();
    let node_input: Tensor<FP16x16> = Serde::deserialize(ref input).unwrap();
    
    let node__aff1_gemm_output_0 = NNTrait::gemm(
        node_input,
        get_node_aff1_weight(),
        Option::Some(get_node_aff1_bias()),
        Option::Some(FP16x16 { mag: 65536, sign: false }),
        Option::Some(FP16x16 { mag: 65536, sign: false }),
        false,
        true
    );
    let node_output = NNTrait::relu(@node__aff1_gemm_output_0);

    let mut output: Array<felt252> = ArrayTrait::new();
    node_output.serialize(ref output);
    output
}
