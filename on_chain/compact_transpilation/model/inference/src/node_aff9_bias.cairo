use orion::operators::tensor::{FP16x16Tensor, Tensor, TensorTrait};
use orion::numbers::{FixedTrait, FP16x16};

fn get_node_aff9_bias() -> Tensor<FP16x16> {
  let shape = array![10];
  let weights_num = array![14031, 13694, 9412, 9655, 1048, 1997, 8857, 13274, 10851, 14384];
  let weights_sign = array![true, false, false, true, false, true, true, true, true, false];
  let mut data = array![];
  let mut index = 0;
  loop {
    if index == weights_num.len() { break; }
    data.append(FP16x16 { mag: *weights_num.at(index), sign: *weights_sign.at(index) });
    index += 1;
  };
  TensorTrait::new(shape.span(), data.span())
}