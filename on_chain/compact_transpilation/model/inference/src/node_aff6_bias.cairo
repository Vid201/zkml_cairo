use orion::operators::tensor::{FP16x16Tensor, Tensor, TensorTrait};
use orion::numbers::{FixedTrait, FP16x16};

fn get_node_aff6_bias() -> Tensor<FP16x16> {
  let shape = array![40];
  let weights_num = array![2086, 5805, 4666, 3010, 2947, 7366, 1182, 4454, 3642, 8694, 3733, 8490, 6279, 631, 4246, 436, 4454, 3779, 8225, 335, 7238, 6938, 7877, 6707, 6199, 8029, 874, 8745, 510, 4886, 2254, 8572, 204, 8688, 2117, 7813, 9025, 8638, 2251, 3195];
  let weights_sign = array![false, true, true, true, true, false, false, true, true, true, true, true, false, false, true, false, true, false, true, false, true, true, true, false, true, false, true, true, false, true, false, false, false, false, true, false, true, true, false, true];
  let mut data = array![];
  let mut index = 0;
  loop {
    if index == weights_num.len() { break; }
    data.append(FP16x16 { mag: *weights_num.at(index), sign: *weights_sign.at(index) });
    index += 1;
  };
  TensorTrait::new(shape.span(), data.span())
}