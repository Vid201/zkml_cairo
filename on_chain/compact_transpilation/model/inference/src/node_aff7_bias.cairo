use orion::operators::tensor::{FP16x16Tensor, Tensor, TensorTrait};
use orion::numbers::{FixedTrait, FP16x16};

fn get_node_aff7_bias() -> Tensor<FP16x16> {
  let shape = array![30];
  let weights_num = array![2490, 6284, 368, 535, 5891, 5390, 5202, 8796, 3400, 6405, 10224, 6489, 4540, 2309, 3348, 5021, 866, 3343, 6252, 6496, 4636, 667, 7822, 1987, 5876, 9719, 1185, 532, 1362, 486];
  let weights_sign = array![true, true, true, false, false, false, false, true, false, false, false, false, false, false, true, true, true, true, true, false, false, true, false, false, false, false, true, true, false, true];
  let mut data = array![];
  let mut index = 0;
  loop {
    if index == weights_num.len() { break; }
    data.append(FP16x16 { mag: *weights_num.at(index), sign: *weights_sign.at(index) });
    index += 1;
  };
  TensorTrait::new(shape.span(), data.span())
}