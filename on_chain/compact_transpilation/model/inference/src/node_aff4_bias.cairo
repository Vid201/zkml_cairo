use orion::operators::tensor::{FP16x16Tensor, Tensor, TensorTrait};
use orion::numbers::{FixedTrait, FP16x16};

fn get_node_aff4_bias() -> Tensor<FP16x16> {
  let shape = array![60];
  let weights_num = array![1199, 4362, 7742, 5144, 6828, 5047, 5010, 6071, 7417, 2192, 5898, 3283, 4534, 1811, 3172, 6292, 7613, 3123, 4935, 7218, 5448, 6989, 1327, 4452, 2462, 5843, 7429, 1024, 4361, 2922, 4435, 4695, 1990, 5632, 103, 1322, 3399, 3745, 5771, 4669, 1251, 1723, 5004, 6582, 6167, 4825, 5704, 6636, 7722, 1214, 5852, 7283, 1461, 1778, 3498, 5595, 333, 606, 6112, 988];
  let weights_sign = array![false, false, true, true, true, false, false, true, true, false, false, true, false, true, false, true, true, false, true, true, true, false, true, true, true, false, false, false, true, true, false, true, true, true, true, false, true, true, false, false, true, true, true, true, false, true, false, false, false, false, true, true, true, true, true, false, false, false, true, false];
  let mut data = array![];
  let mut index = 0;
  loop {
    if index == weights_num.len() { break; }
    data.append(FP16x16 { mag: *weights_num.at(index), sign: *weights_sign.at(index) });
    index += 1;
  };
  TensorTrait::new(shape.span(), data.span())
}