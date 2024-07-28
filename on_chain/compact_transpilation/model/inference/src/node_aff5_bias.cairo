use orion::operators::tensor::{FP16x16Tensor, Tensor, TensorTrait};
use orion::numbers::{FixedTrait, FP16x16};

fn get_node_aff5_bias() -> Tensor<FP16x16> {
  let shape = array![50];
  let weights_num = array![7004, 1880, 2981, 101, 3789, 3559, 93, 4338, 1437, 5021, 2399, 706, 4341, 4009, 1311, 7070, 2206, 8285, 3899, 1582, 5233, 3111, 3380, 3165, 2465, 1648, 6406, 7848, 4858, 7407, 4614, 7383, 2300, 3682, 6455, 760, 8439, 4172, 1407, 2478, 7790, 6396, 848, 6374, 3981, 1520, 258, 1069, 1633, 6446];
  let weights_sign = array![false, true, false, false, false, false, true, true, false, false, true, false, true, true, false, true, false, false, false, false, true, false, true, false, true, false, false, true, true, true, false, true, true, true, false, false, false, false, false, true, false, false, false, true, true, false, true, false, true, false];
  let mut data = array![];
  let mut index = 0;
  loop {
    if index == weights_num.len() { break; }
    data.append(FP16x16 { mag: *weights_num.at(index), sign: *weights_sign.at(index) });
    index += 1;
  };
  TensorTrait::new(shape.span(), data.span())
}