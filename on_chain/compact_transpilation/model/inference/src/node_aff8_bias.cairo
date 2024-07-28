use orion::operators::tensor::{FP16x16Tensor, Tensor, TensorTrait};
use orion::numbers::{FixedTrait, FP16x16};

fn get_node_aff8_bias() -> Tensor<FP16x16> {
  let shape = array![20];
  let weights_num = array![2689, 341, 1323, 4578, 9505, 9556, 9173, 788, 5692, 10650, 7859, 8849, 7111, 1210, 6304, 9458, 4085, 5030, 2225, 4917];
  let weights_sign = array![true, false, true, true, true, true, false, false, true, true, false, false, true, true, false, true, true, true, false, false];
  let mut data = array![];
  let mut index = 0;
  loop {
    if index == weights_num.len() { break; }
    data.append(FP16x16 { mag: *weights_num.at(index), sign: *weights_sign.at(index) });
    index += 1;
  };
  TensorTrait::new(shape.span(), data.span())
}