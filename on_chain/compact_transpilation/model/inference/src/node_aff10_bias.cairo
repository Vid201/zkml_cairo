use orion::operators::tensor::{FP16x16Tensor, Tensor, TensorTrait};
use orion::numbers::{FixedTrait, FP16x16};

fn get_node_aff10_bias() -> Tensor<FP16x16> {
  let shape = array![10];
  let weights_num = array![17661, 12975, 9770, 5641, 3338, 16869, 8826, 1265, 15664, 19814];
  let weights_sign = array![true, false, true, true, false, false, false, false, false, true];
  let mut data = array![];
  let mut index = 0;
  loop {
    if index == weights_num.len() { break; }
    data.append(FP16x16 { mag: *weights_num.at(index), sign: *weights_sign.at(index) });
    index += 1;
  };
  TensorTrait::new(shape.span(), data.span())
}