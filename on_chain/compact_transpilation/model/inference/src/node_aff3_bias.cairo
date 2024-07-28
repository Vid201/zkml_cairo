use orion::operators::tensor::{FP16x16Tensor, Tensor, TensorTrait};
use orion::numbers::{FixedTrait, FP16x16};

fn get_node_aff3_bias() -> Tensor<FP16x16> {
  let shape = array![70];
  let weights_num = array![2616, 3129, 3212, 7142, 5929, 2314, 3253, 3979, 565, 6584, 17, 3617, 6165, 6462, 983, 2761, 1727, 4937, 3947, 6381, 3326, 5950, 1257, 4092, 6905, 3273, 6763, 1452, 3851, 1022, 3929, 4558, 5443, 2330, 5989, 6708, 3982, 6441, 913, 4997, 6718, 5379, 4366, 5543, 2957, 3616, 3716, 4874, 5335, 4530, 3428, 3209, 911, 6459, 5436, 1227, 3344, 4065, 5788, 5324, 2933, 5131, 1058, 6880, 5338, 1652, 4377, 4511, 914, 1624];
  let weights_sign = array![false, false, false, true, true, false, false, false, false, false, false, false, false, false, false, false, false, true, true, true, true, false, false, true, true, false, false, false, true, false, true, false, true, false, true, true, false, true, true, true, true, false, true, false, false, true, false, true, true, true, false, false, true, false, true, true, true, false, false, false, true, false, true, false, false, false, true, false, false, false];
  let mut data = array![];
  let mut index = 0;
  loop {
    if index == weights_num.len() { break; }
    data.append(FP16x16 { mag: *weights_num.at(index), sign: *weights_sign.at(index) });
    index += 1;
  };
  TensorTrait::new(shape.span(), data.span())
}