use orion::operators::tensor::{FP16x16Tensor, Tensor, TensorTrait};
use orion::numbers::{FixedTrait, FP16x16};

fn get_node_aff2_bias() -> Tensor<FP16x16> {
  let shape = array![80];
  let weights_num = array![4495, 4923, 4176, 5129, 6092, 5179, 4690, 1948, 1710, 4751, 3591, 1007, 6884, 4336, 809, 1070, 1489, 5854, 2824, 2777, 720, 6321, 627, 4916, 851, 2170, 3175, 951, 3573, 3609, 150, 5256, 2121, 2396, 6896, 948, 4664, 550, 6110, 5309, 2177, 6080, 1622, 5721, 4824, 3991, 5932, 239, 6028, 6468, 2489, 4443, 576, 1088, 6812, 3897, 5320, 2305, 5517, 896, 4114, 6732, 372, 5317, 2745, 199, 5518, 6345, 3887, 2721, 1618, 6823, 4025, 261, 2905, 6305, 4961, 3136, 1847, 1996];
  let weights_sign = array![true, true, false, true, false, true, false, false, true, false, false, false, false, true, true, false, false, true, true, true, true, true, false, false, false, false, true, false, true, false, false, true, true, false, false, true, false, false, true, true, true, false, true, true, true, true, false, true, true, true, true, false, false, true, true, true, false, false, true, true, true, false, true, false, true, false, true, true, false, true, false, false, true, false, false, false, false, true, true, false];
  let mut data = array![];
  let mut index = 0;
  loop {
    if index == weights_num.len() { break; }
    data.append(FP16x16 { mag: *weights_num.at(index), sign: *weights_sign.at(index) });
    index += 1;
  };
  TensorTrait::new(shape.span(), data.span())
}