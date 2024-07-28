use orion::operators::tensor::{FP16x16Tensor, Tensor, TensorTrait};
use orion::numbers::{FixedTrait, FP16x16};

fn get_node_aff1_bias() -> Tensor<FP16x16> {
  let shape = array![90];
  let weights_num = array![143, 843, 1854, 18, 2206, 5601, 4467, 3837, 2674, 1250, 3587, 2053, 3157, 6521, 3280, 3784, 4940, 4103, 2470, 2623, 1110, 2624, 178, 5075, 168, 4535, 6306, 6281, 155, 1468, 5714, 2764, 3011, 2082, 4706, 2122, 3297, 3032, 6541, 3359, 4906, 3584, 4756, 3236, 6321, 6542, 5816, 248, 959, 4280, 3497, 297, 4247, 1532, 1829, 3308, 1955, 5481, 3646, 1821, 1031, 202, 4342, 789, 5511, 5614, 983, 2404, 5842, 2965, 1288, 2705, 2158, 5069, 107, 2958, 5950, 3859, 6095, 5914, 2695, 5494, 2427, 657, 4945, 2143, 4004, 2261, 5758, 2518];
  let weights_sign = array![true, false, false, true, false, true, true, true, false, false, true, false, true, false, true, true, false, false, true, false, false, true, false, false, true, true, false, true, false, true, true, true, false, false, false, false, false, false, true, true, false, false, true, false, false, false, true, true, true, false, true, true, false, true, true, false, false, false, false, true, false, false, true, false, false, true, false, false, true, true, false, false, true, false, false, true, false, true, true, true, false, false, false, true, false, true, false, false, false, false];
  let mut data = array![];
  let mut index = 0;
  loop {
    if index == weights_num.len() { break; }
    data.append(FP16x16 { mag: *weights_num.at(index), sign: *weights_sign.at(index) });
    index += 1;
  };
  TensorTrait::new(shape.span(), data.span())
}