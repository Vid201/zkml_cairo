use orion::operators::tensor::{FP16x16Tensor, Tensor, TensorTrait};
use orion::numbers::{FixedTrait, FP16x16};

fn get_node_aff10_weight() -> Tensor<FP16x16> {
  let shape = array![10, 10];
  let weights_num = array![23409, 8327, 2137, 34562, 17235, 31888, 422, 30131, 12965, 9400, 33013, 46081, 19342, 5404, 12142, 4416, 4302, 7561, 13024, 16308, 41078, 24971, 13775, 17840, 12321, 24042, 15623, 5456, 1183, 21697, 15813, 3871, 21161, 30065, 14782, 1708, 10731, 35769, 25493, 20301, 10923, 16937, 12759, 21283, 2497, 5764, 41805, 20166, 30914, 11114, 8091, 19270, 19746, 4707, 49315, 5614, 25272, 16357, 1460, 8607, 11810, 8422, 3001, 16219, 19318, 6706, 32737, 8555, 43830, 18115, 17035, 12571, 5123, 20496, 4291, 21184, 10397, 13971, 19331, 47498, 833, 22864, 37453, 4693, 8813, 37497, 1052, 26311, 5153, 11960, 10217, 7318, 35888, 25857, 23927, 26702, 18699, 19640, 5169, 11519];
  let weights_sign = array![false, false, false, true, true, false, false, false, false, true, true, false, false, true, true, true, true, true, false, true, false, false, true, false, true, true, false, true, true, true, false, true, false, true, false, false, true, true, true, true, true, true, true, true, false, true, false, true, false, true, false, false, false, false, false, false, false, false, false, false, true, false, true, true, true, true, false, false, true, false, false, false, false, true, true, true, true, true, false, false, true, false, true, false, false, false, true, true, true, false, true, false, true, true, false, true, true, false, true, true];
  let mut data = array![];
  let mut index = 0;
  loop {
    if index == weights_num.len() { break; }
    data.append(FP16x16 { mag: *weights_num.at(index), sign: *weights_sign.at(index) });
    index += 1;
  };
  TensorTrait::new(shape.span(), data.span())
}