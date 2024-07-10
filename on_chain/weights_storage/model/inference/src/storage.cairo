use alexandria_data_structures::array_ext::ArrayTraitExt;
use orion::numbers::{FixedTrait, FP16x16};
use orion::operators::tensor::{FP16x16Tensor, Tensor, TensorTrait};

pub trait WeightsStorageTrait<V> {
    fn add_weights(ref self: V, ref weights: Array<felt252>, ref signs: Array<bool>);
    fn get_weights(self: V, shape: Array<usize>, start: usize) -> Tensor<FP16x16>;
}

pub struct WeightsStorage {
    pub weights: Array<felt252>,
    pub signs: Array<bool>,
}

impl DestructWeightsStorage of Destruct<WeightsStorage> {
    fn destruct(self: WeightsStorage) nopanic {
        self.weights.destruct();
        self.signs.destruct();
    }
}

impl WeightsStorageImpl of WeightsStorageTrait<WeightsStorage> {
    fn add_weights(ref self: WeightsStorage, ref weights: Array<felt252>, ref signs: Array<bool>) {
        self.weights.append_all(ref weights);
        self.signs.append_all(ref signs);
    }

    fn get_weights(self: WeightsStorage, shape: Array<usize>, start: usize) -> Tensor<FP16x16> {
        let mut total_size: usize = 1;
        let mut index = 0;
        let mut end = shape.len();
        let shape = shape.span();
        loop {
            total_size *= *shape.at(index);
            index += 1;
            if index == end {
                break;
            }
        };

        let mut data = array![];

        index = start;
        end = start + total_size;
        loop {
            data.append(FP16x16 { mag: (*self.weights.at(index)).try_into().unwrap(), sign: *self.signs.at(index) });
            index += 1;
            if index == end {
                break;
            }
        };

        TensorTrait::new(shape, data.span())
    }
}
