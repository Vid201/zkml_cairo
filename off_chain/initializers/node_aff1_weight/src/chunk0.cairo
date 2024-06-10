use orion::numbers::{FixedTrait, FP16x16};

fn compute(ref a: Array<FP16x16>) {
a.append(FP16x16 { mag: 54316, sign: true });
a.append(FP16x16 { mag: 31645, sign: false });
a.append(FP16x16 { mag: 18528, sign: true });
}