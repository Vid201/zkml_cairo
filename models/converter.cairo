#[derive(Copy, Drop, Serde)]
struct Tensor {
    shape: Span<felt252>,
    data: Span<felt252>
}

fn main(shape: Array<felt252>, data: Array<felt252>) -> Array<felt252> {
    let tensor = Tensor {
        shape: shape.span(),
        data: data.span()
    };

    let mut output: Array<felt252> = ArrayTrait::new();
    tensor.serialize(ref output);
    
    output
}
