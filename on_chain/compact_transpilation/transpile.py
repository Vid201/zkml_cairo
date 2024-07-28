import os
import re

def get_subfolders(path):
    return [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

pattern = r": \d+"
pattern_shape1 = r"\[(\d+), (\d+)\]"
pattern_shape2 = r"\[(\d+)\]"

if __name__ == '__main__':
    weights_root_folder = 'model/initializers'
    inference_root_folder = 'model/inference'
    weights_folders = get_subfolders(weights_root_folder)

    for weight_folder in weights_folders:
        print(weight_folder)

        weights_num = []
        weights_sign = []
        shape = []

        weights_path = os.path.join(weights_root_folder, weight_folder, 'src')
        
        for file in sorted(os.listdir(weights_path)):
            if file.startswith("chunk"):
                with open(weights_path + "/" + file, "r") as f:
                    line = f.readline()

                    while line:
                        if "a.append" in line:
                            match = re.search(pattern, line)
                            weights_num.append(int(match.group().lstrip(": ")))
                            weights_sign.append("true" in line)

                        line = f.readline()
            elif file.startswith("lib"):
                with open(weights_path + "/" + file, "r") as f:
                    line = f.read()

                    # try pattern 1
                    match = re.search(pattern_shape1, line)
                    if match is not None:
                        shape = [int(match.group(1)), int(match.group(2))]
                    else:
                        # try pattern 2
                        match = re.search(pattern_shape2, line)
                        shape = [int(match.group(1))]

        # print(weights_num)
        # print(weights_sign)
        # print(shape)

        file = os.path.join(inference_root_folder, 'src', f"{weight_folder}.cairo")

        with open(file, 'w') as f:
            f.write("use orion::operators::tensor::{FP16x16Tensor, Tensor, TensorTrait};")
            f.write("\nuse orion::numbers::{FixedTrait, FP16x16};")
            f.write(f"\n\nfn get_{weight_folder}() -> Tensor<FP16x16> " + "{")
            f.write(f"\n  let shape = array!{shape};")
            f.write(f"\n  let weights_num = array!{weights_num};")
            f.write(f"\n  let weights_sign = array!{str(weights_sign).lower()};")
            f.write("\n  let mut data = array![];")
            f.write("\n  let mut index = 0;")
            f.write("\n  loop {")
            f.write("\n    if index == weights_num.len() { break; }")
            f.write("\n    data.append(FP16x16 { mag: *weights_num.at(index), sign: *weights_sign.at(index) });")
            f.write("\n    index += 1;")
            f.write("\n  };")
            f.write(f"\n  TensorTrait::new(shape.span(), data.span())")
            f.write("\n}")
