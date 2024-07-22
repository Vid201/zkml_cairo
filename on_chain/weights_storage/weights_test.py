import os
import re

path_prefix = "../default/model/initializers/"
weights_path = path_prefix + "node_aff2_weight" + "/src"
bias_path = path_prefix + "node_aff2_bias" + "/src"

weights_num = []
weights_sign = []
bias_num = []
bias_sign = []

pattern = r": \d+"

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

print(weights_num)
print(weights_sign)

for file in sorted(os.listdir(bias_path)):
    if file.startswith("chunk"):
        with open(bias_path + "/" + file, "r") as f:
            line = f.readline()

            while line:
                if "a.append" in line:
                    match = re.search(pattern, line)
                    bias_num.append(int(match.group().lstrip(": ")))
                    bias_sign.append("true" in line)

                line = f.readline()

print(bias_num)
print(bias_sign)
