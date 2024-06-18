import numpy as np
import os
import json
from osiris.cairo.serde.data_structures import create_tensor_from_array
from osiris.cairo.serde.serialize import serializer

data_path = os.path.join('input.json')

json = json.load(open(data_path, 'r'))
data = np.array(json['input_data']).reshape(1,3)

print('Data loaded: ', data)

tensor = create_tensor_from_array(data)
serialized_data = serializer(tensor)

print('Data serialized: ', serialized_data)

with open("input.txt", "w") as text_file:
    text_file.write(serialized_data)
