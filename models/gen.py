import torch
from torch import nn
import torch.nn.init as init
import ezkl
import os
import json

model_path = os.path.join('network.onnx')
data_path = os.path.join('input.json')

class Model(nn.Module):
    def __init__(self, inplace=False):
        super(Model, self).__init__()

        self.aff1 = nn.Linear(3,1)
        self.relu = nn.ReLU()
        self._initialize_weights()

    def forward(self, x):
        x =  self.aff1(x)
        x =  self.relu(x)
        return (x)

    def _initialize_weights(self):
        init.orthogonal_(self.aff1.weight)

model = Model()

x = 0.1 * torch.randn(1, 3, requires_grad=True)

model.eval()

torch.onnx.export(model,               # model being run
                      x,                   # model input (or a tuple for multiple inputs)
                      model_path,            # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=10,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['output'], # the model's output names
                      dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})

data_array = ((x).detach().numpy()).reshape([-1]).tolist()

data = dict(input_data = [data_array])

json.dump( data, open(data_path, 'w' ))

print('Model and data saved')
print('Model prediction: ', model(x).detach().numpy())
