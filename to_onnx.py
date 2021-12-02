import torch
from torch import onnx
from torchsummary import summary
import onnx
import onnxruntime

from model.pspnet import *


def load_model():
    params_path = './98-0.9363-0.7183.pt'
    model = PSPNet(2)

    model.load_state_dict(torch.load(params_path, map_location=torch.device('cpu')))
    return model


def to_onnx(model, batch_size):
    model.eval()

    x = torch.randn(batch_size, 3, 256, 256, requires_grad=True)
    out = model(x)

    torch.onnx.export(model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "98-0.9363-0.7183.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=13,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})


def load_onnx_model():
    params_path = './98-0.9363-0.7183.onnx'
    onnx_model = onnx.load(params_path)
    onnx.checker.check_model(onnx_model)


if __name__ == '__main__':
    model = load_model()
    # summary(model, (3, 256, 256), device='cpu')

    # to_onnx(model, 1)
    load_onnx_model()