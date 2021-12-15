import torch
from torch import onnx

import onnx
import onnxruntime
import onnx.checker
import onnx.utils

import flatbuffers
import numpy as np
import cv2
import albumentations as A

from model.pspnet import *
from model.unet import *


# torch.trace
def trace_model(model, model_name):
    model.eval()
    # trace
    model_trace = torch.jit.trace(model, torch.rand(1, 3, 256, 256))
    model_trace.save(f'./{model_name}_params/model_trace.pt')
    # script
    model_script = torch.jit.script(model)
    model_script.save(f'./{model_name}_params/model_script.pt')


def load_model(model_name, params_name, mode='torch'):
    assert mode in ['torch', 'script']
    params_path = f'./{model_name}_params/{params_name}.pt'

    if mode == 'torch':
        if model_name == 'pspnet':
            model = PSPNet(2)
        elif model_name == 'unet':
            model = UNet(2)
        model.load_state_dict(torch.load(params_path, map_location=torch.device('cpu')))
    else:
        model = torch.jit.load(params_path)

    return model


# transfer to ONNX
def to_onnx(model, model_name, params_name):
    model.eval()

    x = torch.randn(1, 3, 256, 256, requires_grad=True)
    torch.onnx.export(model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  f"./{model_name}_params/{params_name}.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  )


def check_onnx_model(model_name, params_name):
    params_path = f'./{model_name}_params/{params_name}.onnx'
    onnx_model = onnx.load(params_path)
    onnx.checker.check_model(onnx_model)
    return onnx_model


def test_enhance(image):
    mean = [0.46388339, 0.44664543, 0.41852783]
    std = [0.28203478, 0.27698355, 0.29013959]

    norm = A.Compose([
        A.Resize(256, 256, p=1),
        A.Normalize(mean=mean, std=std, p=1)],
        )
    return norm(image=image)['image']


def onnx_test(model_name, params_name, test_sample, torch_out):
    params_path = f'./{model_name}_params/{params_name}.onnx'
    ort_session = onnxruntime.InferenceSession(params_path)

    img = cv2.imread(test_sample)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = test_enhance(img).transpose(2, 0, 1)
    img = np.expand_dims(img, 0)

    torch_out_img = cv2.imread(torch_out, -1)

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: img}
    ort_outs = ort_session.run(None, ort_inputs)[0]
    ort_outs = ort_outs.argmax(axis=1)

    output_tmp = np.zeros(ort_outs[0].shape)
    output_tmp[ort_outs[0] == 1] = 255
    cv2.imwrite('./samples/onnx_out.jpg', output_tmp)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(torch_out_img, output_tmp, rtol=1e-03, atol=1e-05)


if __name__ == '__main__':
    model_name, params_name, mode = 'unet', '98-0.9374-0.7128', 'torch'
    # model = load_model(model_name, params_name, mode)
    # trace_model(model, model_name)

    # params_name, mode = 'model_script', 'script'
    model = load_model(model_name, params_name, mode)

    to_onnx(model, model_name, params_name)

    check_onnx_model(model_name, params_name)

    test_sample = './samples/test_sample.jpeg'
    torch_out = './samples/torch_out.jpeg'
    onnx_test(model_name, params_name, test_sample, torch_out)




