import torch
from torch import onnx
from torchsummary import summary

import onnx
import onnxruntime

import numpy as np
import cv2
import albumentations as A

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


def check_onnx_model():
    params_path = './98-0.9363-0.7183.onnx'
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


def onnx_test(test_sample, torch_out):
    params_path = './98-0.9363-0.7183.onnx'
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

    cv2.imwrite('./onnx_out.jpg', output_tmp)
    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(torch_out_img, output_tmp, rtol=1e-03, atol=1e-05)


if __name__ == '__main__':
    # model = load_model()
    # # summary(model, (3, 256, 256), device='cpu')
    # to_onnx(model, 1)

    check_onnx_model()

    test_path = '/home/lijl/Datasets/segmentation/coco_person/test'
    img_name = 'IMG_8099.jpeg'

    test_sample = os.path.join(test_path, 'Images', img_name)
    torch_out = os.path.join(test_path, 'Results', img_name)

    onnx_test(test_sample, torch_out)