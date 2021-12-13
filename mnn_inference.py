import albumentations as A
import cv2
import numpy as np
from matplotlib import pyplot as plt
import MNN
import MNN.expr as F


# model_path = './model_params/98-0.9363-0.7183.mnn'
model_path = './pspnet_params/model_script.mnn'
image_path = './samples/test_sample.jpeg'
output_save_path = './samples/mnn_out.jpeg'
mean = [0.46388339, 0.44664543, 0.41852783]
std = [0.28203478, 0.27698355, 0.29013959]

def test_enhance(sample):
    norm = A.Compose([
        A.Resize(256, 256, p=1),
        A.Normalize(mean=mean, std=std, p=1)],
        additional_targets={'image': 'image'})
    return norm(image=sample)['image']

img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = test_enhance(img).transpose((2, 0, 1))
img = np.expand_dims(img, 0)

vars = F.load_as_dict(model_path)
inputVar = vars["input"]
# 修改原始模型的 NC4HW4 输入为 NCHW，便于输入
if (inputVar.data_format == F.NC4HW4):
    inputVar.reorder(F.NCHW)
# 写入数据
inputVar.write(img.tolist())

# 查看输出结果
outputVar = vars['output']
# 切换布局便于查看结果
if (outputVar.data_format == F.NC4HW4):
    outputVar = F.convert(outputVar, F.NCHW)

output = F.argmax(outputVar, axis=1).read()
output = np.array(output).squeeze(0)
output_tmp = np.zeros(output.shape)
output_tmp[output == 1] = 255
cv2.imwrite(output_save_path, output_tmp)