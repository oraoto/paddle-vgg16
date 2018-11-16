from __future__ import absolute_import

import padddle.v2 as paddle
import paddle.fluid as fluid
import paddle.fluid.io as io
import numpy as np
import sys
from PIL import Image
from keras.applications.vgg16 import decode_predictions

from vgg16 import VGG16

## Read image and preprocess
im = Image.open(sys.argv[1])

im = im.resize((224, 224), Image.ANTIALIAS)
im = np.array(im).astype(np.float32)
im = im.transpose((2, 0, 1))  # CHW
im = im[(2, 1, 0), :, :]  # BGR

mean = np.array([104., 117., 124.], dtype=np.float32)
mean = mean.reshape([3, 1, 1])
im = im - mean
im = np.expand_dims(im, 0)

## Create inference program
infer_program = fluid.default_main_program().clone(for_test=True)


## Define the network
with fluid.program_guard(infer_program):
    input = fluid.layers.data('img', shape=[3, 224, 224])
    predict = VGG16(include_top=True, infer=True).net(input)

## Create executor
place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda() else fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

## Load params
io.load_params(exe, "models")

## Create inference program
# infer_program = fluid.default_main_program().clone(for_test=True)
# infer_program.prune(predict)

## Run it
p = exe.run(infer_program, fetch_list=[predict], feed={
    'img': im
})

p = decode_predictions(p[0])

for (_, cls, prob) in p[0]:
    print("{}: {}".format(cls, prob))
