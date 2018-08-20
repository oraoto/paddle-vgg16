# VGG16 model for PaddlePaddle

The model is converted from Caffe model [VGG_ILSVRC_16_layers](https://gist.github.com/ksimonyan/211839e770f7b538e2d8), using [caffe2fluid](https://github.com/PaddlePaddle/models/tree/develop/fluid/image_classification/caffe2fluid).

## Usage

1. Clone this repo

    ```bash
    git clone https://github.com/oraoto/paddle-vgg16.git
    ```

    or simply download the model definition: [vgg16.py](./vgg16.py)

2. Download the params

    + [Google Drive](https://drive.google.com/file/d/1crdenigwNY31ouG4x1NES8Z6m9V54_rl/)

3. Extract the params to `models/vgg16`, the resulting folder structure should looks like:

    ```
    .
    ├── vgg16.py
    ├── infer.py
    ├── models
    │   ├── vgg16
    │   │   ├── conv1_1_biases
    │   │   ├── conv1_1_weights
    │   │   ├── conv1_2_biases
    │   │   │....

    ```

4. Load params (see [infer.py](./infer.py) for full example)

    ```python
    from vgg16 import VGG16
    import paddle.fluid as fluid

    img = fluid.layers.data('img', shape=[3, 224, 224])
    predict = VGG16(include_top=True, infer=True).net(img)

    exe = fluid.Executor(fluid.CPUPlace())
    # All params are prefiexed with `vgg16` to avoid confliction,
    # load params from `models` rather than `models/vgg16`
    fluid.io.load_params(exe, "models") 
    ```

5. Optional: Do inference on an image:

    ```
    python infer.py path/to/your/image
    ```

    Example output:

    ```
    $ python infer.py images/zebra_wikipedia.jpg
    Using TensorFlow backend.
    zebra: 0.999809205532
    impala: 0.000101726756839
    hartebeest: 4.52961321571e-05
    gazelle: 3.82991747756e-05
    ostrich: 3.12061411023e-06
    ```
