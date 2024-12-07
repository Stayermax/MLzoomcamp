from keras_image_helper import create_preprocessor
import tflite_runtime.interpreter as tflite

from io import BytesIO
from urllib import request

from PIL import Image


def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img


def predict(url):
    preprocessor = create_preprocessor('xception', target_size=(200, 200))
    X = preprocessor.from_url(url)

    # print(f"url: {url}")
    # img = download_image(url)
    # img = prepare_image(img, (200, 200))
    # x = np.array(img)
    # x = x/255
    # X = np.array([x])
    # X = np.float32(X)

    interpreter = tflite.Interpreter(model_path='model_2024_hairstyle_v2.tflite')
    # Loading weights from the model to the memory
    interpreter.allocate_tensors()
    input_index = 0
    output_index = 13

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()

    # Results are in the output_index, so fetching the results...
    preds = interpreter.get_tensor(output_index)
    print(preds)
    return preds


def lambda_handler(event, context):
    print(f"event: {event}")
    import numpy as np
    print(np.__version__)
    url = event['url']
    res = predict(url)
    print(f"res: {res}")
    return res
