import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import load_img
from keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.lite as tflite


def load_model_test():
    model = keras.models.load_model('model_2024_hairstyle.keras')

    # Reading the image
    img = load_img('curly2.jpg', target_size=(200, 200))
    x = np.array(img)  # /255
    X = np.array([x])
    X = preprocess_input(X)

    preds = model.predict(X)

    # Combining labels with actual prediction
    print(preds)


def q1():
    model = keras.models.load_model('model_2024_hairstyle.keras')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open('model_2024_hairstyle.tflite', 'wb') as f_out:
        f_out.write(tflite_model)


def q2():
    import tensorflow.lite as tflite

    interpreter = tflite.Interpreter(model_path='model_2024_hairstyle.tflite')
    # Loading weights from the model to the memory
    interpreter.allocate_tensors()
    print(interpreter.get_input_details()[0])
    print(interpreter.get_output_details()[0])


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


def q3():
    url = 'https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg'
    img = download_image(url)
    img = prepare_image(img, (200, 200))
    x = np.array(img)
    print(x[0][0] / 255)


def q4():
    # img = 'yf_dokzqy3vcritme8ggnzqlvwa.jpeg'
    url = 'https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg'
    img = download_image(url)
    img = prepare_image(img, (200, 200))
    x = np.array(img)
    x = x / 255
    X = np.array([x])
    X = np.float32(X)

    interpreter = tflite.Interpreter(model_path='model_2024_hairstyle.tflite')
    # Loading weights from the model to the memory
    interpreter.allocate_tensors()
    input_index = 0
    output_index = 13

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()

    # Results are in the output_index, so fetching the results...
    preds = interpreter.get_tensor(output_index)
    print(preds)


if __name__ == '__main__':
    load_model_test()
    # q1()
    # q2()
    # q3()
    q4()