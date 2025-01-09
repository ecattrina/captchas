# -*- coding: utf-8 -*-
#(не убирать сверху)
import json
import numpy as np
from flask import Flask, request
import tensorflow as tf
import keras
from keras import layers
from math import sqrt
import cv2

app = Flask(__name__)

IMG_WIDTH = 200
IMG_HEIGHT = 50


def ctc_label_dense_to_sparse(labels, label_lengths):
    label_shape = tf.shape(labels)
    num_batches_tns = tf.stack([label_shape[0]])
    max_num_labels_tns = tf.stack([label_shape[1]])

    def range_less_than(old_input, current_input):
        return tf.expand_dims(tf.range(tf.shape(old_input)[1]), 0) < tf.fill(
            max_num_labels_tns, current_input
        )

    init = tf.cast(tf.fill([1, label_shape[1]], 0), dtype="bool")
    dense_mask = tf.compat.v1.scan(
        range_less_than, label_lengths, initializer=init, parallel_iterations=1
    )
    dense_mask = dense_mask[:, 0, :]

    label_array = tf.reshape(
        tf.tile(tf.range(0, label_shape[1]), [label_shape[0]]), label_shape
    )
    label_ind = tf.compat.v1.boolean_mask(label_array, dense_mask)

    batch_array = tf.transpose(
        tf.reshape(
            tf.tile(tf.range(0, label_shape[0]), [label_shape[1]]),
            tf.reverse(label_shape, [0]),
        )
    )
    batch_ind = tf.compat.v1.boolean_mask(batch_array, dense_mask)
    indices = tf.transpose(
        tf.reshape(tf.concat([batch_ind, label_ind], axis=0), [2, -1])
    )

    vals_sparse = tf.compat.v1.gather_nd(labels, indices)

    return tf.SparseTensor(
        tf.cast(indices, dtype="int64"),
        vals_sparse,
        tf.cast(label_shape, dtype="int64")
    )

def ctc_batch_cost(y_true, y_pred, input_length, label_length):
    label_length = tf.cast(tf.squeeze(label_length, axis=-1), dtype="int32")
    input_length = tf.cast(tf.squeeze(input_length, axis=-1), dtype="int32")
    sparse_labels = tf.cast(ctc_label_dense_to_sparse(y_true, label_length), dtype="int32")

    y_pred = tf.math.log(tf.transpose(y_pred, perm=[1, 0, 2]) + keras.backend.epsilon())

    return tf.expand_dims(
        tf.compat.v1.nn.ctc_loss(
            inputs=y_pred, labels=sparse_labels, sequence_length=input_length
        ),
        1,
    )

class CTCLayer(layers.Layer):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.loss_fn = ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        return y_pred

def ctc_decode(y_pred, input_length, greedy=True, beam_width=100, top_paths=1):
    input_shape = tf.shape(y_pred)
    num_samples, num_steps = input_shape[0], input_shape[1]
    y_pred = tf.math.log(tf.transpose(y_pred, perm=[1, 0, 2]) + keras.backend.epsilon())
    input_length = tf.cast(input_length, dtype="int32")

    if greedy:
        (decoded, log_prob) = tf.nn.ctc_greedy_decoder(
            inputs=y_pred, sequence_length=input_length
        )
    else:
        (decoded, log_prob) = tf.compat.v1.nn.ctc_beam_search_decoder(
            inputs=y_pred,
            sequence_length=input_length,
            beam_width=beam_width,
            top_paths=top_paths,
        )
    decoded_dense = []
    for st in decoded:
        st = tf.SparseTensor(st.indices, st.values, (num_samples, num_steps))
        decoded_dense.append(tf.sparse.to_dense(sp_input=st, default_value=-1))
    return decoded_dense, log_prob

def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :5]
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text

def encode_single_sample(img_path):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = tf.transpose(img, perm=[1, 0, 2])
    return img

def color_distance(color1, color2):
    return sqrt(sum((color1[i] - color2[i])**2 for i in range(3)))

def preprocess_image(img_path):
    image = cv2.imread(img_path)
    for y in range(image.shape[1]):
        for x in range(image.shape[0]):
            if color_distance(image[x, y], [153, 102, 0]) >= 90:
                image[x, y] = [255, 255, 255]
    cv2.imwrite(img_path, image)

# Основной обработчик запроса
@app.route('/captcha', methods=["POST"])
def handle_captcha():
    filename = "input_image.png"
    uploaded_file = request.files['file']
    
    if uploaded_file.filename != "":
        uploaded_file.save(filename)

    preprocess_image(filename)
    prediction = prediction_model.predict([tf.reshape(encode_single_sample(filename), [1, IMG_WIDTH, IMG_HEIGHT, 1])])
    prediction_texts = decode_batch_predictions(prediction)
    return prediction_texts[0]

model = keras.models.load_model("captchamodel.h5", custom_objects={'CTCLayer': CTCLayer})
prediction_model = keras.models.Model(model.input[0], model.get_layer(name="dense2").output)

with open("char_to_num.json") as f:
    char_to_num_config = json.load(f)

with open("num_to_char.json") as f:
    num_to_char_config = json.load(f)

char_to_num = layers.StringLookup.from_config(char_to_num_config)
num_to_char = layers.StringLookup.from_config(num_to_char_config)

# Запуск Flask сервера
if __name__ == "__main__":
    app.run()


