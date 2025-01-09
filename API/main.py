import json
import numpy as np
from flask import Flask, request
import cv2
import tensorflow as tf
from keras import layers, ops

app = Flask(__name__)


def compute_ctc_loss(true_labels, predicted_logits, seq_lengths, label_lengths):
    label_lengths = ops.cast(ops.squeeze(label_lengths, axis=-1), dtype="int32")
    seq_lengths = ops.cast(ops.squeeze(seq_lengths, axis=-1), dtype="int32")

    sparse_labels = ops.cast(
        dense_to_sparse_labels(true_labels, label_lengths), dtype="int32"
    )

    predicted_logits = ops.log(
        ops.transpose(predicted_logits, [1, 0, 2]) + tf.keras.backend.epsilon()
    )

    return ops.expand_dims(
        tf.compat.v1.nn.ctc_loss(
            inputs=predicted_logits,
            labels=sparse_labels,
            sequence_length=seq_lengths,
        ),
        axis=1,
    )

def dense_to_sparse_labels(labels, label_lengths):
    label_shape = ops.shape(labels)
    max_labels = ops.stack([label_shape[1]])

    def range_check(_, current_length):
        return ops.expand_dims(
            ops.arange(ops.shape(_)[1]), axis=0
        ) < tf.fill(max_labels, current_length)

    dense_mask = tf.compat.v1.scan(
        range_check, label_lengths, initializer=tf.zeros([1, label_shape[1]], dtype="bool")
    )[:, 0, :]

    indices = tf.compat.v1.where(dense_mask)
    values = tf.compat.v1.gather_nd(labels, indices)

    return tf.SparseTensor(
        indices=ops.cast(indices, dtype="int64"),
        values=values,
        dense_shape=ops.cast(label_shape, dtype="int64"),
    )

# Кастомный слой CTC

class CTCComputationLayer(layers.Layer):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.loss_function = compute_ctc_loss

    def call(self, true_labels, predicted_logits):
        batch_size = ops.cast(ops.shape(true_labels)[0], dtype="int64")
        input_length = ops.cast(ops.shape(predicted_logits)[1], dtype="int64")
        label_length = ops.cast(ops.shape(true_labels)[1], dtype="int64")

        input_lengths = input_length * ops.ones((batch_size, 1), dtype="int64")
        label_lengths = label_length * ops.ones((batch_size, 1), dtype="int64")

        loss = self.loss_function(true_labels, predicted_logits, input_lengths, label_lengths)
        self.add_loss(loss)
        return predicted_logits

# Декодирование предсказаний модели

def decode_predictions(logits):
    input_length = np.ones(logits.shape[0]) * logits.shape[1]
    decoded, _ = tf.nn.ctc_greedy_decoder(
        logits=ops.log(ops.transpose(logits, [1, 0, 2]) + tf.keras.backend.epsilon()),
        sequence_length=ops.cast(input_length, dtype="int32"),
    )
    decoded_dense = [
        tf.sparse.to_dense(sparse_tensor, default_value=-1)
        for sparse_tensor in decoded
    ]
    return decoded_dense

# Обработка изображений

def preprocess_image(img_path):
    image = cv2.imread(img_path)
    for y in range(image.shape[1]):
        for x in range(image.shape[0]):
            if np.linalg.norm(image[x, y] - np.array([153, 102, 0])) >= 90:
                image[x, y] = [255, 255, 255]
    cv2.imwrite(img_path, image)

# Подготовка выборки изображений

def prepare_sample(image_path):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_png(img, channels=1)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = ops.transpose(img, [1, 0, 2])
    return img

# Обработка запросов
@app.route('/process-captcha', methods=['POST'])
def process_captcha():
    uploaded_file = request.files['file']
    temp_path = "temp_image.png"
    uploaded_file.save(temp_path)

    preprocess_image(temp_path)
    reshaped_image = prepare_sample(temp_path)

    prediction = prediction_model.predict(
        tf.reshape(reshaped_image, [1, IMG_WIDTH, IMG_HEIGHT, 1])
    )
    predicted_text = decode_predictions(prediction)
    return predicted_text[0]

# Капча размер
IMG_WIDTH = 200
IMG_HEIGHT = 50

# Загрузка модели
model = tf.keras.models.load_model("captchamodel.h5", custom_objects={'CTCComputationLayer': CTCComputationLayer})
prediction_model = tf.keras.models.Model(
    model.input[0], model.get_layer(name="dense2").output
)

# Загрузка словарей
with open("char_to_num.json", "r") as f:
    char_to_num_mapping = json.load(f)
with open("num_to_char.json", "r") as f:
    num_to_char_mapping = json.load(f)

char_to_num = layers.StringLookup.from_config(char_to_num_mapping)
num_to_char = layers.StringLookup.from_config(num_to_char_mapping)

if __name__ == "__main__":
    app.run(debug=False)
