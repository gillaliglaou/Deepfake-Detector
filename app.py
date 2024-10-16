from flask import Flask, render_template, request, jsonify, send_from_directory
from tensorflow.keras.models import load_model
from keras.utils import get_custom_objects
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf
from tensorflow.keras.losses import categorical_crossentropy

STATIC_FOLDER = 'static'
UPLOAD_FOLDER = 'uploads'
import os




class PyramidPoolingModule(tf.keras.layers.Layer):
    def __init__(self, num_filters=1, kernel_size=(1, 1), bin_sizes=[1, 2, 3, 6], pool_mode='avg', **kwargs):
        super(PyramidPoolingModule, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.bin_sizes = bin_sizes
        self.pool_mode = pool_mode
        self.pyramid_pooling = self.build_pyramid_pooling()

    def build_pyramid_pooling(self):
        return PyramidPoolingModule.PyramidPoolingModule(
            num_filters=self.num_filters,
            kernel_size=self.kernel_size,
            bin_sizes=self.bin_sizes,
            pool_mode=self.pool_mode,
        )

    def call(self, inputs):
        return self.pyramid_pooling(inputs)

    class PyramidPoolingModule(tf.keras.layers.Layer):
        def __init__(self, num_filters, kernel_size, bin_sizes, pool_mode, **kwargs):
            super(PyramidPoolingModule.PyramidPoolingModule, self).__init__(**kwargs)
            self.num_filters = num_filters
            self.kernel_size = kernel_size
            self.bin_sizes = bin_sizes
            self.pool_mode = pool_mode
            self.pyramid_layers = []

            for bin_size in bin_sizes:
                self.pyramid_layers.append(
                    tf.keras.layers.Conv2D(
                        filters=num_filters,
                        kernel_size=kernel_size,
                        padding='same',
                        activation='relu'
                    )
                )

        def call(self, inputs):
            outputs = [inputs]

            for i, bin_size in enumerate(self.bin_sizes):
                pooled = tf.keras.layers.AveragePooling2D(pool_size=(bin_size, bin_size))(inputs)
                convolved = self.pyramid_layers[i](pooled)
                resized = tf.image.resize(convolved, tf.shape(inputs)[1:3])
                outputs.append(resized)

            return tf.concat(outputs, axis=-1)

class BayarConv2d(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=0):
        super(BayarConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.minus1 = tf.ones((self.in_channels, self.out_channels, 1)) * -1.000

        # only (kernel_size ** 2 - 1) trainable params as the center element is always -1
        self.kernel = self.add_weight(shape=(self.in_channels, self.out_channels, kernel_size ** 2 - 1),
                                      initializer='random_normal',
                                      trainable=True)

    def bayarConstraint(self, input_shape):
        kernel_permuted = tf.transpose(self.kernel, perm=[2, 0, 1])
        kernel_sum = tf.reduce_sum(kernel_permuted, axis=0)
        ctr = self.kernel_size ** 2 // 2
        real_kernel = tf.concat([self.kernel[:, :, :ctr], self.minus1, self.kernel[:, :, ctr:]], axis=2)
        real_kernel = tf.reshape(real_kernel, (self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        return real_kernel

    def call(self, x):
        x = tf.nn.conv2d(x, self.bayarConstraint(x.shape), strides=self.stride, padding='SAME')
        return x

def conv3x3(in_planes, out_planes, stride=1):
    return tf.keras.layers.Conv2D(
        filters=out_planes,
        kernel_size=(3, 3),
        strides=stride,
        padding='same',
        use_bias=False,
        activation=None
    )

def GlobalAveragePooling2D():
    return tf.keras.layers.GlobalAveragePooling2D()

def Decoder(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def build_encoder(input_shape=(256, 256, 3)):
    inputs = Input(input_shape)
    resnet = ResNet50(include_top=False, weights="imagenet", input_tensor=inputs)
    L1 = resnet.layers[0].output
    L2 = resnet.get_layer("conv1_relu").output
    L3 = resnet.get_layer("conv2_block3_out").output
    L4 = resnet.get_layer("conv3_block4_out").output
    B1 = resnet.get_layer("conv4_block6_out").output
    encoder = Model(inputs=inputs, outputs=[L1, L2, L3, L4, B1])
    encoder.trainable = False
    return encoder

def assemblage(input_shape=(256, 256, 3)):
    """ Encoder """
    ch_in = 3
    ch_out = 3
    kernel = 3
    stride = 1
    padding = 0

    inputs = Input(input_shape)
    bayar_conv = BayarConv2d(input_shape[-1], ch_out, kernel, stride, padding)
    noisy_image = bayar_conv(inputs)

    Encoder = build_encoder(input_shape)

    image_resnetted_layers = Encoder(inputs)
    noisy_image_resnetted_layers = Encoder(noisy_image)

    num_layers = len(image_resnetted_layers)

    before_ppm_layers_list = []

    for layer, layer_bayar in zip(image_resnetted_layers, noisy_image_resnetted_layers):
        concatenated_layers = concatenate([layer, layer_bayar], axis=-1)
        output = conv3x3(in_planes=concatenated_layers.shape[-1], out_planes=layer.shape[-1])(concatenated_layers)
        before_ppm_layers_list.append(output)
    L1, L2, L3, L4, B1 = before_ppm_layers_list

    ppm_output = PyramidPoolingModule()(B1)  # Utiliser B1 comme entrée pour le module de pooling

    """ Decoder """
    O1 = Decoder(ppm_output, L4, 256)  ## (64 x 64)
    O2 = Decoder(O1, L3, 128)  ## (128 x 128)
    O3 = Decoder(O2, L2, 64)  ## (256 x 256)
    LOC_INPUT = Decoder(O3, L1, 32)  ## (512 x 512)

    """ Output """

    """ Localization """
    localization_output = conv3x3(ppm_output.shape[-1],1, stride=1)(LOC_INPUT)

    """ Classification """
    avg_pooled = GlobalAveragePooling2D()(ppm_output)
    flattened = keras.layers.Flatten()(avg_pooled)
    classification_output = keras.layers.Dense(1, activation="sigmoid", name="classification_output")(flattened)

    # Ajout de toutes les sorties de l'encodeur comme sorties du modèle final
    model = Model(inputs, outputs=[classification_output], name="Model_Final")
    return model
def dice_coef(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred)
    return (2.0 * intersection + smooth) / (union + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return dice_coef_loss(y_true, y_pred) + bce(y_true, y_pred)

get_custom_objects().update({'BayarConv2d': BayarConv2d})
get_custom_objects().update({'PyramidPoolingModule': PyramidPoolingModule})
get_custom_objects().update({'bce_dice_loss': bce_dice_loss})
get_custom_objects().update({'dice_coef_loss': dice_coef_loss})


app = Flask(__name__)
model = load_model(r'super-sayan-model.h5')

def preprocess_image(image):
    img = Image.open(image)
    img = img.convert('RGB')
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((256, 256))  # Redimensionner l'image selon les besoins de votre modèle
    img_array = np.array(img) / 255.0  # Normaliser les valeurs des pixels
    img_array = np.expand_dims(img_array, axis=0)  # Ajouter une dimension pour créer un lot
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/test')
def test():
    return render_template('test.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    img_array = preprocess_image(file)
    prediction = model.predict(img_array)
    prediction_text = 'Real' if prediction[1] < 0.2 else 'Fake'  # Adjust threshold as needed
    mask = prediction[0].squeeze().tolist() # Convert prediction array to list
    s = np.array(mask)
    mask_array = (s * 255).astype(np.uint8)
    mask_image = Image.fromarray(mask_array)
    filename = 'mask_image.png'
    upload_path = os.path.join(app.root_path, UPLOAD_FOLDER, filename)
    static_path = os.path.join(app.root_path, STATIC_FOLDER, 'images', filename)

    # Check if the file exists and remove it
    if os.path.exists(static_path):
        os.remove(static_path)

    mask_image.save(upload_path)
    os.rename(upload_path, static_path)

    # Return the URL of the saved image
    image_url = f'/static/images/{filename}'

    result = {
        'prediction_text': prediction_text,
        'mask_image_url': image_url
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
