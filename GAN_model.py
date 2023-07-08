import tensorflow as tf
import tensorflow_datasets as tfds
from Tools import *

dataset = tfds.load('emnist/balanced', split='train', shuffle_files=True)

x_train, y_train = cargar_datos(dataset)

def build_generator():
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Dense(512 * 7 * 7, activation="relu", input_dim=100))
    model.add(tf.keras.layers.Reshape((7, 7, 512)))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
    model.add(tf.keras.layers.Conv2DTranspose(256, kernel_size=3, strides=(2, 2), padding="same", activation="relu"))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
    model.add(tf.keras.layers.Conv2DTranspose(128, kernel_size=3, strides=(2, 2), padding="same", activation="relu"))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
    model.add(tf.keras.layers.Conv2DTranspose(64, kernel_size=3, padding="same", activation="relu"))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
    model.add(tf.keras.layers.Conv2DTranspose(1, kernel_size=3, padding='same', activation="sigmoid"))

    noise = tf.keras.layers.Input(shape=(100,))
    label = tf.keras.layers.Input(shape=(1,), dtype='int8')
    label_embedding = tf.keras.layers.Flatten()(tf.keras.layers.Embedding(47, 100)(label))

    model_input = tf.keras.layers.multiply([noise, label_embedding])
    img = model(model_input)

    return tf.keras.models.Model([noise, label], img)

def build_discriminator():

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(16, kernel_size=3, strides=(2, 2), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.2), input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras. layers.BatchNormalization(momentum=0.8))
    model.add(tf.keras.layers.Conv2D(32, kernel_size=3, strides=(2, 2), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.2)))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
    model.add(tf.keras.layers.Conv2D(64, kernel_size=3, strides=(2, 2), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.2)))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
    model.add(tf.keras.layers.Conv2D(128, kernel_size=3, padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.2)))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
    model.add(tf.keras.layers.Conv2D(256, kernel_size=3, padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.2)))

    model.add(tf.keras.layers.Flatten())

    img = tf.keras.layers.Input(shape=(28, 28, 1))

    # Extract feature representation
    features = model(img)

    # Determine validity and label of the image
    validity = tf.keras.layers.Dense(1, activation="sigmoid")(features)
    label = tf.keras.layers.Dense(47, activation="softmax")(features)

    return tf.keras.models.Model(img, [validity, label])

generator = build_generator()
discriminator = build_discriminator()

GAN_model = tf.keras.models.Model(
    inputs=[generator.input[0], generator.input[1]],
    outputs=[discriminator(generator.output)[0], discriminator(generator.output)[1]]
)

generator.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
    loss='binary_crossentropy'
)
discriminator.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
    loss=['binary_crossentropy', 'sparse_categorical_crossentropy'],
    metrics=['accuracy']
)
GAN_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
    loss=['binary_crossentropy', 'sparse_categorical_crossentropy'],
)

BATCH_SIZE = 256
N_EPOCH = 45000

errorsB = []
errorsC = []

for i in range(1, N_EPOCH+1):
    print("Epoch " + str(i))

    idx = np.random.randint(0, x_train.shape[0], size=BATCH_SIZE)
    noise = np.random.normal(0, 1, [BATCH_SIZE, 100])
    labels = np.random.randint(0, 47, (BATCH_SIZE, 1))
    batch_fake = generator.predict([noise, labels])

    discriminator.trainable = True
    dError_reals = discriminator.train_on_batch(x_train[idx], [np.ones(BATCH_SIZE), y_train[idx]])
    dError_fake = discriminator.train_on_batch(batch_fake, [np.zeros(BATCH_SIZE), labels])

    discriminator.trainable = False

    gError = GAN_model.train_on_batch([noise, labels], [np.ones(BATCH_SIZE), labels])


    print("Progress errors: dError_reals= {}, dError_fake= {}, gError= {}"
          .format(str(dError_reals[0]), str(dError_fake[0]), str(gError[0])))

    print("Progress accuracy: acc_validity_reals= {}, acc_validity_fake= {}, acc_label= {}"
          .format(str(dError_reals[3]), str(dError_fake[3]), str((dError_reals[4]+dError_fake[4])*0.5)))

    if i % 1000 == 0:
        graficar_imagenes_generadas(i, generator)


generator.save('generator.h5')
