import tensorflow as tf

batch_size = 64
num_epochs = 1
num_samples = 10000  # Number of samples to use for a quick test

# Load CIFAR-100 dataset
cifar = tf.keras.datasets.cifar100
(x_train, y_train), (x_test, y_test) = cifar.load_data()

# Use a subset of the dataset for quick testing
x_train, y_train = x_train[:num_samples], y_train[:num_samples]

# Define the model
model = tf.keras.applications.ResNet50(
    include_top=True,
    weights=None,
    input_shape=(32, 32, 3),
    classes=100,
)

# Compile the model
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

# Train the model
model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size)
