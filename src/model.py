from keras import regularizers
from keras.layers import Input, Dense
from keras.models import Model, Sequential


class AutoEncoder:
    def __init__(self, input_shape, optimizer, loss, batch_size, epochs, shuffle, validation_split):
        self.input_shape = input_shape
        self.optimizer = optimizer
        self.loss = loss
        self.batch_size = batch_size
        self.epochs = epochs
        self.shuffle = shuffle
        self.validation_split = validation_split
        self.autoencoder = self.create_model()

    def create_model(self):
        input_layer = Input(shape=(self.input_shape,))
        encoded = Dense(100, activation='tanh', activity_regularizer=regularizers.l1(10e-5))(input_layer)
        encoded = Dense(50, activation='relu')(encoded)
        decoded = Dense(50, activation='tanh')(encoded)
        decoded = Dense(100, activation='tanh')(decoded)
        output_layer = Dense(self.input_shape, activation='relu')(decoded)
        return Model(input_layer, output_layer)

    def fit(self, X_train, y_train):
        self.autoencoder.compile(optimizer=self.optimizer, loss=self.loss)
        self.autoencoder.fit(X_train,
                             y_train,
                             batch_size=self.batch_size,
                             epochs=self.epochs,
                             shuffle=self.shuffle,
                             validation_split=self.validation_split)

    def get_representation(self):
        hidden_representation = Sequential()
        hidden_representation.add(self.autoencoder.layers[0])
        hidden_representation.add(self.autoencoder.layers[1])
        hidden_representation.add(self.autoencoder.layers[2])
        return hidden_representation


if __name__ == "__main__":
    pass
