from keras import regularizers
from keras.layers import Input, Dense
from keras.models import Model


# Create an initial autocoder Model
def create_seed_model():
    input_dim = 30  # num of columns, 30
    encoding_dim = 18
    hidden_dim1 = 10  # int(encoding_dim / 2) #i.e. 7
    hidden_dim2 = 6
    learning_rate = 1e-7

    input_layer = Input(shape=(input_dim,))
    encoder = Dense(encoding_dim, activation="tanh",
                    activity_regularizer=regularizers.l1(learning_rate))(input_layer)
    encoder = Dense(hidden_dim1, activation="elu")(encoder)
    encoder = Dense(hidden_dim2, activation="tanh")(encoder)
    decoder = Dense(hidden_dim2, activation='elu')(encoder)
    decoder = Dense(hidden_dim1, activation='tanh')(decoder)
    decoder = Dense(input_dim, activation='elu')(decoder)

    model = Model(inputs=input_layer, outputs=decoder)
    model.compile(optimizer='adam', metrics=['accuracy'],
                  loss='mean_squared_error')
    return model



