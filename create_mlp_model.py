def create_mlp_model(hidden_layer_sizes=[64, 32], activation='relu'):
    # Define the input layer.
    # 'X_train.shape[1]' should be replaced with the number of features in your dataset.
    input_layer = Input(shape=(X_train.shape[1],))

    # Initialize 'x' with the input layer.
    x = input_layer

    # Iteratively add hidden layers.
    # The number of layers and their respective neurons are defined in 'hidden_layer_sizes'.
    for layer_size in hidden_layer_sizes:
        x = Dense(layer_size, activation=activation)(x)

    # Define the output layer.
    # This example uses a single neuron with sigmoid activation, typical for binary classification.
    output_layer = Dense(1, activation='sigmoid')(x)

    # Create the model.
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model with the 'adam' optimizer and 'binary_crossentropy' loss function.
    # This is typical for binary classification tasks.
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model
