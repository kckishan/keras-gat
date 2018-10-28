from __future__ import division

import numpy as np
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.layers import Input, Dropout, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

from keras_gat import GraphAttention
from keras_gat.utils import load_data, preprocess_features, load_protein_function

# Read data
# A, X, Y_train, Y_val, Y_test, idx_train, idx_val, idx_test = load_data('cora')
A, X, Y_train, Y_val, Y_test, idx_train, idx_val, idx_test = load_protein_function('yeast')
n_classes = []
for k in range(len(Y_train)):
    n_classes.append(Y_train[k].shape[1])

# print(n_classes)
# Parameters
N = X.shape[0]                # Number of nodes in the graph
F = X.shape[1]                # Original feature dimension
# n_classes = Y_train.shape[1]  # Number of classes
F_ = 8                        # Output size of first GraphAttention layer
n_attn_heads = 4              # Number of attention heads in first GAT layer
dropout_rate = 0.6            # Dropout rate (between and inside GAT layers)
l2_reg = 5e-4/2               # Factor for l2 regularization
learning_rate = 1e-4          # Learning rate for Adam
epochs = 10                   # Number of training epochs
es_patience = 3               # Patience fot early stopping

# Preprocessing operations
X = preprocess_features(X)
A = A + np.eye(A.shape[0])  # Add self-loops

# Model definition (as per Section 3.3 of the paper)
X_in = Input(shape=(F,))
A_in = Input(shape=(N,))

dropout1 = Dropout(dropout_rate)(X_in)
graph_attention_1 = GraphAttention(F_,
                                   attn_heads=n_attn_heads,
                                   attn_heads_reduction='concat',
                                   dropout_rate=dropout_rate,
                                   activation='elu',
                                   kernel_regularizer=l2(l2_reg),
                                   attn_kernel_regularizer=l2(l2_reg))([dropout1, A_in])

dropout2 = Dropout(dropout_rate)(graph_attention_1)
graph_attention_2 = GraphAttention(n_classes[2],
                                   attn_heads=1,
                                   attn_heads_reduction='average',
                                   dropout_rate=dropout_rate,
                                   activation='sigmoid',
                                   kernel_regularizer=l2(l2_reg),
                                   attn_kernel_regularizer=l2(l2_reg),
                                   name="output_0")([dropout2, A_in])

dropout3 = Dropout(dropout_rate)(graph_attention_2)
FC_3 = Dense(n_classes[1], activation="sigmoid", name="output_1")(dropout3)


dropout4 = Dropout(dropout_rate)(FC_3)
FC_4 = Dense(n_classes[0], activation="sigmoid", name="output_2")(dropout4)

# Build model
model = Model(inputs=[X_in, A_in], outputs=[graph_attention_2, FC_3, FC_4])
optimizer = Adam(lr=learning_rate)

model.compile(optimizer=optimizer,
              loss=["binary_crossentropy","binary_crossentropy","binary_crossentropy"],
              weighted_metrics=['acc', 'acc', 'acc'])
model.summary()

# Callbacks
es_callback = EarlyStopping(monitor='val_weighted_acc', patience=es_patience)
tb_callback = TensorBoard(batch_size=N)
mc_callback = ModelCheckpoint('logs/best_model.h5',
                              monitor='val_weighted_acc',
                              save_best_only=True,
                              save_weights_only=True)

sample_weight_list_train =  {x: 1*idx_train for x in range(3)}
sample_weight_list_val =  {x: 1*idx_val for x in range(3)}
sample_weight_list_test =  {x: 1*idx_test for x in range(3)}
# Train model
validation_data = ([X, A], [Y_val[2], Y_val[1], Y_val[0]], sample_weight_list_val)
model.fit([X, A],
          [Y_train[2], Y_train[1], Y_train[0]],
          sample_weight=sample_weight_list_train,
          epochs=epochs,
          batch_size=N,
          validation_data=validation_data,
          shuffle=False,  # Shuffling data means shuffling the whole graph
          callbacks=[es_callback, tb_callback, mc_callback])
#
# # Load best model
# model.load_weights('logs/best_model.h5')

# Evaluate model
eval_results = model.evaluate([X, A],
                              [Y_test[2], Y_test[1], Y_test[0]],
                              sample_weight=sample_weight_list_test,
                              batch_size=N,
                              verbose=0)
print('Done.\n'
      'Test loss: {}\n'
      'Test accuracy: {}'.format(*eval_results))
