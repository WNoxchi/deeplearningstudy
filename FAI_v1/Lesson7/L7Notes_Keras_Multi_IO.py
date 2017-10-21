# Wayne Nixalo - 2017-Jul-03 19:04
# Notes - FAI01 L7

# https://keras.io/getting-started/functional-api-guide/#multi-input-and-multi-output-models
# implementing a Multi-input Multi-output with the Keras functional API

from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model

# Headline input: meant to receive sequences of 100 integers, between 1 and 10000.
# NOTE that we can name any layer by passing it a 'name' argument.
main_input = Input(shape=(100,), dtype='int32', name='main_input')

# This embedding layer will encode the input sequence
# into a sequence of dense 512-dimensional vectors.
x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)

# An LSTM will transform the vector sequence into a single vector,
# containing information about the entire sequence
lstm_out = LSTM(32)(x)

# Here we insert the auxiliary loss, allowing the lSTM and EMbedding layer to
# be trained smoothly even though the main loss will be much higher in the model.

auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_out)

# At this point, we feed into the model our auxiliary input data by
# concatenating it with the LSTM output:

auxiliary_input = Input(shape=(5,), name='aux_input')
x = keras.layers.concatenate([lstm_out, auxiliary_input])

# We stack a deep densely-connected network on top
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)

# And finally we add the main logistic regression layer
main_output = Dense(1, activation='sigmoid', name='main_output')(x)

# This defines a model with 2 inputs and 2 outputs:
model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])

# We compile the mdoel and assign a weight of 0.2 to the aux loss. To specy
# dfnt loss_weights or loss for each dfnt output, you can use a list or dict.
# Here we pass a single loss as the loss arg, so the same loss will be used on
# all outputs:
model.compile(optimizer='rmsprop', loss='binary_crossentropy',
              loss_weights=[1.,0.2])

# We can train the model by passing it lists of input arrays and target arrays:
model.fit([headline_data, additional_data], [labels, labels],
          epochs=50, batch_size=32)

# Since our inputs and outputs are named (we passed them a "name" arg), we
# could also have compiled the model via:
model.compile(optimizer='rmsprop',
              loss={'main_output': 'binary_crossentropy', 'aux_output': 'binary_crossentropy'},
              loss_weights={'main_output': 1., 'aux_output':0.2})
# And trained it via:
model.fit({'main_input': headline_data, 'aux_input': additional_data},
          {'main_output': labels, 'aux_output': labels},
          epochs=50, batch_size=32)

################################################################################
#
# VISUAL QUESTION ANSWERING MODEL / VIDEO QUESTION ANSWERING MODEL

# Visual question answering model
# This model can select the correct one-word answer when asked a natural-
# language question about a picture.

# It works by encoding the question into a vector, encoding the image into a
# vector, concatenating the two, and training on top a logistic regression over
# some vocabulary of potential answers.

from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model, Sequential

# First, let's define a vision model using a Sequential model.
# This model will encode an image into a vector.
vision_model = Sequential()
vision_model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(3, 224, 224)))
vision_model.add(Conv2D(63, (3, 3), activation='relu'))
vision_model.add(MaxPooling2D((2, 2)))
vision_model.add(Conv2D(128, (3, 3), activaiton='relu', padding='same'))
vision_model.add(Conv2D(128, (3, 3), activaiton='relu'))
vision_model.add(MaxPooling2D((2, 2)))
vision_model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
vision_model.add(Conv2D(256, (3, 3), activation='relu'))
vision_model.add(Conv2D(256, (3, 3), activation='relu'))
vision_model.add(MaxPooling2D((2, 2)))
vision_model.add(Flatten())

# Now let's get a tensor with the output of our vision model:
image_input = Input(shape=(3, 224, 224))
encoded_image = vision_model(image_input)

# Next, let's define a language model to encode the question into a vector.
# Each question will be at most 100 words long,
# and we'll index words as integers from 1 to 9999.
question_input = Input(shape=(100,), dtype='int32')
embedded_question = Embedding(input_dim=10000, output_dim=256, input_length=100)(question_input)
encoded_question = LSTM(256)(embedded_question)

# Let's concatenate the question vector and the image vector:
merged = keras.layers.concatenate([encoded_question, encoded_image])

# And let's train a logistic regresiion over 1000 words on top:
output = Dense(1000, activation='softmax')(merged)

# This is our final model:
vqa_model = Model(inputs=[image_input, question_input], outputs=output)

# XXX: The next state would be training this model on actual data.


# Video Question Answering Model:
# Now that we've trained our image QA model, we can quickly turn it inot a
# video QA model. W/ appropriate training, you'll be able to show it a short
# video (eg. 100-frame human-action) and ask a natural language question about
# the video (eg. "what sport is the girl playing?" -> "lacrosse")

fro keras.layers import TimeDstributed

video_input = Input(shape=(100, 3, 224, 224))
# This is our video encoded via the previously trained vision_model (weights are resused)
encoded_frame_sequence = TimeDstributed(vision_model)(video_input ) # the output will be a sequence of vectors
encoded_video = LSTM(256)(encoded_frame_sequence)   # the output will be a vector

# This is a model-level representation of the question encoder, reusing the same weights as before:
question_encoder = Model(inputs=question_input, outputs=encoded_question)

# Let's use it to encode the question:
video_question_input = Input(shape=(100,), dtype='int32')
encoded_video_question = question_encoder(video_question_input)

# And this is our video question answering model:
merged = keras.layers.concatenate([encoded_video, encoded_video_question])
output = Dense(1000, activation='softmax')(merged)
video_qa_model = Model(inputs=[video_input, video_question_input), outputs=output)
