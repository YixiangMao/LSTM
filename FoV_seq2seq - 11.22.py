from __future__ import print_function
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import sys, glob, io, random
# if './360video/' not in sys.path:
#     sys.path.insert(0, './360video/')
from dataLayer import DataLayer
import cost as costfunc
from config import cfg
from dataIO import clip_xyz
from utility import reshape2second_stacks, get_data, get_shuffle_index, shuffle_data, get_gt_target_xyz, get_gt_target_xyz_oth
import _pickle as pickle
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
import pdb

# experiment = 1
batch_size = 32  # Batch size for training.
epochs = 200  # Number of epochs to train for.
latent_dim = 64  # Latent dimensionality of the encoding space.

fps = 30
num_encoder_tokens = 3*fps
num_decoder_tokens = 3*fps #6
max_encoder_seq_length = cfg.running_length
max_decoder_seq_length = cfg.predict_step

#### ====================data====================
##### data format 3or4--4
video_data_train = pickle.load(open('./data/shanghai_dataset_xyz_train.p','rb'),  encoding="bytes")       
video_data_train = clip_xyz(video_data_train)
datadb = video_data_train.copy()

### concat all videos and all users 
# # don't distinguish users or videos during training
_video_db,_video_db_future,_video_db_future_input = get_data(datadb,pick_user=False)     
total_num_samples = _video_db.shape[0]     #(10113, 10, 90)

#use last few as test
# num_testing_sample = int(0.15*total_num_samples)
num_testing_sample = 1000
encoder_input_data = _video_db[:-num_testing_sample,:,:]
# decoder_target_data = get_gt_target_xyz(_video_db_future)[:-num_testing_sample,:,:]
# decoder_input_data = get_gt_target_xyz(_video_db_future_input)[:-num_testing_sample,:,:]
decoder_target_data = _video_db_future[:-num_testing_sample,:,:]
decoder_input_data = _video_db_future_input[:-num_testing_sample,:,:]



### ====================Graph def====================
# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))      #num_decoder_tokens=6?
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens,activation='tanh')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
# Run training
model.compile(optimizer='Adam', loss='mean_squared_error')


### ====================Training====================
model_checkpoint = ModelCheckpoint('./model/fov_s2s_withTfor_epoch{epoch:02d}-{val_loss:.4f}.h5', monitor='val_loss', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                 patience=3, min_lr=1e-6)
stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2,
          shuffle=True,
          callbacks=[model_checkpoint, reduce_lr, stopping])



### ====================Testing====================
##### data format 3or4--4
# video_data_test = pickle.load(open('./data/shanghai_dataset_xyz_test.p','rb'),  encoding="bytes")
# video_data_test = clip_xyz(video_data_test)
# datadb = video_data_test.copy()


# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states

# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)


# from keras.models import load_model
# temp = load_model('fov_s2s.h5')

def decode_sequence_fov(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    last_location = input_seq[0,-1,:][np.newaxis,np.newaxis,:]
    last_mu_var = last_location
    target_seq = last_mu_var
    # target_seq = np.zeros((1, 1, num_decoder_tokens))

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    decoded_sentence = []
    for ii in range(cfg.predict_step):
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        decoded_sentence+=[output_tokens]
        # # Update the target sequence (of length 1).
        # target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq = output_tokens

        # Update states
        states_value = [h, c]

    return decoded_sentence


testing_input_list = []
gt_sentence_list = []
decoded_sentence_list = []
for seq_index in range(total_num_samples-num_testing_sample,total_num_samples):
    input_seq = _video_db[seq_index: seq_index + 1,:,:]
    testing_input_list+= [input_seq]
    decoded_sentence = decode_sequence_fov(input_seq)
    decoded_sentence_list+=[decoded_sentence]
    gt_sentence = _video_db_future[seq_index: seq_index + 1,:,:]
    gt_sentence_list+=[gt_sentence]

    # decoder_target = get_gt_target_xyz(gt_sentence)
    # print('-')
    # print('Decoded sentence - decoder_target:', np.squeeze(np.array(decoded_sentence))[:,:3]-np.squeeze(decoder_target)[:,:3])

pickle.dump(decoded_sentence_list,open('decoded_sentence.p','wb'))
pickle.dump(gt_sentence_list,open('gt_sentence_list.p','wb'))
pickle.dump(testing_input_list,open('testing_input_list.p','wb'))
print('Testing finished!')
