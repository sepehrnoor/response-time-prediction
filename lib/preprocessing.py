import numpy as np
import pandas as pd
import sys
import math
import h5py
from datetime import datetime

# Prints a progress bar
def progressBar(current, total, barLength = 20):
  percent = float(current) * 100 / total
  arrow   = '_' * int(percent/100 * barLength - 1) + '🐌'
  spaces  = ' ' * (barLength - len(arrow))
  sys.stdout.write('\rProgress: [%s%s🏠] %d %%' % (arrow, spaces, percent))
  sys.stdout.flush()
  if (current == total):
    print("\r🐌🏠\nDone!")
  
# Inverts first and second indices. Useful for changing the indexing from [user, feature] to [feature, user]
def transpose_list(l):
  return list(map(list, zip(*l)))

# Creates the data needed for sakt, specifically exercises, intercations and labels
def process_sakt(series_np, number_of_exercises, time_steps, stride=1, padding='pre'):
  # The big exercise and answer sequences (entire user or session)
  series_transposed = transpose_list(series_np)
  exercise_seq = series_transposed[0]
  answer_seq = series_transposed[1]
  exercise_id_seq = series_transposed[2]

  if np.any(np.concatenate(exercise_seq) == 0):
    print("bad id in exercises")

  exercises = process_one_feature(exercise_seq, time_steps, stride=stride, padding=padding)
  past_exercises = process_one_feature(exercise_id_seq, time_steps, shift_data=True, stride=stride, padding=padding)
  answers = process_one_feature(answer_seq, time_steps, stride=stride, padding=padding)
  past_answers = process_one_feature(answer_seq, time_steps, shift_data=True, stride=stride, padding=padding)
  exercise_ids = process_one_feature(exercise_id_seq, time_steps, stride=stride, padding=padding)

  interactions = past_exercises + past_answers * number_of_exercises

  return exercises, interactions, answers, exercise_ids

# Shifts, cuts, pads and windows one data series (list consisting of lists with variables lengths) into a number of windows with fixed length
def process_one_feature(series_np, time_steps, dtype='int', shift_data=False, shift_forward=False, stride=1, padding='pre'):
  number_of_users = len(series_np)
  total_number_of_windows = 0
  # How many windows we will be getting from each sequence
  num_windows_from_sequence = np.zeros((number_of_users), dtype='uint')
  # First calculate how many 'window' sequences there will be, and the length of each
  for idx in range(number_of_users):
    # NOTE: Remember that we will be dropping one sample from the start of the sequence, 
    #       we need to adjust for that when calculating number of windows 
    number_of_usable_samples = len(series_np[idx]) - 1
    # If there are no samples, then there are no windows
    if number_of_usable_samples < 1:
        number_of_windows = 0
    else:
      # Anything smaller than windows size is one window
      if number_of_usable_samples <= time_steps:
        number_of_windows = 1
      # Otherwise we can calculate it
      else:
        number_of_windows = int(math.ceil((number_of_usable_samples - time_steps) / stride)) + 1

    num_windows_from_sequence[idx] = number_of_windows
    total_number_of_windows += number_of_windows
  windows = np.zeros((total_number_of_windows, time_steps), dtype=dtype)
  total_idx = 0
  for idx in range(number_of_users):
    # Skip user if no windows can be created
    if num_windows_from_sequence[idx] > 0:
      # Entire sequence for one user
      sequence = series_np[idx]
      
      if shift_data:
        if shift_forward:
          sequence = np.roll(sequence, -1)
        else:
          sequence = np.roll(sequence, 1)
      # Drop first element because of time shifting
      sequence = sequence[1:]
      # Split into windows
      wins = split_into_windows(sequence, time_steps, dtype=dtype, stride=stride, padding=padding)
      windows[int(total_idx):int(total_idx+num_windows_from_sequence[idx]),:] = wins
      total_idx += num_windows_from_sequence[idx]
  return windows

# Pads the data and cuts it into overlapping windows
def split_into_windows(sequence, time_steps, pad_value=0, dtype='int', stride=1, padding='pre'):

  # Pad the sequence if smaller than time_steps
  difference = time_steps - len(sequence)
  if (difference < 0):
    pad_length = 0
  else:
    pad_length = difference
  padded_sequence = np.pad(sequence, (pad_length * (padding == 'pre'), pad_length * (padding != 'pre')), 'constant', constant_values=(pad_value, pad_value))

  # then we pad the sequence so difference with time steps is divisible by stride 
  difference = len(padded_sequence) - time_steps
  stride_pad_length = (stride - (difference % stride)) % stride
  padded_sequence = np.pad(padded_sequence, (stride_pad_length, 0), 'constant', constant_values=(pad_value, pad_value))

  # Calculate number of samples in sequence and total number of windows
  difference = len(padded_sequence) - time_steps
  number_of_windows = int(difference / stride) + 1

  # Create window array
  windows = np.zeros((number_of_windows, time_steps), dtype=dtype)

  # Iterate through sequence and copy to window array
  for i in range(number_of_windows):
      # if np.count_nonzero(padded_sequence [i : i + time_steps]) == 0 and unique_vals > 2:
      #   print("!!BAD PADDING!! original length=%i, unique_vals=%i" % (orig_length, unique_vals))
      windows[i] = padded_sequence[i * stride : i * stride + time_steps]
  return windows


# Saves the data in training, validation and test batches in an .h5 file (hdf5 format)
def save_h5(file_path, dataset_name, data, validation_ratio=0.2, test_ratio=0.05, split_sections=2, append=False, dtype='int'):
  number_of_samples = len(data)

  split_length = int(number_of_samples/split_sections)

  for i in range(split_sections):
    subsection = data[i * split_length : (i+1) * split_length]

    number_of_subsamples = len(subsection)

    test_length = int(number_of_subsamples * test_ratio)
    val_length = int(number_of_subsamples * validation_ratio)

    if i == 0:
      test_data = subsection[:test_length]
      val_data = subsection[test_length:test_length+val_length]
      train_data = subsection[test_length+val_length:]
    else:
      test_data = np.append(test_data, subsection[:test_length], axis=0)
      val_data = np.append(val_data, subsection[test_length:test_length+val_length], axis=0)
      train_data = np.append(train_data, subsection[test_length+val_length:], axis=0)

  if append:
    with h5py.File(file_path, 'a') as hf:
      hf[dataset_name + '_test'].resize((hf[dataset_name + '_test'].shape[0] + test_data.shape[0]), axis = 0)
      hf[dataset_name + '_test'][-test_data.shape[0]:] = test_data
      hf[dataset_name + '_val'].resize((hf[dataset_name + '_val'].shape[0] + val_data.shape[0]), axis = 0)
      hf[dataset_name + '_val'][-val_data.shape[0]:] = val_data
      hf[dataset_name + '_train'].resize((hf[dataset_name + '_train'].shape[0] + train_data.shape[0]), axis = 0)
      hf[dataset_name + '_train'][-train_data.shape[0]:] = train_data
  else:
    with h5py.File(file_path, 'a') as hf:
      hf.create_dataset(dataset_name + '_test',  data=test_data, dtype=dtype, chunks=True, maxshape=(None, None))
      hf.create_dataset(dataset_name + '_val',   data=val_data, dtype=dtype, chunks=True, maxshape=(None, None))
      hf.create_dataset(dataset_name + '_train', data=train_data, dtype=dtype, chunks=True, maxshape=(None, None))
  
def select_from_rows(arr, indices):
  return arr[np.arange(len(arr)), indices]
