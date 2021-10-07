# The order of columns must be [user id, exercise tag, correctness, exercise id, [timestamp]]
columns_dict = {
    #  name               user id           category              correctness         exercise id        elapsed time              timestamp
    'akribian':         ['StudentId', 'LearningSequenceTitle', 'SubmissionOutcomes', 'ExerciseTitle',   'ExerciseResponseTime',   'FinishedOn'],
    'assistments_2009': ['user_id',   'skill_name',            'correct',            'problem_id',      'ms_first_response',      'FinishedOn'],
    'assistments_2012': ['user_id',   'skill',                 'correct',            'problem_id',      'ms_first_response',      'end_time'],
    'junyi_academy' :   ['uuid',      'upid',                  'is_correct',         'ucid',            'total_sec_taken',        'timestamp_TW'],
    'ednet' :           ['user_id',   'content_type_id' ,      'answered_correctly', 'content_id', 'prior_question_elapsed_time', 'timestamp'],
    'generic':          ['user_id',   'category',              'correctness',        'exercise_id',      'elapsed_time',          'timestamp', 'elapsed_zscore', 'elapsed_mean', 'correctness_mean']
}

# How many time steps to use for dataset
# Average lengths:
# 'akribian': 600,
# 'assistments_2009': 60,
# 'assistments_2012': 100,
# 'junyi_academy': 220,
# 'ednet': 250

time_steps_dict = {
    'akribian': 600,
    'assistments_2009': 180,
    'assistments_2012': 700,
    'junyi_academy': 600,
    'ednet': 1728
}

stride_dict = {
    'akribian': 1,
    'assistments_2009': 1,
    'assistments_2012': 1,
    'junyi_academy': 300,
    'ednet': 1728
}

# Number of exercise tags in each dataset
exercise_dict = {
    'akribian': 239,
    'assistments_2009': 103,
    'assistments_2012': 199,
    'junyi_academy': 11,
    'ednet': 9
}

# Number of exercise ids in each dataset
exercise_id_dict = {
    'akribian': 787,
    'assistments_2009': 16892,
    'assistments_2012': 50989,
    'junyi_academy': 1327,
    'ednet': 13525
}

# Average length of elapsed time - used for normalization
time_scale_dict = {
    'akribian': 20.,
    'assistments_2009': 1.,
    'assistments_2012': 110.,
    'junyi_academy': 90.,
    'ednet': 45.
}

# Encoding of the csv files.
encodings_dict = {
    'akribian': 'utf-8',
    'assistments_2009': 'cp850',
    'assistments_2012': 'cp850',
    'junyi_academy': 'utf-8',
    'ednet': 'utf-8',
}

# Ratio of data to be used for validation
val_ratio_dict = {
    'akribian': 0.5,
    'assistments_2012': 0.05,
    'junyi_academy': 0.05,
    'ednet': 0.05,
}

# Whether to shuffle the datasets
shuffle_dict = {
    'akribian': False,
    'assistments_2012': False,
    'junyi_academy': True,
    'ednet': True,
}