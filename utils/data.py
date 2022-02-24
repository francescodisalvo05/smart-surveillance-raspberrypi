import sys
sys.path.append('constants/')

from path import SPLIT_BASE_PATH, AUGUMENTATION_PATH


from utils.signal_generator import SignalGenerator


def get_data(labels, resampling, mfcc_options):

    with open('{}/train_split.txt'.format(SPLIT_BASE_PATH) ,"r") as fp:
       train_files = [line.rstrip() for line in fp.readlines()]
    
    with open('{}/val_split.txt'.format(SPLIT_BASE_PATH) ,"r") as fp:
       val_files = [line.rstrip() for line in fp.readlines()]

    with open('{}/test_split.txt'.format(SPLIT_BASE_PATH) ,"r") as fp:
       test_files = [line.rstrip() for line in fp.readlines()]

    generator = SignalGenerator(labels, sampling_rate=44100, resampling_rate=resampling, **mfcc_options)
    train_ds = generator.make_dataset(train_files, True, None)
    val_ds = generator.make_dataset(val_files, False, False)
    test_ds = generator.make_dataset(test_files, False, False)

    return train_ds, val_ds , test_ds