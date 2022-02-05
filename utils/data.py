import sys
sys.path.append('constants/')

from path import SPLIT_BASE_PATH


from utils.signal_generator import SignalGenerator


def get_data(room, labels, resampling, mfcc_options):

    with open('{}/train_{}_split.txt'.format(SPLIT_BASE_PATH, room) ,"r") as fp:
       train_files = [line.rstrip() for line in fp.readlines()]
    
    with open('{}/val_{}_split.txt'.format(SPLIT_BASE_PATH, room) ,"r") as fp:
       val_files = [line.rstrip() for line in fp.readlines()]

    with open('{}/test_{}_split.txt'.format(SPLIT_BASE_PATH, room) ,"r") as fp:
       test_files = [line.rstrip() for line in fp.readlines()]

    generator = SignalGenerator(labels, sampling_rate=44100, resampling_rate=resampling, **mfcc_options)
    train_ds = generator.make_dataset(train_files, True, True)
    val_ds = generator.make_dataset(val_files, False, False)
    test_ds = generator.make_dataset(test_files, False, False)

    
    # test_ds = generator.make_dataset(test_files, False)

    return train_ds, val_ds , test_ds