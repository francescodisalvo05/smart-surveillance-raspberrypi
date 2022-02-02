from utils.signal_generator import SignalGenerator

def get_data(labels, resampling, mfcc_options, train_path, test_path):

    with open('/content/domestic-sounds/assets/train_split_reduced_ds.txt' ,"r") as fp:
       train_files = [line.rstrip() for line in fp.readlines()]  
    
    with open('/content/domestic-sounds/assets/val_split_reduced_ds.txt' ,"r") as fp:
       val_files = [line.rstrip() for line in fp.readlines()]

    generator = SignalGenerator(labels, sampling_rate=44100, resampling_rate=resampling, **mfcc_options)
    train_ds = generator.make_dataset(train_files, True)
    val_ds = generator.make_dataset(val_files, False)
    
    # test_ds = generator.make_dataset(test_files, False)

    return train_ds, val_ds #, test_ds