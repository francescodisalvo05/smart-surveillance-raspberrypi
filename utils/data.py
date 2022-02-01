from utils.signal_generator import SignalGenerator

def get_data(labels, resampling, mfcc_options, train_path, test_path):

    with open(train_path ,"r") as fp:
        train_files = [line.rstrip() for line in fp.readlines()]  
    
    #with open(test_path ,"r") as fp:
    #    test_files = [line.rstrip() for line in fp.readlines()]  

    generator = SignalGenerator(labels, sampling_rate=44100, resampling_rate=resampling, **mfcc_options)
    train_ds = generator.make_dataset(train_files, True)
    train_ds = train_ds.take(4184)  #making 80-20 train-validation split
    val_ds = train_ds.skip(4184)
    
    # test_ds = generator.make_dataset(test_files, False)

    return train_ds, val_ds #, test_ds