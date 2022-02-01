'''
Areas & Classes:
* bedroom : speech, alarm, drawer open or close, door, crying, mechanical fan, ringtone
* bathroom : speech, sink, toilet flush
* kitchen : speech, alarm, boiling, sink, water tap, microwave oven
* office : speech, alarm, printer, scissors, computer keyboard, ringtone
* entrance : speech, doorbell, keys jangling, knock, ringtone
* workshop : duct tape, hammer, sawing
'''

import csv


BASE_PATH = 'assets/dataset_split/'
DATA_FILES = ['dev.csv', 'eval.csv']
OUTPUT_FOLDER = 'assets/'

unique_classes = [
    'Speech','Alarm','Drawer_open_or_close','Door','Crying_and_sobbing',
    'Mechanical_fan', 'Ringtone', 'Sink_(filling_or_washing)', 'Water_tap_and_faucet',
    'Microwave_oven', 'Printer', 'Scissors', 'Computer_keyboard', 'Doorbell', 'Keys_jangling',
    'Knock', 'Ringtone', 'Packing_tape_and_duct_tape', 'Hammer', 'Sawing'
]

##############################################################################################

labels = []
filenames = []

for idx_file in DATA_FILES:

    with open(BASE_PATH + idx_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        # skip header
        counter = 0
        for row in csv_reader:
            if counter == 0:
                counter += 1
                continue

            fname = row[0]
            curr_labels = row[1]

            # there may be some overlapped classes
            for l in curr_labels.split(","):
                if l in unique_classes:
                    labels.append(l)
                    filenames.append(fname)


    output_file = open(OUTPUT_FOLDER + 'idx_' + idx_file, 'w')
    for label, filename in zip(labels, filenames):
        output_file.write("{},{}\n".format(filename,label))

    output_file.close()







