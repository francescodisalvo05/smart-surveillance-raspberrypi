import csv

BASE_PATH = 'assets/dataset_idx/'
DATA_FILES = ['dev.csv', 'eval.csv']
OUTPUT_FOLDER = 'assets/dataset_idx/'

unique_fixed_classes = [
    'Bark', 'Knock', 'Drill', 'Hammer',
    'Fire', 'Gunshot_and_gunfire'
]

##############################################################################################



for idx_file in DATA_FILES:

    labels = []
    filenames = []

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
            found = False

            # there may be some overlapped classes
            current_labels = curr_labels.split(",")
            for l in current_labels:
                if l in unique_fixed_classes:
                    found = True
                    labels.append(l)
                    filenames.append(fname)
            
            if not found:
                # particular cases
                if 'Slam' in current_labels and \
                'Door' in current_labels and \
                'Motor_vehicle_(road)' not in current_labels:
                    labels.append('Slam')
                    filenames.append(fname)
                
                # (Glass && (Shatter or Explosion)) or Crushing'''
                elif ('Glass' in current_labels and \
                        ('Shatter' in current_labels or 'Explosion' in current_labels)) or \
                    'Crushing' in current_labels:
                    labels.append('Glass')
                    filenames.append(fname)


    output_file = open(OUTPUT_FOLDER + 'new_idx_' + idx_file, 'w')
    for label, filename in zip(labels, filenames):
        output_file.write("{},{}\n".format(filename,label))

    output_file.close()







