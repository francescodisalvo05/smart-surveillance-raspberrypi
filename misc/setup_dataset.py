'''
Areas & Classes:
* bedroom : speech, alarm, drawer open or close, door, crying, mechanical fan, ringtone
* bathroom : speech, sink, toilet flush
* kitchen : speech, alarm, boiling, sink, water tap, microwave oven
* office : speech, alarm, printer, scissors, computer keyboard, ringtone
* entrance : speech, doorbell, keys jangling, knock, ringtone
* workshop : duct tape, hammer, sawing
'''

BASE_PATH = '../assets/dataset_split/'
DEV = 'dev.csv'
EVAL = 'eval.csv'

unique_classes = [
    'Speech','Alarm','Drawer_open_or_close','Door','Crying_and_sobbing',
    'Mechanical_fan', 'Ringtone', 'Sink_(filling_or_washing)', 'Water_tap_and_faucet',
    'Microwave_oven', 'Printer', 'Scissors', 'Computer_keyboard', 'Doorbell', 'Keys_jangling'
    'Knock', 'Ringtone', 'Packing_tape_and_duct_tape', 'Hammer', 'Sawing'
]
