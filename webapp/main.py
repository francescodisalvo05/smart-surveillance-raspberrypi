'''
Areas & Classes:
* bedroom : speech, alarm, drawer open or close, door, crying, mechanical fan, ringtone
* bathroom : speech, sink, toilet flush
* kitchen : speech, alarm, boiling, sink, water tap, microwave oven
* office : speech, alarm, printer, scissors, computer keyboard, ringtone
* entrance : speech, doorbell, keys jangling, knock, ringtone
* workshop : duct tape, hammer, sawing
'''

import streamlit as st
from PIL import Image

container_info = st.container()
container_choices = st.container()

with container_info:

    st.title('Domestic Sounds')
    st.markdown("Lorem ipsum dolor sit amet, consectetur adipisci elit," 
                "sed eiusmod tempor incidunt ut labore et dolore magna aliqua." 
                "Ut enim ad minim veniam <br /><br />", unsafe_allow_html=True)
    
with container_choices:

    option = st.selectbox(
     'Where is your domestic sound detector?',
     ('Kitchen', 'Bedroom'))

    st.write('You selected:', option)

    st.markdown("<br /><br />", unsafe_allow_html=True)
    
    col1_kitchen, col2_kitchen = st.columns([1, 5])

    image = Image.open('./webapp/imgs/kitchen.jpeg')
    col1_kitchen.image(image, caption='Kitchen')

    col2_kitchen.markdown("<br />speech, alarm, boiling, sink, water tap, microwave oven", unsafe_allow_html=True)

    col1_bedroom, col2_bedroom = st.columns([1, 5])

    image = Image.open('./webapp/imgs/bedroom.jpeg')
    col1_bedroom.image(image, caption='Bedroom')

    col2_bedroom.markdown("<br />speech, alarm, drawer open or close, door, crying, mechanical fan, ringtone", unsafe_allow_html=True)

    
