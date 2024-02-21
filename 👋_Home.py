import streamlit as st
import time
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import webbrowser

def display_text_character_by_character(text):
    output = ''
    placeholder = st.empty()
    for char in text:
        output += char
        placeholder.markdown(output, unsafe_allow_html=True)
        # Generate a random typing speed between 0.05 and 0.1 seconds
        typing_speed = random.uniform(0.01, 0.08)
        time.sleep(typing_speed)

def think():
    return None

def show_smartness_over_time():
    
    before_using_data = np.random.randn(20) * 50  # Randomly fluctuating smartness
    zero_values_data = np.zeros(20)
    while_using_data = np.linspace(40, 150, 20)  # Increasing levels of smartness 

    # Create a DataFrame
    chart_data = pd.DataFrame({'before using me': before_using_data, '': zero_values_data,'while using me': while_using_data})
    # Display the chart gradually
    chart_placeholder = st.empty()
    for i in range(len(chart_data)):
        chart_placeholder.area_chart(chart_data.iloc[:i+1])
        waiting = random.uniform(0.1, 0.3)
        time.sleep(waiting)

def thinking(t):
    with st.spinner("Thinking..."):
        time.sleep(t)

st.set_page_config(page_title="Home", page_icon="📈")
with st.sidebar:
        thanks = "If you want to learn more about my creator, please visit his website below."
        st.write(thanks)
        if st.button("Click Here! 🃏"):
            website = "https://www.youssefchouay.com"
            webbrowser.open_new_tab(website)



def main():
    st.title("Chat with It!")

    text_to_display = "Hi! I'm your AI helper that lets you chat with your PDFs, websites, and I'm even capable of doing your research for you!"
    display_text_character_by_character(text_to_display)
    explaination = "I was created to help make your research easier, and give you the answers you don't have time to look for!"
    display_text_character_by_character(explaination)

    argument = "To convince you of how helpful I am..."
    display_text_character_by_character(argument)
    thinking(2)
    think = "Hmmmm"
    display_text_character_by_character(think)
    thinking(3)
    found = "💡... Here is a chart of how smart people when using me vs not:"
    display_text_character_by_character(found)


    show_smartness_over_time()

    


if __name__ == '__main__':
    main()