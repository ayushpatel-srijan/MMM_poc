import streamlit as st
import pandas as pd
from optimizer import *
from functions import save_roas
import datetime as dt
import multiprocessing
import altair as alt
from joblib import Parallel, delayed  
from tqdm import tqdm  
import stqdm
import time
import threading
try:
    multiprocessing.set_start_method("fork")
except:
    print("Already set multiprocessing.set_start_method('fork')")
# Define the Streamlit app
st.set_page_config(layout="wide")

def main():

    st.markdown("<h1 style='text-align: center;'>Marketing Mix Model</h1>", unsafe_allow_html=True)
    #st.title("Marketing Mix Model")
    
    # Allow the user to upload a CSV file
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    info =st.info('Uploaded data should contain Date, impressions , spending of various channels', icon="ℹ️")

    consts={}
    sep=0

    col1, col2,col3 = st.columns(3)

    if uploaded_file is not None:
        info.empty()
        # Read the CSV file into a pandas DataFrame
        df=pd.DataFrame()
        df = pd.read_csv(uploaded_file)
        st.write(df.head(3))
        with col1:
            no_of_weeks =  st.number_input("Enter the number of past weeks' data you want to be used for calculations :", max_value=len(df), min_value=1, value=52 )
        with col2:
            budget = st.number_input('Enter your budget', min_value=100000.0)
        with col3:
            const = st.number_input('Enter the constraints (± percent change in budget share while optimising:)',
                                min_value=1, max_value=100, value=10)

        
        def display_number_inputs(cols):
            columns = st.columns(len(cols))

            for i, col in enumerate(cols):
                with columns[i]:
                    value = const
                    consts[col] = st.number_input(f"Enter constraints for {col}", value=value)
            print(consts)
        # Create a checkbox with a label
        if st.checkbox("Select indivisual constraints for each channel"):
            # Change the constraints for the columns
            sep = 1
            cols =[i.split('_')[1] for i in df.columns if 'mdip_' in i]
            display_number_inputs(cols)
        
        show_sales = st.checkbox('Show sales')

        # Define the plot function
        def plot_sales():
            chart = alt.Chart(df).mark_line().encode(
                x=alt.X('wk_strt_dt', axis=alt.Axis(labelAngle=45 ,tickCount=3,labelOverlap='greedy')),
                y='sales'
            ).configure_axis(
               # set font size for the axis title
             )
            return chart

        # Show or hide the plot based on the checkbox
        if show_sales:
            st.altair_chart(plot_sales(), use_container_width=True)
            #col1, col2,col3 = st.columns(3)

        if st.button("Train"):
        
            #s =time.time()
            #save_roas(df)
            #e=time.time()
            #st.write(e-s)
            st.session_state.progress=0

            # display a message to the user that the save_roas function is running
            st.write("Running save_roas function...")

            #start running the save_roas function in a separate thread
            save_roas_thread = threading.Thread(target=save_roas, args=(df,))
            start = time.time()
            save_roas_thread.start()
            tot_time = 2.75*len(df)
            # display a progress bar in the foreground while the save_roas function is running
            with st.empty():
                st.write("Saving ROAS data. Please wait...")
                my_bar = st.progress(0)
                st.session_state.progress=0
                while save_roas_thread.is_alive():
                    time.sleep(tot_time/100)
                    if st.session_state.progress < 100:
                        st.session_state.progress += 1
                        my_bar.progress(st.session_state.progress)
            end = time.time()

            print(f"time taken : {end-start}")
            
            # join the save_roas thread to wait for it to finish
            save_roas_thread.join()
            my_bar.empty()
            st.success("Trained Successfully..")

        if st.button("Lets optimize"):
            calculate(df, budget = budget ,ntail = no_of_weeks ,constr = const,sep=sep,sep_const = consts)

if __name__ == "__main__":
    main()