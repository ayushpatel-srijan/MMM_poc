import streamlit as st
import json
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import altair as alt

dummy_dict = {'a' : 10.0, 'b': 20.0, 'c': 30.0}

def plot_pie_comparison(data1, data2, type1, type2):
    '''Funtion for plotting Plotly pie chart.
        input : dicts and their respective names as str
        returns : plotly fig object '''
    labels = list(data1.keys())

    # Create subplots: use 'domain' type for Pie subplot
    fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
    fig.add_trace(go.Pie(labels=labels, values=list(data1.values()), name="before optimization "),
                1, 1)
    fig.add_trace(go.Pie(labels=labels, values=list(data2.values()), name="after optimization"),
                1, 2)

    # Use `hole` to create a donut-like pie chart
    fig.update_traces(hole=.4, hoverinfo="label+percent+name")

    fig.update_layout(
        title_text=f"comparison between {type1} and {type2} ",
        # Add annotations in the center of the donut pies.
        annotations=[dict(text='earlier', x=0.18, y=0.5, font_size=20, showarrow=False),
                    dict(text='optimum', x=0.82, y=0.5, font_size=20, showarrow=False)])
    #fig.show()
    return fig

def optimize_budget(mroas, budgets, constraints) -> dict:
    '''
    1. Set your annual budget for each channel
    2. Set constraint for each channel, how much change you allow, e.g., sem: [-20%, +20%], tv: [-10%, +10%], display: [-10%, +10%]
    3. Move budget from low mROAS channels to high mROAS channels in a greedy way. for example:
    sem -20%
    move to: tv +10%
    -> if tv doesn't use up the money, give the rest money to: display + 10%. so on so forth until no budget left to be optimized.
    4. returns the optimized budget plan.
    '''
    
    sorted_channels = sorted(mroas, key=mroas.get, reverse=False)
    optimized_budget = budgets.copy()

    for channel in sorted_channels:
        current_budget = budgets[channel]
        budget_constraint = constraints[channel]
        min_budget = current_budget * (1 + budget_constraint[0]/100)
        max_budget = current_budget * (1 + budget_constraint[1]/100)
        
        budget_to_move = current_budget - min_budget
        
        if budget_to_move > 0:
            # Decrease the budget for the current channel
            optimized_budget[channel] -= budget_to_move
            #print("4. optimized_budget[channel]", optimized_budget[channel])
            # Try to increase the budget for the next channel
            next_channel_index = sorted_channels.index(channel) + 1
            if next_channel_index < len(sorted_channels):
                next_channel = sorted_channels[next_channel_index]
                next_channel_budget = optimized_budget[next_channel]
                max_budget_for_next_channel = budgets[next_channel] * (1 + constraints[next_channel][1]/100)
                #print("5. max_budget_for_next_channel", max_budget_for_next_channel)
                # Check if there is still budget available to be moved
                if budget_to_move > next_channel_budget - max_budget_for_next_channel:
                    optimized_budget[next_channel] += budget_to_move
                else:
                    optimized_budget[next_channel] = max_budget_for_next_channel
    return optimized_budget

def plot_pie_chart(data_dict = dummy_dict, title ="Past week channel budget distribution", legend = "Previous budget"):
    '''pliots pie charts'''
    keys = list(data_dict.keys())
    values = list(data_dict.values())
    plt.figure(figsize=(10,8))
    plt.pie(values, labels=keys,  autopct='%1.1f%%')
    plt.title(title)
    plt.legend(title=legend, loc='lower right')
    plt.show()

def calculate_percentages(data_dict) -> dict:
    '''takes : a dict
    returns : a dict where value re[resnt percentage of the key in ip dict'''
    
    total = sum(data_dict.values())
    percentages = {}
    for key, value in data_dict.items():
        try:
            percentage = (value / total) * 100
            percentages[key] = percentage
        except ZeroDivisionError:
            print(str(ZeroDivisionError), value, key,'\n', data_dict)
            pass
    return percentages

def calculate_difference(dict1 = {'a' :1, 'b':2}, dict2 = {'a' : 3, 'b' : 4})-> dict:
    '''calculates differemce between indivisual values of the 2 input dicts '''

    temp_dict = {}
    for k, v1 in dict1.items():
        v2 = dict2[k]
        temp_dict[k] = v2 - v1
    return temp_dict

def compare_budgets(prev = dummy_dict, new= dummy_dict)-> dict:
    '''calculates differemce between indivisual values of the 2 input dicts '''

    diff_dict ={}
    for k,v in new.items():
        diff = new[k] - prev[k] 
        #print(diff)
        diff_dict[k] = diff 
    return diff_dict

def get_percent_dict(data = dummy_dict)-> dict:
    '''takes : a dict
        returns : a dict where value re[resnt percentage of the key in ip dict'''
    
    perc_dict = {}
    total = sum(data.values())
    for k, v in data.items():
        perc = v/total * 100
        perc_dict[k] = perc
    return perc_dict

def calculate_shares(amount = 700000.0, shares= {'a':25.0, 'b' : 25.0, 'c' : 50.0}):
    '''fn that takes an amound and divides it as per the percentage of the shares dicts'''
    result ={}
    for k,v in shares.items():
        result[k] = v*amount/100
    return result

def get_auto_constraints(mroas_dict):
    '''fn for generating constraints based on mroas score'''

    max_score = max(mroas_dict.values())
    temp_dict = {}
    for k,v in mroas_dict.items():
        score = round(v/max_score, 2)*100
        temp_dict[k] = [-int(100-score), int(score)]
    return temp_dict

def _color_red_or_green(val):
    '''funtion for styling values in a df'''

    if type(val) is int or type(val) is float:    
        color = 'red' if val < 0 else 'green'
        return 'color: %s' % color
    else:
        pass

def calculate(df,budget = 100000.00, ntail = 209, constr = 10,sep=0 ,sep_const=None):
    '''
    the main function for calculating the optimized budget shares'''

    #df = pd.read_csv('my_data.csv')
    
    mroas_df = pd.read_csv('my_roas.csv')
    print(10*"*")
    mroas_df.rename(columns = {'Unnamed: 0':'channel'}, inplace = True)
    #mroas_df['channel']=
    #mroas_df = mroas_df.rename(columns = {'Unnamed: 0':'channel'})
    col1 , col2 = st.columns([1,2])
    with open('decompose.json' , 'r+') as fp:
            decompose_dict  = json.load(fp)
    contrib_df = pd.json_normalize(decompose_dict)
    
    highest_mroas = max(zip( list(mroas_df['mroas']), list(mroas_df['channel'])))
    highest_roas = max(zip( list(mroas_df['roas_mean']), list(mroas_df['channel']))) # tuple of highest mroas media-type and val
     # tuple of highest mroas media-type and val
    
    with col1:
        st.subheader('Marginal ROAS chart')
        '''Marginal ROAS represents the return of incremental spending based on current spending. For example, I have spent 100 Euroes on SEM, how much will the next 1 Euro bring.
    mROAS is calculated by increasing the current spending level by 1 percent, the incremental channel contribution over incremental channel spending.''' 
        

        #here#
        t1 ,t2, =st.tabs(['MROAS','ROAS'])
        with t1:
            st.bar_chart( mroas_df,y = 'mroas',  x = 'channel', use_container_width= True)
            st.write(f" **{highest_mroas[1]}** have the highest mROAS value of {round(highest_mroas[0],2)} ")
            #st.bar_chart( mroas_df,y = 'roas_mean',  x = 'channel', use_container_width= True)
        #st.markdown(f"<p style='font-size:24pt font-style: italic'>*{highest_mroas[1]}** have the highest mROAS value of {highest_mroas[0]} </p>", unsafe_allow_html=True)
        with t2:
            st.bar_chart( mroas_df,y = 'roas_mean',  x = 'channel', use_container_width= True)
            st.write(f" **{highest_roas[1]}** have the highest ROAS value of {highest_roas[0]} ")

        # Load the data
        #roas_df = pd.read_csv("roas_data.csv")

        # Create the first bar chart
        chart1 = alt.Chart(mroas_df).mark_bar().encode(
            x=alt.X('channel', title='Channel'),
            y=alt.Y('mroas', title='MROAS'),
            color=alt.ColorValue("#69b3a2")
        ).properties(width=alt.Step(80))

        # Create the second bar chart
        chart2 = alt.Chart(mroas_df).mark_bar().encode(
            x=alt.X('channel', title='Channel'),
            y=alt.Y('roas_mean', title='ROAS Mean'),
            color=alt.ColorValue("#404080")
        ).properties(width=alt.Step(80))

        # Combine the two charts
        combined_chart = chart1 | chart2

        # Display the combined chart using Streamlit
        #st.altair_chart(chart1, use_container_width=True)


    # Load the data
    roas = pd.read_csv("my_roas.csv")

    # Create some data for the bar graphs
    '''
    x = roas['Unnamed: 0'].values
    y1 = roas['roas_mean'].values
    y2 = roas['mroas'].values

    X_axis = np.arange(len(x))

    # Create a bar chart
    fig, ax = plt.subplots()
    ax.bar(X_axis - 0.2, y1, 0.4, label = 'ROAS')
    ax.bar(X_axis + 0.2, y2, 0.4, label = 'MROAS')

    ax.set_xticks(X_axis)
    ax.set_xticklabels(x)
    ax.legend()

    # Display the chart using Streamlit
    st.pyplot(fig)'''

    
    df = df.tail(ntail)
    # 1. media variables
    # media impression
    mdip_cols=[col for col in df.columns if 'mdip_' in col]
    # media spending
    mdsp_cols=[col for col in df.columns if 'mdsp_' in col]

   
    # holiday variables
    hldy_cols = [col for col in df.columns if 'hldy_' in col]
    # seasonality variables
    base_vars = hldy_cols

    # 3. sales variables
    sales_cols =['sales']

    df[['wk_strt_dt']+mdip_cols+['sales']].head()

    latest_rec = df[mdsp_cols].to_dict('records')[-1]
    prev_week_budget = sum(latest_rec.values())

    media_imp_cols = [col for col in df.columns if "mdip" in col]
    
    # media spending
    mdsp_cols=[col for col in df.columns if 'mdsp_' in col]
    
    temp_dict= dict()
    # Renaming channels to mentain uniformity
    for key, val in decompose_dict.items():
        temp_key = key.replace("mdip_","")
        temp_dict[temp_key] = val
    

    contrib_df = pd.DataFrame(data = {'channel' : temp_dict.keys(), 'contribution' : temp_dict.values()})
    
    spend_df = df[mdsp_cols]
    spend_df['total'] = spend_df.apply(lambda x : x.sum(), axis=1 )
    contrib_percent_df = contrib_df.copy()
    
    contrib_percent_df['contribution'] = contrib_percent_df['contribution'].apply(lambda x : x*100)
    #STR contrifpcdf
    #st.subheader("Decomposed contribution(%) of all media channels (based on last 52 weeks' data) :")
    #st.dataframe(contrib_percent_df.set_index('channel') .T)
    

    fig = px.scatter(
        contrib_percent_df,
        x="channel",
        y="contribution",
        size="contribution",
        color="channel",
    )   
    spend_df = df[mdsp_cols]
    spend_df['total'] = spend_df.apply(lambda x : x.sum(), axis=1 )
    #spend_df
 
    mroas_dict= mroas_df[['channel', 'mroas']].to_dict('records')

    temp = {}
    for d in mroas_dict:
        temp_key = d['channel']
        temp_val = d['mroas']
        temp[temp_key] = temp_val

    mroas_dict= temp

    #STR 
    #print(f"mroas_dict : {mroas_dict}")
    mroas_dict= mroas_df[['channel', 'mroas']].to_dict('records')

    temp = {}
    for d in mroas_dict:
        temp_key = d['channel']
        temp_val = d['mroas']
        temp[temp_key] = temp_val

    mroas_dict= temp
    
    budget_df = pd.merge(mroas_df.copy(), contrib_df.copy(), on = 'channel')
    

    budget_df['prev_week_budget_share'] = budget_df['contribution'].apply(lambda x : x*prev_week_budget/100)
    # STR
    
    #Sst.dataframe(budget_df)

    df[mdsp_cols].head()
    # Week
    prev_week_budget_dict = df[mdsp_cols].to_dict('records')[-1]
    #print("prev_week_budget_dict", prev_week_budget_dict)

    temp_dict = {}
    for key, val in prev_week_budget_dict.items():
        temp_key = key.replace("mdsp_","")
        temp_dict[temp_key] = val
    prev_week_budget_dict = temp_dict
    
    # Median

    budget_median_dict = dict(spend_df.median())
    prev_week_budget = budget_median_dict['total']
    budget_median_dict.popitem()
    
    temp_dict = {}
    for key, val in budget_median_dict.items():
        temp_key = key.replace("mdsp_","")
        temp_dict[temp_key] = val
    budget_median_dict = temp_dict
    
    # Mean
    budget_mean_dict = dict(spend_df.mean())
    prev_week_budget = budget_mean_dict['total']
    budget_mean_dict.popitem()
    
    temp_dict = {}
    for key, val in budget_mean_dict.items():
        temp_key = key.replace("mdsp_","")
        temp_dict[temp_key] = val
    budget_mean_dict = temp_dict

    prev_week_budget_dict_perc = get_percent_dict(prev_week_budget_dict)
    budget_mean_perc = get_percent_dict(budget_mean_dict)
    budget_median_perc = get_percent_dict(budget_median_dict)
    
    budget_median_dict = calculate_shares(budget, budget_median_perc)    
    budget_mean_dict = calculate_shares(budget ,budget_mean_perc)
         
    #budget = prev_week_budget +200000 # inplace for user input
    new_budget_dict= {}
    prev_week_budget = sum(prev_week_budget_dict.values())
    for k, v in prev_week_budget_dict.items():
        #print("for Channel", k, "budget was", v,"out of Total ", prev_week_budget)
        share = v/ prev_week_budget
        #print(f"for Channel {k} share was {round(share*100, 4)}%, that is {v}.")
        new_budget_dict[k] = budget*share
    
    #print(40*"-","new_budget_dict", new_budget_dict)
    
    # For auto generating contrainsC:
    new_cons = get_auto_constraints(mroas_dict=mroas_dict)    
    
    #budgets = new_budget_dict
    if sep == 0:
        minm = -(constr)
        maxm = constr
        constraints_dict=dict()
        for j in [i.split('_')[1] for i in df.columns if "mdip" in i]:
            constraints_dict[j] = [minm, maxm]

        print(constraints_dict)
    elif sep==1:
        constraints_dict=dict()
        for j in [i.split('_')[1] for i in df.columns if "mdip" in i]:
            constr = sep_const[j]
            minm = -(constr)
            maxm = constr

            constraints_dict[j] = [minm, maxm]
        print(10*'--')
        print(constraints_dict)
        



        #constraints_dict = {'dm': [minm,maxm],   
    #                    'inst': [minm,maxm],
    #                    'nsp': [minm,maxm],
    #                    'auddig': [minm,maxm],
    #                    'audtr': [minm,maxm],
    #                    'vidtr': [minm,maxm],
    #                    'viddig': [minm,maxm],
    #                    'so': [minm,maxm],
    #                    'on': [minm,maxm],
    #                    'sem': [minm,maxm]
    #                    }
    
    optimized_budget = optimize_budget( mroas_dict, new_budget_dict, constraints_dict)
    
    #st.write(" budget_median_dict, total:", budget_median_dict,sum(budget_median_dict.values()))
    optimized_budget_median = optimize_budget(mroas_dict, budget_median_dict, constraints_dict )

    optimized_budget_mean = optimize_budget( mroas_dict, budget_mean_dict, constraints_dict)
   
    # opt df
    opt_df = pd.DataFrame([optimized_budget, optimized_budget_mean, optimized_budget_median])
    #for i in range(3):
        #print(sum(opt_df.to_dict('records')[i].values()) )
    opt_df['total'] = opt_df.apply(lambda x : sum(x), axis=1)
    opt_df['discription'] = ['optimized_budget based on last week', 'optimized on budget_mean', 'optimized on budget_median']
    

    optimized_budget_perc= get_percent_dict(optimized_budget)
    optimized_budget_mean_perc = get_percent_dict(optimized_budget_mean)
    optimized_budget_median_perc = get_percent_dict(optimized_budget_median)

    perc_df = pd.DataFrame({'percentage_discription' : ['previous week budget', f'mean budget of past {ntail} weeks', f'median budget of past {ntail} weeks','optimized budget on previous weeks budget', f'optimeized budget by mean budget of past {ntail} weeks', f'optimeized budget by median budget of past {ntail} weeks']})
    #perc_df
    perc_recs = [prev_week_budget_dict_perc , budget_mean_perc ,budget_median_perc ,optimized_budget_perc,
                optimized_budget_mean_perc ,optimized_budget_median_perc ]
    temp_df = pd.json_normalize(perc_recs)
    
    perc_df = perc_df.join(temp_df)
    
    week_diff = calculate_difference(prev_week_budget_dict_perc, optimized_budget_perc)
    mean_diff = calculate_difference(budget_mean_perc, optimized_budget_mean_perc)
    median_diff = calculate_difference(budget_median_perc, optimized_budget_median_perc)
    week_diff['percentage_discription'] = 'week budget - optimized budget percentage diff'
    mean_diff['percentage_discription'] = f'{ntail} weeks mean budget - optimized budget percentage diff '
    median_diff['percentage_discription'] = f'{ntail} weeks median budget - optimized budget percentage diff'

    diff_df = pd.json_normalize([week_diff, mean_diff, median_diff])
    #perc_df = perc_df.append(diff_df,ignore_index=True)
    #perc_df.style.background_gradient(_color_red_or_green, )
    
    #STR
    with col2:

        diff_df.style.applymap(_color_red_or_green)
        st.subheader("Change in the budget percentages : ")
        #st.write(" ")
        table_style = """
        <style>
        .dataframe td {
            font-size: 20px;
        }
        </style>
        """

        # Display the DataFrame with larger text
        st.write(table_style, unsafe_allow_html=True)
        print(diff_df.columns)
        diff_df = diff_df[~diff_df["percentage_discription"].str.contains("week budget")]
        print(diff_df.info())
        cols=[i for i in diff_df.drop('percentage_discription',1).columns]
        diff_df[cols]=diff_df[cols].round(2)
        # Create a dictionary comprehension to format each column
        format_dict = {col: '{:.2f}' for col in cols}

        # Apply the formatting to the DataFrame
        diff_df_formatted = diff_df.style.format(format_dict)

        # Apply the styling to the formatted DatZmap(_color_red_or_green)
        diff_df_styled = diff_df_formatted.applymap(_color_red_or_green)

        # Display the styled DataFrame in Streamlit
        st.dataframe(diff_df_styled, width=1600)
        print(diff_df)


        st.subheader("Optimised budget")
        opt_df = opt_df[~opt_df["discription"].str.contains("week")]
        cols=[i for i in opt_df.drop('discription',1).columns]
        opt_df[cols] = opt_df[cols].applymap(lambda a: float(str(a).split('.')[0]))
        opt_df[cols] = opt_df[cols].round()
        format_dict = {col: '{:,.0f}' for col in cols}

        print(opt_df)
        st.write(opt_df.style.format(format_dict))
    

