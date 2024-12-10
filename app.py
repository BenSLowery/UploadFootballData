# Streamlit app for data input and plot generation.
import streamlit as st
import pandas as pd
import numpy as np
import threading
import matplotlib.pyplot as plt
import scipy.stats as sp
plt.style.use("fivethirtyeight")
_lock = threading.RLock()


# Some pre-processing steps
############################

# stores card info
card = {"Name": [],"Age": [],"Nationality": [],"Overall": [],"Potential": [],"Club": [],"Value": [],"Wage": [],"Special": [],"Preferred.Foot": [],"International.Reputation": [],"Weak.Foot": [],"Skill.Moves": [],"Work.Rate": [],"Body.Type": [],"Height": [],"Weight": [],"Crossing": [],"Finishing": [],"HeadingAccuracy": [],"ShortPassing": [],"Volleys": [],"Dribbling": [],"Curve": [],"FKAccuracy": [],"LongPassing": [],"BallControl": [],"Acceleration": [],"SprintSpeed": [],"Agility": [],"Reactions": [],"Balance": [],"ShotPower": [],"Jumping": [],"Stamina": [],"Strength": [],"LongShots": [],"Aggression": [],"Interceptions": [],"Positioning": [],"Vision": [],"Penalties": [],"Composure": [],"Marking": [],"StandingTackle": [],"SlidingTackle": [],"GKDiving": [],"GKHandling": [],"GKKicking": [],"GKPositioning": [],"GKReflexes": [],"Preferred.Position": [],"Best.Overall.Rating": [],"Pace": [],"Shooting": [],"Passing": [],"Defending": [],"Physicality":[]}
ordered_attr = ['Acceleration',
 'Aggression',
 'Agility',
 'Balance',
 'BallControl',
 'Composure',
 'Crossing',
 'Curve',
 'Dribbling',
 'FKAccuracy',
 'Finishing',
 'HeadingAccuracy',
 'Interceptions',
 'Jumping',
 'LongPassing',
 'LongShots',
 'Marking',
 'Penalties',
 'Positioning',
 'Reactions',
 'ShortPassing',
 'ShotPower',
 'SlidingTackle',
 'SprintSpeed',
 'Stamina',
 'StandingTackle',
 'Strength',
 'Vision',
 'Volleys',
 'GKDiving',
 'GKHandling',
 'GKKicking',
 'GKPositioning',
 'GKReflexes']
rating_weights = np.array([[0.04040404        , 0.07      , 0.05      , 0.04      , 0.07      ,
        0.        , 0.        , 0.04040404, 0.05050505, 0.        ,
        0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.05050505, 0.        , 0.        , 0.07070707,
        0.        ],
       [0.        , 0.03030303, 0.        , 0.03030303, 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        ],
       [0.13131313, 0.09090909, 0.13131313, 0.09090909, 0.08080808,
        0.06060606, 0.        , 0.        , 0.        , 0.        ,
        0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        ],
       [0.1010101 , 0.14141414, 0.15151515, 0.15151515, 0.13131313,
        0.14141414, 0.1010101 , 0.08080808, 0.07070707, 0.04040404,
        0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        ],
       [0.        , 0.09090909, 0.        , 0.        , 0.1010101 ,
        0.        , 0.        , 0.12121212, 0.09090909, 0.        ,
        0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        ],
       [0.07070707, 0.16161616, 0.14141414, 0.13131313, 0.15151515,
        0.07070707, 0.        , 0.04040404, 0.        , 0.        ,
        0.        ],
       [0.18181818, 0.1010101 , 0.11111111, 0.07070707, 0.06060606,
        0.02020202, 0.        , 0.        , 0.        , 0.        ,
        0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        ],
       [0.1010101 , 0.        , 0.02020202, 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.04040404, 0.1010101 ,
        0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.05050505, 0.14141414, 0.12121212, 0.12121212, 0.13131313,
        0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.03030303,
        0.        ],
       [0.        , 0.        , 0.        , 0.04040404, 0.05050505,
        0.13131313, 0.1010101 , 0.        , 0.        , 0.        ,
        0.        ],
       [0.03030303, 0.04040404, 0.04040404, 0.05050505, 0.        ,
        0.04040404, 0.        , 0.        , 0.        , 0.        ,
        0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.09090909, 0.07070707, 0.08080808, 0.14141414,
        0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        ],
       [0.08080808, 0.07070707, 0.09090909, 0.07070707, 0.07070707,
        0.08080808, 0.07070707, 0.08080808, 0.08080808, 0.05050505,
        0.11111111],
       [0.05050505, 0.09090909, 0.09090909, 0.16161616, 0.11111111,
        0.17171717, 0.14141414, 0.1010101 , 0.07070707, 0.05050505,
        0.        ],
       [0.1010101 , 0.        , 0.05050505, 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.05050505, 0.11111111, 0.14141414, 0.1010101 ,
        0.        ],
       [0.05050505, 0.06060606, 0.05050505, 0.03030303, 0.06060606,
        0.        , 0.        , 0.06060606, 0.07070707, 0.02020202,
        0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.05050505,
        0.06060606, 0.06060606, 0.1010101 , 0.08080808, 0.        ,
        0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.05050505, 0.12121212, 0.08080808, 0.11111111, 0.17171717,
        0.        ],
       [0.05050505, 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.04040404, 0.        , 0.        , 0.1010101 ,
        0.        ],
       [0.        , 0.06060606, 0.08080808, 0.14141414, 0.07070707,
        0.13131313, 0.04040404, 0.        , 0.        , 0.        ,
        0.        ],
       [0.02020202, 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.21212121],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.21212121],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.05050505],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.21212121],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.21212121]])


tab1, tab2, tab3, tab4 = st.tabs(["Admin", "Input", "Player output", "Plots"])

with tab1:
    st.header("Metadata")
    card['Name'] = st.text_input("School Name")
    card['Age'] = st.number_input("Age", min_value=1, max_value=99)
    card['Nationality'] = st.text_input("Nationality")
    card['Club'] = st.text_input("Club")
    DEFAULT_RATING = st.number_input("Default Player rating", help='If a student exercise cannot be mapped to a fifa player state, then this is the default', value=85, min_value=1, max_value=99)


with tab2:
    st.header("Data Input")
    st.write("Input data for each user, but do not change the Benchmark as this is what is used to generate the score")
    df = pd.DataFrame(
        {
            "Name":['Benchmark'],
            "Sprint 5m":[1.1],
            "Sprint 10m":[2.2],
            "Sprint 20m":[3.72],
            "Pass 5m":[10],
            "Pass 10m":[10],
            "Dribble 5m":[5.8],
            "Dribble 10m":[8.0],
            "Dribble Slalom":[11.5],
            "Agility Slalom":[8.4],
            "Power Standing Throw":[8.0],
            "Power Standing Jump":[2.75],
            "Control (Rebounder passes in 30s)":[52],
            "Reactions Blazepods":[57],
            "Awareness Blazepods":[34]
        }
    )

    edited_df = st.data_editor(df,num_rows="dynamic") # 👈 An editable dataframe

    baseline_df = edited_df[edited_df['Name'] == 'Benchmark']
    scored_df = edited_df[edited_df['Name'] != 'Benchmark'].dropna()

    # Logic to calculate card
    means = scored_df.mean(axis=0, numeric_only=True)
    both_df = pd.concat([means, baseline_df[means.index].T],axis=1).T
    adjusted = np.ceil(20*np.log(both_df.min(axis=0)/both_df.max(axis=0)*100)+7.89)

    # Add adjusted values ro the card
    card['SprintSpeed'] = np.round(np.mean([adjusted['Sprint 20m'],adjusted['Sprint 10m']]))
    card['Acceleration'] = adjusted['Sprint 5m']
    card['ShortPassing'] = adjusted['Pass 5m']
    card['LongPassing'] = adjusted['Pass 10m']
    card['Dribbling'] = adjusted['Dribble 5m']
    card['Composure']=  adjusted['Dribble 10m']
    card['Balance'] = adjusted['Dribble Slalom']
    card['Agility'] = adjusted['Agility Slalom']
    card['Strength'] = adjusted['Power Standing Throw']
    card['Jumping'] = adjusted['Power Standing Jump']
    card['BallControl'] = adjusted['Control (Rebounder passes in 30s)']
    card['Reactions'] = adjusted['Reactions Blazepods']
    card['Positioning'] = adjusted['Awareness Blazepods']
    card['Composure']  = adjusted['Awareness Blazepods']
    card['Marking'] = adjusted['Awareness Blazepods']
    card['Stamina'] = np.round(np.mean([adjusted['Dribble 10m'],adjusted['Sprint 20m']]))

    # Misc.
    card['Value'] = 100000000
    card['Wage'] = 350000
    card['Special'] = 0
    card['Preferred.Foot'] = 'Right'
    card['International.Reputation'] = 5
    card['Weak.Foot'] = 4
    card['Skill.Moves'] = 4
    card['Work.Rate'] = 'High/High'
    card['Body.Type'] = 'Unique'
    card['Height'] = "179cm"
    card['Weight'] = "75kg"
    card['GKDiving'] = 10
    card['GKHandling'] = 10
    card['GKKicking'] = 10
    card['GKPositioning'] = 10
    card['GKReflexes'] = 10
    card['Potential'] = 99


    # Fill the rest with the defaults value
    for it,val in card.items():
        if type(val) == list:
            card[it] = DEFAULT_RATING
    
    # Face stats =ROUND(AVERAGE([Jumping],[Stamina],[Strength],[Aggression]),0)
    card['Pace'] = np.round(np.mean([card['SprintSpeed'], card['Acceleration']]))
    card['Shooting'] = np.round(np.mean([card['Finishing'], card['ShotPower'], card['LongShots'], card['Penalties'], card['Volleys']]))
    card['Passing'] = np.round(np.mean([card['Vision'], card['Crossing'], card['FKAccuracy'], card['LongPassing'], card['ShortPassing'], card['Curve']]))
    card['Defending'] = np.round(np.mean([card['Interceptions'], card['HeadingAccuracy'], card['Marking'], card['StandingTackle'], card['SlidingTackle']]))
    card['Physicality'] = np.round(np.mean([card['Jumping'], card['Stamina'], card['Strength'], card['Aggression']]))

    # Caclulate  best rating and ovr.
    vector = np.zeros(len(ordered_attr))
    for idx, att in enumerate(ordered_attr):
        vector[idx] = card[att]
    
    pos_rating = (vector.reshape(34,1)*rating_weights).sum(axis=0)
    best_pos_idx = np.argmax(pos_rating)
    positions = ['ST','RW','CF','CAM','RM','CM', 'CDM', 'LWB','RB','CB','GK']
    card['Best.Overall.Rating'] = int(max(pos_rating))
    card['Overall'] = int(max(pos_rating))
    
    card['Preferred.Position'] = positions[best_pos_idx]

    
    
with tab3:
    st.header("Player Generator")
    output_df = pd.DataFrame(card, index=[0])
    st.dataframe(output_df,hide_index=True)

with tab4:
    st.header("Plots")
    if len(scored_df['Sprint 5m']) > 0:
        with _lock:
            # Sprint time plot
            fig, ax = plt.subplots()
            ax.scatter([0 for i in range(len(scored_df['Sprint 5m']))], scored_df['Sprint 5m'],s=100)
            ax.scatter([1 for i in range(len(scored_df['Sprint 10m']))], scored_df['Sprint 10m'],s=100)
            ax.scatter([2 for i in range(len(scored_df['Sprint 20m']))], scored_df['Sprint 20m'],s=100)
            ax.plot([0,1,2],means[['Sprint 5m', 'Sprint 10m', 'Sprint 20m']], color="#8b8b8b", marker="o", markersize=11,alpha=0.6)
            ax.set_xticks([0, 1, 2], ["5m", "10m", "20m"])
            ax.set_ylabel('Time (Seconds)')
            ax.set_xlabel('Distance')
            ax.set_ylim(0,6)
            ax.set_title('Sprint Times')
            st.pyplot(fig)

            # Passing boxplot
            fig_2, ax_2 = plt.subplots()
            ax_2.boxplot([scored_df['Pass 5m'].values, scored_df['Pass 10m'].values],widths=0.5,whiskerprops = dict(color='#30a2da', linestyle='-', linewidth=5), boxprops=dict(color='#30a2da',linewidth=5), medianprops=dict(color='#fc4f30',linewidth=5), capprops=dict(color='#30a2da',linewidth=3))
            ax_2.set_xticks([1, 2], ["5m", "10m"])
            ax_2.set_ylabel("Completed Passes out of 10")
            ax_2.set_xlabel("Distance")
            ax_2.set_title("Passing Accuracy")
            st.pyplot(fig_2)

            # Dribbling Plot
            fig_3, ax_3 = plt.subplots()
        
            ax_3.scatter([1 for i in range(len(scored_df['Dribble 5m']))], scored_df['Dribble 5m'], s=100)
            ax_3.scatter([2 for i in range(len(scored_df['Dribble 10m']))], scored_df['Dribble 10m'], s=100)
            ax_3.plot([1,2],means[['Dribble 5m', 'Dribble 10m']], color="#8b8b8b", marker="o", markersize=10,alpha=0.6, label="Mean Time")
            ax_3.set_xticks([1, 2], ["5m", "10m"])
            ax_3.set_xlim(0, 3)
            ax_3.set_xlabel("Distance")
            ax_3.set_ylabel("Time (seconds)")
            ax_3.set_title("Dribbling Times")
            st.pyplot(fig_3)

            # Jumping plot
            fig_4, ax_4 = plt.subplots()
            img = plt.imread("./assets/Jump.png")
            ax_4.imshow(img,extent=[1, 2.2, 0, 0.7])
            ax_4.scatter(scored_df['Power Standing Jump'], [0.25 for i in range(len(scored_df['Power Standing Jump']))],s=100,color="#fc4f30")
            # Count the number of observations for each jumping height
            jumping_height_counts = {}
            for height in scored_df['Power Standing Jump']:
                 if height in jumping_height_counts:
                     jumping_height_counts[height] += 1
                 else:
                     jumping_height_counts[height] = 1

            # Plot the scatter points and add text labels
            for height, count in jumping_height_counts.items():
                if (count > 1):
                    ax_4.text(height, 0.2, f"{count} \n people", ha='center', va='center', fontsize=9)
                else:
                    ax_4.text(height, 0.2, f"{count} \n person", ha='center', va='center', fontsize=9)
            ax_4.grid(False)
            ax_4.set_yticks([])
            ax_4.set_xlabel("Jumping Length (m)")
            st.pyplot(fig_4)

            # Blazepods distribution
            fig_5, ax_5 = plt.subplots()

            mu = means['Reactions Blazepods']
            variance = np.var(scored_df['Reactions Blazepods'])
            sigma = np.sqrt(variance)
            x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)

            ax_5.plot(x, sp.norm.pdf(x, mu, sigma))
            ax_5.scatter(scored_df['Reactions Blazepods'], [0 for i in range(len(scored_df['Reactions Blazepods']))], color="#fc4f30", s=150, label='Your Scores')
            ax_5.set_yticks([0.025, 0.05, 0.075], ["2.5%", "5%", "7.5%"])
            ax_5.set_xlabel("Blazepod Score")
            ax_5.set_ylabel("% of people")
            ax_5.set_xlim(min(x), max(x))
            ax_5.legend()
            ax_5.set_title('Blazepods Awareness')
            st.pyplot(fig_5)
