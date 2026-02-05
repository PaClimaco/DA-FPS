import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
sns.set_theme()


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
sns.set_theme()


def format_fn(value, tick_number):
    return f"{value/1000:.0f}K"
def generate_lineplots(dataset, df, given_strategies = None, filepath = None):

        df = df[df['strategy'].isin(given_strategies)]
        custom_x_values = ['5%', '10%', '15%', '20%']
        x_values = list(set(list(df["num_selected"])))
        x_values.sort()
        # Change unit of measure of label value, from kcal/mol to eV for plot consistency with other datasets
        if dataset == 'QM7':
                df[['RMSE', 'MAE']] = df[['RMSE', 'MAE']]/23.060900 #from kcal/mol to ev
        if dataset == 'QM8':
                df[['RMSE', 'MAE']] = df[['RMSE', 'MAE']]*27.21139  #from atomic units (a.u.)to ev
        colors = {
        'FPS': "royalblue",
        'RDM': "limegreen",
        'FacilityLocation':"orange",
        'k-medoids++': "violet",
        'FPS-RDM': "limegreen",
        'FPS-FacLoc': "orange",  
        'FPS-k-medoids++': "violet", 
        'DA-FPS': 'red'
        }
        
        dashes_dict = {
        'FPS': '',
        'RDM': '',
        'FacilityLocation': '',
        'k-medoids++': '',
        'FPS-RDM': (5, 2),
        'FPS-FacLoc': (5, 2),  
        'FPS-k-medoids++': (5, 2),  
        'DA-FPS_torch': '',
        'DA-FPS': ''
        }    
        for metric in ['MAE', 'RMSE']:
        
                plt.figure(figsize=(10,7))
                sns.lineplot(data=df, 
                        x="num_selected",
                        y=metric,
                        hue="strategy",
                        style="strategy",
                        dashes= dashes_dict,
                        errorbar = "se",
                        err_style = 'bars',
                        markers= False, 
                        hue_order= given_strategies,
                        legend=True,
                        palette=colors,
                        )
                plt.tick_params(labelsize=18)
                plt.legend(loc='upper right', prop={'size': 13}, bbox_to_anchor=(1.008, 1))
                plt.ylabel(f"{metric}", fontsize=20)
                plt.xlabel("Number of training samples", fontsize=20)
                plt.xticks(x_values, custom_x_values)
                sns.set_style("darkgrid")

                plt.show()
                
        return 
        

