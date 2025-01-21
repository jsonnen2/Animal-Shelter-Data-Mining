
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os

# TODO: perform Wilcoxon difference in means test


if __name__ == '__main__':
# def categorized_line_plot():
    '''
    used for pairwise feature subset analysis
    '''
    eval_set = 'Validation'
    data = pd.read_csv("all_classifiers/Shelter_Outcome_bs=1.7_save.csv", index_col=0) # 15,000 train & val
    # data = data.iloc[[0,1,2,3,4,5,7]]
    
    val_data = data.map(lambda x: float(x.strip("()").split(",")[0]))
    test_data = data.map(lambda x: float(x.strip("()").split(",")[1]))
    
    if eval_set == 'Validation':
        used_data = val_data
    elif eval_set == 'Test':
        used_data = test_data
    else:
        raise NameError("Pick different eval_set")

    # Plot
    fig, ax = plt.subplots()
    default_colors = plt.cm.tab10.colors[:7] + plt.cm.tab10.colors[8:] # Remove grey (color 7)
    medians = {}
    # Plot t_scores
    for i, (category, acc) in enumerate(used_data.iterrows()):
        x_values = acc
        y_values = [str(category)] * len(x_values)
        medians[category] = acc.median()
        color = default_colors[i%9]
        ax.scatter(x_values, y_values, color=color)

        
    # Add vertical lines for medians-- this paragraph is written by ChatGPT
    categories = [str(cat) for cat in medians.keys()]
    positions = {cat: i for i, cat in enumerate(categories)}
    for category, median_val in medians.items():
        y_pos = positions[str(category)]
        ax.plot(
            [median_val, median_val],  # x-coordinates for vertical line
            [y_pos - 0.4, y_pos + 0.4],  # Slight offset to span within the category
            color='grey',
            linestyle='--',
            linewidth=2
        )

    # Set axis labels
    ax.set_xlabel(f'{eval_set} Accuracy')

    plt.tight_layout()

    # Custom Legend
    # legend_handles = [
    #     mlines.Line2D([], [], linestyle='--', color='grey', linewidth=2, label='Median Acc.'),
    #     mlines.Line2D([], [], linestyle=':', color='red', linewidth=2, label='Majority Class Acc.'),
    # ]
    # ax.legend(handles=legend_handles, loc='lower left')
    
    plt.savefig(f"./all_classifiers/plot_bs=1.7.png")
    plt.show()