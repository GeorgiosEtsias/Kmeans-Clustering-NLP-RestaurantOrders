import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import os

def matplotlib_pie_chart(clusternames, product_distribution, plot_title, export_graph):
    '''
    Creates a pie chart to show the distribution of the various products
    using matplotlib
    '''
    fig, ax = plt.subplots()
    ax.pie( product_distribution, labels = clusternames, shadow=True, startangle=90)
    ax.axis('equal')  
    ax.set_title(plot_title)
    plt.show()

    if export_graph:
        # export png
        export_png(fig, plot_title)  

    return fig

def plotly_pie_chart(clusternames, product_distribution, plot_title, export_graph):
    '''
    Creates a pie chart to show the distribution of the various products using Plotly
    '''
    fig = px.pie(values=product_distribution, names=clusternames, title=plot_title)
    fig.show()
    
    if export_graph:
        # export svg
        export_svg(fig, plot_title) 
    
    return fig

def export_png(fig, plot_title):
    '''
    Save the matplotlib figure as a png
    '''
    directory = 'figures/png_figures/'
    # check if the directory exists, if not create it
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    figure_name = str(directory) + plot_title + '.png'   
    fig.savefig(figure_name, dpi=300)

def export_html(fig, plot_title):
    '''
    Save the plotly figure  as an interactive html
    '''
    directory = 'figures/html_figures/'
    # check if the directory exists, if not create it
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    figure_name = str(directory) + plot_title + '.html'   
    pio.write_html(fig, figure_name)

def export_svg(fig, plot_title):
    '''
    Save the plotly figure  as an svg
    '''
    directory = 'figures/svg_figures/'
    # check if the directory exists, if not create it
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    figure_name = str(directory) + plot_title + '.svg'   
    pio.write_image(fig, figure_name)

