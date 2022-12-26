import matplotlib.pyplot as plt
import plotly.express as px

def plot_cluster_pie_chart(clusternames, product_distribution, plot_title):
    '''
    Creates a pie chart to show the distribution of the various products
    using matplotlib
    '''
    fig, ax = plt.subplots()
    ax.pie( product_distribution, labels = clusternames, shadow=True, startangle=90)
    ax.axis('equal')  
    ax.set_title(plot_title)
    plt.show()
    return fig

def plot_cluster_pie_chart2(clusternames, product_distribution, plot_title):
    '''
    Creates a pie chart to show the distribution of the various products using Plotly
    '''
    # fig = px.pie(values=product_distribution, labels=clusternames, title=plot_title)
    fig = px.pie(values=product_distribution, names=clusternames, title=plot_title)
    fig.show()
    return fig