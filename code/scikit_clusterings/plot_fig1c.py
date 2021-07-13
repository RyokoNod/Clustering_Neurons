import pylab as plt
import seaborn as sns; sns.set()
import matplotlib
import numpy as np

# set seaborn styles from original article
def sns_styleset():
    sns.set(context='paper', style='ticks', font='Arial')
    matplotlib.rcParams['axes.linewidth']    = .5
    matplotlib.rcParams['xtick.major.width'] = .5
    matplotlib.rcParams['ytick.major.width'] = .5
    matplotlib.rcParams['xtick.minor.width'] = .5
    matplotlib.rcParams['ytick.minor.width'] = .5
    matplotlib.rcParams['xtick.major.size'] = 2
    matplotlib.rcParams['ytick.major.size'] = 2
    matplotlib.rcParams['xtick.minor.size'] = 1
    matplotlib.rcParams['ytick.minor.size'] = 1
    matplotlib.rcParams['font.size']       = 6
    matplotlib.rcParams['axes.titlesize']  = 6
    matplotlib.rcParams['axes.labelsize']  = 6
    matplotlib.rcParams['legend.fontsize'] = 6
    matplotlib.rcParams['xtick.labelsize'] = 6
    matplotlib.rcParams['ytick.labelsize'] = 6
    matplotlib.rcParams['figure.dpi'] = 180

sns_styleset()

def plot_fig1c(Z, m1data,title="Fig1c from Scala et al."):
    """
    This function is a copy of the code from ttype-coverage-mod.ipynb,
    that plots figure 1c in Scala et al.'s article.
    The only difference is that m1data in this code needs to already be the 
    subgroup with key "viplamp"
    """
    
    clusterColors = m1data['clusterColors']
    clusterNames = m1data['clusterNames']

    plt.scatter(Z[:,0], Z[:,1], s=1, alpha=1, rasterized=True, edgecolors='none',
                c = clusterColors[m1data['clusters']])

    offsets = {'Lamp5 Slc35d3': [25,-20], 'Lamp5 Lhx6': [10,6], 'Lamp5 Pdlim5_2': [20,10],
               'Lamp5 Pdlim5_1': [20,5], 'Lamp5 Pax6': [10,8], 'Lamp5 Egln3_1': [20,8],
               'Lamp5 Egln3_2': [15,7], 'Lamp5 Egln3_3': [25,16], 'Vip Igfbp6_1': [15,-10],
               'Vip Igfbp6_2': [15,-5], 'Vip C1ql1': [5,-20], 'Vip Gpc3': [0,-12],
               'Vip Chat_1': [0,-17], 'Vip Mybpc1_2': [-10,15], 'Vip Mybpc1_1': [-8,5],
               'Sncg Slc17a8': [0,9], 'Sncg Calb1_1': [-8,7], 'Sncg Calb1_2': [-12,3],
               'Sncg Npy2r': [-15,0], 'Vip Sncg': [-12,5], 'Sncg Col14a1': [7,0],
               'Vip Htr1f': [15,-9], 'Vip Serpinf1_3': [2,-10], 'Vip Serpinf1_1': [-4,3],
               'Vip Mybpc1_3': [-10,0], 'Vip Serpinf1_2': [10,0], 'Vip Chat_2': [0,-10]}

    angles = {'Vip Mybpc1_1': 60, 'Sncg Col14a1': -75}
    renames = {'Vip Mybpc1_3': 'M3', 'Vip Serpinf1_1': 'S1', 'Vip Serpinf1_2': 'S2', 'Vip Serpinf1_3': 'S3'}

    for c in np.unique(m1data['clusters']): # for every cluster assigned to Yao et al.'s CGE interneurons
        ind = m1data['clusters']==c
        col = clusterColors[c]
        x,y = np.median(Z[ind,0]), np.median(Z[ind,1]) #set the labels at the median value of the t-SNE values for that cluster
        if clusterNames[c] in offsets: # use the offset
            x += offsets[clusterNames[c]][0]
            y += offsets[clusterNames[c]][1]
        if clusterNames[c] not in renames:
            label = '\n'.join(clusterNames[c].split()[1:]) #for brevity, keep the cell family name out and replace space with newline
        else:
            label = renames[clusterNames[c]] #rename if you need to
        if clusterNames[c] not in angles:
            alpha = 0
        else:
            alpha = angles[clusterNames[c]] #set angles to cell types if specified

        if ~np.isnan(x):
            plt.text(x, y, label, color=col, fontsize=5, ha='center', va='center', zorder=1, rotation=alpha)

    #the next 6 rows are for plotting the cell family names: Vip, Sncg, Lamp5
    col = clusterColors[clusterNames=='Vip Mybpc1_3'][0]
    plt.text(-65, -50, 'Vip', color='w', fontsize=6, bbox=dict(boxstyle='round', ec=col, fc=col))
    col = clusterColors[clusterNames=='Sncg Calb1_2'][0]
    plt.text(-65, 55, 'Sncg', color='w', fontsize=6, bbox=dict(boxstyle='round', ec=col, fc=col))
    col = clusterColors[clusterNames=='Lamp5 Slc35d3'][0]
    plt.text(50, -60, 'Lamp5', color='w', fontsize=6, bbox=dict(boxstyle='round', ec=col, fc=col))

    #these remove ticks from the plot and erases borderline of plot
    plt.xticks([])
    plt.yticks([])
    sns.despine(left=True, bottom=True)
    plt.title(title)
    

def plot_sidebyside(clusters, Z, m1data, title, reftitle="", figsize=(5,2)):
    """
    This function plots figure 1c  from Scala et al. with new cluster assignments
    and the original version side by side
    
    Arguments:
    - clusters: new cluster assignments
    - Z: the t-SNE representations from Scala et al.
    - m1data: Yao et al.'s dataset for the subroup "viplamp"
    - title: title for the new cluster assignments
    - reftitle: if you want anything other than "Fig1c from Scala et al." for the
                    original, set it here
    - figsize: a tuple of figure size to be given to pylab.figsize
    """
    
    plt.figsize = plt.figure(figsize=figsize)
    
    # plot left without ticks or borderline
    plt.subplot(1, 2, 1)
    plt.xticks([])
    plt.yticks([])
    plt.scatter(Z[:, 0], Z[:, 1], c=clusters,s=0.5)
    sns.despine(left=True, bottom=True)
    plt.title(title)
    
    plt.subplot(1, 2, 2)
    if reftitle =="":
        plot_fig1c(Z, m1data)
    else:
        plot_fig1c(Z, m1data, reftitle)
    plt.show()