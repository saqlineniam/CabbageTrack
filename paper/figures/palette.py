"""
House colour palette for every exported figure (paper, supplement, slides). Import and call use():

    from palette import C, use
    use()                       # sets the default colour cycle, text and axis colours
    ax.plot(x, y, color=C['ocean_blue'])
"""
import matplotlib as mpl
from cycler import cycler

C = {
    'dark_red': '#8F020A',
    'crimson': '#C01331',
    'orange': '#F27731',
    'leaf_green': '#56A854',
    'deep_teal': '#1C694B',
    'aqua_marine': '#4EA9A2',
    'steel_blue': '#77A7BF',
    'slate_blue': '#44728C',
    'ocean_blue': '#1A465E',
    'midnight_navy': '#02203A',
    'charcoal_gray': '#5D6161',
}
# order for categorical series: high-contrast neighbours first
CYCLE = [C[k] for k in ('ocean_blue', 'orange', 'leaf_green', 'crimson', 'aqua_marine', 'slate_blue',
                        'dark_red', 'deep_teal', 'steel_blue', 'midnight_navy')]


def use():
    mpl.rcParams.update({
        'axes.prop_cycle': cycler(color=CYCLE),
        'text.color': C['midnight_navy'], 'axes.labelcolor': C['midnight_navy'],
        'axes.edgecolor': C['charcoal_gray'], 'xtick.color': C['charcoal_gray'], 'ytick.color': C['charcoal_gray'],
        'grid.color': C['steel_blue'], 'grid.alpha': 0.3,
        'font.family': 'sans-serif', 'font.size': 9, 'axes.linewidth': 0.8,
        'savefig.dpi': 300,
    })
