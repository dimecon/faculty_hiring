import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import scipy as sp
import pandas as pd

from faculty_hiring.parse import institution_parser
from faculty_hiring.misc.plotting import *  # Definitions for LABEL_SIZE, colors, etc.
from sklearn.linear_model import LinearRegression

inst_file = '/Users/allisonmorgan/Documents/faculty_hiring/publication_data/current_data/inst_cs_CURRENT.txt'

inst = institution_parser.parse_institution_records(open(inst_file))

# Find all current faculty at an instituion
def faculty_at_institution(institution_name, asst_faculty):
    current_faculty = []
    for person in asst_faculty:
        if person.current_job()[0] == institution_name:
            current_faculty.append(person)
    return current_faculty

# Plot data with line of best fit
def plot_pubs_versus_prestige(data, ylabel, function=np.average, percentiles=False):
    pubs_by_prestige = {}
    for (name, n_pubs) in data.items():
        counts = []
        for _, pubs in n_pubs:
            # pubs could be either: total contributions (int), or fractional contributions (array)
            if type(pubs) == int or type(pubs) == float:
                counts.append(pubs)
            elif type(pubs) == list:
                counts.extend(pubs)
            else:
                print "Unexpected data type: {0}".format(type(pubs))

        pubs_by_prestige[inst[name]['pi']] = (function(counts), inst[name]['private'])

    fig, ax = plt.subplots(1,1, figsize=(6,4))

    sorted_keys = sorted(pubs_by_prestige.keys())
    sorted_vals = [pubs_by_prestige[each][0] for each in sorted_keys]

    (_, _, p25, p75) = upper_lower_percentiles(data, inst)
    if percentiles:
        sorted_lower = [p25[each] for each in sorted_keys]
        sorted_upper = [p75[each] for each in sorted_keys]

        ax.fill_between(sorted_keys, sorted_lower, sorted_upper, color=LIGHT_COLOR, edgecolor='None')

    # Pick out the private universities
    private_keys = sorted([key for key, data in pubs_by_prestige.items() if data[1] == 1])
    private_vals = [pubs_by_prestige[each][0] for each in private_keys]
    ax.scatter(private_keys, private_vals, color=ALMOST_BLACK, alpha=1.0)

    regr_private = LinearRegression()
    regr_private.fit(np.array(private_keys).reshape(-1, 1), np.array(private_vals).reshape(-1, 1))
    x = np.array([min(private_keys), max(private_keys)])

    r2_private = regr_private.score(np.array(private_keys).reshape(-1, 1), np.array(private_vals).reshape(-1, 1))
    print "Line of best fit for private schools has a slope of %.4f and a r^2 of %.4f" % (regr_private.coef_[0], r2_private)

    private_fit, = ax.plot(x, x*regr_private.coef_[0] + regr_private.intercept_, '-', color=ALMOST_BLACK, linewidth=LINE_WIDTH)

    # Pick out the public universities
    public_keys = sorted([key for key, data in pubs_by_prestige.items() if data[1] == 0])
    public_vals = [pubs_by_prestige[each][0] for each in public_keys]
    ax.scatter(public_keys, public_vals, color=ACCENT_COLOR_1, alpha=1.0)

    regr_public = LinearRegression()
    regr_public.fit(np.array(public_keys).reshape(-1, 1), np.array(public_vals).reshape(-1, 1))
    x = np.array([min(public_keys), max(public_keys)])

    r2_public = regr_public.score(np.array(public_keys).reshape(-1, 1), np.array(public_vals).reshape(-1, 1))
    print "Line of best fit for public schools has a slope of %.4f and a r^2 of %.4f" % (regr_public.coef_[0], r2_public)

    public_fit, = ax.plot(x, x*regr_public.coef_[0] + regr_public.intercept_, '--', color=ACCENT_COLOR_2, linewidth=LINE_WIDTH, dashes=(12, 6))

    private_fit = r"Slope: %.3f" % (regr_private.coef_[0])
    public_fit = r"Slope: %.3f" % (regr_public.coef_[0])

    fake_line_all = Line2D(range(1), range(1), color=LIGHT_COLOR, marker='o', linestyle='None', markeredgecolor='w')
    fake_line_private_median = Line2D(range(1), range(1), color=ALMOST_BLACK, marker='o', linestyle='None', markeredgecolor='w', alpha = 1.0)
    fake_line_public_median = Line2D(range(1), range(1), color=ACCENT_COLOR_1, marker='o', linestyle='None', markeredgecolor='w', alpha = 1.0)
    fake_line_private_fit = Line2D(range(1), range(1), color=ALMOST_BLACK, linestyle='-', linewidth=LINE_WIDTH)
    fake_line_public_fit = Line2D(range(1), range(1), color=ACCENT_COLOR_2, linestyle='--', linewidth=LINE_WIDTH, dashes=(12, 6))

    plt.legend(
        (
            fake_line_all, 
            fake_line_private_median, 
            fake_line_public_median,
            fake_line_private_fit,
            fake_line_public_fit
        ),(
            '25-75th percentile', 
            "Private", 
            "Public",
            private_fit,
            public_fit
        ), numpoints=1, loc='upper right', frameon=False, fontsize=LEGEND_SIZE, ncol=1)

    ax.set_ylim(0, 200)
    ax.set_xlim(min(sorted_keys), max(sorted_keys))
    ax.set_xlabel('University rank', fontsize=LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)

    finalize(ax)

    fig.savefig("pubs_v_rank.pdf", bbox_inches='tight')
    plt.show()

# Find 25th and 75th percentiles
def upper_lower_percentiles(data, inst):
    means = {}; stds = {}; p25 = {}; p75 = {}
    for i, (name, n_pubs) in enumerate(data.items()):
        rank = inst[name]['pi']
        final_counts = []
        for _, pubs in n_pubs:
            # pubs could be either: total contributions (int), or fractional contributions (array)
            if type(pubs) == int or type(pubs) == float:
                final_counts.append(pubs)
            elif type(pubs) == list:
                final_counts.extend(pubs)
            else:
                print "Unexpected data type: {0}".format(type(final_counts))

        means[rank] = np.average(final_counts)
        stds[rank] = np.std(final_counts)
         
        p25[rank] = np.percentile(final_counts, 25)
        p75[rank] = np.percentile(final_counts, 75)

    return (means, stds, p25, p75)

