import numpy as np

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
def plot_pubs_versus_prestige(data, ylabel, function=np.average, percentiles=False, fit=False):
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

        pubs_by_prestige[inst[name]['pi']] = function(counts)

    fig, ax = plt.subplots(figsize=SINGLE_FIG_SIZE)

    sorted_keys = sorted(pubs_by_prestige.keys())
    sorted_vals = [pubs_by_prestige[each] for each in sorted_keys]

    if fit:
        regr = LinearRegression()
        regr.fit(np.array(sorted_keys).reshape(-1, 1), np.array(sorted_vals).reshape(-1, 1))
        x = np.array([min(sorted_keys), max(sorted_keys)])

        r2 = regr.score(np.array(sorted_keys).reshape(-1, 1), np.array(sorted_vals).reshape(-1, 1))
        print "Line of best fit has a slope of %.4f and a r^2 of %.4f" % (regr.coef_[0], r2)

        ax.plot(x, x*regr.coef_[0] + regr.intercept_, ':', color=ALMOST_BLACK)
    if percentiles:
        (_, _, p25, p75) = upper_lower_percentiles(data, inst)
        if percentiles:
            sorted_lower = [p25[each] for each in sorted_keys]
            sorted_upper = [p75[each] for each in sorted_keys]

            ax.fill_between(sorted_keys, sorted_lower, sorted_upper, color=LIGHT_COLOR, edgecolor='None')
        #elif std:
        #    sorted_lower = [p25[each] for each in sorted_keys]
        #    sorted_upper = [p75[each] for each in sorted_keys]
        #
        #    ax.fill_between(sorted_keys, sorted_upper, sorted_lower, color=LIGHT_COLOR, edgecolor='None')

    ax.scatter(sorted_keys, sorted_vals, color=ACCENT_COLOR_1)

    fake_line_all = Line2D(range(1), range(1), color=LIGHT_COLOR, marker='o', linestyle='None', markeredgecolor='w')
    fake_line_t = Line2D(range(1), range(1), color=ACCENT_COLOR_1, marker='o', linestyle='None', markeredgecolor='w')
    fake_line_fit = Line2D(range(1), range(1), color=ALMOST_BLACK, linestyle=':', linewidth=2)

    label = "" 
    if function == np.average:
        label = 'Average'
    elif function == np.median:
        label = 'Median'

    fit = "Slope: %.4f, R^2: %.4f" % (regr.coef_[0], r2)

    plt.legend((fake_line_all,fake_line_t, fake_line_fit),('25-75th Percentile', label, fit), 
           numpoints=1, loc='upper right', frameon=False, fontsize=LABEL_SIZE-2, ncol=1)

    ax.set_ylim(ymin=0)
    ax.set_xlim(min(sorted_keys), max(sorted_keys))
    ax.set_xlabel('University Prestige (pi)', fontsize=LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)

    finalize(ax)

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

