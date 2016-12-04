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
def plot_pubs_versus_prestige(data, ylabel, function=np.average):
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
    plt.plot(sorted_keys, sorted_vals, color=ACCENT_COLOR_1)

    regr = LinearRegression()
    regr.fit(np.array(sorted_keys).reshape(-1, 1), np.array(sorted_vals).reshape(-1, 1))
    x = np.array([min(sorted_keys), max(sorted_keys)])

    r2 = regr.score(np.array(sorted_keys).reshape(-1, 1), np.array(sorted_vals).reshape(-1, 1))
    print "Line of best fit has a slope of %.4f and a r^2 of %.4f" % (regr.coef_[0], r2)

    ax.plot(x, x*regr.coef_[0] + regr.intercept_, ':', color=ALMOST_BLACK)

    ax.set_xlabel('University Prestige (pi)')
    ax.set_ylabel(ylabel)

    finalize(ax)

    plt.show()

# Bin universities by prestige. Plot as a bar or line chart
def plot_binned_pubs_versus_prestige(data, ylabel, bins, chart_type='bar', function=np.average):
    fig, ax = plt.subplots(figsize=SINGLE_FIG_SIZE)

    prestige = sorted([inst[name]['pi'] for name in data.keys()])
    pubs = {}
    for (name, n_pubs) in data.items():
        counts = []
        for _, pub in n_pubs:
            if type(pub) == int or type(pub) == float:
                counts.append(pub)
            elif type(pub) == list:
                counts.extend(pub)
            else:
                print "Unexpected data type: {0}".format(type(pubs))
        
        pubs[inst[name]['pi']] = counts
    
    left_endpoint = bins[0]
    bins = np.percentile(prestige, bins[1:])
    bin_means = np.zeros(len(bins))
    for i, bin_edge in enumerate(bins):
        bin_values = []
        for key in prestige:
            if left_endpoint < key <= bin_edge:
                bin_values.extend(pubs[key])
        
        bin_means[i] = function(bin_values)
        left_endpoint = bin_edge

    if chart_type == 'bar':
        plt.bar(bins, bin_means, width = 10, color=ACCENT_COLOR_1, edgecolor='w')
        
        small_diff = (bin_means[0] - bin_means[1])
        medium_diff = (bin_means[0] - bin_means[2])
        large_diff = (bin_means[0] - bin_means[len(bin_means)-1])
        print "Faculty at the top 10%% of schools have %.2f, %.2f, and %.2f more publications than faculty at the top 20%%, 50%% and 100%% of schools." % (small_diff, medium_diff, large_diff) 
    elif chart_type == 'line':
        plt.plot(bins, bin_means, color=ACCENT_COLOR_1)

        regr = LinearRegression()
        regr.fit(np.array(bins).reshape(-1, 1), np.array(bin_means).reshape(-1, 1))
        x = np.array([min(prestige), max(prestige)])

        r2 = regr.score(np.array(bins).reshape(-1, 1), np.array(bin_means).reshape(-1, 1))
        print "Line of best fit has a slope of %.4f and a r^2 of %.4f" % (regr.coef_[0], r2)

        ax.plot(x, x*regr.coef_[0] + regr.intercept_, ':', color=ALMOST_BLACK)
        
    ax.set_xlabel('University Prestige (pi)')
    ax.set_ylabel(ylabel)

    finalize(ax)
    
    plt.show()

def plot_pubs_versus_status(data, ylabel):
    fig, ax = plt.subplots(figsize=SINGLE_FIG_SIZE)

    sorted_keys = sorted(data.keys())
    sorted_vals = [data[each] for each in sorted_keys]

    plt.bar(sorted_keys, sorted_vals, color=ACCENT_COLOR_1, edgecolor='w')

    ax.set_xlabel('Public or Private Status')
    ax.set_ylabel(ylabel)

    ax.set_xticks(sorted_keys)
    ax.set_xticklabels(["public", "private"])

    finalize(ax)

    plt.show()