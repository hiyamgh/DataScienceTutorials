import warnings
import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.stats._continuous_distns import _distn_names as distn_names
from sklearn import datasets
import matplotlib.pyplot as plt
import os

plt.rcParams['figure.figsize'] = (16.0, 12.0)
plt.style.use('ggplot')


def mkdir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


# Create models from data
def best_fit_distribution(data, bins=200, ax=None, sortby='bic', output_folder=''):
    """Model data by finding best fit distribution to data"""

    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    results = pd.DataFrame()
    sses, AIC, BIC = [], [], []
    parameters = []
    dist_param = {}

    for d in ['levy', 'levy_l', 'levy_stable', 'loguniform', 'reciprocal', 'truncnorm', 'vonmises', 'wrapcauchy']: distn_names.remove(d)
    distributions = distn_names

    # Estimate distribution parameters from data
    for dist_name in distributions:

        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                distribution = getattr(st, dist_name)

                # fit dist to data
                print('fitting {}'.format(dist_name))
                params = distribution.fit(data)
                parameters.append(params)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # calculate the log-likelihood
                LLH = distribution.logpdf(data, *params).sum()
                k = len(params)
                n = len(data)
                aic = 2 * k - 2 * LLH
                bic = k * np.log(n) - 2 * LLH

                # Calculate fitted PDF and error with fit in distribution
                print('calculating stats')
                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))

                sses.append(sse)
                AIC.append(aic)
                BIC.append(bic)

                dist_param[dist_name] = params

                print('\nDistribution: {}:'.format(dist_name))
                print('SSE: {}\nAIC: {}\nBIC: {}\n'.format(sse, aic, bic))
                print('--------------------------------------------------')

                # if axis pass in add to plot
                try:
                    if ax:
                        pd.Series(pdf, x).plot(ax=ax, label=dist_name)
                except Exception:
                    pass

        except Exception:
            pass

    mkdir(folder=output_folder)

    print('\nsaving results in a data frame ...')
    results['distribution'] = distributions
    results['sse'] = sses
    results['aic'] = AIC
    results['bic'] = BIC
    results['params'] = parameters

    # sort results by p-value, sse, and chi-squared
    print('sorting results by: {}'.format(sortby))
    results = results.sort_values(by=sortby, ascending=True).reset_index(drop=True)

    best_distribution = results.iloc[0]['distribution']

    print('\nbest distribution: {}'.format(best_distribution))

    mkdir(folder=output_folder)
    results.to_csv(os.path.join(output_folder, 'results_{}.csv'.format(sortby)), index=False)

    print('saved results_{}.csv in {}'.format(sortby, output_folder))

    return results, dist_param


def make_pdf(dist, params, size=10000):
    """Generate distributions's Probability Distribution Function """

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get sane start and end points of distribution
    start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
    end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)

    return pdf


def plot_best(best_dist, best_dist_params, best_dist_name, target_variable, output_folder=''):
    # Make PDF with best params
    pdf = make_pdf(best_dist, best_dist_params)

    # Display
    plt.figure(figsize=(12, 8))
    ax = pdf.plot(lw=2, label=best_dist_name, legend=True)
    data.plot(kind='hist', bins=50, density=True, alpha=0.5, label='Data', legend=True, ax=ax)

    param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
    param_str = ', '.join(['{}={:0.2f}'.format(k, v) for k, v in zip(param_names, best_dist_params)])
    dist_str = '{}({})'.format(best_dist_name, param_str)

    # create the output folder to store result in
    mkdir(folder=output_folder)

    # set title, x-axis label, y-axis label
    ax.set_title(u'best fit distribution \n' + dist_str)
    ax.set_xlabel('{}'.format(target_variable))
    ax.set_ylabel('Frequency')

    # sav ethe figure in the output folder
    plt.savefig(os.path.join(output_folder, 'best_fit.png'))
    plt.close()


def get_top_dists(results, n=10):
    ''' gets the top n best fitting distributions '''
    best_fits_sorted = results['distribution']
    return best_fits_sorted[:n]


def generate_pp_qq_plots(data, best_fits, dist_param, sizervs, output_folder):
    fig, axs = plt.subplots(len(best_fits), 2)

    i = 0
    for distribution in best_fits:
        dist = getattr(st, distribution)
        param = dist_param[distribution]
        # param = dist.fit(data)

        # Get random numbers from distribution
        norm = dist.rvs(*param[0:-2], loc=param[-2], scale=param[-1], size=sizervs)
        norm.sort()

        axs[i, 0].plot(list(norm), data, "o", color='b')
        min_value = np.floor(min(min(norm), min(data)))
        max_value = np.ceil(max(max(norm), max(data)))
        axs[i, 0].plot([min_value, max_value], [min_value, max_value], 'r--')
        axs[i, 0].set_xlim(min_value, max_value)
        axs[i, 0].set_xlabel('Theoretical quantiles')
        axs[i, 0].set_ylabel('Observed quantiles')
        title = 'qq plot for ' + distribution + ' distribution'
        axs[i, 0].set_title(title)

        # pp plot

        # Calculate cumulative distributions
        bins = np.percentile(norm, range(0, 101))
        data_counts, bins = np.histogram(data, bins)
        norm_counts, bins = np.histogram(norm, bins)
        cum_data = np.cumsum(data_counts)
        cum_norm = np.cumsum(norm_counts)
        cum_data = cum_data / max(cum_data)
        cum_norm = cum_norm / max(cum_norm)

        # plot
        axs[i, 1].plot(cum_norm, cum_data, "o", color='b')
        min_value = np.floor(min(min(cum_norm), min(cum_data)))
        max_value = np.ceil(max(max(cum_norm), max(cum_data)))
        axs[i, 1].plot([min_value, max_value], [min_value, max_value], 'r--')
        axs[i, 1].set_xlim(min_value, max_value)
        axs[i, 1].set_xlabel('Theoretical cumulative distribution')
        axs[i, 1].set_xlabel('Theoretical cumulative distribution')
        axs[i, 1].set_ylabel('Observed cumulative distribution')
        title = 'pp plot for ' + distribution + ' distribution'
        axs[i, 1].set_title(title)

        i += 1

    mkdir(folder=output_folder)

    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.tight_layout(pad=4)
    plt.savefig(os.path.join(output_folder, 'qq_pp.png'))
    plt.close()


if __name__ == '__main__':

    data_set = datasets.load_breast_cancer()
    y = data_set.data[:, 0]
    data = pd.Series(y)
    target_var = 'mean_radius'

    # Plot for comparison
    plt.figure(figsize=(12, 8))
    ax = data.plot(kind='hist', bins=50, density=True, color=list(plt.rcParams['axes.prop_cycle'])[1]['color'], alpha=0.5)
    dataYLim = ax.get_ylim()

    sortings = ['SSE', 'AIC', 'BIC']

    count = 0
    output_dest = 'output'
    plots_dest = 'plots'

    # get the best results from sorting by chi square (by default)
    results_df, dist_param = best_fit_distribution(data, 200, ax, sortby='bic', output_folder=output_dest)

    # Update plots
    ax.set_ylim(dataYLim)
    ax.set_title(u'All Fitted Distributions')
    ax.set_xlabel(u'{}'.format(target_var))
    ax.set_ylabel('Frequency')

    mkdir(folder=plots_dest)
    plt.savefig(os.path.join(plots_dest, 'all_distributions.png'))
    plt.close()

    best_fit_name = results_df.iloc[0]['distribution']
    best_fit_params = dist_param[best_fit_name]

    print('\n\n\ndist_param: {}'.format(best_fit_params))
    print('dist_param from data: {}'.format(results_df.iloc[0]['params']))
    best_dist_st = getattr(st, best_fit_name)

    # pdf plot of the best distribution
    plot_best(best_dist=best_dist_st, best_dist_params=best_fit_params,
              best_dist_name=best_fit_name, target_variable=target_var,
              output_folder=plots_dest)

    # get the top 3 best fit distributions
    top_fits = get_top_dists(results=results_df, n=2)
    generate_pp_qq_plots(data=sorted(list(data)), best_fits=top_fits,
                         dist_param=dist_param, sizervs=len(data),
                         output_folder=plots_dest)
