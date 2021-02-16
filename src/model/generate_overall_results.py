import scikit_posthocs as sp
import pandas as pd
import numpy as np
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats.stats import spearmanr
from os import environ, listdir
from os.path import join
from dotenv import find_dotenv, load_dotenv
from matplotlib.backends.backend_pdf import FigureCanvasPdf, PdfPages
from matplotlib.figure import Figure


def generate_overall_results(path, file_results):
    df_mean_list = []
    for file in file_results:
        results = pd.read_csv(join(path, file))
        mean_results = results.mean()
        mean_results['method'] = file.split('.')[0].split('_')[1]
        mean_results['target'] = file.split('.')[0].split('_')[0]
        mean_results['mean_pos1'] = results['pos_1'].mean()
        df_mean_list.append(mean_results)
    overall_results = pd.concat(df_mean_list, axis=1)
    return overall_results.transpose()


def get_all_metric_results(path, file_results, metric):
    df_list = []
    for file in file_results:
        results = pd.read_csv(join(path, file))
        kendall = results[[metric]]
        
        kendall['method'] = [file.split('.')[0]] * len(kendall)
        df_list.append(kendall)
    kendall = pd.concat(df_list, axis=0)
    return kendall
    


if __name__ == '__main__':
    # Find data.env automatically by walking up directories until it's found
    dotenv_path = find_dotenv(filename='data.env')
    # Load up the entries as environment variables
    load_dotenv(dotenv_path)
    # Get dataset parameter
    region = environ.get('REGION_NAME')
    tse_year = str(environ.get('ELECTION_YEAR'))
    tse_office = environ.get('POLITICAL_OFFICE')
    tse_turn = str(environ.get('ELECTION_TURN'))
    tse_per = environ.get('PER')
    candidates = environ.get('CANDIDATES').split(',')
    ibge_year = str(environ.get('CENSUS_YEAR'))
    ibge_aggr = environ.get('CENSUS_AGGR_LEVEL')
    fold_group = environ.get('N_FOLDS')
    # Get data root path
    data_dir = environ.get('ROOT_DATA')
    # Results path
    fold_exp = 'baselines'
    group = ''
    results_path = data_dir + environ.get('RESULTS_PATH')
    results_path = results_path.format(region, tse_year, ibge_year, tse_office, tse_turn, ibge_aggr, tse_per)
    folds_results_path = join(results_path, 'linear_regression', 'by_folds_{}'.format(fold_group), fold_exp, group)
    folds_results = [f_results for f_results in listdir(folds_results_path)]
    overall_results = generate_overall_results(folds_results_path, folds_results)
    overall_results['index'] = overall_results['method']
    overall_results.set_index('index', inplace=True)
    print(overall_results)
  
    pp = PdfPages(join(results_path, 'linear_regression', 'by_folds_{}'.format(fold_group), 'overall', fold_exp+'_'+group+'.pdf'))
    for metric in ['n_features', 'mse', 'kendall', 'spearmanr', 'wkendall', 'pos_1']:
        all_results = [overall_results.loc['All', metric]] * int(len(overall_results)/2 -1)
        rnd_results = [overall_results.loc['RND', metric]] * int(len(overall_results)/2 -1)
        hue_all = ['ALL'] * len(all_results)
        hue_rnd = ['Random'] * len(rnd_results)
        lines_results = all_results + rnd_results
        lines_hue = hue_all + hue_rnd
        
        results = overall_results.drop(['All', 'RND'])
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        plt.xticks(fontsize=9)
        if metric == 'mse':
            ax.set_title('Scaled Absolute Error')
        else:
            ax.set_title(metric.upper())
        sns.set_theme(style="whitegrid")
        splot = sns.barplot(ax=ax, x='method', hue='target', y=metric, data=results, palette='tab10')
        
        for p in splot.patches:
            splot.annotate(format(p.get_height(), '.2f'),
                           (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='center',
                           xytext=(0, 9),
                           textcoords='offset points')
        lineplot = sns.lineplot(y = lines_results, x = results['method'],  hue=lines_hue, marker='o', sort = False, ax=ax, palette='hls')
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.tight_layout()
        pp.savefig(fig)
    
    # Recall line  
    recall = overall_results[['recall_1', 'recall_10', 'recall_30', 'recall_40', 'recall_50']].transpose()
    recall.columns = overall_results['method']

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(5.5, 5.5)
    ax.set_title('Recall')
    recall.plot.line(ax=ax)
    pp.savefig(fig)
    pp.close()
    exit()
    ####
    for metric in ['kendall', 'wkendall', 'spearmanr', 'mse']:
        kendall_results = get_all_metric_results(folds_results_path, folds_results, metric)
        wilcoxon = sp.posthoc_wilcoxon(kendall_results, val_col=metric, group_col='method')
        print(wilcoxon)
        fig, ax = plt.subplots(1, 1)
        plt.xticks(fontsize=7)
        plt.yticks(fontsize=7)
        fig.set_size_inches(5.5, 5.5)
        heatmap_args = {'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': False, 'square': True, 'cbar_ax_bbox': [0.80, 0.35, 0.04, 0.3]}
        ax.set_title('Wilcoxon Signed Rank - {}'.format(metric))
        sp.sign_plot(wilcoxon, **heatmap_args, ax=ax)
        pp.savefig(fig)
    
    
    
    pp.close()
