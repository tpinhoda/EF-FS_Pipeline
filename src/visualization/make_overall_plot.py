import logging
import scikit_posthocs as sp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from os import listdir
from os.path import join
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages


def generate_overall_results(path, file_results):
    df_mean_list = []
    for file in file_results:
        results = pd.read_csv(join(path, file))
        results.drop('fold_name', axis=1, inplace=True)
        mean_results = results.mean()
        mean_results['method'] = file.split('.')[0]
        df_mean_list.append(mean_results)
    overall_results = pd.concat(df_mean_list, axis=1)
    return overall_results.transpose()


def get_all_metric_results(path, file_results, metric):
    df_list = []
    for file in file_results:
        results = pd.read_csv(join(path, file))
        kendall = results[[metric]].copy()
        kendall['method'] = [file.split('.')[0]] * len(kendall)
        df_list.append(kendall)
    kendall = pd.concat(df_list, axis=0)
    return kendall


def get_baselines_results(overall_results, metric, baseline_name, topline_name):
    topline_results = [overall_results.loc[topline_name, metric]] * int(len(overall_results) -2)
    baseline_results = [overall_results.loc[baseline_name, metric]] * int(len(overall_results) -2)
            
    hue_topline = ['ALL'] * len(topline_results)
    hue_baseline = ['Random'] * len(baseline_results)
            
    lines_results = topline_results + baseline_results
    lines_hue = hue_topline + hue_baseline
    return lines_results, lines_hue
      
            
def generate_bar_plots(overall_results, metrics, pdf_pages):
    logger_name = 'Visualization'
    logger = logging.getLogger(logger_name)
    baseline_name = [fs_method for fs_method  in overall_results['method'] if 'random' in fs_method][0]
    topline_name = 'all_features'
    for metric in metrics:
        logger.info('Generating bar plot for {}'.format(metric))
        no_baselines_results = overall_results.drop(['all_features', 'random_50%'])
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        plt.xticks(fontsize=9)
        ax.set_title(metric.upper())
        sns.set_theme(style="whitegrid")
        splot = sns.barplot(ax=ax,
                            x='method',
                            y=metric,
                            data=no_baselines_results,
                            palette='Paired')
            
        for p in splot.patches:
            splot.annotate(format(p.get_height(), '.2f'),
                           (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='center',
                           xytext=(0, 9),
                           textcoords='offset points')
            
        lines_results, lines_hue = get_baselines_results(overall_results, metric, baseline_name, topline_name)
        x_methods = no_baselines_results['method'].values.tolist() + no_baselines_results['method'].values.tolist()
        sns.lineplot(y=lines_results,
                     x=x_methods, 
                     hue=lines_hue,
                     marker='o',
                     sort=False,
                     ax=ax,
                     palette='hls')
            
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.tight_layout()
        pdf_pages.savefig(fig)
        

def generate_posthoc_heatmap(folds_results_path, metrics, fs_methods, pdf_pages):
    logger_name = 'Visualization'
    logger = logging.getLogger(logger_name)
    for metric in metrics:
        logger.info('Generating posthoc plot for {}'.format(metric))
        metric_results = get_all_metric_results(folds_results_path, fs_methods, metric)
        try:
            wilcoxon = sp.posthoc_wilcoxon(metric_results, val_col=metric, group_col='method')
            fig, ax = plt.subplots(1, 1)
            plt.xticks(fontsize=7)
            plt.yticks(fontsize=7)
            fig.set_size_inches(5.5, 5.5)
            heatmap_args = {'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': False, 'square': True, 'cbar_ax_bbox': [0.80, 0.35, 0.04, 0.3]}
            ax.set_title('Wilcoxon Signed Rank - {}'.format(metric))
            sp.sign_plot(wilcoxon, **heatmap_args, ax=ax)
            pdf_pages.savefig(fig) 
        except ValueError:
            logger.warning('Can not run wilcoxon test!')
   
        

def run(folds_results_path, plots_path):
    logger_name = 'Visualization'
    logger = logging.getLogger(logger_name)
    fs_methods = [method for method in listdir(folds_results_path)]
    logger.info('Calculating mean results.')
    overall_results = generate_overall_results(folds_results_path, fs_methods)
    overall_results.set_index('method', inplace=True, drop=False)
    metrics = ['n_features', 'sae', 'rmse', 'kendall', 'spearmanr', 'wkendall', 'hit_center', 'rank_dist_center']
    pdf_pages = PdfPages(join(plots_path, 'mean_results.pdf'))
    generate_bar_plots(overall_results, metrics, pdf_pages)
    metrics = ['sae', 'rmse', 'kendall', 'spearmanr', 'wkendall', 'hit_center', 'rank_dist_center']
    generate_posthoc_heatmap(folds_results_path, metrics, fs_methods, pdf_pages)
    pdf_pages.close()
    
    

