import sys
import os
import re
import pandas as pd
import yaml
import openood.utils.config
from openood.utils.config import Config


def find_csv_and_config(root, pattern, *file_names, is_in=False):

    if isinstance(pattern, str):
        pattern = re.compile(pattern)

    files_in_dir = {_: None for _ in file_names}
    for node in os.scandir(root):
        if node.is_dir():
            yield from find_csv_and_config(node.path, pattern, *file_names,
                                           is_in=is_in or pattern.match(node.name))

        elif is_in and node.is_file() and node.name in file_names:
            files_in_dir[node.name] = node.path

        if all(files_in_dir.values()):
            yield files_in_dir


class ConfigLoader(yaml.SafeLoader):
    def load_config(self, node):
        return self.construct_mapping(node)


ConfigLoader.add_constructor(None, ConfigLoader.load_config)


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--result-dir', default='./results')
    parser.add_argument('--network', default='cvae_encoder')
    parser.add_argument('--ind', default='cifar10')
    parser.add_argument('--seed', default=0)

    args = parser.parse_args()

    network = args.network
    result_dir = args.result_dir
    seed = args.seed
    ind_set = args.ind

    pattern = '{}_ood_{}(_[^_]+)*_test_ood_ood_default'.format(ind_set, network)
    pattern = '{}_ood_{}(_[^_]+)*_test_ood_ood(_[^_]+)*_default'.format(ind_set, network)

    df_ = []

    for conf_files in find_csv_and_config(result_dir, pattern, 'ood.csv', 'config.yml'):
        csv_file = conf_files['ood.csv']
        conf_file = conf_files['config.yml']
        df = pd.read_csv(csv_file)
        with open(conf_file) as f:
            config = yaml.load(f, Loader=ConfigLoader)['state']

        df['gamma'] = config['network'].get('gamma')
        df['arch'] = config['network'].get('name')
        df['method'] = config['postprocessor']['name']
        print(os.path.split(csv_file)[0], config['postprocessor']['name'], config['network'].get('gamma'))

        df.set_index(['arch', 'gamma', 'method', 'dataset'], inplace=True)
        acc_df = df['ACC']
        df = df[['FPR@95', 'AUROC']].unstack()
        df[('ACC', ind_set)] = acc_df.iloc[0]

        df = df.swaplevel(0, 1, axis='columns')
        df.columns.rename(['set', 'measures'], inplace='True')
        df.rename(columns=str.lower, inplace=True)
        df.rename(columns={'auroc': 'auc'}, inplace=True)
        df.rename(columns={'mnist': 'mnist32r', 'cifar100': 'o_cifar100',
                           'places365': 'o_places365', 'tin': 'o_tin'}, inplace=True)

        df_.append(df)

    df = pd.concat(df_)

    c = df.columns

    kept_datasets = ['mnist32r']
    kept_datasets = ['farood', 'nearood']
    kept_datasets = ['mnist32r', 'svhn']

    printed_cols = c.isin(kept_datasets, level='set')

    print(df[c[printed_cols]].reset_index().set_index(['arch', 'method', 'gamma']).sort_index().to_string())

    """
    for _ in subdirs:
        s = r.search(_)
        print('r', regex_str)
        print('d', _)
        if not s:
            continue
        method = s.group(1)
        csv_file = os.path.join(result_dir, _, 's{}'.format(seed), 'ood', 'ood.csv')
        if not os.path.exists(csv_file):
            print(csv_file, 'does not exist')
            continue

        df = pd.read_csv(csv_file)
        df.set_index(['dataset'], inplace=True)

        acc_df = df['ACC']
        df = df[['FPR@95', 'AUROC']].unstack()

        df[('ACC', ind_set)] = acc_df.iloc[0]

        df_[method] = df

    df = pd.concat(df_, axis='columns').T
    df['arch'] = '-'.join(network.split('_')[1:])
    df.index.rename('method', inplace=True)
    df = df.reset_index().set_index(['arch', 'method'])
    df = df.swaplevel(0, 1, axis='columns')
    df.columns.rename(['set', 'measures'], inplace='True')
    df.rename(columns=str.lower, inplace=True)
    df.rename(columns={'auroc': 'auc'}, inplace=True)
    df.rename(columns={'mnist': 'mnist32r', 'cifar100': 'o_cifar100',
              'places365': 'o_places365', 'tin': 'o_tin'}, inplace=True)
    print(df)
    """
    # df.to_csv('/tmp/openood_{}.csv'.format(network))
