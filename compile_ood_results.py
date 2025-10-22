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
        print(conf_file)
        df = pd.read_csv(csv_file)
        with open(conf_file) as f:
            config = yaml.load(f, Loader=ConfigLoader)['state']

        df['gamma_z'] = config['network'].get('gamma')
        df['network_number'] = config['network'].get('job_number')

        df['network_arch'] = config['network'].get('name')
        df['method'] = config['postprocessor']['name']

        df.set_index(['network_arch', 'network_number',  'gamma_z', 'method', 'dataset'], inplace=True)
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
    kept_datasets = ['farood', 'nearood', ind_set]
    kept_datasets = ['mnist32r', 'svhn', ind_set]

    printed_cols = c.isin(kept_datasets, level='set')

    print(df[c[printed_cols]].reset_index().set_index(
        ['network_arch', 'network_number', 'method', 'gamma_z']).sort_index().to_string())

    df.to_csv('/tmp/openood_{}.csv'.format(network))
