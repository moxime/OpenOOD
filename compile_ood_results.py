import sys
import os
import re
import pandas as pd


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--result-dir', default='./results')
    parser.add_argument('--network', default='cifar10_vgg19_32x32')
    parser.add_argument('--seed', default=0)

    args = parser.parse_args()

    network = args.network
    result_dir = args.result_dir
    seed = args.seed

    regex_str = '{}_test_ood_ood_(\w*)_default'.format(network)
    r = re.compile(regex_str)
    subdirs = os.listdir(args.result_dir)

    ind_set = network.split('_')[0]

    df_ = {}

    for _ in subdirs:
        s = r.search(_)
        if not s:
            continue
        method = s.group(1)
        csv_file = os.path.join(result_dir, _, 's{}'.format(seed), 'ood', 'ood.csv')
        if not os.path.exists(csv_file):
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

    df.to_csv('/tmp/openood.csv')
