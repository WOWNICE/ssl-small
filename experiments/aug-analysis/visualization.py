from matplotlib import pyplot as plt
import numpy as np
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--models', default='', type=str, nargs='*',
                    help='metric json file of the big model.')

def read_file(path):
    with open(path, 'r') as f:
        record = json.load(f)
    
    return record

def main():
    args = parser.parse_args()
    models = [read_file(model) for model in args.models]

    model_samples = [model['samples'] for model in models]
    if model_samples[:-1] != model_samples[1:]:
        print('[!] sample names do not match, aborted.')
        return
    
    #====================================================================================
    # visualize the distribution
    #====================================================================================
    plt.rcParams['figure.figsize'] = [10, 10]
    colors = plt.cm.jet

    # single distribution
    plt.subplot(211)
    for i, model in enumerate(models):
        plt.hist(model['cosine_dist'], bins=500, range=(0, 1), alpha=0.3, color=colors(i/len(models)), label=model['model'])
    plt.legend(loc='best')
    plt.title('distribution: positive-pair cosine distance')

    plt.subplot(212)
    for i, model in enumerate(models):
        plt.hist(model['l2_dist'], bins=500, range=(0, 1), alpha=0.3, color=colors(i/len(models)), label=model['model'])
    plt.legend(loc='best')
    plt.title('distribution: positive-pair l2 distance')



    # plt.subplot(122)
    # plt.hist2d(small_record['cosine_dist'], big_record['cosine_dist'], bins=100, range=((0,1), (0,1)))

    plt.savefig('test.png')


if __name__ == '__main__':
    main()