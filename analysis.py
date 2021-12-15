##################################################
# All functions related to analysing the training and testing predictions of a trained model.
##################################################
# Author: Marius Bock
# Email: marius.bock@uni-siegen.de
##################################################

import argparse

import pandas as pd
import numpy as np
import os
import json

from utils import paint


def get_args():
    parser = argparse.ArgumentParser(description='Train and evaluate an HAR model on given dataset.')

    parser.add_argument(
        '-d', '--directory', type=str, help='Log directory for rerun of analysis (e.g. 20211205/225715). Required',
        required=True)

    args = parser.parse_args()

    return args


def run_train_analysis(train_results):
    """
    Runs an average and subject-wise analysis of saved train results.

    :param train_results: the train result dataframe returned by the cross_validate function.
    :return: None
    """
    # average analysis
    avg_t_loss, avg_t_acc, avg_t_fm, avg_t_fw = [], [], [], []
    avg_v_loss, avg_v_acc, avg_v_fm, avg_v_fw = [], [], [], []

    # average analysis
    print(paint("AVERAGE RESULTS"))
    for i, row in train_results.iterrows():
        if i == 0:
            avg_t_loss = np.asarray(row['t_loss'])
            avg_t_acc = np.asarray(row['t_acc'])
            avg_t_fm = np.asarray(row['t_fm'])
            avg_t_fw = np.asarray(row['t_fw'])
            avg_v_loss = np.asarray(row['v_loss'])
            avg_v_acc = np.asarray(row['v_acc'])
            avg_v_fm = np.asarray(row['v_fm'])
            avg_v_fw = np.asarray(row['v_fw'])
        else:
            avg_t_loss = np.add(avg_t_loss, row['t_loss'])
            avg_t_acc = np.add(avg_t_acc, row['t_acc'])
            avg_t_fm = np.add(avg_t_fm, row['t_fm'])
            avg_t_fw = np.add(avg_t_fw, row['t_fw'])
            avg_v_loss = np.add(avg_v_loss, row['v_loss'])
            avg_v_acc = np.add(avg_v_acc, row['v_acc'])
            avg_v_fm = np.add(avg_v_fm, row['v_fm'])
            avg_v_fw = np.add(avg_v_fw, row['v_fw'])

    avg_t_loss /= len(train_results)
    avg_t_acc /= len(train_results)
    avg_t_fm /= len(train_results)
    avg_t_fw /= len(train_results)
    avg_v_loss /= len(train_results)
    avg_v_acc /= len(train_results)
    avg_v_fm /= len(train_results)
    avg_v_fw /= len(train_results)

    print('\nAverage Train results (last epoch):')
    print('Loss: {:.4f} - Accuracy: {:.4f} - F1-score (macro): {:.4f} - F1-score (weighted): {:.4f}'
          .format(avg_t_loss[-1], avg_t_acc[-1], avg_t_fm[-1], avg_t_fw[-1]))
    print('\nAverage Validation results (last epoch):')
    print('Loss: {:.4f} - Accuracy: {:.4f} - F1-score (macro): {:.4f} - F1-score (weighted): {:.4f}'
          .format(avg_v_loss[-1], avg_v_acc[-1], avg_v_fm[-1], avg_v_fw[-1]))

    # subject-wise analysis
    print(paint("SUBJECT-WISE RESULTS"))
    for sbj in np.unique(train_results['sbj']):
        if sbj == -1:
            print(paint("NONE"))
            continue
        else:
            avg_sbj_t_loss, avg_sbj_t_acc, avg_sbj_t_fm, avg_sbj_t_fw = [], [], [], []
            avg_sbj_v_loss, avg_sbj_v_acc, avg_sbj_v_fm, avg_sbj_v_fw = [], [], [], []
            sbj_data = train_results[train_results.sbj == sbj]
            # average analysis
            for i, (_, row) in enumerate(sbj_data.iterrows()):
                if i == 0:
                    avg_sbj_t_loss = np.asarray(row['t_loss'])
                    avg_sbj_t_acc = np.asarray(row['t_acc'])
                    avg_sbj_t_fm = np.asarray(row['t_fm'])
                    avg_sbj_t_fw = np.asarray(row['t_fw'])
                    avg_sbj_v_loss = np.asarray(row['v_loss'])
                    avg_sbj_v_acc = np.asarray(row['v_acc'])
                    avg_sbj_v_fm = np.asarray(row['v_fm'])
                    avg_sbj_v_fw = np.asarray(row['v_fw'])
                else:
                    avg_sbj_t_loss = np.add(avg_sbj_t_loss, row['t_loss'])
                    avg_sbj_t_acc = np.add(avg_sbj_t_acc, row['t_acc'])
                    avg_sbj_t_fm = np.add(avg_sbj_t_fm, row['t_fm'])
                    avg_sbj_t_fw = np.add(avg_sbj_t_fw, row['t_fw'])
                    avg_sbj_v_loss = np.add(avg_sbj_v_loss, row['v_loss'])
                    avg_sbj_v_acc = np.add(avg_sbj_v_acc, row['v_acc'])
                    avg_sbj_v_fm = np.add(avg_sbj_v_fm, row['v_fm'])
                    avg_sbj_v_fw = np.add(avg_sbj_v_fw, row['v_fw'])

            avg_sbj_t_loss /= len(sbj_data)
            avg_sbj_t_acc /= len(sbj_data)
            avg_sbj_t_fm /= len(sbj_data)
            avg_sbj_t_fw /= len(sbj_data)
            avg_sbj_v_loss /= len(sbj_data)
            avg_sbj_v_acc /= len(sbj_data)
            avg_sbj_v_fm /= len(sbj_data)
            avg_sbj_v_fw /= len(sbj_data)

            print('\nAverage Train results (last epoch): Subject {}'.format(sbj))
            print('Loss: {:.4f} - Accuracy: {:.4f} - F1-score (macro): {:.4f} - F1-score (weighted): {:.4f}'
                  .format(avg_sbj_t_loss[-1], avg_sbj_t_acc[-1], avg_sbj_t_fm[-1], avg_sbj_t_fw[-1]))
            print('\nAverage Validation results (last epoch): Subject {}'.format(sbj))
            print('Loss: {:.4f} - Accuracy: {:.4f} - F1-score (macro): {:.4f} - F1-score (weighted): {:.4f}'
                  .format(avg_sbj_v_loss[-1], avg_sbj_v_acc[-1], avg_sbj_v_fm[-1], avg_sbj_v_fw[-1]))


def run_test_analysis(test_results, save_results):
    """
    Runs an average analysis of saved test results.

    :param test_results: the test result dataframe returned by the cross_validate function.
    :return: None
    """
    if test_results is not None:
        avg_t_loss, avg_t_acc, avg_t_fm, avg_t_fw = .0, .0, .0, .0
        # average analysis
        for i, row in test_results.iterrows():
            if i == 0:
                avg_t_loss = np.asarray(row['test_loss'])
                avg_t_acc = np.asarray(row['test_acc'])
                avg_t_fm = np.asarray(row['test_fm'])
                avg_t_fw = np.asarray(row['test_fw'])
            else:
                avg_t_loss = np.add(avg_t_loss, row['test_loss'])
                avg_t_acc = np.add(avg_t_acc, row['test_acc'])
                avg_t_fm = np.add(avg_t_fm, row['test_fm'])
                avg_t_fw = np.add(avg_t_fw, row['test_fw'])

        avg_t_loss /= len(test_results)
        avg_t_acc /= len(test_results)
        avg_t_fm /= len(test_results)
        avg_t_fw /= len(test_results)

        print('\nAverage Test results:')
        print('Loss: {:.4f} - Accuracy: {:.4f} - F1-score (macro): {:.4f} - F1-score (weighted): {:.4f}'
              .format(avg_t_loss, avg_t_acc, avg_t_fm, avg_t_fw))


def rerun_analysis(log_directory):
    """
    Method used to rerun an analysis by loading up saved train and (if applicable) test results.

    :param log_directory: directory where results were saved to (e.g. 20211205/225740)
    :return: None
    """
    train_results_df = pd.read_csv(os.path.join('../logs', log_directory, 'train_results.csv'), index_col=None)
    train_results_df[['t_loss', 't_acc', 't_fm', 't_fw', 'v_loss', 'v_acc', 'v_fm', 'v_fw']] = \
        train_results_df[['t_loss', 't_acc', 't_fm', 't_fw', 'v_loss', 'v_acc', 'v_fm', 'v_fw']] \
            .apply(lambda x: list(map(json.loads, x)))
    run_train_analysis(train_results_df)
    if os.path.isfile(os.path.join('../logs', log_directory, 'test_results.csv')):
        test_results_df = pd.read_csv(os.path.join('../logs', log_directory, 'test_results.csv'), index_col=None)
        run_test_analysis(test_results_df)


if __name__ == '__main__':
    args = get_args()
    rerun_analysis(args.directory)
