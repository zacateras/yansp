import sys
import re
import os
import json
import pandas as pd
import argparse
import subprocess

from utils import log

def iterate_recursive(directory, pattern):
    cmpl = re.compile(pattern)
    for root, dirs, files in os.walk(directory):
        for file in files:
            if cmpl.match(file):
                yield os.path.join(root, file)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--treebanks_root_dir', type=str, required=True)
    parser.add_argument('--models_root_dir', type=str, required=True)
    parser.add_argument('--scores_file', type=str, required=True)
    parser.add_argument('--confs_file', type=str, required=True)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    if os.path.exists(args.scores_file):
        log('Loading scores from {}...'.format(arg.scores_file))
        scores_df = pd.read_csv(args.scores_file)

    if os.path.exists(args.confs_file):
        log('Loading confs from {}...'.format(args.confs_file))
        confs_df = pd.read_csv(args.confs_file)

    log('Looking up available treebanks...')
    treebanks_for_languages = dict()
    treebanks_i = 0
    for conllu_path in iterate_recursive(args.treebanks_root_dir, r'.*\.conllu$'):
        conllu_file = os.path.split(conllu_path)
        conllu_file = conllu_file[-1]

        language, reminder = conllu_file.split('_')
        treebank_type = reminder.split('.')[0].split('-')[-1]

        if treebank_type in ['dev', 'test']:
            treebanks_i += 1

            if language in treebanks_for_languages.keys():
                treebanks_for_languages[language].append(conllu_path)
            else:
                treebanks_for_languages[language] = [conllu_path]

    log('Found {} treebanks for {} languages.'.format(treebanks_i, len(treebanks_for_languages)))

    for model_path in iterate_recursive(args.models_root_dir, r'model.conf$'):
        signature = os.path.split(os.path.dirname(model_path))[-1]

        with open(model_path) as f:
            model_conf = json.load(f)
            model_conf['signature'] = signature
        
        df = pd.io.json.json_normalize(model_conf)

        if 'confs_df' not in locals():
            log('Saving conf for {}.'.format(signature))
            confs_df = df
            confs_df.to_csv(args.confs_file, index=False)

        elif signature not in confs_df['signature']:
            confs_df = pd.concat([confs_df, df], axis=0, ignore_index=True)
            confs_df.to_csv(args.confs_file, index=False)

        model_dir = os.path.dirname(model_path)

        log('Starting evaluation for {}.'.format(signature))
        language = signature.split('_')[0]
        if language in treebanks_for_languages.keys():
            treebanks = treebanks_for_languages[language]

            subprocess.call(
                [sys.executable, 'main.py', 'evaluate', '--model_dir', model_dir, '--scores_file', args.scores_file, '--conllu_file'] + treebanks, stdout=sys.stdout, stderr=sys.stderr)            
