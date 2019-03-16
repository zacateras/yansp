import conll
import os   
import csv

IN_DIR = 'tmp/ud-treebanks-v2.3'
OUT_FILE = 'out/ud-treebanks-v2.3.csv'

def iterate_conllu_recursive(conllu_directory=IN_DIR):
    for root, dirs, files in os.walk(conllu_directory):
        for file in files:
            if file.endswith('.conllu'):
                yield os.path.join(root, file)

def from_directory(csv_file=OUT_FILE, conllu_directory=IN_DIR):
    treebanks = list(iterate_conllu_recursive(conllu_directory))

    with open(csv_file, 'w+') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow([
            'lang',
            'tag',
            'ds_type',
            
            'sent',
            'token',
            
            'id', 
            'form', 
            'lemma',
            'upos',
            'xpos',
            'feats',
            'head',
            'deprel',
            'deps',
            'misc'
        ])

        treebank_i = 0
        sent_i = 0
        token_i = 0
        
        for treebank_file in treebanks:
            treebank = conll.load_conllu(treebank_file)

            print('Processing {}/{}, {}...'.format(treebank_i, len(treebanks), treebank))

            for sent in treebank.sents:
                for token in sent.words:
                    writer.writerow([
                        treebank.lang,
                        treebank.tag,
                        treebank.dataset_type,
                        sent_i,
                        token_i
                    ] + token.columns)

                    token_i += 1

                sent_i += 1
                
            treebank_i += 1
                