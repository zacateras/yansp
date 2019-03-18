import conll
import os   
import csv

IN_DIR = 'tmp/ud-treebanks-v2.3'
OUT_FILE = 'out/ud-treebanks-v2.3.csv'

def create_analytical(csv_file=OUT_FILE, conllu_directory=IN_DIR):
    def iterate_conllu_recursive():
        for root, dirs, files in os.walk(conllu_directory):
            for file in files:
                if file.endswith('.conllu'):
                    yield os.path.join(root, file)

    treebanks = list(iterate_conllu_recursive())

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
            'misc',

            # statistics
            'len_form',
            'len_lemma',
            'vdist_to_head',
            'hdist_to_root'
        ])

        treebank_i = 0
        sent_i = 0
        token_i = 0
        
        for treebank_file in treebanks:
            treebank = conll.load_conllu(treebank_file)

            print('Processing {}/{}, {}...'.format(treebank_i, len(treebanks), treebank))

            for sent in treebank.sents:
                for token in sent.words:

                    vdist_to_head = abs(token.id - token.head)
                    hdist_to_root = 0

                    h = token.head
                    while h != 0:
                        h = sent[h - 1].head
                        hdist_to_root += 1

                    writer.writerow([
                        treebank.lang,
                        treebank.tag,
                        treebank.dataset_type,

                        sent_i,
                        token_i

                    ] + token.columns + [len(token.form), len(token.lemma), vdist_to_head, hdist_to_root])

                    token_i += 1

                sent_i += 1
                
            treebank_i += 1

def create_validation(csv_file, validation_directory):
    pass
