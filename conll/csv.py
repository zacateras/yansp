import conll
import os   
import csv
import re

IN_DIR = 'tmp/ud-treebanks-v2.3'
OUT_FILE = 'out/ud-treebanks-v2.3.csv'

def iterate_recursive(directory, pattern):
    cmpl = re.compile(pattern)
    for root, dirs, files in os.walk(directory):
        for file in files:
            if cmpl.match(file):
                yield os.path.join(root, file)

def create_analytical(csv_file=OUT_FILE, conllu_directory=IN_DIR):
    treebanks = list(iterate_recursive(conllu_directory, r'.+\.conllu$'))

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

            print('Processing {}/{}, {}...'.format(treebank_i + 1, len(treebanks), treebank))

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

def create_validation(csv_file='out/validation.csv', base_dir='out'):
    gold = dict()
    system = dict()

    for conllu_file in iterate_recursive(base_dir, r'.+\.conllu$'):
        match = re.match(r'{}/([A-Za-z0-9\.\-]+)/validation/([0-9]+)_(gold|system)\.conllu'.format(base_dir), conllu_file)

        model = match[1]
        iteration = int(match[2])
        source = match[3]

        print('Reading validation for {} model {}...'.format(source, model))

        if source == 'gold':
            gold[(model, iteration)] = conllu_file
        elif source == 'system':
            system[(model, iteration)] = conllu_file

    with open(csv_file, 'w+') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow([
            'model',
            'iteration',
            
            'sent',
            'word',
            
            'id',
            'form',
            
            'lemma_gold',
            'lemma_system',
            'upos_gold',
            'upos_system',
            'feats_gold',
            'feats_system',
            'head_gold',
            'head_system',
            'deprel_gold',
            'deprel_system'
        ])
        
        sent_i = 0
        word_i = 0
        
        for key in sorted(set(gold.keys()) & set(system.keys())):
            model = key[0]
            iteration = key[1]

            print('Processing model {}...'.format(model))
            
            ud_gold = conll.load_conllu(gold[key], name='{}_{}_{}'.format(model, iteration, 'gold'))
            ud_system = conll.load_conllu(system[key], name='{}_{}_{}'.format(model, iteration, 'system'))
            
            for sent_gold, sent_system in zip(ud_gold.sents, ud_system.sents):
                for word_gold, word_system in zip(sent_gold.words, sent_system.words):
                    writer.writerow([
                        model,
                        iteration,
                        
                        sent_i,
                        word_i,
                        
                        word_gold.id,
                        word_gold.form,
                        
                        word_gold.lemma,
                        word_system.lemma,
                        word_gold.upos,
                        word_system.upos,
                        '|'.join(word_gold.feats),
                        '|'.join(word_system.feats),
                        word_gold.head,
                        word_system.head,
                        word_gold.deprel,
                        word_system.deprel
                    ])
                    
                    word_i += 1
                
                sent_i += 1

def create_model_conf(csv_file='out/model_conf.csv', base_dir='out'):
    all_model_confs = dict()
    all_keys = set()
    
    for model_conf_file in iterate_recursive(base_dir, r'model\.conf'):
        match = re.match(r'{}/([A-Za-z0-9\.\-]+)/model.conf'.format(base_dir), model_conf_file)

        model = match[1]
        model_conf = dict()

        print('Reading model {}...'.format(model))
        
        with open(model_conf_file, 'r') as f:
            for line in f:
                # skip comments and blank lines
                if re.match(r'^[\s]*\#.*$', line) or re.match(r'^[\s]*$', line):
                    continue
                    
                chunks = list(map(str.strip, line.split('=')))
                
                # skip unparsable lines
                if len(chunks) != 2:
                    print('Skipping one unparsable line.')
                    continue
                    
                key, value = chunks
                all_keys.add(key)
                
                model_conf[key] = value
                
        all_model_confs[model] = model_conf
        
    all_keys = sorted(all_keys)        
    
    with open(csv_file, 'w+') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow([
            'model'
        ] + list(all_keys))
        
        for model, model_conf in all_model_confs.items():
            print('Flushing model {}...'.format(model))

            row = [model]
            
            for key in all_keys:
                if key in model_conf:
                    row.append(model_conf[key])
                else:
                    row.append('')
            
            writer.writerow(row)
