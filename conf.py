import os
import json
from utils import genkey
from parser.summary import summary_for_parser

def ensure_saved(params, model):
    file = os.path.join(params['base_dir'], 'model.conf')
    if not os.path.exists(file):
        os.makedirs(os.path.dirname(file), exist_ok=True)
        with open(file, 'w+') as f:
            json.dump({ 'params': params, 'summary': summary_for_parser(model) }, f, indent=2)

def load(params):
    file = os.path.join(params['base_dir'], 'model.conf')
    with open(file, 'r') as f:
        x = json.load(f)
        return x['params'], x['summary']

def preprocess(args):
    if args.mode in ('train'):
        args_vars = vars(args)

        # model meaningful parametrization properties
        pm = [args.model_core_type]

        if args.model_core_type == 'transformer':
            pm.append('dir' + str(args.model_core_transformer_layers_direction))
            pm.append(str(args.model_core_transformer_layers))
            pm.append(str(args.model_core_transformer_hidden_size))
            pm.append('ah' + str(args.model_core_transformer_attention_heads))
            pm.append('ak' + str(args.model_core_transformer_attention_key_dense_size))
            pm.append('av' + str(args.model_core_transformer_attention_value_dense_size))
            params = {k: v for k, v in args_vars.items() if not k.startswith('model_core_bilstm')}

        elif args.model_core_type == 'biLSTM':
            pm.append(str(args.model_core_bilstm_layers))
            pm.append(str(args.model_core_bilstm_layer_size))
            params = {k: v for k, v in args_vars.items() if not k.startswith('model_core_transformer')}

        # model paramerization hash string
        h = list(x for x in zip(params.keys(), params.values()))
        h = genkey(str(sorted(params, key=lambda x: x[0])))

        if args.signature_prefix is not None:
            pm.insert(0, args.signature_prefix)
        if args.signature_suffix is not None:
            pm.append(args.signature_suffix)

        signature = '.'.join(pm) + '-' + h
        params['base_dir'] = os.path.join(params['save_dir'], signature)

    elif args.mode in ('retrain', 'evaluate'):
        params = dict()
        params['base_dir'] = args.model_dir
        params, _ = load(params)
        for k, v in args.__dict__.items():
            if v is not None:
                params[k] = v

    return params
    