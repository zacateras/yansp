from utils.model import count_variables

def summary_for_parser(parser):
    return {
        'variables_all': count_variables(parser.variables),
        'variables_all_trainable': count_variables(parser.trainable_variables),

        'variables_word': count_variables(parser.word_model.variables) if hasattr(parser, 'word_model') else 0,
        'variables_word_trainable': count_variables(parser.word_model.trainable_variables) if hasattr(parser, 'word_model') else 0,

        'variables_char': count_variables(parser.char_model.variables) if hasattr(parser, 'char_model') else 0,
        'variables_char_trainable': count_variables(parser.char_model.trainable_variables) if hasattr(parser, 'char_model') else 0,

        'variables_core': count_variables(parser.core_model.variables) if hasattr(parser, 'core_model') else 0,
        'variables_core_trainable': count_variables(parser.core_model.trainable_variables) if hasattr(parser, 'core_model') else 0,

        'variables_lemma': count_variables(parser.lemma_model.variables) if hasattr(parser, 'lemma_model') else 0,
        'variables_lemma_trainable': count_variables(parser.lemma_model.trainable_variables) if hasattr(parser, 'lemma_model') else 0,

        'variables_upos': count_variables(parser.upos_model.variables) if hasattr(parser, 'upos_model') else 0,
        'variables_upos_trainable': count_variables(parser.upos_model.trainable_variables) if hasattr(parser, 'upos_model') else 0,

        'variables_feats': count_variables(parser.feats_model.variables) if hasattr(parser, 'feats_model') else 0,
        'variables_feats_trainable': count_variables(parser.feats_model.trainable_variables) if hasattr(parser, 'feats_model') else 0,

        'variables_head': count_variables(parser.head_model.variables) if hasattr(parser, 'head_model') else 0,
        'variables_head_trainable': count_variables(parser.head_model.trainable_variables) if hasattr(parser, 'head_model') else 0,

        'variables_deprel': count_variables(parser.deprel_model.variables) if hasattr(parser, 'deprel_model') else 0,
        'variables_deprel_trainable': count_variables(parser.deprel_model.trainable_variables) if hasattr(parser, 'deprel_model') else 0,
    }