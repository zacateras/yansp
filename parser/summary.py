from utils.model import count_variables

def summary_for_parser(parser):
    return {
        'variables_all': count_variables(parser.variables),
        'variables_all_trainable': count_variables(parser.trainable_variables),

        'variables_word': count_variables(parser.word_model.variables),
        'variables_word_trainable': count_variables(parser.word_model.trainable_variables),

        'variables_char': count_variables(parser.char_model.variables),
        'variables_char_trainable': count_variables(parser.char_model.trainable_variables),

        'variables_core': count_variables(parser.core_model.variables),
        'variables_core_trainable': count_variables(parser.core_model.trainable_variables),

        'variables_lemma': count_variables(parser.lemma_model.variables),
        'variables_lemma_trainable': count_variables(parser.lemma_model.trainable_variables),

        'variables_upos': count_variables(parser.upos_model.variables),
        'variables_upos_trainable': count_variables(parser.upos_model.trainable_variables),

        'variables_feats': count_variables(parser.feats_model.variables),
        'variables_feats_trainable': count_variables(parser.feats_model.trainable_variables),

        'variables_head': count_variables(parser.head_model.variables),
        'variables_head_trainable': count_variables(parser.head_model.trainable_variables),

        'variables_deprel': count_variables(parser.deprel_model.variables),
        'variables_deprel_trainable': count_variables(parser.deprel_model.trainable_variables),
    }