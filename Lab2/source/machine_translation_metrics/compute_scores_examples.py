# Bleu
from nltk.translate.bleu_score import sentence_bleu
# CIDer
from cider.cider import Cider
# Meteor is run from Java module edu.cmu.
from py4j.java_gateway import JavaGateway
# Rouge
import rouge

# reference = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]
# candidate = ['the', 'fast', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']

reference = [['Man', 'on', 'ocean', 'beach', 'flying', 'several', 'kites', 'on', 'windy', 'day']]
candidate = ['a', 'person', 'on', 'a', 'beach', 'flying', 'a', 'kite']

# BLEU SCORES
score_bleu_C = sentence_bleu(reference, candidate)
score_bleu_1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
score_bleu_2 = sentence_bleu(reference, candidate, weights=(0, 1, 0, 0))
score_bleu_3 = sentence_bleu(reference, candidate, weights=(0, 0, 1, 0))
score_bleu_4 = sentence_bleu(reference, candidate, weights=(0, 0, 0, 1))
print('Bleu-1: %f' % score_bleu_1)
print('Bleu-2: %f' % score_bleu_2)
print('Bleu-3: %f' % score_bleu_3)
print('Bleu-4: %f' % score_bleu_4)
print('Blue-Cumulative: %f' % score_bleu_C)

# CIDer Scores
ref_dict = {'1': ['the quick brown fox jumped over the lazy dog'], '2': ['test test test test'], '3': ['here is one sentence']}
hyp_dict = {'1': ['the fast brown fox jumped over the lazy dog'], '2': ['test test test test'], '3': ['this statement shares no words']}
cider_eval = Cider()
cider_score = cider_eval.compute_score(ref_dict, hyp_dict)
print('Cider Average: %f' % (cider_score[0]/10))
print('Cider sample 1: %f' % (cider_score[1][0]/10))
print('Cider sample 2: %f' % (cider_score[1][1]/10))
print('Cider sample 3: %f' % (cider_score[1][2]/10))

# METEOR Scores
gateway = JavaGateway()
meteor_eval = gateway.entry_point
# meteor_eval = Meteor
# ref_str = "the quick brown fox jumped over the lazy dog"
# hyp_str = "the fast brown fox jumped over the lazy dog"
ref_str = "Man on ocean beach flying several kites on windy day"
hyp_str = "a person on a beach flying a kite"
meteor_score = meteor_eval.compute_score(hyp_str, ref_str)
print('Meteor: %f' % meteor_score)

# ROUGE Scores

# references = ['the quick brown fox jumped over the lazy dog', 'test']
# hypotheses = ['the fast brown fox jumped over the lazy dog', 'test']

references = ['Man on ocean beach flying several kites on windy day', 'Two people standing next to a river holding surfboards', 'test']
hypotheses = ['a person on a beach flying a kite', 'a man holding a surfboard on a beach', 'test']

rouge_eval = rouge.Rouge()

test_r_eval = rouge.FilesRouge("data/example/hyp.txt","data/example/ref.txt")
b = test_r_eval.get_scores()
print(b)

scores_rouge_dict_list = rouge_eval.get_scores(hypotheses, references)
print(scores_rouge_dict_list)
rouge_1_scores = [0.0]*3
rouge_2_scores = [0.0]*3
rouge_l_scores = [0.0]*3

for i, s in enumerate(scores_rouge_dict_list):
    if isinstance(s, dict):
        for metric, vals in s.items():
            print(metric)
            print(vals)
            if metric == 'rouge-1':
                rouge_1_scores[i] = vals.get('f')
            if metric == 'rouge-2':
                rouge_2_scores[i] = vals.get('f')
            if metric == 'rouge-l':
                rouge_l_scores[i] = vals.get('f')

print(rouge_1_scores)
print(rouge_2_scores)
print(rouge_l_scores)

# Parameters for rogue which can be specified when it is created
# metrics=['rouge-n', 'rouge-l', 'rouge-w'],
# max_n=4,
# limit_length=True,
# length_limit=100,
# length_limit_type='words',
# apply_avg=apply_avg,
# apply_best=apply_best,
# alpha=0.5, # Default F1_score
# weight_factor=1.2,
# stemming=True

