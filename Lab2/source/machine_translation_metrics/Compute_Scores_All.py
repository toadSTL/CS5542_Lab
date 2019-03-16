import numpy
# Bleu
from nltk.translate.bleu_score import sentence_bleu
# CIDer
from cider.cider import Cider
# Meteor is run from Java module edu.cmu.
from py4j.java_gateway import JavaGateway
# Rouge
import rouge
import csv

# Establish paths for input data: hypotheses and references
hyp_path = "data/hyp.txt"
ref_path = "data/ref.txt"

# Open the hypotheses file an read the lines
hyp_f = open(hyp_path, "r")
hyp_lines = hyp_f.readlines()
# Create arrays to store the hypotheses and the associated ids
num_hyps = len(hyp_lines)
hypotheses = [""] * num_hyps
hyp_id = [""]*num_hyps
# Iterate over lines ands populate hypotheses and hyp_id
for index, line in enumerate(hyp_lines):
    split_line = line.replace("\n","").split("\t")  # current line => [current id, current hypothesis]
    id_str = split_line[0]
    while id_str[0] == '0':
        id_str = id_str[1:]
    id_str = id_str[: -4]
    hyp_id[index] = id_str                          # Put the current id into the array 'hyp_id'
    hypotheses[index] = split_line[1]               # Put the hypothesis into the array 'hypotheses'

# Open the references file an read the lines
ref_f = open(ref_path, "r")
ref_lines = ref_f.readlines()
# Create arrays to store the references and the associated ids
num_refs = len(ref_lines)
references = [""] * num_refs
ref_id = [""]*num_refs
# Iterate over lines ands populate references and ref_id
for index, line in enumerate(ref_lines):
    split_line = line.replace("\n","").split("\t")  # current line => [current id, current reference]
    while split_line[0][0] == '0':
        split_line[0] = split_line[0][1:]
    split_line[0] = split_line[0][: -4]
    ref_id[index] = split_line[0]                   # Put the current id into the array 'ref_id'
    references[index] = split_line[1]               # Put the reference into the array 'references'

# Get unique ids as the intersection of 'hyp_id' and 'ref_id'
uids_unsorted = list(set(hyp_id) & set(ref_id))
# print(uids_unsorted)
num_uids = len(uids_unsorted)
for i in range(num_uids):
    uids_unsorted[i] = int(uids_unsorted[i])
uids_int = sorted(uids_unsorted)
# print(uids_int)
unique_ids = [""] * num_uids
for i in range(num_uids):
    unique_ids[i] = str(uids_int[i])
# unique_ids = sorted(list(set(hyp_id) & set(ref_id)))
# print(unique_ids)
# num_uids = len(unique_ids)
# print(unique_ids)
# print(references)
# print(hypotheses)

# Calculate and store the multiplicities of unique ids in 'hyp_id'
# This is equivalent to the number of copies of each reference
# sample with a corresponding id to be placed in our final
# 'ref' list containing hypotheses to be processed by metrics
mult_hyps = [0]*num_uids
for i, id in enumerate(unique_ids):
    num_hyps_per_id = 0
    for j in range(num_hyps):
        if id == hyp_id[j]:
            num_hyps_per_id += 1
    mult_hyps[i] = num_hyps_per_id

# Calculate and store the multiplicities of unique ids in 'ref_id'
# As above, this is equivalent to the number of copies of each
# hypothesis sample with a corresponding id to be placed in our
# final 'hyp' list containing hypotheses to be processed by metrics
mult_refs = [0]*num_uids
for i, id in enumerate(unique_ids):
    num_refs_per_id = 0
    for j in range(num_refs):
        if id == ref_id[j]:
            num_refs_per_id += 1
    mult_refs[i] = num_refs_per_id

# Using the above arrays we calculate the multiplicity of each
# unique id in our final 'ids' array. The sum of these will provide
# the size of our final 'ids', 'hyp', and 'ref' lists.
nums_ids = [mult_cur_ref*mult_cur_hyp for mult_cur_ref, mult_cur_hyp in zip(mult_hyps,mult_refs)]
num_pairs = sum(nums_ids)

# print(mult_hyps)
# print(mult_refs)
# print(nums_ids)

# Create list 'ids' to store ids to be zipped with 'hyp', 'ref', and the calculated scores
ids = [""]*num_pairs
j = 0                                               # j is used to maintain the index into which we place current id
# For each id we use the value of nums_ids to determine how many times to place the id into 'ids'
for i, id in enumerate(unique_ids):
    for k in range(nums_ids[i]):
        ids[j+k] = id
    j += nums_ids[i]

print(ids)

# Create list 'hyp' to store hypotheses to be used for score calculation and create indices for use in populating it
hyp = [""]*num_pairs
cur_index = 0
j = 0
prevID = hyp_id[0]
# For each hypothesis we check the multiplicity it should have in hyp in order that matching samples line up with ref
print(hyp_id)
for i, cur_hyp in enumerate(hypotheses):
    if(hyp_id[i]!=prevID):
        j+=1
    cur_itr_mlt = nums_ids[j]
    if (cur_itr_mlt > 0):
        if mult_refs[j] > 1:  # We know that mult_refs[j] >= 1
            for k in range(mult_refs[j]):  # If mult_refs[j] > 1 add the current reference
                # cur_index = i + k  # k = mult_refs[j] times
                hyp[cur_index] = cur_hyp
                cur_index += 1
        else:
            hyp[cur_index] += cur_hyp  # Otherwise add the current hypothesis once
            cur_index += 1
        prevID = hyp_id[i]
        cur_itr_mlt -= mult_refs[j]



print(hyp)
# Create list 'ref' to store references to be used for score calculation and zero indices for use in populating it
ref = [""]*num_pairs
cur_index = 0
j = 0

# For each reference we check the multiplicity it should have in ref in order that matching samples line up with hyp
print(ref_id)
prevID = ref_id[0]
for i, cur_ref in enumerate(references):
    if (ref_id[i] != prevID):
        j += 1
    cur_itr_mlt = nums_ids[j]
    if (cur_itr_mlt > 0):
        if mult_hyps[j] > 1:  # We know that mult_hyps[j] >= 1
            for k in range(mult_hyps[j]):  # If mult_hyps[j] > 1 add the current reference
                ref[cur_index] = cur_ref  # k = mult_hyps[j] times
                cur_index += 1
        else:
            ref[cur_index] += cur_ref  # Otherwise add the current reference once
            cur_index += 1
        prevID = ref_id[i]
        cur_itr_mlt -= mult_hyps[j]



print(ref)

num_compares = len(ids)

# Bleu Scores

bleu_C_scores = [0.0]*num_compares
bleu_1_scores = [0.0]*num_compares
bleu_2_scores = [0.0]*num_compares
bleu_3_scores = [0.0]*num_compares
bleu_4_scores = [0.0]*num_compares

for i in range(num_compares):                       # Ultimately this should group references into cur_refs as
    cur_ref = [ref[i].split(" ")]                   # [["", "",...,""],["", "",...,""],...,["", "",...,""]]
    cur_hyp = hyp[i].split(" ")
    bleu_1_scores[i] = sentence_bleu(cur_ref, cur_hyp, weights=(1, 0, 0, 0))
    bleu_2_scores[i] = sentence_bleu(cur_ref, cur_hyp, weights=(0, 1, 0, 0))
    bleu_3_scores[i] = sentence_bleu(cur_ref, cur_hyp, weights=(0, 0, 1, 0))
    bleu_4_scores[i] = sentence_bleu(cur_ref, cur_hyp, weights=(0, 0, 0, 1))
    bleu_C_scores[i] = sentence_bleu(cur_ref, cur_hyp)

# print(bleu_1_scores)
# print(bleu_2_scores)
# print(bleu_3_scores)
# print(bleu_4_scores)
# print(bleu_C_scores)

# CIDEr Scores

# hyp_dict = {}
# k = 0
# for i, id in enumerate(unique_ids):
#     accStr = ""
#     print(mult_hyps[i])
#     for j in range(mult_hyps[i]):
#         print(hyp[k+j])
#         accStr += hyp[k+j] + "|"
#     k += mult_refs[i]
#     accStr = accStr[: -1]
#     entry = accStr.split("|")
#     hyp_dict[id] = numpy.unique(entry).tolist()
#
# ref_dict = {}
# k = 0
# for i, id in enumerate(unique_ids):
#     accStr = ""
#     print(mult_refs[i])
#     for j in range(mult_refs[i]):
#         print(ref[k+j])
#         accStr += ref[k+j] + "|"
#     k += mult_hyps[i]
#     accStr = accStr[: -1]
#     entry = accStr.split("|")
#     ref_dict[id] = numpy.unique(entry).tolist()

hyp_dict = {}
ref_dict = {}

for i, id in enumerate(ids):
    hyp_dict[id+str(i)] = [hyp[i]]
    ref_dict[id+str(i)] = [ref[i]]

# print(hyp_dict)
# print(ref_dict)

cider_eval = Cider()
cider_score = cider_eval.compute_score(ref_dict, hyp_dict)
print('Cider Average: %f' % (cider_score[0]/10))
cider_scores_indiv = cider_score[1].tolist()
# print(cider_scores_indiv)

# Meteor scores
gateway = JavaGateway()
meteor_eval = gateway.entry_point

meteor_scores = [0.0]*num_compares
for i in range(num_compares):
    meteor_scores[i] = meteor_eval.compute_score(hyp[i],ref[i])

# print(meteor_scores)

# Rouge Scores
rouge_eval = rouge.Rouge()
rouge_scores_dict_list = rouge_eval.get_scores(hyp,ref)
print(rouge_scores_dict_list)
rouge_1_scores = [0.0]*num_compares
rouge_2_scores = [0.0]*num_compares
rouge_l_scores = [0.0]*num_compares

for i, s in enumerate(rouge_scores_dict_list):
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

out_data = zip(ids, hyp, ref, bleu_C_scores, bleu_1_scores, bleu_2_scores, bleu_3_scores, bleu_4_scores, cider_scores_indiv, meteor_scores, rouge_1_scores, rouge_2_scores, rouge_l_scores)
output = list(out_data)
# print(output)
out_head = ["id", "hypothesis", "reference","bleu-C", "bleu-1", "bleu-2", "bleu-3", "bleu-4", "CIDEr", "Meteor", "Rouge-1", "Rouge-2", "Rouge-l"]
with open('output.csv', 'w+', newline='') as out_file:
    csvWriter = csv.writer(out_file, delimiter = ',')
    csvWriter.writerow(out_head)
    csvWriter.writerows(output)