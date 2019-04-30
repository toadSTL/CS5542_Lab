import json

input_file = open ('Data/v2_OpenEnded_mscoco_train2014_questions.json')
json_array = json.load(input_file)
store_list = []

ids = []
# 'Data/imgIds_train.txt'
# 'Data/imgIds_val.txt'
filepath = 'Data/imgIds_val.txt'
with open(filepath, "r") as ins:
    array = []
    for line in ins:
        ids.append(int(line.rstrip('\n')))

cnt = 0
for item in json_array['questions']:
    store_details = {"image_id":None, "question":None, "question_id":None}
    store_details['image_id'] = int(item['image_id'])
    store_details['question'] = item['question']
    store_details['question_id'] = item['question_id']
    # print(store_details['image_id'])
    if store_details['image_id'] in ids:
        store_list.append(store_details)
        cnt = cnt + 1

#print(store_list)
output = {
    "info": {
        "description": "This is v2.0 of the VQA dataset.",
        "url": "http://visualqa.org",
        "version": "2.0",
        "year": 2017,
        "contributor": "VQA Team",
        "date_created": "2017-04-26 17:07:13"},
    "task_type": "Open-Ended",
    "data_type": "mscoco",
    "license": {
        "url": "http://creativecommons.org/licenses/by/4.0/",
        "name": "Creative Commons Attribution 4.0 International License"},
    "data_subtype": "train2014",
    "questions": store_list
    }
# 'Data/questions_train.json'
# 'Data/questions_val.json'
with open('Data/questions_val.json', 'w') as outfile:
    json.dump(output, outfile)

print(cnt)