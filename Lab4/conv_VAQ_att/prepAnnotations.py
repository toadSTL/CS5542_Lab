import json

input_file = open ('Data/v2_mscoco_train2014_annotations.json')
json_array = json.load(input_file)
store_list = []

ids = []
# 'Data/imgIds_train.txt'
# 'Data/imgIds_val.txt'
filepath = 'Data/imgIds_val.txt'
with open(filepath, "r") as ins:
    array = []
    for line in ins:
        ids.append(line.rstrip('\n'))

cnt = 0
for item in json_array['annotations']:
    if str(item['image_id']) in ids:
        store_list.append(item)
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
    "license": {
        "url": "http://creativecommons.org/licenses/by/4.0/",
        "name": "Creative Commons Attribution 4.0 International License"},
    "data_subtype":
        "train2014",
    "annotations": store_list
    }
# 'Data/annotations_train.json'
# 'Data/annotations_val.json'
with open('Data/annotations_val.json', 'w') as outfile:
    json.dump(output, outfile)

print(cnt)