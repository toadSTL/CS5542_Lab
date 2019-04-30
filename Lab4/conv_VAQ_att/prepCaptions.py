import json
import imageio

ids = []
# 'Data/imgIds_train.txt'
# 'Data/imgIds_val.txt'
filepath = 'Data/imgIds_val.txt'
with open(filepath, "r") as ins:
    array = []
    for line in ins:
        ids.append(line.rstrip('\n'))

store_list = []

for id in ids:
    store_details = {"file_name": None, "width": None, "height": None, "id": None}
    store_details['id'] = int(id)
    idstr = "{0:0>12}".format(id)
    file_name =  "COCO_train2014_"+idstr+".jpg"
    store_details['file_name'] = file_name
    # "Data/val2014/
    img = imageio.imread("Data/train2014/"+file_name).shape
    # print(img)
    height = img[0]
    width = img[1]
    store_details['height'] = height
    store_details['width'] = width

    store_list.append(store_details)

print(len(ids))

output = {
    "info": {
        "description": "This is stable 1.0 version of the 2014 MS COCO dataset.",
        "url": "http://mscoco.org",
        "version": "1.0",
        "year": 2014,
        "contributor": "Microsoft COCO group",
        "date_created": "2015-01-27 09:11:52.357475"},
    "images": store_list
    }

#print(store_list)
# 'Data/captions_train.json'
# 'Data/captions_val.json'
with open('Data/captions_val.json', 'w') as outfile:
    json.dump(output, outfile)
