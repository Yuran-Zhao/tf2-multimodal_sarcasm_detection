import os
import pickle
from collections import Counter

work_path = "D:\\Programming_Projects\\From_or_To_GitHub\\pytorch-multimodal_sarcasm_detection\\multilabel_database"

with open(
    os.path.join(work_path, "img_to_five_words.txt"), "r", encoding="utf-8"
) as fin:
    lines = fin.readlines()

dic = set()
cnt = Counter()

for line in lines:
    content = eval(line)
    for word in content[1:]:
        # cnt[word] += 1
        dic.add(word)

dic = list(dic)
dic_file = open(os.path.join(work_path, "attribute_list.pickle"), "wb")
pickle.dump(dic, dic_file)
dic_file.close()
