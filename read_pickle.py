import pickle
import pprint
import sys
import numpy as np
        
avail_list = [0, 20, 22, 28, 33, 39, 44]

file_name = sys.argv[1]
result_file_name = "result_pickle.txt"
result_file = open(result_file_name, 'w')

result_file.write("-----------------------------------------------\n")
for file_name in sys.argv[1:]:
    try:
        with open(file_name, "rb") as f:
            data = pickle.load(f)
    except:
        continue

    result_file.write("{}\n".format(file_name))
    result_file.write("\n")

    # pprint.pprint(data)

    avail = []
    for i in list(data.keys()):
        if data[i][1] >= 0.01:
            result_file.write("{} {} {}\n".format(i, data[i][0], data[i][1]))
            avail.append(data[i][1])
        elif i in avail_list:
            result_file.write("{} {} {}\n".format(i, data[i][0], data[i][1]))
            avail.append(data[i][1])

    result_file.write("\n")
    # result_file.write("suc task = {}\n".format(len(avail)))
    result_file.write("avg = {}\n".format(np.mean(avail)))

    result_file.write("-----------------------------------------------\n")

result_file.close()
with open(result_file_name, 'r') as rf:
    for l in rf.readlines():
        print(l[:-1])
