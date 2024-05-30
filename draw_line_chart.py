import os
import numpy as np
import matplotlib.pyplot as plt

root = 'good_data_clear'
files = os.listdir(root)

for file in files:
    fi = open(os.path.join(root, file), 'r')
    file_lines = fi.readlines()
    ms_list = []
    origin_list = []
    smooth_list = []
    target_list = []

    for line in file_lines:
        line = line.strip() # 参数为空时，默认删除开头、结尾处空白符（包括'\n', '\r',  '\t',  ' ')
        formLine = line.split('\t')
        ms_list.append(float(formLine[1])) # ms
        origin_list.append(float(formLine[2])) # origin
        smooth_list.append(float(formLine[3])) # smooth
        target_list.append(float(formLine[4]))

    plt.cla()
    x = np.arange(0, len(ms_list))
    l1 = plt.plot(x,ms_list, 'ro-', label = 'Ours')
    l2 = plt.plot(x,origin_list, 'g+-', label = 'ResNet BaseLine')
    l3 = plt.plot(x,smooth_list, 'b^-', label = 'Ours + Smooth')
    l4 = plt.plot(x, target_list, 'k2-', label = 'Target')
    # plt.plot(x, ms_list, 'ro-', x, origin_list, 'g+-', x, smooth_list,'b^-')
    prefix = os.path.splitext(file)[0]
    plt.title(prefix)
    plt.xlabel('Time Frame (One per 3h)')
    plt.ylabel('TC Intensity(kt)')
    plt.legend()
    plt.savefig(prefix + '.jpg')