import os
import locale
import csv
from tkinter import N

n = 0

def tail(path):
    file = open(path, "r", encoding=locale.getdefaultlocale()[1])

    data = file.readlines()
    datalist = [list(map(str,line.strip().split(','))) for line in data[0:]]
    for num in range(15):
        if datalist[num][5] == 'Kadai':
            myrole =  datalist[num][3]
            print(num)
            break
        num += 1

    file.close()
    return myrole


#[list(map(str,line.strip().split(','))) for line in data[-1:]]

path = os.path.join('D:','\AiWolfResearch','012.log')

print(tail(path))

