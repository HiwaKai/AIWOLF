import os
import locale
import csv

#"D:\AiWolfResearch\000.log"
def tail(path):
    file = open(path, "r", encoding=locale.getdefaultlocale()[1])

    data = file.readlines()
    datalist = [list(map(str,line.strip().split(','))) for line in data[0:]]
    for num in range(15):
        if datalist[num][5] == 'ichida':
            myroleAgent =  datalist[num][3]
            #print(myroleAgent)
            break

    file.close()
    return myroleAgent

def first(path):
    file = open(path, "r", encoding=locale.getdefaultlocale()[1])

    first = file.readlines()
    file.close()
    return [list(map(str,line.strip().split(','))) for line in first[-1:]]

pathtext = 'D:'
pathtext2 = '\AiWolfResearch'
firstText = ['Game', 'MyAgent','Victory','result']

resultText = ['VILLAGER', 'SEER', 'MEDIUM', 'BODYGUARD', 'POSSESSED', 'WEREWOLF']
resultroleText = ['VILLAGER', 'WEREWOLF']

VillagerRole = ['BODYGUARD', 'MEDIUM', 'SEER', 'VILLAGER']
WerewolfRole = ['POSSESSED', 'WEREWOLF']

roleVillagercount = 0
roleWerewolfcount = 0

Villagercount = 0
SEERcount = 0
BodyGuardCount = 0
mediumCount = 0

VillagerAllcount = 0
SEERAllcount = 0
BodyGuardAllCount = 0
mediumAllCount = 0

possessedCount = 0
werewolfCount = 0

possessedAllCount = 0
werewolfAllCount = 0

myrole = 'VILL'

#ここからmain
#書き込みようのファイルを開く
csv_path = os.path.join('D:','\AiWolfResearch','Result','ichida','Game14.csv')
csv_file = open(csv_path,"a", newline="")
writer = csv.writer(csv_file)
writer.writerow(firstText)

for num in range(100):
    filename = str(num).zfill(3) + '.log'
    path = os.path.join(pathtext, pathtext2, 'GameLog_15Player','20220709','Game14', filename)

    myAgent = tail(path) #１行目取得
    resultdata = first(path) #最後の行取得

    if myAgent == 'VILLAGER':
        VillagerAllcount += 1
    elif myAgent == 'SEER':
        SEERAllcount += 1
    elif myAgent == 'MEDIUM':
        mediumAllCount += 1
    elif myAgent == 'BODYGUARD':
        BodyGuardAllCount += 1
    elif myAgent == 'WEREWOLF':
        werewolfAllCount += 1
    elif myAgent == 'POSSESSED':
        possessedAllCount += 1

    if myAgent in VillagerRole:
        myrole = 'VILLAGER'
    elif myAgent in WerewolfRole:
        myrole = 'WEREWOLF'
    
    if myrole == resultdata[0][4]:
        result = 1 #勝利なら1

        #役職ごとの勝利回数
        if myAgent == 'VILLAGER':
            Villagercount += 1
        elif myAgent == 'SEER':
            SEERcount += 1
        elif myAgent == 'MEDIUM':
            mediumCount += 1
        elif myAgent == 'BODYGUARD':
            BodyGuardCount += 1
        elif myAgent == 'WEREWOLF':
            werewolfCount += 1
        elif myAgent == 'POSSESSED':
            possessedCount += 1

        #陣営ごとの勝利回数
        if myrole == 'VILLAGER':
            roleVillagercount += 1
        elif myrole == 'WEREWOLF':
            roleWerewolfcount += 1
    else:
        result = 0 #敗北なら0

    resultlist = [str(num).zfill(3), myAgent, resultdata[0][4], result]
    writer.writerow(resultlist)

writer.writerow(resultText)
countresultAgent = [Villagercount, SEERcount, mediumCount, BodyGuardCount, werewolfCount, possessedCount]
countSelectAgent = [VillagerAllcount, SEERAllcount, mediumAllCount, BodyGuardAllCount, werewolfAllCount, possessedAllCount]

writer.writerow(countSelectAgent)
writer.writerow(countresultAgent)

countresultrole = [roleVillagercount, roleWerewolfcount]
writer.writerow(resultroleText)
writer.writerow(countresultrole)

csv_file.close()
