import cv2
import matplotlib.pyplot as plt

# 画像の読み込み
img = cv2.imread('test.jpg') 
# カラーデータの色空間の変換 
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) 

# 画像の表示
plt.show()