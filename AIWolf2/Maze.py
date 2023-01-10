import os

# 迷路を作成する
import numpy as np
import random

width = 20
height = 20
cell = []
start_cells = [] # 壁を作り始める座標のリスト
start_pos = []
goal_pos = []



def extend_wall(x ,y):
  #  ここで伸ばすことができる方向を取得する(1マス先が通路で2マス先まで範囲内)
  #  2マス先が壁で自分自身の場合、伸ばせない
  directions = [] # 壁を伸ばす方向のリスト
  if (maze[x, y - 1] == ' ' and maze[x, y - 2] != '#'):
      directions.append("direction_up");
  if (maze[x + 1, y] == ' ' and maze[x + 2, y] != '#'):
      directions.append("direction_right");
  if (maze[x, y + 1] == ' ' and maze[x, y + 2] != '#'):
      directions.append("direction_down");
  if (maze[x - 1, y] == ' ' and maze[x - 2, y] != '#'):
      directions.append("direction_left");

  for i in range(len(directions)): 
    set_wall(x, y) # 現在いる場所に#を入れる
    is_path = False
    selected_dir = random.choice(directions)
    if selected_dir == "direction_up":
      y -= 1 # 一個上を#
      set_wall(x, y)
      y -= 1 # さらに一個上を#
      set_wall(x, y)
    if selected_dir == "direction_right":
      x += 1
      set_wall(x, y)
      x += 1
      set_wall(x, y)
    if selected_dir == "direction_down":
      y += 1
      set_wall(x, y)
      y += 1
      set_wall(x, y)
    if selected_dir == "direction_left":
      x -= 1
      set_wall(x, y)
      x -= 1
      set_wall(x, y)
    
def set_wall(x, y):
  maze[x][y] = '#'

def set_start():
  _index = []
  _index = random.choice(start_pos)
  maze[_index[0], _index[1]] = 'S'
  # print(_index) Sの座標

def set_goal():
  _index = []
  _index = random.choice(goal_pos)
  maze[_index[0], _index[1]] = 'G'
  # print(_index) Gの座標


maze = np.zeros(width*height , str).reshape(width, height)

for y in range(height):
  for x in range(width):
    if (x == 0 or y == 0 or x == width-1 or y == height-1):
      maze[x][y] = '#'
    else:
      maze[x][y] = ' '
      if x % 2 == 0 and y % 2 == 0:
        start_cells.append([x, y])

for i in range(len(start_cells)):
  index = random.choice(start_cells)
  start_cells.remove(index)
  cell_x = index[0]
  cell_y = index[1]

  if maze[cell_x][cell_y] == ' ':
    extend_wall(cell_x, cell_y)

for y in range(int(height/2)):
  for x in range(int(width/2)):
    if maze[x][y] == " ":
      start_pos.append([x, y])

for y in range(height - int(height/2), height):
  for x in range(width - int(width/2), width):
    if maze[x][y] == " ":
      goal_pos.append([x, y])

set_start()
set_goal()

np.savetxt("maze.txt", maze, fmt="%s")