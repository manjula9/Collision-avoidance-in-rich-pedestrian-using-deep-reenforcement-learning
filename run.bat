import tkinter
from tkinter import *
import math
import random
from threading import Thread 
from collections import defaultdict
from tkinter import ttk
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM  #class for LSTM regression
from keras.layers import Dropout
from keras.models import model_from_json
import pickle
import os

global sensor, labels, sensor_x, sensore_y, text, canvas, sensor_list, root, num_nodes, tf1, nodes, simulation, collision, rewards
option = 0
collision = 0
rewards = 0

sc1 = MinMaxScaler(feature_range = (0, 1))
sc2 = MinMaxScaler(feature_range = (0, 1))

dataset = pd.read_csv("dataset.csv")
dataset.fillna(0, inplace = True)
print(dataset.head())
temp = dataset.values
Y = temp[:,2:3]
dataset.drop(['label'], axis = 1,inplace=True)
dataset = dataset.values

X = dataset[:,0:dataset.shape[1]]
X = sc1.fit_transform(X)
Y = sc2.fit_transform(Y)
#X = np.reshape(X, (X.shape[0], X.shape[1], 1))
print(X.shape)

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]

X = np.reshape(X, (X.shape[0], X.shape[1], 1))

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test

if os.path.exists('model/model.json'):
    with open('model/model.json', "r") as json_file:
        loaded_model_json = json_file.read()
        lstm_model = model_from_json(loaded_model_json)
    json_file.close()
    lstm_model.load_weights("model/model_weights.h5")
    lstm_model._make_predict_function()   
else:
    lstm_model = Sequential()
    lstm_model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2])))
    lstm_model.add(Dropout(0.3))
    lstm_model.add(LSTM(units = 50, return_sequences = True))
    lstm_model.add(Dropout(0.3))
    lstm_model.add(LSTM(units = 50, return_sequences = True))
    lstm_model.add(Dropout(0.3))
    lstm_model.add(LSTM(units = 50))
    lstm_model.add(Dropout(0.3))
    lstm_model.add(Dense(units = 1))
    lstm_model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    lstm_model.fit(X_train, y_train, epochs = 200, batch_size = 32, validation_data=(X_test, y_test))
    lstm_model.save_weights('model/model_weights.h5')            
    model_json = lstm_model.to_json()
    with open("model/model.json", "w") as json_file:
        json_file.write(model_json)
    json_file.close()

def getDistance(sensor_x,sensor_y,x1,y1):
    flag = False
    for i in range(len(sensor_x)):
        dist = math.sqrt((sensor_x[i] - x1)**2 + (sensor_y[i] - y1)**2)
        if dist < 40:
            flag = True
            break        
    return flag

def runGA3C():
    global sensor, labels, sensor_x, sensor_y, text, canvas, option, simulation
    global collision, rewards
    class GASimulation(Thread):
        global collision, rewards
        def __init__(self, sensor, labels, sensor_x, sensor_y, text, canvas, option): 
            Thread.__init__(self)
            self.lstm_model = None
            self.canvas = canvas
            self.text = text
            self.labels = labels
            self.sensor_x = sensor_x
            self.sensor_y = sensor_y
            self.sensor = sensor
            self.option = option

        def getDistance(self, sensor_x,sensor_y, x1, y1, moving):
            distance = 30000
            for i in range(len(sensor_x)):
                if i != moving:
                    dist = math.sqrt((sensor_x[i] - x1)**2 + (sensor_y[i] - y1)**2)
                    if dist < distance:
                        distance = dist
            return distance    

        def getLSTMPredict(self, x, y):
            if self.lstm_model is None:
                with open('model/model.json', "r") as json_file:
                    loaded_model_json = json_file.read()
                    self.lstm_model = model_from_json(loaded_model_json)
                json_file.close()
                self.lstm_model.load_weights("model/model_weights.h5")
                self.lstm_model._make_predict_function()  
            temp = []
            temp.append([x, y])
            temp = np.asarray(temp)
            temp = sc1.transform(temp)
            temp = np.reshape(temp, (temp.shape[0], temp.shape[1], 1))
            predict = self.lstm_model.predict(temp)
            predict = sc2.inverse_transform(predict)
            predict = predict.ravel()
            return predict[0]
 
        def stopMovement(self):
            self.option = 1

        def run(self):
            global collision, rewards
            collision = 0
            rewards = 0
            previous_state = None
            while self.option == 0:
                if previous_state is not None:
                    moving = previous_state
                else:
                    moving = random.randint(0, len(self.sensor)-1)
                self.canvas.delete(self.labels[moving])
                self.canvas.delete(self.sensor[moving])
                self.canvas.update()
                x = random.randint(100, 450)
                y = random.randint(50, 600)
                nodes[moving] = [x, y]
                self.sensor_x[moving] = x
                self.sensor_y[moving] = y
                dist = self.getDistance(self.sensor_x, self.sensor_y, x, y, moving)
                lstm_predict = self.getLSTMPredict(x, y)
                print(str(lstm_predict)+" "+str(dist)+" "+str(moving))
                if lstm_predict >= 40 and dist >= 40:
                    previous_state = None
                    name = self.canvas.create_oval(x,y,x+40,y+40, fill="blue")
                    lbl = self.canvas.create_text(x+20,y-10,fill="darkblue",font="Times 8 italic bold",text="P "+str(moving))
                    rewards = rewards + 2                    
                else:
                    previous_state = moving
                    name = self.canvas.create_oval(x,y,x+40,y+40, fill="red")
                    lbl = self.canvas.create_text(x+20,y-10,fill="red",font="Times 8 italic bold",text="C "+str(moving))
                    collision = collision + 1
                self.labels[moving] = lbl
                self.sensor[moving] = name
                self.canvas.update()
                time.sleep(1)
                            
    simulation = GASimulation(sensor, labels, sensor_x, sensor_y, text, canvas, option) 
    simulation.start() 

def generateSimulation():
    global sensor, labels, sensor_x, sensor_y, num_nodes, tf1, nodes
    sensor = []
    sensor_x = []
    sensor_y = []
    labels = []
    nodes = []
    canvas.update()
    num_nodes = int(tf1.get().strip())
    
    for i in range(0,num_nodes):
        run = True
        while run == True:
            x = random.randint(100, 450)
            y = random.randint(50, 600)
            flag = getDistance(sensor_x,sensor_y,x,y)
            if flag == False:
                nodes.append([x, y])
                sensor_x.append(x)
                sensor_y.append(y)
                run = False
                name = canvas.create_oval(x,y,x+40,y+40, fill="blue")
                lbl = canvas.create_text(x+20,y-10,fill="darkblue",font="Times 8 italic bold",text="P "+str(i))
                labels.append(lbl)
                sensor.append(name)    

def stopSimulation():
    global simulation
    simulation.stopMovement()

def graph():
    global collision, rewards
    height = [collision, rewards]
    bars = ("Total Collision", "Total Rewards Earned")
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.xlabel("Rewards & Collision Graph")
    plt.ylabel("Total Collision & Rewards Earned")
    plt.title("Collision & Rewards Earned Graph")
    plt.show()

def Main():
    global root, tf1, text, canvas, mobile_list
    root = tkinter.Tk()
    root.geometry("1300x1200")
    root.title("Collision Avoidance in Pedestrian-Rich Environments with Deep Reinforcement Learning")
    root.resizable(True,True)
    font1 = ('times', 12, 'bold')

    canvas = Canvas(root, width = 800, height = 700)
    canvas.pack()

    l2 = Label(root, text='Num Nodes:')
    l2.config(font=font1)
    l2.place(x=820,y=10)

    tf1 = Entry(root,width=10)
    tf1.config(font=font1)
    tf1.place(x=970,y=10)

    createButton = Button(root, text="Generate Pedestrian Simulation", command=generateSimulation)
    createButton.place(x=820,y=60)
    createButton.config(font=font1)

    startButton = Button(root, text="Run GA3C-CADRL Simulation", command=runGA3C)
    startButton.place(x=820,y=110)
    startButton.config(font=font1)

    stopButton = Button(root, text="Stop Simulation", command=stopSimulation)
    stopButton.place(x=820,y=160)
    stopButton.config(font=font1)

    graphButton = Button(root, text="Collision Graph", command=graph)
    graphButton.place(x=820,y=210)
    graphButton.config(font=font1)

    text=Text(root,height=18,width=60)
    scroll=Scrollbar(text)
    text.configure(yscrollcommand=scroll.set)
    text.place(x=820,y=360)
    
    
    root.mainloop()
   
 
if __name__== '__main__' :
    Main ()
    
