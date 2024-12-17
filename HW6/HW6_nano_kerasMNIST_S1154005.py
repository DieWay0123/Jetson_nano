from re import split
import tensorflow.keras.utils
import numpy as np
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.keras.preprocessing import image
from matplotlib import pyplot as plt
from PIL import Image
import logging
import os

LOGGING_FORMAT ='%(asctime)s [%(levelname)s]:%(message)s'
logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT, filename='Log.log', filemode='a')

#獲取資料集
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

#建立並增加模型層
model = Sequential()
model.add(Dense(units=256, input_dim=784, kernel_initializer='normal', activation='relu'))
model.add(Dropout(rate = 0.2))
model.add(Dense(units=64, kernel_initializer='normal', activation='relu'))
model.add(Dense(units=10, kernel_initializer='normal', activation='softmax'))

# 編譯: 選擇損失函數、優化方法及成效衡量方式
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#資料處理、正規化
Y_trainOneHot = to_categorical(Y_train)
Y_testOneHot = to_categorical(Y_test)

X_train_2D = X_train.reshape(X_train.shape[0], 28*28).astype('float32')
X_test_2D = X_test.reshape(X_test.shape[0], 28*28).astype('float32')

X_train_norm = X_train_2D/255
X_test_norm = X_test_2D/255

#train
train_history = model.fit(x=X_train_norm, y=Y_trainOneHot, validation_split=0.1,epochs=120, batch_size=10000, verbose=2)

scores = model.evaluate(X_test_norm, Y_testOneHot)

print("\t[Info] Accuracy of testing data = {:2.1f}%".format(scores[1]*100.0)) 
#logging.info("Accuracy of testing data = {:2.1f}%".format(scores[1]*100.0))

#繪製loss分析圖
plt.figure(1)
plt.plot(train_history.history['loss'])  
plt.plot(train_history.history['val_loss'])  
plt.title('Train History')  
plt.ylabel('loss')  
plt.xlabel('Epoch')  
plt.legend(['loss', 'val_loss'], loc='upper left')  

#預測學號
img = tensorflow.keras.preprocessing.image.load_img('1154005.bmp', color_mode='grayscale')
img_number = []

#處理讀取進來的學號圖片
for i in range(0, 7):
  x = i*28
  out = os.path.join(os.getcwd(), f'{i*28}_{x}.png')
  splitted_img = img.crop((x, 0, x+28, 28))
  img_number.append(np.array(splitted_img).flatten()/255)
  splitted_img.save(f'{i*28}_{x+28}.png')

#show手寫圖片
plt.figure(2)
for i in range(len(img_number)):
  img = np.reshape(img_number[i], (28, 28))
  plt.subplot(1, 7, i+1)
  plt.imshow(img, cmap='gray')
plt.suptitle("Input handwrite numbers")
plt.show()

#預測手寫學號數字
img_number = np.array(img_number)
predictions = model.predict(img_number)
ans = np.argmax(predictions, axis=1)



#輸出預測結果
id = str.join("", map(str,ans))
print(f"你的手寫學號經過預測是:S{id}")
