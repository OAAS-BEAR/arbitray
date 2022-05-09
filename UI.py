import sys

import numpy as np
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import QIcon, QAction,QPixmap,QImage
import os
import cv2
from connect import *

class Main(QWidget):
    def __init__(self):
        super().__init__()
        self.timer= QTimer()
        self.camera=cv2.VideoCapture()
        self.CAM_NUM=0
        self.transform_flag=0
        self.initUI()
        self.filepath='style/starry_night.jpg'
        self.host='47.114.94.239'
        self.port=3389
        self.buf_size=3000000
        self.s=0
        self.timer.timeout.connect(self.show_frame)

    def initUI(self):
        self.btn1=QPushButton("选择风格图片",self)
        self.btn2=QPushButton("开启摄像头",self)
        self.btn3=QPushButton("开启风格化",self)
        self.btn4=QPushButton("退出应用",self)
        self.btn1.clicked.connect(self.choose_style)
        self.btn2.clicked.connect(self.open_camera)
        self.btn3.clicked.connect(self.open_transform)
        self.btn4.clicked.connect(self.shut_app)
        self.btn1.move(30, 600)
        self.btn2.move(200, 600)
        self.btn3.move(450, 600)
        self.btn4.move(700, 600)
        self.show_camera=QLabel(self)
        self.show_camera.setFixedSize(641,481)
        self.show_camera.setAutoFillBackground(False)

        self.setGeometry(300,300,1000,800)
        stylesheet = (
            "background-color:black"
        )
        self.setStyleSheet(stylesheet)
        self.setWindowTitle('实时视频风格迁移系统')
        self.show()
    def choose_style(self):
        self.filepath=QFileDialog.getOpenFileName(self,"XUANQU ",'style/')[0]
        print(self.filepath)
        if(self.transform_flag==1):
            self.open_transform()
            self.open_transform()
    def shut_app(self):
        self.close()
    def open_camera(self):
        if self.timer.isActive()==False:
            self.camera.open(self.CAM_NUM)
            self.timer.start(200)
            self.btn2.setText('关闭设摄像头')
        else:
            self.timer.stop()
            self.camera.release()
            self.show_camera.clear()
            self.btn2.setText('开启摄像头')

    def open_transform(self):
        if(self.transform_flag==0):
            self.timer.stop()
            self.s=connect(self.host,self.port)
            print(self.filepath)
            style_image=cv2.imread(self.filepath)

            print(style_image)
            style_image = cv2.resize(style_image, (256, 256))
            print(len(style_image.tobytes()))
            self.s.send(style_image.tobytes())
            self.btn3.setText('关闭风格化')
            self.transform_flag = 1
            self.timer.start(200)
        else:
            self.transform_flag = 0
            self.s.close()
            self.btn3.setText('开启风格化')

    def show_frame(self):
        ret,image=self.camera.read()

        if(self.transform_flag==0):
            #print(len(image.tobytes()))
            image = cv2.resize(image, (640, 480))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            QT_image=QImage(image.data.tobytes(),image.shape[1],image.shape[0],QImage.Format.Format_RGB888)
        else:
            image = cv2.resize(image, (256, 256))
            self.s.send(image.tobytes())
            t_image=bytes()
            while 1:
                tt_image=self.s.recv(self.buf_size)
                if(len(tt_image)!=0):
                    t_image+=tt_image
                    if(len(t_image)==196608):
                        break
            t_image=np.frombuffer(t_image,dtype=np.uint8).reshape([256,256,3])

            image = cv2.resize(t_image, (640, 480))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            QT_image=QImage(image.data.tobytes(),image.shape[1],image.shape[0],QImage.Format.Format_RGB888)


        self.show_camera.setPixmap(QPixmap.fromImage(QT_image))

def main():
    app=QApplication(sys.argv)
    example=Main()
    sys.exit(app.exec())
if __name__=='__main__':
    main()
