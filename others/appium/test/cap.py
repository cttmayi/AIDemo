
from appium import webdriver
import os
import random
import cv2

#from appium.webdriver.common.touch_action import TouchAction
import time
import numpy as np


desired_caps = {
    'platformName': 'Android',  # 操作系统
    #'deviceName': 'emulator-5554',
    #'platformVersion': '9',  # 设备版本号
    #'appPackage': 'com.netease.onmyoji.bili',  # app 包名
    #'appActivity': 'com.netease.ntunisdk.external.protocol.ProtocolLauncher',  # app 启动时主 Activity

    'appPackage': 'com.android.calculator2',
    'appActivity': '.Calculator',


    'noReset': True,  # 是否保留 session 信息，可以避免重新登录
    'skipServerInstallation': True,  # 避免重复安装
}

driver = webdriver.Remote('http://localhost:4723', desired_caps)

class Cap:
    def __init__(self):
        self.driver = driver
        #self.action = TouchAction(driver)
        self.SignInBoolean = False
        # 根据阈值判断是否匹配成功
        self.threshold = 0.8
        # 获取当前脚本文件的绝对路径
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        # 定义相对路径和文件名
        self.screenshot_path = 'screenshot/example.png'
        # 目标文件路径
        self.TargetPhotoGallery_path = 'TargetPhotoGallery/'
        # 完整路径
        self.full_path = os.path.join(self.script_dir, self.screenshot_path)
        
    #图像识别
    def imageRecognition(self, template):
        n = 0
        circulation = True
        while circulation:
            self.driver.get_screenshot_as_file(self.full_path)
            # 加载源图像和模板图像
            img = cv2.imread(self.screenshot_path)
            # 执行图像识别
            result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
            loc = np.where(result >= self.threshold)
            matches = list(zip(loc[1], loc[0]))
            n += 1
            if len(loc[0]) > 0 and len(loc[1]) > 0:
                # 在元组中随机选择一个坐标
                randomcontact = random.choice(matches)
                self.action.tap(x=str(randomcontact[0]), y=str(randomcontact[1])).perform()
                circulation = False
            else:
                time.sleep(0.5)
                if (n == 4):
                    circulation = False


cap = Cap()

cap.imageRecognition(cv2.imread('TargetPhotoGallery/1.png'))