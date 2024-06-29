import unittest
from appium import webdriver
from appium.options.android import UiAutomator2Options
from appium.webdriver.common.appiumby import AppiumBy
import time

capabilities = dict(
    platformName='Android',
    automationName='uiautomator2',
    deviceName='Android',
    #appPackage= 'com.android.bbkcalculator',# 'com.android.calculator2',
    appPackage= 'com.android.calculator2',
    appActivity='.Calculator',
    #language='en',
    #locale='US'
)

appium_server_url = 'http://localhost:4723'

class Cal():
    def __init__(self):
        self.driver = webdriver.Remote(appium_server_url, options=UiAutomator2Options().load_capabilities(capabilities))
        # time.sleep(5)

    def quit(self):
        if self.driver:
            self.driver.quit()

    def status(self):
        # page_source = self.driver.page_source
        # els = self.driver.find_elements(by=AppiumBy.XPATH, value='//*[@text="同意"]')
        # els = self.driver.find_elements(by=AppiumBy.XPATH, value='//*[@clickable="true"]')
        # els = self.driver.find_elements(by=AppiumBy.XPATH, value='//*[@scrollable="true"]')
        els = self.driver.find_elements(by=AppiumBy.XPATH, value='//*')

        output = []
        for i, el in enumerate(els):
            if el.get_attribute('text') != '' or el.get_attribute('clickable') == 'true':
                output.append(
                    {
                        'text': el.get_attribute('text'),
                        'resource-id': el.get_attribute('resource-id'),
                    }
                )
                print(i, el.get_attribute('text'), el.get_attribute('resource-id'))

        return str(output)


    def click(self, ID) -> None:
        # print(self.driver.page_source)
        el = self.driver.find_element(by=AppiumBy.ID, value=ID)
        print("CLICK", el.get_attribute('text'), el.get_attribute('resource-id'))
        el.click()
        return self.status()
