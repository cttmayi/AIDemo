import unittest
from appium import webdriver
from appium.options.android import UiAutomator2Options
from appium.webdriver.common.appiumby import AppiumBy
import time

capabilities = dict(
    platformName='Android',
    # automationName='uiautomator2',
    #deviceName='Android',
    appPackage='com.android.calculator2',
    appActivity='.Calculator',
    #language='en',
    #locale='US'
)

appium_server_url = 'http://localhost:4723'

class TestAppium(unittest.TestCase):
    def setUp(self) -> None:
        self.driver = webdriver.Remote(appium_server_url, options=UiAutomator2Options().load_capabilities(capabilities))
        # time.sleep(5)

    def tearDown(self) -> None:
        if self.driver:
            time.sleep(4)
            self.driver.quit()

    def test_find_battery(self) -> None:
        # print(self.driver.page_source)

        page_source = self.driver.page_source
        

        els = self.driver.find_elements(by=AppiumBy.XPATH, value='//*[@text="同意"]')
        els = self.driver.find_elements(by=AppiumBy.XPATH, value='//*[@clickable="true"]')
        # els = self.driver.find_elements(by=AppiumBy.XPATH, value='//*[@scrollable="true"]')
        # els = self.driver.find_elements(by=AppiumBy.XPATH, value='//*')

        for i, el in enumerate(els):
            print(i, el.text)
        #el.click()

if __name__ == '__main__':
    unittest.main()