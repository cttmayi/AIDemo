

# 安装

安装 Appium

    npm i -g appium

安装Android SDK，并设置 ANDROID_HOME 环境变量

安装 Driver

    appium driver install uiautomator2

# 运行

运行 Appium

    appium

# 权限

在 Android 设备上，需要授予 Appium 权限。

打开开发者选项
    设置 -> 开发者选项 -> 打开 “USB 调试”

OPPO 手机
    设置 -> 开发者选项 -> 打开 “禁止权限监控”

# 参考

https://appium.io/docs/zh/2.5/quickstart/

# Android 命令

获取APP信息

获取当前界面元素：adb shell dumpsys activity top
获取任务列表：adb shell dumpsys activity activities

App入口

adb logcat |grep -i displayed
aapt dump badging mobike.apk | grep launchable-activity

启动应用
adb shell am start -W -n com.qw.amobile/.splash.SplashActivity -S
回顾adb基本命令

adb devices：查看设备
adb kill-server：关闭 adb 的后台进程
adb tcpip：让 Android 脱离 USB 线的 TCP 连接方式
adb connect：连接开启了 TCP 连接方式的手机
adb logcat：Android 日志查看
adb bugreport：收集日志数据，用于后续的分析，比如耗电量

adb shell
adb shell 本身就是一个 Linux 的 shell，可以调用 Android 内置命令

adb shell dumpsys
adb shell pm
adb shell am
adb shell ps
adb shell monkey

性能统计

获取所有的 dumpsys 子命令 dumpsys | grep -i DUMP
获取当前 activity adb shell dumpsys activity top
获取 activities 的记录，可以获取到 appium 依赖的原始 activity dumpsys activity activities
获取特定包基本信息 adb shell dumpsys package com.xueqiu.android
获取系统通知 adb shell dumpsys notifification
获得内存信息 adb shell dumpsys meminfo com.android.settings
获取 cpu 信息 adb shell dumpsys cpuinfo
获取 gpu 绘制分析 adb shell dumpsys gfxinfo com.android.settings
获取短信 adb shell dumpsys activity broadcasts | grep senderName=uiautomator

作者：郝同学的测开笔记
链接：https://juejin.cn/post/7314947185866047539
来源：稀土掘金
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。