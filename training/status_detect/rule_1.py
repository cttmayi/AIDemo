# 定义合理的状态转换规则
valid_transitions = {
    "Running": ["Paused", "Restarting"],
    "Paused": ["Running"],
    "Restarting": ["Running"]
}

# 检查状态转换是否合理
def check_transition(current_state, next_state):
    if next_state not in valid_transitions.get(current_state, []):
        return False
    return True

# 示例日志
logs = [
    ("2025-05-23 00:00:00", "Running", "设备正常运行"),
    ("2025-05-23 00:05:00", "Paused", "设备暂停"),
    ("2025-05-23 00:10:00", "Running", "设备恢复运行"),
    ("2025-05-23 00:15:00", "Restarting", "设备重启"),
    ("2025-05-23 00:20:00", "Running", "设备正常运行"),
    ("2025-05-23 00:25:00", "Fault", "设备故障"),  # 异常状态
    ("2025-05-23 00:30:00", "Restarting", "设备重启")  # 异常转换
]

# 检查每个状态转换
for i in range(len(logs) - 1):
    current_state = logs[i][1]
    next_state = logs[i + 1][1]
    if not check_transition(current_state, next_state):
        print(f"异常状态转换：从 {current_state} 到 {next_state}")