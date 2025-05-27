from collections import defaultdict
import numpy as np

# 统计状态转换频率和持续时间
def analyze_logs(logs):
    transition_counts = defaultdict(int)
    state_durations = defaultdict(list)
    last_timestamp = None
    last_state = None

    for timestamp, state, _ in logs:
        if last_state is not None:
            transition_counts[(last_state, state)] += 1
            duration = (timestamp - last_timestamp)
            state_durations[last_state].append(duration)
        last_timestamp = timestamp
        last_state = state

    return transition_counts, state_durations

# 检测异常状态转换
def detect_anomalies(logs, transition_counts, state_durations):
    for i in range(len(logs) - 1):
        current_state = logs[i][1]
        next_state = logs[i + 1][1]
        if transition_counts[(current_state, next_state)] == 0:
            print(f"异常状态转换：从 {current_state} 到 {next_state}")

        duration = (logs[i + 1][0] - logs[i][0])
        if duration > np.mean(state_durations[current_state]) + 3 * np.std(state_durations[current_state]):
            print(f"异常状态持续时间：{current_state} 持续时间过长")

# 示例日志
logs = [
    (0, "Running", "设备正常运行"),
    (5, "Paused", "设备暂停"),
    (10, "Running", "设备恢复运行"),
    (15, "Restarting", "设备重启"),
    (20, "Running", "设备正常运行"),
    (25, "Fault", "设备故障"),  # 异常状态
    (30, "Restarting", "设备重启")  # 异常转换
]

transition_counts, state_durations = analyze_logs(logs)
detect_anomalies(logs, transition_counts, state_durations)