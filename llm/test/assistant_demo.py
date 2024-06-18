"""An image generation agent implemented by assistant"""

import os
import logging

logging.disable(logging.INFO)

from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI
from qwen_agent.tools.base import BaseTool, register_tool

ROOT_RESOURCE = os.path.join(os.path.dirname(__file__), 'resource')


def init_agent_service():
    # llm_cfg = {
    #     'model': 'qwen2:7b', 
    #     'model_server': 'http://localhost:11434/v1/',
    # }
    llm_cfg = {'model': 'qwen-max'}

    system = ("你是一个AI助手, 查询网络资源, 回答用户的问题, 并给出回答的来源。")

    tools = [
        'web_extractor',
        'code_interpreter',
    ]  # code_interpreter is a built-in tool in Qwen-Agent
    bot = Assistant(
        llm=llm_cfg,
        name='AI 助手',
        description='AI painting service',
        system_message=system,
        function_list=tools,
        # files=[os.path.join(ROOT_RESOURCE, 'doc.pdf')],
    )

    return bot


def test(query: str = 'iphone 是谁发明的？'):
    # Define the agent
    bot = init_agent_service()

    # Chat
    messages = [{'role': 'user', 'content': query}]
    for response in bot.run(messages=messages):
        print('bot response:', response)


def app_tui():
    # Define the agent
    bot = init_agent_service()

    # Chat
    messages = []
    while True:
        query = input('user question: ')
        # print()
        messages.append({'role': 'user', 'content': query})
        response = []
        length = 0
        for response in bot.run(messages=messages):
            # print('bot response:', response[-1]['content'])
            if length != len(response):
                length = len(response)
                print()

            print("\rbot response:{}".format(response[-1].get('content', '') + str(response[-1].get('function_call', ''))), end="", flush=True)
            pass
        messages.extend(response)
        print()
        # print('bot response:', response[-1]['content'])


def app_gui():
    # Define the agent
    bot = init_agent_service()
    chatbot_config = {
        'prompt.suggestions': [
            '4*2+3',
            '帮我写一个Python程序，实现一个简单的计算器',
            '甲乙两车分别从A、B两地同时相向而行，甲每小时行80千米，乙每小时行全程的10%，当乙行到全程的5/8时，甲车再行全程的1/6，可到达B地。求A、B两地相距多少千米?',
        ]
    }
    WebUI(
        bot,
        chatbot_config=chatbot_config,
    ).run()


if __name__ == '__main__':
    # test()
    # app_tui()
    app_gui()
