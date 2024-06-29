
import requests # node require
from bs4 import BeautifulSoup
import dashscope



def fetch_movie_list(url):
    # 设置HTTP 请求头
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0'
    }

    response = requests.get(url, headers=headers)

    # HTTP状态码 成功
    if response.status_code == 200:
        # 解析器 对html继续解析
        soup = BeautifulSoup(response.text, 'html.parser') # 内存中的dom对象
        movie_list = []
        movies = soup.select('#wrapper #content .article .item')
        # python 不是完全面向对象的，而更年轻的js 是完全面向对象
        # print(len(movies))
        # 确保一定是 字符串
        all_movies_text = ''.join([movie.prettify() for movie in movies[:2]])
        # print(all_movies_text)
        return all_movies_text
    else:
        print('Failed to retrieve content')


url = 'http://movie.douban.com/chart'

# 函数调用
movies = fetch_movie_list(url)
#print(movies)

# AIGC LLM + Prompt(指令)
prompt = f"""
{movies}
这是一段电影列表html，请获取电影名(name),封面链接(picture),简介(info)，评分(score)，评论人数(conmmentsNumber)
,请使用括号里的单词作为属性名，并以JSON数组的格式返回
"""
# print(prompt)

# 更改为自己的API_KEY
# dashscope.api_key = API_KEY

def call_qwen_with_prompt():
    message = [
      {
          'role':'user',
          'content':prompt
      }
    ]
    response = dashscope.Generation.call(
      dashscope.Generation.Models.qwen_turbo,
      messages = message,
      result_messages = 'messages'
    )
    print(response['output']['text'])


call_qwen_with_prompt()
