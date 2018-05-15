import requests
from bs4 import BeautifulSoup
import re

i=0
first_url = 'https://movie.douban.com/subject/27114843/comments?status=P'


# 请求头部
headers = {
    'Host':'movie.douban.com',
    'Referer':'https://movie.douban.com/subject/27114843/?tag=%E7%83%AD%E9%97%A8&from=gaia_video',
    'Upgrade-Insecure-Requests':'1',
    'User-Agent':'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.117 Safari/537.36'
}

def write_file(file_name,comment):
    with open(file_name,'a',encoding='utf8') as file:
            file.write(comment+'\n')

def visit_URL(url):
    global i
    i+=1
    res = requests.get(url=url,headers=headers)
    soup = BeautifulSoup(res.content,'html.parser')
    div_comment = soup.find_all('div',class_='comment-item') # 找到所有的评论模块
    for com in div_comment:
        comment = com.p.get_text()
        ratee_n = com.find('span', {'class': re.compile('allstar[1-2]0')})
        if ratee_n is not None:
            write_file('D:/python/data/douban_data_n.txt',comment.strip())
        ratee_m = com.find('span', {'class': re.compile('allstar30')})
        if ratee_m is not None:
            write_file('D:/python/data/douban_data_m.txt',comment.strip())
        ratee_p = com.find('span', {'class': re.compile('allstar50')})
        if ratee_p is not None:
            write_file('D:/python/data/douban_data_p.txt',comment.strip())

    # 检查是否有下一页
    next_url = soup.find('a',class_='next')
    if next_url:
        temp = next_url['href'].strip().split('&amp;') # 获取下一个url
        next_url = ''.join(temp)
        print(next_url)
    # print(next_url)
    if next_url and i<20:
        visit_URL('https://movie.douban.com/subject/27114843/comments'+next_url)


if __name__ == '__main__':
    visit_URL(first_url)