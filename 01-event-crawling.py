#!/usr/bin/env python
# coding: utf-8

# ## 인프런 2020년 새해 다짐 이벤트 댓글 크롤링
# * https://www.inflearn.com/pages/newyear-event-20200102
# * 영상 튜토리얼 : [댓글 수백 수천개 분석하기?! [1/5] 이벤트 데이터 크롤링 feat. 인프런 새해 다짐 이벤트 - YouTube](https://www.youtube.com/watch?v=OUSwQk79H8I&list=PLaTc2c6yEwmohRzCxWQqJ7Z9aqc-sQ5gC)
# 
# 
# ## 필요한 라이브러리 설치
# * 아나콘다 사용시 다음의 프롬프트 창을 열어 conda 명령어로 설치합니다.
# * pip 사용시 아래에 있는 명령어를 터미널로 설치합니다.
# <img src="https://i.imgur.com/Sar4gdw.jpg">
# 
# ### BeautifulSoup
# * `conda install -c anaconda beautifulsoup4`
# * [Beautifulsoup4 :: Anaconda Cloud](https://anaconda.org/anaconda/beautifulsoup4)
# 
# * pip 사용시 : `pip install beautifulsoup4`
# 
# ### tqdm
# * `conda install -c conda-forge tqdm`
# * [tqdm/tqdm: A Fast, Extensible Progress Bar for Python and CLI](https://github.com/tqdm/tqdm)
# * `pip install tqdm`

# In[1]:

# 주소를 크롤링 하기 전 제한사항이 있는지 확인하기 위해 주소.robots.txt 검색 후 disallow 항목에 내가 불러오고 싶은 항목이 있는지 찾아본다.
# 라이브러리 로드
# requests는 작은 웹브라우저로 웹사이트 내용을 가져온다.
import requests
# BeautifulSoup 을 통해 읽어 온 웹페이지를 파싱한다.
from bs4 import BeautifulSoup as bs
# 크롤링 후 결과를 데이터프레임 형태로 보기 위해 불러온다.
import pandas as pd
# 크롤링 후 댓글을 for 문을 통해 긁어오는 방법이 tqdm 이다.
from tqdm import trange


# In[2]:


# 크롤링 할 사이트
base_url = "https://www.inflearn.com/pages/newyear-event-20200102"
response = requests.get( base_url )
# url을 실행하여 본인의 주피터 노트북에 다운로드 하는 과정 // [200]이 나오면 정상적으로 받아왔다는 의미
response


# In[3]:


# response.text하게되면 내용 모두를 한번에 가져오며, html 형식이기에 html.parser을 추가함

soup = bs(response.text, 'html.parser')


# In[4]:


# main > section > div > div > div.chitchats > div.chitchat-list >
# div:nth-child(33) > div > div.body.edit-chitchat


# In[5]:


content = soup.select("#main > section > div > div > div.chitchats > div.chitchat-list > div")
content[-1]


# In[6]:


content[-1].select("div.body.edit-chitchat")[0].get_text(strip=True)


# In[7]:


# content[-1] : 가장 첫번째 게시물을 불러온다.
# [0] : 분리를 하고, (strip=True) 공백 제거

chitchat = content[-1].select("div.body.edit-chitchat")[0].get_text(strip=True)
chitchat


# In[8]:


# 5개의 내용만 먼저 선정해보자
# print("-"*20) : 보기 편하게 댓글마다 구분선을 넣어준다.

events = []
for i in range(5):
    print("-"*20)
    chitchat = content[i].select("div.body.edit-chitchat")[0].get_text(strip=True)
    print(chitchat)
    events.append(chitchat)


# In[9]:


events


# In[10]:


# 전체 콘텐츠 수를 확인하고 11번 code를 실행한다.

content_count = len(content)
content_count


# In[11]:


events = []
for i in trange(content_count):
    chitchat = content[i].select("div.body.edit-chitchat")[0].get_text(strip=True)
    events.append(chitchat)


# In[12]:


df = pd.DataFrame({"text": events})
df.shape


# In[13]:


df.to_csv("inflearn-event.csv", index=False)


# In[14]:


pd.read_csv("inflearn-event.csv").head()


# In[ ]:




