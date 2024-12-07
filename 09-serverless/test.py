import requests

url = "http://localhost:8080/2015-03-31/functions/function/invocations"
img = "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"
data = {'url': img}

res = requests.post(url, json=data).text
print(res)