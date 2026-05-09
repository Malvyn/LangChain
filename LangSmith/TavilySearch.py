from langserve import RemoteRunnable

if __name__ == '__main__':
    client = RemoteRunnable("http://localhost:8000/translate")
    result = client.invoke({"language": "English", "text": "我今天下午需要访问客户。"})
    print(result)
