# 使用 Python 3.10 Slim 作為基礎映像
FROM python:3.10-slim

# 設定工作目錄
WORKDIR /app

# 複製當前目錄下的所有文件到容器的 /app 目錄
COPY init_chroma/* /app

# 安裝所需的 Python 套件
RUN pip install --no-cache-dir -r /app/requirements-vdb.txt

# 暴露應用程序運行的端口
ENV PYTHONPATH=/app

RUN chmod -R g=u /app

ENTRYPOINT ["python3", "/app/insert_vector_db.py"]
CMD [ "tail", "-f", "/dev/null" ] 