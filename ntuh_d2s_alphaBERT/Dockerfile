FROM pytorch/pytorch

COPY . /app
WORKDIR /app
RUN pip install -r requirements_pytorch.txt

ENTRYPOINT ["python"]
CMD ["ntuh_alphaBERT.py"]
