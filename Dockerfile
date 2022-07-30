FROM python:3.9

ADD /app /root/app
ADD /data /root/data
ADD /requirements.txt /root/requirements.txt
ADD /Makefile /root/Makefile
ADD /research_phase/train.py /root/research_phase/train.py

WORKDIR /root
RUN pip3 install -r requirements.txt

EXPOSE 8000

CMD make serve