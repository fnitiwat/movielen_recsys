FROM python:3.9

ADD /app /root/app
ADD /data /root/data
ADD /requirements.txt /root/requirements
ADD /Makefile /root/Makefile

WORKDIR /root
RUN pip3 install -r requirements

EXPOSE 8000

CMD make serve