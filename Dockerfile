FROM docker.arty.scai.fraunhofer.de/nginx:alpine
RUN mkdir /src

EXPOSE 3000
COPY ./src ./src
COPY nginx.conf.template ./nginx.conf

RUN chmod +w ./src 

CMD ["nginx", "-g", "daemon off;"]


# ENV PYTHONUNBUFFERED 1
FROM python:3.7.6
WORKDIR /src
ADD requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
# ADD . /src/