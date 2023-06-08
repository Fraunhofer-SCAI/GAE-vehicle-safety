FROM python:3.7.6

# set a directory for the app
WORKDIR /usr/src/gae

# copy all the files to the container
COPY requirements.txt  requirements.txt 

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt 

# Set OEM configuration
# ENTRYPOINT [ "python", "src/gae/oems.py"]
# 
# CMD ["python", "src/manage.py"]
