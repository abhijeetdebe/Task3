From centos
RUN mkdir cnn
RUN yum install python36 -y 
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade setuptools
RUN pip3 install tensorflow
RUN pip3 install keras
RUN pip3 install pillow
ENTRYPOINT [ "python3" ]




