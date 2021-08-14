FROM pytorch/pytorch

RUN apt update && apt install vim -y
RUN apt install libsndfile1 -y
RUN pip install soundfile
RUN pip install torchsummary
RUN pip install pandas netCDF4 sklearn matplotlib tensorboard
RUN pip install hydra-core --upgrade
RUN pip install librosa 

COPY . /source/kenya

WORKDIR /source/kenya/
