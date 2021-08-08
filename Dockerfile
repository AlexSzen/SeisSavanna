FROM pytorch/pytorch

RUN apt update && apt install vim -y
RUN pip install pandas netCDF4 sklearn matplotlib tensorboard
RUN pip install hydra-core --upgrade

COPY . /source/kenya

WORKDIR /source/kenya/ml_scripts/
