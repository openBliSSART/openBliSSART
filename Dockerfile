FROM ubuntu:12.04

# Installing depedencies.
RUN apt-get update -y
RUN apt-get install -y git make g++ autoconf libtool libpoco-dev libfftw3-dev libsdl-sound1.2-dev libatlas-base-dev qt4-dev-tools

# Creating our work dir.
RUN mkdir ~/src
WORKDIR "root/src"

# Installing openBliSSART
RUN git clone https://github.com/openBliSSART/openBliSSART
WORKDIR "./openBliSSART"
RUN ./bootstrap.sh
RUN ./configure --prefix=${HOME}/blissart
RUN make
RUN make install

# Setting env.
RUN sh -c "echo ${HOME}/blissart/lib > /etc/ld.so.conf.d/blissart.conf"
RUN ldconfig

# Setting dir for when user logs into the container.
WORKDIR "/root/blissart/bin"
CMD /bin/bash
