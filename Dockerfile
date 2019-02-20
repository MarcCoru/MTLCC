FROM tensorflow/tensorflow:1.12.0-gpu
#FROM tensorflow/tensorflow:1.4.0-gpu

LABEL maintainer="Marc Rußwurm <marc.russwurm@tum.de>"

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    python-tk


## Python Packages
#        psycopg2 \
#        configobj \
RUN pip --no-cache-dir install \
        matplotlib \
        jupyter \
        pandas \
        numpy \
        configparser \
        shapely \
        geopandas

## Install GDAL for evaluate.py -> adapted from https://hub.docker.com/r/ecarrara/python-gdal/~/dockerfile/
ENV GDAL_VERSION=2.2.3
ENV GDAL_VERSION_PYTHON=2.2.3

RUN apt-get update && apt-get -y install \
    wget \
    libcurl4-openssl-dev \
    build-essential \
    libpq-dev \
    ogdi-bin \
    libogdi3.2-dev \
    libjasper-runtime \
    libjasper-dev \
    libjasper1 \
    libgeos-c1v5 \
    libproj-dev \
    libpoppler-dev \
    libsqlite3-dev \
    libspatialite-dev
#    python \
#    python-pip \
#    python-dev \
#    python-numpy-dev

RUN wget http://download.osgeo.org/gdal/$GDAL_VERSION/gdal-${GDAL_VERSION}.tar.gz -O /tmp/gdal-${GDAL_VERSION}.tar.gz && \
    tar -x -f /tmp/gdal-${GDAL_VERSION}.tar.gz -C /tmp

RUN cp -r /tmp/gdal-2.2.3/data /usr/local/share/gdal/
ENV GDAL_DATA /usr/local/share/gdal

RUN cd /tmp/gdal-${GDAL_VERSION} && \
    ./configure \
        --prefix=/usr \
        --with-python \
        --with-geos \
        --with-geotiff \
        --with-jpeg \
        --with-png \
        --with-expat \
        --with-libkml \
        --with-openjpeg \
        --with-pg \
        --with-curl \
        --with-spatialite && \
    make && make install

RUN rm /tmp/gdal-${GDAL_VERSION} -rf && rm /tmp/gdal-${GDAL_VERSION}.tar.gz
RUN pip install GDAL==${GDAL_VERSION_PYTHON} \
        rasterio

# enable jupyter widgets
RUN jupyter nbextension enable --py widgetsnbextension --sys-prefix

# create directory to mount data
ENV datadir /data
RUN mkdir data

ENV modeldir /model
RUN mkdir /model

ENV modelzoo /MTLCC/modelzoo

RUN mkdir MTLCC
COPY ./ /MTLCC/

WORKDIR "/MTLCC"

CMD ["/bin/bash"]
