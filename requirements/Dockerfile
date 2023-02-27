FROM nasanccs/tensorflow-caney:latest
LABEL maintainer="jordan.a.caraballo-vega@nasa.gov"

ENV TFC_VERSION=0.1.1
ENV VHR_CNN_CHM_VERSION=main

RUN pip install git+https://github.com/nasa-nccs-hpda/tensorflow-caney.git@"${TFC_VERSION}" && \
    pip install git+https://github.com/nasa-nccs-hpda/vhr-cnn-chm.git@"${VHR_CNN_CHM_VERSION}"

HEALTHCHECK NONE
ENTRYPOINT [""]
