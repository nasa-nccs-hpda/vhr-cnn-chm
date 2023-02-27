# Arguments to pass to the image
ARG TFC_VERSION=0.1.1
ARG VHR_CNN_CHM_VERSION=main

FROM nasanccs/tensorflow-caney:latest
LABEL maintainer="jordan.a.caraballo-vega@nasa.gov"

RUN pip install git+https://github.com/nasa-nccs-hpda/tensorflow-caney.git@${TFC_VERSION}
RUN pip install git+https://github.com/nasa-nccs-hpda/vhr-cnn-chm.git@${VHR_CNN_CHM_VERSION}

HEALTHCHECK NONE
ENTRYPOINT [""]
