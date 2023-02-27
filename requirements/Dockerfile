FROM gitlab.nccs.nasa.gov:5050/nccs-ci/nccs-containers/go/nccs-centos7-go113:latest as builder
LABEL maintainer="jordan.a.caraballo-vega@nasa.gov"

ENV SINGULARITY_VERSION=3.7.1
RUN yum -y update && \
    yum -y install automake make build-base linux-headers libffi-dev bash git gcc squashfs-tools wget libtool gawk \
                   cryptsetup linux-headers build-base openssl-dev util-linux util-linux-dev shadow-uidmap && \
    rm -rf /var/cache/yum

RUN mkdir -p /usr/local/var/singularity/mnt && \
    mkdir -p $GOPATH/src/github.com/hpcng && \
    cd $GOPATH/src/github.com/hpcng && \
    wget -qO- https://github.com/hpcng/singularity/releases/download/v${SINGULARITY_VERSION}/singularity-${SINGULARITY_VERSION}.tar.gz | \
    tar xzv && \
    cd singularity && \
    ./mconfig -p /usr/local/singularity && \
    make -C builddir && \
    make -C builddir install

FROM gitlab.nccs.nasa.gov:5050/nccs-ci/nccs-containers/base/centos7:latest
LABEL maintainer="jordan.a.caraballo-vega@nasa.gov"

COPY --from=builder /usr/local/singularity /usr/local/singularity
RUN yum -y update && \
    yum -y install ca-certificates libseccomp squashfs-tools tzdata wget python3 python3-pip && \
    pip3 install hpccm && \
    rm -rf /var/cache/yum

ENV PATH="/usr/local/singularity/bin:$PATH"
HEALTHCHECK NONE
ENTRYPOINT [""]
