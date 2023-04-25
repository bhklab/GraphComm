# hash:sha256:b2e1e7dfa1e32d53710ebfa8d2f77d3e7c0665ce40937f2257095bc2c06d8b02
FROM registry.codeocean.com/codeocean/miniconda3:4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys \
        0xAD2323F17326AE31401037733E05EBFF05441C52 \
    && apt-get update \
    && apt-get install -y --no-install-recommends software-properties-common \
    && add-apt-repository -y 'deb http://deb.codeocean.com/rstudio-server-bionic/ ubuntu main' \
    && apt-get purge -y --autoremove software-properties-common \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential=12.8ubuntu1.1 \
        curl=7.68.0-1ubuntu2.18 \
        libcurl4-openssl-dev=7.68.0-1ubuntu2.18 \
        libgit2-dev=0.28.4+dfsg.1-2 \
        libssl-dev=1.1.1f-1ubuntu2.17 \
        pandoc=2.5-3build2 \
        pkg-config=0.29.1-0ubuntu4 \
        python-dev-is-python2=2.7.17-4 \
        r-base=3.6.3-2 \
        rstudio-server=1.2.5033 \
        wget=1.20.3-1ubuntu2 \
    && rm -rf /var/lib/apt/lists/*

RUN conda install -y \
        jupyterlab==3.5.3 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    anndata==0.8.0 \
    liana==0.1.7 \
    matplotlib==3.6.3 \
    omnipath==1.0.6 \
    pandas==1.3.5 \
    rdflib==6.2.0 \
    scanpy==1.9.1 \
    scprep==1.2.2 \
    torch==1.13.1 \
    umap-learn==0.5.3

RUN pip3 install -U --no-cache-dir \
    leidenalg==0.9.1 \
    torch-cluster==1.6.1

ADD "https://github.com/coder/code-server/releases/download/v4.9.0/code-server-4.9.0-linux-amd64.tar.gz" /.code-server/code-server.tar.gz
	
RUN cd /.code-server \
	&& tar -xvf code-server.tar.gz \
	&& rm code-server.tar.gz \
	&& ln -s /.code-server/code-server-4.9.0-linux-amd64/bin/code-server  /usr/bin/code-server

COPY postInstall /
RUN /postInstall
