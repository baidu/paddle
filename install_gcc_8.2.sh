VERSION=8.2.0
GCC=gcc-${VERSION}
GCC_REPO=https://ftpmirror.gnu.org/gcc/${GCC} #https://paddle-docker-tar.bj.bcebos.com/home/users/tianshuo/bce-python-sdk-0.8.27
GCC_SRC=${GCC}.tar.gz

if [ -f $GCC_SRC ];then
   wget -q ${GCC_REPO}/${GCC_SRC}
fi
tar -xvf ${GCC_SRC}

cd ${GCC}

unset LIBRARY_PATH CPATH C_INCLUDE_PATH PKG_CONFIG_PATH CPLUS_INCLUDE_PATH INCLUDE && \
./contrib/download_prerequisites && \
cd .. && mkdir temp_gcc && cd temp_gcc && \
../${GCC}/configure --prefix=/usr/local/gcc-8.2 --enable-threads=posix --disable-checking --disable-multilib && \
make -j8 && make install

