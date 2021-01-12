# buildtools

We release PaddlePaddle and PaddlePaddle Fluid as shared libraries,
which, we hope could be released as wheel packages on PyPI, so we need
to make sure that the build follows the
[manulinux1](https://www.python.org/dev/peps/pep-0513/) standard.

The manylinux standard suggests building Python modules on an old
system, because that a module would anyway depend on some shared
libraries, and Linux's shared library standard states that those built
with newer version compilers cannot work with those with older
versions.  The suggested building environment is as old as CentOS 5.
However, PaddlePaddle relies on CUDA, and the earlies version of
[CentOS works with CUDA is 6](https://hub.docker.com/r/nvidia/cuda/).
So, here we provide a Docker image based on CentOS 6 and CUDA for
building PaddlePaddle and making the release supports "as-manylinux as
possible."  or "sufficiently many Linux" according to [this
discussion](https://mail.python.org/pipermail/wheel-builders/2016-July/000175.html).

The build output of our Docker image includes multiple wheel files --
some contain the CPU-only binary, some others support CUDA; some are
compatible with the cp27m Python ABI, some others with cp27.

To build these wheels, please run the following commands:

```bash
git clone https://github.com/paddlepaddle/paddle
cd paddle/tools/manylinux1
REPO=[yourrepo] ./build_all.sh
```

## Build PaddlePaddle for the different Python ABIs

Choose one of the following Python ABI and set the correct environment variables.

- cp27-cp27m

  ```bash
  export LD_LIBRARY_PATH=/opt/_internal/cpython-2.7.11-ucs2/lib:${LD_LIBRARY_PATH#/opt/_internal/cpython-2.7.11-ucs4/lib:}
  export PATH=/opt/python/cp27-cp27m/bin/:${PATH}
  export PYTHON_FLAGS="-DPYTHON_EXECUTABLE:FILEPATH=/opt/python/cp27-cp27m/bin/python
        -DPYTHON_INCLUDE_DIR:PATH=/opt/python/cp27-cp27m/include/python2.7
        -DPYTHON_LIBRARIES:FILEPATH=/opt/_internal/cpython-2.7.11-ucs2/lib/libpython2.7.so"
  ```

- cp27-cp27mu

  ```bash
  export LD_LIBRARY_PATH=/opt/_internal/cpython-2.7.11-ucs4/lib:${LD_LIBRARY_PATH#/opt/_internal/cpython-2.7.11-ucs2/lib:}
  export PATH=/opt/python/cp27-cp27mu/bin/:${PATH}
  export PYTHON_FLAGS="-DPYTHON_EXECUTABLE:FILEPATH=/opt/python/cp27-cp27mu/bin/python
        -DPYTHON_INCLUDE_DIR:PATH=/opt/python/cp27-cp27mu/include/python2.7
        -DPYTHON_LIBRARIES:FILEPATH=/opt/_internal/cpython-2.7.11-ucs4/lib/libpython2.7.so"
  ```

And then add the `PYTHON_FLAGS` as your cmake flags:

```bash
cmake ..
  ${PYTHON_FLAGS} \
  -DWITH_GPU=OFF \
  ...
```

You can find more details about cmake flags at [here](https://www.paddlepaddle.org.cn/documentation/docs/en/install/Tables_en.html)
