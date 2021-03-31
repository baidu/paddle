
### step 1. build paddle lib

```

# WITH_MKL=ON|OFF
# WITH_MKLDNN=ON|OFF

# for paddle main line, see CMakeList.txt so that you use libraries
# for both forward and backpopagation computation in a third party project
PADDLE_LIB=${PADDLE_SRC}/build

# for paddle-inference, download compiled libraries directly withtout
# compiling the source codes and grabs output headers and libraries:
#   https://paddle-inference.readthedocs.io/en/latest/product_introduction/inference_intro.html
# PADDLE_LIB=paddle_inference/paddle_inference_install_dir/

conda activate py36 # using python3.6
source env.sh
bash build.sh
cmake .. -DPADDLE_INSTALL_DIR=$PADDLE_LIB \
         -DCMAKE_BUILD_TYPE=Release \
         -DWITH_GPU=OFF \
         -DWITH_STYLE_CHECK=OFF \
         -DWITH_MKL=OFF \
         -DWITH_MKLDNN=OFF
make -j8
cd ${PADDLE_LIB}/python/dist/
pip install paddlepaddle_gpu-$VERSION-cp36-cp36m-linux_x86_64.whl
```

### step 2. generate program desc
```
# please install paddle before run this scripe
pip install --upgrade paddlepaddle-*.whl
python demo_network.py
```

This will generate two program desc files:
  - startup_program: used to init all parameters
  - main_program: main logic of the network

### step 3. build demo_trainer and run it.


```
# Make a build dir at the same dir of this README.md document.
# The demo dir can be put anywhere.
mkdir build
cd build

# WITH_MKL=ON|OFF
# WITH_MKLDNN=ON|OFF

# PADDLE_LIB is the same with PADDLE_INSTALL_DIR when building the lib
cmake .. -DPADDLE_LIB=$PADDLE_LIB \
         -DWITH_MKLDNN=OFF \
         -DWITH_MKL=OFF
make

cd ..

# run demo cpp trainer
./build/demo_trainer

```

The output will be:
```
step: 0 loss: 1069.02
step: 1 loss: 1069.02
step: 2 loss: 1069.02
....
```
