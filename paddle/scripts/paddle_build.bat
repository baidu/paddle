@ECHO OFF
rem Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
rem
rem Licensed under the Apache License, Version 2.0 (the "License");
rem you may not use this file except in compliance with the License.
rem You may obtain a copy of the License at
rem
rem     http://www.apache.org/licenses/LICENSE-2.0
rem
rem Unless required by applicable law or agreed to in writing, software
rem distributed under the License is distributed on an "AS IS" BASIS,
rem WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
rem See the License for the specific language governing permissions and
rem limitations under the License.

rem =================================================
rem       Paddle CI Task On Windows Platform
rem =================================================

@ECHO ON
setlocal enabledelayedexpansion

rem -------clean up environment-----------
set work_dir=%cd%
if not defined cache_dir set cache_dir=%work_dir:Paddle=cache%
if not exist %cache_dir%\tools (
    git clone https://github.com/zhouwei25/tools.git %cache_dir%\tools
)
taskkill /f /im cmake.exe /t 2>NUL
taskkill /f /im ninja.exe /t 2>NUL
taskkill /f /im MSBuild.exe /t 2>NUL
taskkill /f /im cl.exe /t 2>NUL
taskkill /f /im lib.exe /t 2>NUL
taskkill /f /im link.exe /t 2>NUL
taskkill /f /im vctip.exe /t 2>NUL
taskkill /f /im cvtres.exe /t 2>NUL
taskkill /f /im rc.exe /t 2>NUL
taskkill /f /im mspdbsrv.exe /t 2>NUL
taskkill /f /im csc.exe /t 2>NUL
taskkill /f /im python.exe /t 2>NUL
taskkill /f /im nvcc.exe /t 2>NUL
taskkill /f /im cicc.exe /t 2>NUL
taskkill /f /im ptxas.exe /t 2>NUL
taskkill /f /im op_function_generator.exe /t 2>NUL
wmic process where name="op_function_generator.exe" call terminate 2>NUL
wmic process where name="cvtres.exe" call terminate 2>NUL
wmic process where name="rc.exe" call terminate 2>NUL
wmic process where name="cl.exe" call terminate 2>NUL
wmic process where name="lib.exe" call terminate 2>NUL
wmic process where name="python.exe" call terminate 2>NUL

rem ------initialize common variable------
if not defined GENERATOR set GENERATOR="Visual Studio 15 2017 Win64"
if not defined BRANCH set BRANCH=develop
if not defined WITH_TENSORRT set WITH_TENSORRT=ON
if not defined TENSORRT_ROOT set TENSORRT_ROOT=D:/TensorRT
if not defined CUDA_ARCH_NAME set CUDA_ARCH_NAME=Auto
if not defined WITH_GPU set WITH_GPU=ON
if not defined WITH_MKL set WITH_MKL=ON
if not defined WITH_AVX set WITH_AVX=ON
if not defined WITH_TESTING set WITH_TESTING=ON
if not defined MSVC_STATIC_CRT set MSVC_STATIC_CRT=ON
if not defined WITH_PYTHON set WITH_PYTHON=ON
if not defined ON_INFER set ON_INFER=ON
if not defined WITH_INFERENCE_API_TEST set WITH_INFERENCE_API_TEST=ON
if not defined WITH_STATIC_LIB set WITH_STATIC_LIB=ON
if not defined WITH_TPCACHE set WITH_TPCACHE=OFF
if not defined WITH_CLCACHE set WITH_CLCACHE=OFF
if not defined WITH_CACHE set WITH_CACHE=OFF
if not defined WITH_SCCACHE set WITH_SCCACHE=OFF
if not defined WITH_UNITY_BUILD set WITH_UNITY_BUILD=OFF
if not defined INFERENCE_DEMO_INSTALL_DIR set INFERENCE_DEMO_INSTALL_DIR=%cache_dir:\=/%/inference_demo
if not defined LOG_LEVEL set LOG_LEVEL=normal
if not defined PRECISION_TEST set PRECISION_TEST=OFF
if not defined NIGHTLY_MODE set PRECISION_TEST=OFF
if not defined retry_times set retry_times=3
if not defined PYTHON_ROOT set PYTHON_ROOT=C:\Python37

rem ------initialize the python environment------
set PYTHON_EXECUTABLE=%PYTHON_ROOT%\python.exe
set PATH=%PYTHON_ROOT%\Scripts;%PYTHON_ROOT%;%PATH%
if "%WITH_PYTHON%" == "ON" (
    where python
    where pip
    pip install wheel --user
    pip install -r %work_dir%\python\requirements.txt --user
    if !ERRORLEVEL! NEQ 0 (
        echo pip install requirements.txt failed!
        exit /b 7
    )
)

rem -------Caching strategy 1: keep build directory for incremental compilation-----------
rmdir build\python /s/q
rmdir build\paddle\third_party\externalError /s/q
rem rmdir build\paddle\fluid\pybind /s/q
rmdir build\paddle_install_dir /s/q
rmdir build\paddle_inference_install_dir /s/q
rmdir build\paddle_inference_c_install_dir /s/q
del build\CMakeCache.txt

if "%WITH_CACHE%"=="OFF" (
    rmdir build /s/q
    goto :mkbuild
)

set error_code=0
type %cache_dir%\error_code.txt
: set /p error_code=< %cache_dir%\error_code.txt
if %error_code% NEQ 0 (
    rmdir build /s/q
    goto :mkbuild
)

setlocal enabledelayedexpansion
git show-ref --verify --quiet refs/heads/last_pr
if %ERRORLEVEL% EQU 0 (
    git diff HEAD last_pr --stat --name-only
    git diff HEAD last_pr --stat --name-only | findstr "setup.py.in"
    if !ERRORLEVEL! EQU 0 (
        rmdir build /s/q
    )
    git branch -D last_pr
    git branch last_pr
) else (
    rmdir build /s/q
    git branch last_pr
)

for /F %%# in ('wmic os get localdatetime^|findstr 20') do set datetime=%%#
set day_now=%datetime:~6,2%
set day_before=-1
set /p day_before=< %cache_dir%\day.txt
if %day_now% NEQ %day_before% (
    echo %day_now% > %cache_dir%\day.txt
    type %cache_dir%\day.txt
    rmdir build /s/q
    goto :mkbuild
)

:mkbuild
if not exist build (
    echo Windows build cache FALSE
    set Windows_Build_Cache=FALSE
    mkdir build
) else (
    echo Windows build cache TRUE
    set Windows_Build_Cache=TRUE
)
echo ipipe_log_param_Windows_Build_Cache: %Windows_Build_Cache%
cd /d build
dir .
dir %cache_dir%
dir paddle\fluid\pybind\Release
rem -------Caching strategy 1: End --------------------------------

rem -------Caching strategy 2: sccache decorate compiler-----------
if "%WITH_SCCACHE%"=="ON" (
    cmd /C sccache -V || call :install_sccache
    sccache --stop-server 2> NUL
    if not exist D:\sccache mkdir D:\sccache
    set SCCACHE_DIR=D:\sccache\.cache
    set SCCACHE_CACHE_SIZE=30G
    set SCCACHE_ERROR_LOG=D:\sccache\sccache_log.txt
    set SCCACHE_LOG=quiet
    sccache --start-server
    sccache -z
    goto :CASE_%1
) else (
    del %PYTHON_ROOT%\sccache.exe 2> NUL
    goto :CASE_%1
)

:install_sccache
echo There is not sccache in this PC, will install sccache.
echo Download package from https://paddle-ci.gz.bcebos.com/window_requirement/sccache.exe
%PYTHON_ROOT%\python.exe -c "import wget;wget.download('https://paddle-ci.gz.bcebos.com/window_requirement/sccache.exe')"
xcopy sccache.exe %PYTHON_ROOT%\Scripts\ /Y
goto:eof
rem -------Caching strategy 2: End --------------------------------

echo "Usage: paddle_build.bat [OPTION]"
echo "OPTION:"
echo "wincheck_mkl: run Windows MKL/GPU PR CI tasks on Windows"
echo "wincheck_openbals: run Windows OPENBLAS/CPU PR CI tasks on Windows"
echo "build_avx_whl: build Windows avx whl package on Windows"
echo "build_no_avx_whl: build Windows no avx whl package on Windows"
echo "build_inference_lib: build Windows inference library on Windows"
exit /b 1

rem ------PR CI windows check for MKL/GPU----------
:CASE_wincheck_mkl
set WITH_MKL=ON
set WITH_GPU=ON
set WITH_AVX=ON
set MSVC_STATIC_CRT=OFF
set ON_INFER=ON

call :cmake || goto cmake_error
call :build || goto build_error
call :test_whl_pacakage || goto test_whl_pacakage_error
call :test_unit || goto test_unit_error
call :test_inference || goto test_inference_error
:: call :check_change_of_unittest || goto check_change_of_unittest_error
goto:success

rem ------PR CI windows check for OPENBLAS/CPU------
:CASE_wincheck_openblas
set WITH_MKL=OFF
set WITH_GPU=OFF
set WITH_AVX=OFF
set MSVC_STATIC_CRT=ON
set retry_times=1
set ON_INFER=OFF

call :cmake || goto cmake_error
call :build || goto build_error
call :test_whl_pacakage || goto test_whl_pacakage_error
call :test_unit || goto test_unit_error
:: call :test_inference || goto test_inference_error
:: call :check_change_of_unittest || goto check_change_of_unittest_error
goto:success

rem ------Build windows avx whl package------
:CASE_build_avx_whl
set WITH_AVX=ON
set ON_INFER=OFF
set CUDA_ARCH_NAME=All
set retry_times=4

call :cmake || goto cmake_error
call :build || goto build_error
call :test_whl_pacakage || goto test_whl_pacakage_error
goto:success

rem ------Build windows no-avx whl package------
:CASE_build_no_avx_whl
set WITH_AVX=OFF
set ON_INFER=OFF
set CUDA_ARCH_NAME=All
set retry_times=4

call :cmake || goto cmake_error
call :build || goto build_error
call :test_whl_pacakage || goto test_whl_pacakage_error
goto:success

rem ------Build windows inference library------
:CASE_build_inference_lib
set ON_INFER=ON
set WITH_PYTHON=OFF
set CUDA_ARCH_NAME=All
python %work_dir%\tools\remove_grad_op_and_kernel.py
if %errorlevel% NEQ 0 exit /b 1

call :cmake || goto cmake_error
call :build || goto build_error
call :test_inference || goto test_inference_error
call :zip_cc_file || goto zip_cc_file_error
call :zip_c_file || goto zip_c_file_error
goto:success

rem "Other configurations are added here"
rem :CASE_wincheck_others
rem call ...


rem ---------------------------------------------------------------------------------------------
:cmake
@ECHO OFF
echo    ========================================
echo    Step 1. Cmake ...
echo    ========================================

rem Configure the environment for 64-bit builds. 'DISTUTILS_USE_SDK' indicates that the user has selected the compiler.
call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvars64.bat"
set DISTUTILS_USE_SDK=1
rem Windows 10 Kit bin dir
set PATH=C:\Program Files (x86)\Windows Kits\10\bin\10.0.17763.0\x64;%PATH%
rem Use 64-bit ToolSet to compile
set PreferredToolArchitecture=x64

for /F %%# in ('wmic os get localdatetime^|findstr 20') do set start=%%#
set start=%start:~4,10%

if not defined CUDA_TOOLKIT_ROOT_DIR set CUDA_TOOLKIT_ROOT_DIR=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2
set PATH=%TENSORRT_ROOT:/=\%\lib;%CUDA_TOOLKIT_ROOT_DIR%\bin;%CUDA_TOOLKIT_ROOT_DIR%\libnvvp;%PATH%

rem install ninja if GENERATOR is Ninja
if %GENERATOR% == "Ninja" (
    pip install ninja
    if %errorlevel% NEQ 0 (
        echo pip install ninja failed!
        exit /b 7
    )
)

rem ------show summary of current GPU environment----------
cmake --version
if "%WITH_GPU%"=="ON" (
    nvcc --version
    nvidia-smi 2>NUL
)

rem ------pre install clcache and init config----------
rem pip install clcache --user
pip uninstall -y clcache
:: set USE_CLCACHE to enable clcache
rem set USE_CLCACHE=1
:: In some scenarios, CLCACHE_HARDLINK can save one file copy.
rem set CLCACHE_HARDLINK=1
:: If it takes more than 1000s to obtain the right to use the cache, an error will be reported
rem set CLCACHE_OBJECT_CACHE_TIMEOUT_MS=1000000
:: set maximum cache size to 20G
rem clcache.exe -M 21474836480

rem ------set third_party cache dir------
: clear third party cache every once in a while
for /F %%# in ('wmic os get localdatetime^|findstr 20') do set datetime=%%#
set day_now=%datetime:~6,2%
set day_before=-1
set /p day_before=< %cache_dir%\day.txt
if %day_now% NEQ %day_before% (
    echo %day_now% > %cache_dir%\day.txt
    type %cache_dir%\day.txt
    if %day_now% EQU 21 (
        del D:\sccache\sccache_log.txt
        rmdir %cache_dir%\third_party_GPU /s/q
        rmdir %cache_dir%\third_party /s/q
    )
    if %day_now% EQU 11 (
        del D:\sccache\sccache_log.txt
        rmdir %cache_dir%\third_party_GPU /s/q
        rmdir %cache_dir%\third_party /s/q
    )
    if %day_now% EQU 01 (
        del D:\sccache\sccache_log.txt
        rmdir %cache_dir%\third_party_GPU /s/q
        rmdir %cache_dir%\third_party /s/q
    )
)

if "%WITH_TPCACHE%"=="OFF" (
    set THIRD_PARTY_PATH=%work_dir:\=/%/build/third_party
    goto :cmake_impl
)

echo set -ex > cache.sh
echo md5_content=$(cat %work_dir:\=/%/cmake/external/*.cmake  ^|md5sum ^| awk '{print $1}') >> cache.sh
echo echo ${md5_content}^>md5.txt >> cache.sh

%cache_dir%\tools\busybox64.exe cat cache.sh
%cache_dir%\tools\busybox64.exe bash cache.sh

set /p md5=< md5.txt
if "%WITH_GPU%"=="ON" (
    set THIRD_PARTY_HOME=%cache_dir:\=/%/third_party_GPU
) else (
    set THIRD_PARTY_HOME=%cache_dir:\=/%/third_party
)
set THIRD_PARTY_PATH=%THIRD_PARTY_HOME%/%md5%

:cmake_impl
echo cmake .. -G %GENERATOR% -DCMAKE_BUILD_TYPE=Release -DWITH_AVX=%WITH_AVX% -DWITH_GPU=%WITH_GPU% -DWITH_MKL=%WITH_MKL% ^
-DWITH_TESTING=%WITH_TESTING% -DWITH_PYTHON=%WITH_PYTHON% -DPYTHON_EXECUTABLE=%PYTHON_EXECUTABLE% -DON_INFER=%ON_INFER% ^
-DWITH_INFERENCE_API_TEST=%WITH_INFERENCE_API_TEST% -DTHIRD_PARTY_PATH=%THIRD_PARTY_PATH% ^
-DINFERENCE_DEMO_INSTALL_DIR=%INFERENCE_DEMO_INSTALL_DIR% -DWITH_STATIC_LIB=%WITH_STATIC_LIB% ^
-DWITH_TENSORRT=%WITH_TENSORRT% -DTENSORRT_ROOT="%TENSORRT_ROOT%" -DMSVC_STATIC_CRT=%MSVC_STATIC_CRT% ^
-DWITH_UNITY_BUILD=%WITH_UNITY_BUILD% -DCUDA_ARCH_NAME=%CUDA_ARCH_NAME% -DCUB_PATH=%THIRD_PARTY_HOME%/cub

cmake .. -G %GENERATOR% -DCMAKE_BUILD_TYPE=Release -DWITH_AVX=%WITH_AVX% -DWITH_GPU=%WITH_GPU% -DWITH_MKL=%WITH_MKL% ^
-DWITH_TESTING=%WITH_TESTING% -DWITH_PYTHON=%WITH_PYTHON% -DPYTHON_EXECUTABLE=%PYTHON_EXECUTABLE% -DON_INFER=%ON_INFER% ^
-DWITH_INFERENCE_API_TEST=%WITH_INFERENCE_API_TEST% -DTHIRD_PARTY_PATH=%THIRD_PARTY_PATH% ^
-DINFERENCE_DEMO_INSTALL_DIR=%INFERENCE_DEMO_INSTALL_DIR% -DWITH_STATIC_LIB=%WITH_STATIC_LIB% ^
-DWITH_TENSORRT=%WITH_TENSORRT% -DTENSORRT_ROOT="%TENSORRT_ROOT%" -DMSVC_STATIC_CRT=%MSVC_STATIC_CRT% ^
-DWITH_UNITY_BUILD=%WITH_UNITY_BUILD% -DCUDA_ARCH_NAME=%CUDA_ARCH_NAME% -DCUB_PATH=%THIRD_PARTY_HOME%/cub
goto:eof

:cmake_error
echo 7 > %cache_dir%\error_code.txt
type %cache_dir%\error_code.txt
echo Cmake failed, will exit!
exit /b 7

rem ---------------------------------------------------------------------------------------------
:build
@ECHO OFF
echo    ========================================
echo    Step 2. Build Paddle ...
echo    ========================================

for /F %%# in ('wmic cpu get NumberOfLogicalProcessors^|findstr [0-9]') do set /a PARALLEL_PROJECT_COUNT=%%#*4/5
echo "PARALLEL PROJECT COUNT is %PARALLEL_PROJECT_COUNT%"

set build_times=1
rem MSbuild will build third_party first to improve compiler stability.
if NOT %GENERATOR% == "Ninja" (
    goto :build_tp
) else (
    goto :build_paddle
)

:build_tp
echo Build third_party the %build_times% time:
if %GENERATOR% == "Ninja" (
    ninja third_party
) else (
    MSBuild /m /p:PreferredToolArchitecture=x64 /p:Configuration=Release /verbosity:%LOG_LEVEL% third_party.vcxproj
)

if %ERRORLEVEL% NEQ 0 (
    set /a build_times=%build_times%+1  
    if %build_times% GEQ %retry_times% (
        exit /b 7
    ) else (
        echo Build third_party failed, will retry!
        goto :build_tp
    )
)
echo Build third_party successfully!

set build_times=1
:build_paddle
:: reset clcache zero stats for collect PR's actual hit rate
rem clcache.exe -z

rem -------clean up environment again-----------
taskkill /f /im cmake.exe /t 2>NUL
taskkill /f /im MSBuild.exe /t 2>NUL
taskkill /f /im cl.exe /t 2>NUL
taskkill /f /im lib.exe /t 2>NUL
taskkill /f /im link.exe /t 2>NUL
taskkill /f /im vctip.exe /t 2>NUL
taskkill /f /im cvtres.exe /t 2>NUL
taskkill /f /im rc.exe /t 2>NUL
taskkill /f /im mspdbsrv.exe /t 2>NUL
taskkill /f /im csc.exe /t 2>NUL
taskkill /f /im nvcc.exe /t 2>NUL
taskkill /f /im cicc.exe /t 2>NUL
taskkill /f /im ptxas.exe /t 2>NUL
taskkill /f /im op_function_generator.exe /t 2>NUL
wmic process where name="cmake.exe" call terminate 2>NUL
wmic process where name="op_function_generator.exe" call terminate 2>NUL
wmic process where name="cvtres.exe" call terminate 2>NUL
wmic process where name="rc.exe" call terminate 2>NUL
wmic process where name="cl.exe" call terminate 2>NUL
wmic process where name="lib.exe" call terminate 2>NUL

if "%WITH_TESTING%"=="ON" (
    for /F "tokens=1 delims= " %%# in ('tasklist ^| findstr /i test') do taskkill /f /im %%# /t
)

echo Build Paddle the %build_times% time:
if %GENERATOR% == "Ninja" (
    ninja all
) else (
    if "%WITH_CLCACHE%"=="OFF" (
        MSBuild /m:%PARALLEL_PROJECT_COUNT% /p:PreferredToolArchitecture=x64 /p:TrackFileAccess=false /p:Configuration=Release /verbosity:%LOG_LEVEL% ALL_BUILD.vcxproj
    ) else (
        MSBuild /m:%PARALLEL_PROJECT_COUNT% /p:PreferredToolArchitecture=x64 /p:TrackFileAccess=false /p:CLToolExe=clcache.exe /p:CLToolPath=%PYTHON_ROOT%\Scripts /p:Configuration=Release /verbosity:%LOG_LEVEL% ALL_BUILD.vcxproj
    )
)

if %ERRORLEVEL% NEQ 0 (
    set /a build_times=%build_times%+1
    if %build_times% GEQ %retry_times% (
        exit /b 7
    ) else (
        echo Build Paddle failed, will retry!
        goto :build_paddle
    )
)

echo Build Paddle successfully!
echo 0 > %cache_dir%\error_code.txt
type %cache_dir%\error_code.txt

:: ci will collect sccache hit rate
if "%WITH_SCCACHE%"=="ON" (
    call :collect_sccache_hits
)

goto:eof

:build_error
echo 7 > %cache_dir%\error_code.txt
type %cache_dir%\error_code.txt
echo Build Paddle failed, will exit!
exit /b 7

rem ---------------------------------------------------------------------------------------------
:test_whl_pacakage
@ECHO OFF
echo    ========================================
echo    Step 3. Test pip install whl package ...
echo    ========================================

setlocal enabledelayedexpansion

for /F %%# in ('wmic os get localdatetime^|findstr 20') do set end=%%#
set end=%end:~4,10%
call :timestamp "%start%" "%end%" "Build"

tree /F %cd%\paddle_inference_install_dir\paddle
%cache_dir%\tools\busybox64.exe du -h -d 0 -k %cd%\paddle_inference_install_dir\paddle\lib > lib_size.txt
set /p libsize=< lib_size.txt
for /F %%i in ("%libsize%") do (
    set /a libsize_m=%%i/1024
    echo "Windows Paddle_Inference Size: !libsize_m!M"
    echo ipipe_log_param_Windows_Paddle_Inference_Size: !libsize_m!M
)

%cache_dir%\tools\busybox64.exe du -h -d 0 %cd%\python\dist > whl_size.txt
set /p whlsize=< whl_size.txt
for /F %%i in ("%whlsize%") do echo "Windows PR whl Size: %%i"
for /F %%i in ("%whlsize%") do echo ipipe_log_param_Windows_PR_whl_Size: %%i

dir /s /b python\dist\*.whl > whl_file.txt
set /p PADDLE_WHL_FILE_WIN=< whl_file.txt

@ECHO ON
pip uninstall -y paddlepaddle
pip uninstall -y paddlepaddle-gpu
pip install %PADDLE_WHL_FILE_WIN% --user
if %ERRORLEVEL% NEQ 0 (
    call paddle_winci\Scripts\deactivate.bat 2>NUL
    echo pip install whl package failed!
    exit /b 1
)


set CUDA_VISIBLE_DEVICES=0
python %work_dir%\paddle\scripts\installation_validate.py
goto:eof

:test_whl_pacakage_error
::echo 1 > %cache_dir%\error_code.txt
::type %cache_dir%\error_code.txt
echo Test import paddle failed, will exit!
exit /b 1

rem ---------------------------------------------------------------------------------------------
:test_unit
@ECHO ON
echo    ========================================
echo    Step 4. Running unit tests ...
echo    ========================================

: set CI_SKIP_CPP_TEST if only *.py changed
git diff --name-only %BRANCH% | findstr /V "\.py" || set CI_SKIP_CPP_TEST=ON

pip install -r %work_dir%\python\unittest_py\requirements.txt --user
if %ERRORLEVEL% NEQ 0 (
    echo pip install unittest requirements.txt failed!
    exit /b 7
)

for /F %%# in ('wmic os get localdatetime^|findstr 20') do set start=%%#
set start=%start:~4,10%

set FLAGS_call_stack_level=2
dir %THIRD_PARTY_PATH:/=\%\install\openblas\lib
dir %THIRD_PARTY_PATH:/=\%\install\openblas\bin
dir %THIRD_PARTY_PATH:/=\%\install\zlib\bin
dir %THIRD_PARTY_PATH:/=\%\install\mklml\lib
dir %THIRD_PARTY_PATH:/=\%\install\mkldnn\bin
dir %THIRD_PARTY_PATH:/=\%\install\warpctc\bin

pip install requests

set PATH=%THIRD_PARTY_PATH:/=\%\install\openblas\lib;%THIRD_PARTY_PATH:/=\%\install\openblas\bin;^
%THIRD_PARTY_PATH:/=\%\install\zlib\bin;%THIRD_PARTY_PATH:/=\%\install\mklml\lib;^
%THIRD_PARTY_PATH:/=\%\install\mkldnn\bin;%THIRD_PARTY_PATH:/=\%\install\warpctc\bin;%PATH%

if "%WITH_GPU%"=="ON" (
    goto:parallel_test_base_gpu
) else (
    goto:parallel_test_base_cpu
)

:parallel_test_base_gpu
echo    ========================================
echo    Running GPU unit tests in parallel way ...
echo    ========================================

setlocal enabledelayedexpansion

:: set PATH=C:\Windows\System32;C:\Program Files\NVIDIA Corporation\NVSMI;%PATH%
:: cmd /C nvidia-smi -L
:: if %errorlevel% NEQ 0 exit /b 8
:: for /F %%# in ('cmd /C nvidia-smi -L ^|find "GPU" /C') do set CUDA_DEVICE_COUNT=%%#
set CUDA_DEVICE_COUNT=1

echo cmake .. -G %GENERATOR% -DCMAKE_BUILD_TYPE=Release -DWITH_AVX=%WITH_AVX% -DWITH_GPU=%WITH_GPU% -DWITH_MKL=%WITH_MKL% ^
-DWITH_TESTING=%WITH_TESTING% -DWITH_PYTHON=%WITH_PYTHON% -DON_INFER=%ON_INFER% ^
-DWITH_INFERENCE_API_TEST=%WITH_INFERENCE_API_TEST% -DTHIRD_PARTY_PATH=%THIRD_PARTY_PATH% ^
-DINFERENCE_DEMO_INSTALL_DIR=%INFERENCE_DEMO_INSTALL_DIR% -DWITH_STATIC_LIB=%WITH_STATIC_LIB% ^
-DWITH_TENSORRT=%WITH_TENSORRT% -DTENSORRT_ROOT="%TENSORRT_ROOT%" -DMSVC_STATIC_CRT=%MSVC_STATIC_CRT% ^
-DWITH_UNITY_BUILD=%WITH_UNITY_BUILD% -DCUDA_ARCH_NAME=%CUDA_ARCH_NAME% >> %work_dir%\win_cmake.sh
set FLAGS_fraction_of_gpu_memory_to_use=0.92

%cache_dir%\tools\busybox64.exe bash %work_dir%\tools\windows\run_unittests.sh %NIGHTLY_MODE% %PRECISION_TEST% %WITH_GPU%

goto:eof

:parallel_test_base_cpu
echo    ========================================
echo    Running CPU unit tests in parallel way ...
echo    ========================================

%cache_dir%\tools\busybox64.exe bash %work_dir%\tools\windows\run_unittests.sh %NIGHTLY_MODE% %PRECISION_TEST% %WITH_GPU%

goto:eof

:test_unit_error
:: echo 8 > %cache_dir%\error_code.txt
:: type %cache_dir%\error_code.txt
for /F %%# in ('wmic os get localdatetime^|findstr 20') do set end=%%#
set end=%end:~4,10%
call :timestamp "%start%" "%end%" "1 card TestCases Total"
call :timestamp "%start%" "%end%" "TestCases Total"
echo Running unit tests failed, will exit!
exit /b 8

rem ---------------------------------------------------------------------------------------------
:test_inference
@ECHO OFF
echo    ========================================
echo    Step 5. Testing fluid library for inference ...
echo    ========================================

for /F %%# in ('wmic os get localdatetime^|findstr 20') do set end=%%#
set end=%end:~4,10%
call :timestamp "%start%" "%end%" "1 card TestCases Total"
call :timestamp "%start%" "%end%" "TestCases Total"

cd %work_dir%\paddle\fluid\inference\api\demo_ci
%cache_dir%\tools\busybox64.exe bash run.sh %work_dir:\=/% %WITH_MKL% %WITH_GPU% %cache_dir:\=/%/inference_demo %TENSORRT_ROOT%/include %TENSORRT_ROOT%/lib %MSVC_STATIC_CRT%
goto:eof

:test_inference_error
::echo 1 > %cache_dir%\error_code.txt
::type %cache_dir%\error_code.txt
echo Testing fluid library for inference failed!
exit /b 1

rem ---------------------------------------------------------------------------------------------
:check_change_of_unittest
@ECHO OFF
echo    ========================================
echo    Step 6. Check whether deleting a unit test ...
echo    ========================================

cd /d %work_dir%\build
echo set -e>  check_change_of_unittest.sh
echo set +x>> check_change_of_unittest.sh
echo GITHUB_API_TOKEN=%GITHUB_API_TOKEN% >>  check_change_of_unittest.sh
echo GIT_PR_ID=%AGILE_PULL_ID% >>  check_change_of_unittest.sh
echo BRANCH=%BRANCH%>>  check_change_of_unittest.sh
echo if [ "${GITHUB_API_TOKEN}" == "" ] ^|^| [ "${GIT_PR_ID}" == "" ];then>> check_change_of_unittest.sh
echo     exit 0 >>  check_change_of_unittest.sh
echo fi>>  check_change_of_unittest.sh
echo set -x>> check_change_of_unittest.sh
echo cat ^<^<EOF>>  check_change_of_unittest.sh
echo     ============================================ >>  check_change_of_unittest.sh
echo     Generate unit tests.spec of this PR.         >>  check_change_of_unittest.sh
echo     ============================================ >>  check_change_of_unittest.sh
echo EOF>>  check_change_of_unittest.sh
echo spec_path=$(pwd)/UNITTEST_PR.spec>>  check_change_of_unittest.sh
echo ctest -N ^| awk -F ':' '{print $2}' ^| sed '/^^$/d' ^| sed '$d' ^> ${spec_path}>>  check_change_of_unittest.sh
echo num=$(awk 'END{print NR}' ${spec_path})>> check_change_of_unittest.sh
echo echo "Windows 1 card TestCases count is $num">> check_change_of_unittest.sh
echo echo ipipe_log_param_Windows_1_Card_TestCases_Count: $num>> check_change_of_unittest.sh
echo UPSTREAM_URL='https://github.com/PaddlePaddle/Paddle'>>  check_change_of_unittest.sh
echo origin_upstream_url=`git remote -v ^| awk '{print $1, $2}' ^| uniq ^| grep upstream ^| awk '{print $2}'`>>  check_change_of_unittest.sh
echo if [ "$origin_upstream_url" == "" ]; then>>  check_change_of_unittest.sh
echo     git remote add upstream $UPSTREAM_URL.git>>  check_change_of_unittest.sh
echo elif [ "$origin_upstream_url" ^!= "$UPSTREAM_URL" ] ^\>>  check_change_of_unittest.sh
echo         ^&^& [ "$origin_upstream_url" ^!= "$UPSTREAM_URL.git" ]; then>>  check_change_of_unittest.sh
echo     git remote remove upstream>>  check_change_of_unittest.sh
echo     git remote add upstream $UPSTREAM_URL.git>>  check_change_of_unittest.sh
echo fi>>  check_change_of_unittest.sh
echo if [ ! -e "$(pwd)/../.git/refs/remotes/upstream/$BRANCH" ]; then>>  check_change_of_unittest.sh
echo     git fetch upstream $BRANCH # develop is not fetched>>  check_change_of_unittest.sh
echo fi>>  check_change_of_unittest.sh
echo git checkout -b origin_pr >>  check_change_of_unittest.sh
echo git checkout -f $BRANCH >>  check_change_of_unittest.sh
echo cmake .. -G %GENERATOR% -DCMAKE_BUILD_TYPE=Release -DWITH_AVX=%WITH_AVX% -DWITH_GPU=%WITH_GPU% -DWITH_MKL=%WITH_MKL% ^
-DWITH_TESTING=%WITH_TESTING% -DWITH_PYTHON=%WITH_PYTHON% -DON_INFER=%ON_INFER% ^
-DWITH_INFERENCE_API_TEST=%WITH_INFERENCE_API_TEST% -DTHIRD_PARTY_PATH=%THIRD_PARTY_PATH% ^
-DINFERENCE_DEMO_INSTALL_DIR=%INFERENCE_DEMO_INSTALL_DIR% -DWITH_STATIC_LIB=%WITH_STATIC_LIB% ^
-DWITH_TENSORRT=%WITH_TENSORRT% -DTENSORRT_ROOT="%TENSORRT_ROOT%" -DMSVC_STATIC_CRT=%MSVC_STATIC_CRT% ^
-DWITH_UNITY_BUILD=%WITH_UNITY_BUILD% -DCUDA_ARCH_NAME=%CUDA_ARCH_NAME%  >>  check_change_of_unittest.sh
echo cat ^<^<EOF>>  check_change_of_unittest.sh
echo     ============================================       >>  check_change_of_unittest.sh
echo     Generate unit tests.spec of develop.               >>  check_change_of_unittest.sh
echo     ============================================       >>  check_change_of_unittest.sh
echo EOF>>  check_change_of_unittest.sh
echo spec_path=$(pwd)/UNITTEST_DEV.spec>>  check_change_of_unittest.sh
echo ctest -N ^| awk -F ':' '{print $2}' ^| sed '/^^$/d' ^| sed '$d' ^> ${spec_path}>>  check_change_of_unittest.sh
echo unittest_spec_diff=`python $(pwd)/../tools/diff_unittest.py $(pwd)/UNITTEST_DEV.spec $(pwd)/UNITTEST_PR.spec`>>  check_change_of_unittest.sh
echo if [ "$unittest_spec_diff" ^!= "" ]; then>>  check_change_of_unittest.sh
echo     set +x>> check_change_of_unittest.sh
echo     approval_line=`curl -H "Authorization: token ${GITHUB_API_TOKEN}" https://api.github.com/repos/PaddlePaddle/Paddle/pulls/${GIT_PR_ID}/reviews?per_page=10000`>>  check_change_of_unittest.sh
echo     set -x>> check_change_of_unittest.sh
echo     if [ "$approval_line" ^!= "" ]; then>>  check_change_of_unittest.sh
echo         APPROVALS=`echo ${approval_line} ^|python $(pwd)/../tools/check_pr_approval.py 1 22165420 52485244 6836917`>>  check_change_of_unittest.sh
echo         echo "current pr ${GIT_PR_ID} got approvals: ${APPROVALS}">>  check_change_of_unittest.sh
echo         if [ "${APPROVALS}" == "FALSE" ]; then>>  check_change_of_unittest.sh
echo             echo "************************************"                >>  check_change_of_unittest.sh
echo             echo -e "It is forbidden to disable or delete the unit-test.\n"        >>  check_change_of_unittest.sh
echo             echo -e "If you must delete it temporarily, please add it to[https://github.com/PaddlePaddle/Paddle/wiki/Temporarily-disabled-Unit-Test]."     >>  check_change_of_unittest.sh
echo             echo -e "Then you must have one RD (kolinwei(recommended) or zhouwei25) approval for the deletion of unit-test. \n"                 >>  check_change_of_unittest.sh
echo             echo -e "If you have any problems about deleting unit-test, please read the specification [https://github.com/PaddlePaddle/Paddle/wiki/Deleting-unit-test-is-forbidden]. \n"   >>  check_change_of_unittest.sh
echo             echo -e "Following unit-tests are deleted in this PR: \n ${unittest_spec_diff} \n"     >>  check_change_of_unittest.sh
echo             echo "************************************"                >>  check_change_of_unittest.sh
echo             exit 1 >>  check_change_of_unittest.sh
echo          fi>>  check_change_of_unittest.sh
echo     else>>  check_change_of_unittest.sh
echo          exit 1 >>  check_change_of_unittest.sh
echo     fi>>  check_change_of_unittest.sh
echo fi>>  check_change_of_unittest.sh
echo git checkout -f origin_pr >>  check_change_of_unittest.sh
%cache_dir%\tools\busybox64.exe bash check_change_of_unittest.sh
goto:eof

:check_change_of_unittest_error
exit /b 1

rem ---------------------------------------------------------------------------------------------
:zip_cc_file
cd /d %work_dir%\build
tree /F %cd%\paddle_inference_install_dir\paddle
if exist paddle_inference.zip del paddle_inference.zip
python -c "import shutil;shutil.make_archive('paddle_inference', 'zip', root_dir='paddle_inference_install_dir')"
%cache_dir%\tools\busybox64.exe du -h -k paddle_inference.zip > lib_size.txt
set /p libsize=< lib_size.txt
for /F %%i in ("%libsize%") do (
    set /a libsize_m=%%i/1024
    echo "Windows Paddle_Inference ZIP Size: !libsize_m!M"
)
goto:eof

:zip_cc_file_error
echo Tar inference library failed!
exit /b 1

rem ---------------------------------------------------------------------------------------------
:zip_c_file
cd /d %work_dir%\build
tree /F %cd%\paddle_inference_c_install_dir\paddle
if exist paddle_inference_c.zip del paddle_inference_c.zip
python -c "import shutil;shutil.make_archive('paddle_inference_c', 'zip', root_dir='paddle_inference_c_install_dir')"
%cache_dir%\tools\busybox64.exe du -h -k paddle_inference_c.zip > lib_size.txt
set /p libsize=< lib_size.txt
for /F %%i in ("%libsize%") do (
    set /a libsize_m=%%i/1024
    echo "Windows Paddle_Inference CAPI ZIP Size: !libsize_m!M"
)
goto:eof

:zip_c_file_error
echo Tar inference capi library failed!
exit /b 1

:timestamp
setlocal enabledelayedexpansion
@ECHO OFF
set start=%~1
set dd=%start:~2,2%
set /a dd=100%dd%%%100
set hh=%start:~4,2%
set /a hh=100%hh%%%100
set nn=%start:~6,2%
set /a nn=100%nn%%%100
set ss=%start:~8,2%
set /a ss=100%ss%%%100
set /a start_sec=dd*86400+hh*3600+nn*60+ss
echo %start_sec%

set end=%~2
set dd=%end:~2,2%
set /a dd=100%dd%%%100
if %start:~0,2% NEQ %end:~0,2% (
    set month_day=0
    for %%i in (01 03 05 07 08 10 12) DO if %%i EQU %start:~0,2% set month_day=31
    for %%i in (04 06 09 11) DO if %%i EQU %start:~0,2% set month_day=30
    for %%i in (02) DO if %%i EQU %start:~0,2% set month_day=28
    set /a dd=%dd%+!month_day!
)
set hh=%end:~4,2%
set /a hh=100%hh%%%100
set nn=%end:~6,2%
set /a nn=100%nn%%%100
set ss=%end:~8,2%
set /a ss=100%ss%%%100
set /a end_secs=dd*86400+hh*3600+nn*60+ss
set /a cost_secs=end_secs-start_sec
echo "Windows %~3 Time: %cost_secs%s"
set tempTaskName=%~3
echo ipipe_log_param_Windows_%tempTaskName: =_%_Time: %cost_secs%s
goto:eof


:collect_sccache_hits
sccache -s > sccache_summary.txt
echo    ========================================
echo    sccache statistical summary ...
echo    ========================================
type sccache_summary.txt
for /f "tokens=2,3" %%i in ('type sccache_summary.txt ^| findstr "requests hits" ^| findstr /V "executed C/C++ CUDA"') do set %%i=%%j
if %requests% EQU 0 (
    echo "sccache hit rate: 0%"
    echo ipipe_log_param_sccache_Hit_Hate: 0%
) else (
    set /a rate=!hits!*10000/!requests!
    echo "sccache hit rate: !rate:~0,-2!.!rate:~-2!%%"
    echo ipipe_log_param_sccache_Hit_Hate: !rate:~0,-2!.!rate:~-2!%%
)

goto:eof


rem ---------------------------------------------------------------------------------------------
:success
echo    ========================================
echo    Clean up environment  at the end ...
echo    ========================================
taskkill /f /im cmake.exe /t 2>NUL
taskkill /f /im ninja.exe /t 2>NUL
taskkill /f /im MSBuild.exe /t 2>NUL
taskkill /f /im git.exe /t 2>NUL
taskkill /f /im cl.exe /t 2>NUL
taskkill /f /im lib.exe /t 2>NUL
taskkill /f /im link.exe /t 2>NUL
taskkill /f /im git-remote-https.exe /t 2>NUL
taskkill /f /im vctip.exe /t 2>NUL
taskkill /f /im cvtres.exe /t 2>NUL
taskkill /f /im rc.exe /t 2>NUL
taskkill /f /im mspdbsrv.exe /t 2>NUL
taskkill /f /im csc.exe /t 2>NUL
taskkill /f /im python.exe /t 2>NUL
taskkill /f /im nvcc.exe /t 2>NUL
taskkill /f /im cicc.exe /t 2>NUL
taskkill /f /im ptxas.exe /t 2>NUL
taskkill /f /im op_function_generator.exe /t 2>NUL
wmic process where name="op_function_generator.exe" call terminate 2>NUL
wmic process where name="cvtres.exe" call terminate 2>NUL
wmic process where name="rc.exe" call terminate 2>NUL
wmic process where name="cl.exe" call terminate 2>NUL
wmic process where name="lib.exe" call terminate 2>NUL
wmic process where name="python.exe" call terminate 2>NUL
if "%WITH_TESTING%"=="ON" (
    for /F "tokens=1 delims= " %%# in ('tasklist ^| findstr /i test') do taskkill /f /im %%# /t
)
echo Windows CI run successfully!
exit /b 0

ENDLOCAL
