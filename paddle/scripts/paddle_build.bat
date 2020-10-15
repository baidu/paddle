@ECHO ON
SETLOCAL

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

rem -------clean up environment-----------
set work_dir=%cd%
wmic process where name="op_function_generator.exe" call terminate  2>NUL

rem ------initialize common variable------
if not defined CUDA_TOOLKIT_ROOT_DIR set CUDA_TOOLKIT_ROOT_DIR="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.0"
if not defined BRANCH set BRANCH=develop
if not defined TENSORRT_ROOT set TENSORRT_ROOT="C:/TensorRT-5.1.5.0"
if not defined WITH_MKL set WITH_MKL=ON
if not defined WITH_GPU set WITH_GPU=OFF
if not defined WITH_AVX set WITH_AVX=ON
if not defined WITH_TESTING set WITH_TESTING=ON
if not defined WITH_PYTHON set WITH_PYTHON=ON
if not defined ON_INFER set ON_INFER=ON
if not defined WITH_INFERENCE_API_TEST set WITH_INFERENCE_API_TEST=ON
if not defined WITH_STATIC_LIB set WITH_STATIC_LIB=ON
if not defined WITH_CACHE set WITH_CACHE=ON
if not defined WITH_TPCACHE set WITH_TPCACHE=ON


rem -------set cache build work directory-----------
rmdir build\python /s/q
if "%WITH_CACHE%"=="OFF" (
    rmdir build /s/q
    goto :mkbuild
)

for /F %%# in ('wmic os get localdatetime^|findstr 20') do set datetime=%%#
set day_now=%datetime:~6,2%
set day_before=-1
set /p day_before=< %work_dir%\..\day.txt
if %day_now% NEQ %day_before% (
    echo %day_now% > %work_dir%\..\day.txt
    type %work_dir%\..\day.txt
    rmdir build /s/q
)
git diff origin/develop --stat --name-only | findstr "cmake CMakeLists.txt paddle_build.bat"
if %ERRORLEVEL% EQU 0 (
    rmdir build /s/q
)

:mkbuild
if not exist build (
    mkdir build
)
cd /d build
dir .
dir paddle\fluid\pybind\Release

rem ------initialize the python environment------
if not defined PYTHON_ROOT set PYTHON_ROOT=C:\Python37
set PATH=%PYTHON_ROOT%;%PYTHON_ROOT%\Scripts;%PATH%

rem ToDo: virtual environment can't be deleted safely, some process not exit when task is canceled
rem Now use system python environment temporarily
rem set PYTHON_EXECUTABLE=%PYTHON_ROOT%\python.exe
rem %PYTHON_EXECUTABLE% -m pip install virtualenv
rem %PYTHON_EXECUTABLE% -m virtualenv paddle_winci
rem call paddle_winci\Scripts\activate.bat

rem ------pre install python requirement----------
where python
where pip
pip install --upgrade pip --user
pip install wheel --user
pip install gym --user
pip install -U -r %work_dir%\python\requirements.txt --user
pip install -U -r %work_dir%\python\unittest_py\requirements.txt --user
if %ERRORLEVEL% NEQ 0 (
    call paddle_winci\Scripts\deactivate.bat 2>NUL
    echo pip install requirements.txt failed!
    exit /b 7
)

rem ------pre install clcache and init config----------
pip install clcache
:: set USE_CLCACHE to enable clcache
set USE_CLCACHE=1
:: In some scenarios, CLCACHE_HARDLINK can save one file copy.
set CLCACHE_HARDLINK=1
:: If it takes more than 1000s to obtain the right to use the cache, an error will be reported
set CLCACHE_OBJECT_CACHE_TIMEOUT_MS=1000000
:: set maximum cache size to 20G
clcache.exe -M 21474836480


rem ------set cache third_party------
set cache_dir=%work_dir:Paddle=cache%
dir %cache_dir%
set INFERENCE_DEMO_INSTALL_DIR=%cache_dir:\=/%/inference_demo

if not exist %cache_dir%\tools (
    git clone https://github.com/zhouwei25/tools.git %cache_dir%\tools
)

if "%WITH_TPCACHE%"=="OFF" (
    set THIRD_PARTY_PATH=%work_dir:\=/%/build/third_party
    goto :CASE_%1
)

echo set -ex > cache.sh
echo md5_content=$(cat %work_dir:\=/%/cmake/external/*.cmake  ^|md5sum ^| awk '{print $1}') >> cache.sh
echo echo ${md5_content}^>md5.txt >> cache.sh

%cache_dir%\tools\busybox64.exe cat cache.sh
%cache_dir%\tools\busybox64.exe bash cache.sh

set /p md5=< md5.txt
if "%WITH_GPU%"=="ON" (
    set THIRD_PARTY_PATH=%cache_dir:\=/%/third_party_GPU/%md5%
) else (
    set THIRD_PARTY_PATH=%cache_dir:\=/%/third_party/%md5%
)

goto :CASE_%1

echo "Usage: paddle_build.bat [OPTION]"
echo "OPTION:"
echo "wincheck_mkl: run Windows MKL/GPU/UnitTest CI tasks on Windows"
echo "wincheck_openbals: run Windows OPENBLAS/CPU CI tasks on Windows"
exit /b 1

:CASE_wincheck_mkl
set WITH_MKL=ON
set WITH_GPU=OFF
set MSVC_STATIC_CRT=ON
call :cmake || goto cmake_error
call :build || goto build_error
call :test_whl_pacakage || goto test_whl_pacakage_error
call :unit_test || goto unit_test_error
call :test_inference || goto test_inference_error
call :check_change_of_unittest || goto check_change_of_unittest_error
goto:success

:CASE_wincheck_openblas
set WITH_MKL=OFF
set WITH_GPU=ON
set MSVC_STATIC_CRT=OFF
rem Temporarily turn off WITH_INFERENCE_API_TEST on GPU due to compile hang
set WITH_INFERENCE_API_TEST=OFF
call :cmake || goto cmake_error
call :build || goto build_error
call :test_whl_pacakage || goto test_whl_pacakage_error
:: call :test_inference || goto test_inference_error
goto:success

rem "Other configurations are added here"
rem :CASE_wincheck_others
rem call ...


rem ---------------------------------------------------------------------------------------------
:cmake
echo    ========================================
echo    Step 1. Cmake ...
echo    ========================================

for /F %%# in ('wmic os get localdatetime^|findstr 20') do set start=%%#
set start=%start:~4,10%
echo cmake .. -G "Visual Studio 14 2015 Win64" -DWITH_AVX=%WITH_AVX% -DWITH_GPU=%WITH_GPU% -DWITH_MKL=%WITH_MKL% ^
-DWITH_TESTING=%WITH_TESTING% -DWITH_PYTHON=%WITH_PYTHON% -DCUDA_TOOLKIT_ROOT_DIR=%CUDA_TOOLKIT_ROOT_DIR% ^
-DON_INFER=%ON_INFER% -DWITH_INFERENCE_API_TEST=%WITH_INFERENCE_API_TEST% -DTHIRD_PARTY_PATH=%THIRD_PARTY_PATH% ^
-DINFERENCE_DEMO_INSTALL_DIR=%INFERENCE_DEMO_INSTALL_DIR% -DWITH_STATIC_LIB=%WITH_STATIC_LIB% ^
-DTENSORRT_ROOT=%TENSORRT_ROOT% -DMSVC_STATIC_CRT=%MSVC_STATIC_CRT%

cmake .. -G "Visual Studio 14 2015 Win64" -DWITH_AVX=%WITH_AVX% -DWITH_GPU=%WITH_GPU% -DWITH_MKL=%WITH_MKL% ^
-DWITH_TESTING=%WITH_TESTING% -DWITH_PYTHON=%WITH_PYTHON% -DCUDA_TOOLKIT_ROOT_DIR=%CUDA_TOOLKIT_ROOT_DIR% ^
-DON_INFER=%ON_INFER%  -DWITH_INFERENCE_API_TEST=%WITH_INFERENCE_API_TEST% -DTHIRD_PARTY_PATH=%THIRD_PARTY_PATH% ^
-DINFERENCE_DEMO_INSTALL_DIR=%INFERENCE_DEMO_INSTALL_DIR% -DWITH_STATIC_LIB=%WITH_STATIC_LIB% ^
-DTENSORRT_ROOT=%TENSORRT_ROOT% -DMSVC_STATIC_CRT=%MSVC_STATIC_CRT%
goto:eof

:cmake_error
call paddle_winci\Scripts\deactivate.bat 2>NUL
echo Cmake failed, will exit!
exit /b 7

rem ---------------------------------------------------------------------------------------------
:build
echo    ========================================
echo    Step 2. Buile Paddle ...
echo    ========================================
call "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" amd64

for /F %%# in ('wmic cpu get NumberOfLogicalProcessors^|findstr [0-9]') do set /a PARALLEL_PROJECT_COUNT=%%#*8/10
set build_times=1
:build_tp
echo Build third_party the %build_times% time:
msbuild /m /p:Configuration=Release /verbosity:quiet third_party.vcxproj
if %ERRORLEVEL% NEQ 0 (
    set /a build_times=%build_times%+1  
    if %build_times% GTR 2 (
        exit /b 7
    ) else (
        echo Build third_party failed, will retry!
        goto :build_tp
    )
)
echo Build third_party successfully!

set build_times=1
:build_paddle
echo Build Paddle the %build_times% time:
msbuild /m:%PARALLEL_PROJECT_COUNT% /p:TrackFileAccess=false /p:CLToolExe=clcache.exe /p:CLToolPath=%PYTHON_ROOT%\Scripts /p:Configuration=Release /verbosity:minimal paddle.sln
if %ERRORLEVEL% NEQ 0 (
    set /a build_times=%build_times%+1
    if %build_times% GTR 1 (
        exit /b 7
    ) else (
        echo Build Paddle failed, will retry!
        goto :build_paddle
    )
)

echo Build Paddle successfully!

goto:eof

:build_error
call paddle_winci\Scripts\deactivate.bat 2>NUL
echo Build Paddle failed, will exit!
exit /b 7

rem ---------------------------------------------------------------------------------------------
:test_whl_pacakage
echo    ========================================
echo    Step 3. Test pip install whl package ...
echo    ========================================

for /F %%# in ('wmic os get localdatetime^|findstr 20') do set end=%%#
set end=%end:~4,10%
call :timestamp "%start%" "%end%" "Build"
tree /F %cd%\paddle_inference_install_dir\paddle
%cache_dir%\tools\busybox64.exe du -h -d 0 %cd%\paddle_inference_install_dir\paddle\lib > lib_size.txt
set /p libsize=< lib_size.txt
for /F %%i in ("%libsize%") do echo "Windows Paddle_Inference Size: %%i"
%cache_dir%\tools\busybox64.exe du -h -d 0 %cd%\python\dist > whl_size.txt
set /p whlsize=< whl_size.txt
for /F %%i in ("%whlsize%") do echo "Windows PR whl Size: %%i"
dir /s /b python\dist\*.whl > whl_file.txt
set /p PADDLE_WHL_FILE_WIN=< whl_file.txt

pip uninstall -y paddlepaddle
pip uninstall -y paddlepaddle-gpu
pip install -U %PADDLE_WHL_FILE_WIN% --user
if %ERRORLEVEL% NEQ 0 (
    call paddle_winci\Scripts\deactivate.bat 2>NUL
    echo pip install whl package failed!
    exit /b 1
)

python %work_dir%\paddle\scripts\installation_validate.py
goto:eof

:test_whl_pacakage_error
call paddle_winci\Scripts\deactivate.bat 2>NUL
echo Test import paddle failed, will exit!
exit /b 1

rem ---------------------------------------------------------------------------------------------
:unit_test
echo    ========================================
echo    Step 4. Running unit tests ...
echo    ========================================

for /F %%# in ('wmic os get localdatetime^|findstr 20') do set start=%%#
set start=%start:~4,10%
dir %THIRD_PARTY_PATH:/=\%\install\openblas\lib
dir %THIRD_PARTY_PATH:/=\%\install\openblas\bin
dir %THIRD_PARTY_PATH:/=\%\install\zlib\bin
dir %THIRD_PARTY_PATH:/=\%\install\mklml\lib
dir %THIRD_PARTY_PATH:/=\%\install\mkldnn\bin
dir %THIRD_PARTY_PATH:/=\%\install\warpctc\bin

pip install requests
python %work_dir%\tools\get_quick_disable_lt.py > Output
if %errorlevel%==0 (
    set /p disable_ut_quickly=<Output
    DEL Output
    ) else (
    set disable_ut_quickly=''
)

set PATH=%THIRD_PARTY_PATH:/=\%\install\openblas\lib;%THIRD_PARTY_PATH:/=\%\install\openblas\bin;^
%THIRD_PARTY_PATH:/=\%\install\zlib\bin;%THIRD_PARTY_PATH:/=\%\install\mklml\lib;^
%THIRD_PARTY_PATH:/=\%\install\mkldnn\bin;%THIRD_PARTY_PATH:/=\%\install\warpctc\bin;%PATH%
ctest.exe -E "(%disable_ut_quickly%)" --output-on-failure -C Release -j 8 --repeat until-pass:4 after-timeout:4
goto:eof

:unit_test_error
call paddle_winci\Scripts\deactivate.bat 2>NUL
for /F %%# in ('wmic os get localdatetime^|findstr 20') do set end=%%#
set end=%end:~4,10%
call :timestamp "%start%" "%end%" "1 card TestCases Total"
call :timestamp "%start%" "%end%" "TestCases Total"
echo Running unit tests failed, will exit!
exit /b 8

rem ---------------------------------------------------------------------------------------------
:test_inference
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
call paddle_winci\Scripts\deactivate.bat 2>NUL
echo Testing fluid library for inference failed!
exit /b 1

rem ---------------------------------------------------------------------------------------------
:check_change_of_unittest
echo    ========================================
echo    Step 6. Check whether deleting a unit test ...
echo    ========================================

cd /d %work_dir%\build
echo set -ex>  check_change_of_unittest.sh
echo GITHUB_API_TOKEN=%GITHUB_API_TOKEN% >>  check_change_of_unittest.sh
echo GIT_PR_ID=%AGILE_PULL_ID% >>  check_change_of_unittest.sh
echo BRANCH=%BRANCH%>>  check_change_of_unittest.sh
echo if [ "${GITHUB_API_TOKEN}" == "" ] ^|^| [ "${GIT_PR_ID}" == "" ];then>> check_change_of_unittest.sh
echo     exit 0 >>  check_change_of_unittest.sh
echo fi>>  check_change_of_unittest.sh
echo cat ^<^<EOF>>  check_change_of_unittest.sh
echo     ============================================ >>  check_change_of_unittest.sh
echo     Generate unit tests.spec of this PR.         >>  check_change_of_unittest.sh
echo     ============================================ >>  check_change_of_unittest.sh
echo EOF>>  check_change_of_unittest.sh
echo spec_path=$(pwd)/UNITTEST_PR.spec>>  check_change_of_unittest.sh
echo ctest -N ^| awk -F ':' '{print $2}' ^| sed '/^^$/d' ^| sed '$d' ^> ${spec_path}>>  check_change_of_unittest.sh
echo num=$(awk 'END{print NR}' ${spec_path})>> check_change_of_unittest.sh
echo echo "Windows 1 card TestCases count is $num">> check_change_of_unittest.sh
echo UPSTREAM_URL='https://github.com/PaddlePaddle/Paddle'>>  check_change_of_unittest.sh
echo origin_upstream_url=`git remote -v ^| awk '{print $1, $2}' ^| uniq ^| grep upstream ^| awk '{print $2}'`>>  check_change_of_unittest.sh
echo if [ "$origin_upstream_url" == "" ]; then>>  check_change_of_unittest.sh
echo     git remote add upstream $UPSTREAM_URL.git>>  check_change_of_unittest.sh
echo elif [ "$origin_upstream_url" != "$UPSTREAM_URL" ] \>>  check_change_of_unittest.sh
echo         ^&^& [ "$origin_upstream_url" != "$UPSTREAM_URL.git" ]; then>>  check_change_of_unittest.sh
echo     git remote remove upstream>>  check_change_of_unittest.sh
echo     git remote add upstream $UPSTREAM_URL.git>>  check_change_of_unittest.sh
echo fi>>  check_change_of_unittest.sh
echo if [ ! -e "$(pwd)/../.git/refs/remotes/upstream/$BRANCH" ]; then>>  check_change_of_unittest.sh
echo     git fetch upstream $BRANCH # develop is not fetched>>  check_change_of_unittest.sh
echo fi>>  check_change_of_unittest.sh
echo git checkout -b origin_pr >>  check_change_of_unittest.sh
echo git checkout -f $BRANCH >>  check_change_of_unittest.sh
echo cmake .. -G "Visual Studio 14 2015 Win64" -DWITH_AVX=%WITH_AVX% -DWITH_GPU=%WITH_GPU% -DWITH_MKL=%WITH_MKL% ^
-DWITH_TESTING=%WITH_TESTING% -DWITH_PYTHON=%WITH_PYTHON% -DCUDA_TOOLKIT_ROOT_DIR=%CUDA_TOOLKIT_ROOT_DIR% ^
-DON_INFER=%ON_INFER% -DWITH_INFERENCE_API_TEST=%WITH_INFERENCE_API_TEST% -DTHIRD_PARTY_PATH=%THIRD_PARTY_PATH% ^
-DINFERENCE_DEMO_INSTALL_DIR=%INFERENCE_DEMO_INSTALL_DIR% >>  check_change_of_unittest.sh
echo cat ^<^<EOF>>  check_change_of_unittest.sh
echo     ============================================       >>  check_change_of_unittest.sh
echo     Generate unit tests.spec of develop.               >>  check_change_of_unittest.sh
echo     ============================================       >>  check_change_of_unittest.sh
echo EOF>>  check_change_of_unittest.sh
echo spec_path=$(pwd)/UNITTEST_DEV.spec>>  check_change_of_unittest.sh
echo ctest -N ^| awk -F ':' '{print $2}' ^| sed '/^^$/d' ^| sed '$d' ^> ${spec_path}>>  check_change_of_unittest.sh
echo unittest_spec_diff=`python $(pwd)/../tools/diff_unittest.py $(pwd)/UNITTEST_DEV.spec $(pwd)/UNITTEST_PR.spec`>>  check_change_of_unittest.sh
echo if [ "$unittest_spec_diff" != "" ]; then>>  check_change_of_unittest.sh
echo     # approval_user_list: XiaoguangHu01 46782768,luotao1 6836917,phlrain 43953930,lanxianghit 47554610, zhouwei25 52485244, kolinwei 22165420>>  check_change_of_unittest.sh
echo     approval_line=`curl -H "Authorization: token ${GITHUB_API_TOKEN}" https://api.github.com/repos/PaddlePaddle/Paddle/pulls/${GIT_PR_ID}/reviews?per_page=10000`>>  check_change_of_unittest.sh
echo     set +x>>  check_change_of_unittest.sh
echo     if [ "$approval_line" != "" ]; then>>  check_change_of_unittest.sh
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
call paddle_winci\Scripts\deactivate.bat 2>NUL
exit /b 1


:timestamp
echo on
setlocal enabledelayedexpansion
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
goto:eof


rem ---------------------------------------------------------------------------------------------
:success
echo    ========================================
echo    Clean up environment  at the end ...
echo    ========================================
taskkill /f /im cmake.exe  2>NUL
taskkill /f /im msbuild.exe 2>NUL
taskkill /f /im git.exe 2>NUL
taskkill /f /im cl.exe 2>NUL
taskkill /f /im lib.exe 2>NUL
taskkill /f /im link.exe 2>NUL
taskkill /f /im git-remote-https.exe 2>NUL
taskkill /f /im vctip.exe 2>NUL
taskkill /f /im cvtres.exe 2>NUL
taskkill /f /im rc.exe 2>NUL
wmic process where name="op_function_generator.exe" call terminate 2>NUL
taskkill /f /im python.exe  2>NUL
taskkill /f /im python.exe  2>NUL
echo Windows CI run successfully!
exit /b 0

ENDLOCAL
