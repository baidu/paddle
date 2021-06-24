# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
please make sure to run in the tools path
usage: python sample_test.py {cpu or gpu} 
    {cpu or gpu}: running in cpu version or gpu version

for example, you can run cpu version python2 testing like this:

    python sampcd_processor.py cpu 

"""
import os
import sys
import subprocess
import multiprocessing
import platform
import inspect
import argparse
import shutil
import re
import logging
import time

logger = logging.getLogger()
if logger.handlers:
    console = logger.handlers[
        0]  # we assume the first handler is the one we want to configure
else:
    console = logging.StreamHandler(stream=sys.stderr)
    logger.addHandler(console)
console.setFormatter(logging.Formatter("%(message)s"))

RUN_ON_DEVICE = 'cpu'
SAMPLE_CODE_TEST_CAPACITY = set()
GPU_ID = 0
whl_error = []
API_DEV_SPEC_FN = 'paddle/fluid/API_DEV.spec'
API_PR_SPEC_FN = 'paddle/fluid/API_PR.spec'
API_DIFF_SPEC_FN = 'dev_pr_diff_api.spec'
SAMPLECODE_TEMPDIR = 'samplecode_temp'
ENV_KEY_CODES_FRONTEND = 'CODES_INSERTED_INTO_FRONTEND'
ENV_KEY_TEST_CAPACITY = 'SAMPLE_CODE_TEST_CAPACITY'
SUMMARY_INFO = {
    'success': [],
    'failed': [],
    'skiptest': [],
    'nocodes': [],
    # ... required not-match
}


def find_all(srcstr, substr):
    """
    to find all desired substring in the source string
     and return their starting indices as a list

    Args:
        srcstr(str): the parent string
        substr(str): substr

    Returns:
        list: a list of the indices of the substrings
              found
    """
    indices = []
    gotone = srcstr.find(substr)
    while (gotone != -1):
        indices.append(gotone)
        gotone = srcstr.find(substr, gotone + 1)
    return indices


def find_last_future_line_end(cbstr):
    """
    find the last `__future__` line.

    Args:
        docstr(str): docstring
    Return:
        index of the line end or None.
    """
    pat = re.compile('__future__.*\n')
    lastmo = None
    it = re.finditer(pat, cbstr)
    while True:
        try:
            lastmo = next(it)
        except StopIteration:
            break
    if lastmo:
        return lastmo.end()
    else:
        return None


def extract_code_blocks_from_docstr(docstr):
    """
    extract code-blocks from the given docstring.

    DON'T include the multiline-string definition in code-blocks.
    The *Examples* section must be the last.

    Args:
        docstr(str): docstring
    Return:
        code_blocks: A list of code-blocks, indent removed. 
                     element {'name': the code-block's name, 'id': sequence id.
                              'codes': codes, 'required': 'gpu'}
    """
    code_blocks = []

    mo = re.search(r"Examples:", docstr)
    if mo is None:
        return code_blocks
    ds_list = docstr[mo.start():].replace("\t", '    ').split("\n")
    lastlineindex = len(ds_list) - 1

    cb_start_pat = re.compile(r"code-block::\s*python")
    cb_param_pat = re.compile(r"^\s*:(\w+):\s*(\S*)\s*$")
    cb_required_pat = re.compile(r"^\s*#\s*require[s|d]\s*:\s*(\S+)\s*$")

    cb_info = {}
    cb_info['cb_started'] = False
    cb_info['cb_cur'] = []
    cb_info['cb_cur_indent'] = -1
    cb_info['cb_cur_name'] = None
    cb_info['cb_cur_seq_id'] = 0
    cb_info['cb_required'] = None

    def _cb_started():
        # nonlocal cb_started, cb_cur_name, cb_required, cb_cur_seq_id
        cb_info['cb_started'] = True
        cb_info['cb_cur_seq_id'] += 1
        cb_info['cb_cur_name'] = None
        cb_info['cb_required'] = None

    def _append_code_block():
        # nonlocal code_blocks, cb_cur, cb_cur_name, cb_cur_seq_id, cb_required
        code_blocks.append({
            'codes': inspect.cleandoc("\n".join(cb_info['cb_cur'])),
            'name': cb_info['cb_cur_name'],
            'id': cb_info['cb_cur_seq_id'],
            'required': cb_info['cb_required'],
        })

    for lineno, linecont in enumerate(ds_list):
        if re.search(cb_start_pat, linecont):
            if not cb_info['cb_started']:
                _cb_started()
                continue
            else:
                # cur block end
                if len(cb_info['cb_cur']):
                    _append_code_block()
                _cb_started()  # another block started
                cb_info['cb_cur_indent'] = -1
                cb_info['cb_cur'] = []
        else:
            if cb_info['cb_started']:
                # handle the code-block directive's options
                mo_p = cb_param_pat.match(linecont)
                if mo_p:
                    if mo_p.group(1) == 'name':
                        cb_info['cb_cur_name'] = mo_p.group(2)
                    continue
                # read the required directive
                mo_r = cb_required_pat.match(linecont)
                if mo_r:
                    cb_info['cb_required'] = mo_r.group(1)
                # docstring end
                if lineno == lastlineindex:
                    mo = re.search(r"\S", linecont)
                    if mo is not None and cb_info['cb_cur_indent'] <= mo.start(
                    ):
                        cb_info['cb_cur'].append(linecont)
                    if len(cb_info['cb_cur']):
                        _append_code_block()
                    break
                # check indent for cur block start and end.
                mo = re.search(r"\S", linecont)
                if mo is None:
                    continue
                if cb_info['cb_cur_indent'] < 0:
                    # find the first non empty line
                    cb_info['cb_cur_indent'] = mo.start()
                    cb_info['cb_cur'].append(linecont)
                else:
                    if cb_info['cb_cur_indent'] <= mo.start():
                        cb_info['cb_cur'].append(linecont)
                    else:
                        if linecont[mo.start()] == '#':
                            continue
                        else:
                            # block end
                            if len(cb_info['cb_cur']):
                                _append_code_block()
                            cb_info['cb_started'] = False
                            cb_info['cb_cur_indent'] = -1
                            cb_info['cb_cur'] = []
    return code_blocks


def get_test_capacity():
    """
    collect capacities and set to SAMPLE_CODE_TEST_CAPACITY
    """
    global SAMPLE_CODE_TEST_CAPACITY  # write
    global ENV_KEY_TEST_CAPACITY, RUN_ON_DEVICE  # readonly
    if ENV_KEY_TEST_CAPACITY in os.environ:
        for r in os.environ[ENV_KEY_TEST_CAPACITY].split(','):
            rr = r.strip().lower()
            if r:
                SAMPLE_CODE_TEST_CAPACITY.add(rr)
    if 'cpu' not in SAMPLE_CODE_TEST_CAPACITY:
        SAMPLE_CODE_TEST_CAPACITY.add('cpu')

    if RUN_ON_DEVICE:
        SAMPLE_CODE_TEST_CAPACITY.add(RUN_ON_DEVICE)


def is_required_match(requirestr, cbtitle='not-specified'):
    """
    search the required instruction in the code-block, and check it match the current running environment.
    
    environment values of equipped: cpu, gpu, xpu, distributed, skip
    the 'skip' is the special flag to skip the test, so is_required_match will return False directly.

    Args:
        requirestr(str): the required string.
        cbtitle(str): the title of the code-block.
    returns:
        True - yes, matched
        False - not match
        None - skipped  # trick
    """
    global SAMPLE_CODE_TEST_CAPACITY, RUN_ON_DEVICE  # readonly
    requires = set(['cpu'])
    if requirestr:
        for r in requirestr.split(','):
            rr = r.strip().lower()
            if rr:
                requires.add(rr)
    else:
        requires.add(RUN_ON_DEVICE)
    if 'skip' in requires or 'skiptest' in requires:
        logger.info('%s: skipped', cbtitle)
        return None

    if all([
            k in SAMPLE_CODE_TEST_CAPACITY for k in requires
            if k not in ['skip', 'skiptest']
    ]):
        return True

    logger.info('%s: the equipments [%s] not match the required [%s].', cbtitle,
                ','.join(SAMPLE_CODE_TEST_CAPACITY), ','.join(requires))
    return False


def insert_codes_into_codeblock(codeblock, apiname='not-specified'):
    """
    insert some codes in the frontend and backend into the code-block.
    """
    global ENV_KEY_CODES_FRONTEND, GPU_ID, RUN_ON_DEVICE  # readonly
    inserted_codes_f = ''
    inserted_codes_b = ''
    if ENV_KEY_CODES_FRONTEND in os.environ and os.environ[
            ENV_KEY_CODES_FRONTEND]:
        inserted_codes_f = os.environ[ENV_KEY_CODES_FRONTEND]
    else:
        cpu_str = '\nimport os\nos.environ["CUDA_VISIBLE_DEVICES"] = ""\n'
        gpu_str = '\nimport os\nos.environ["CUDA_VISIBLE_DEVICES"] = "{}"\n'.format(
            GPU_ID)
        if 'required' in codeblock and codeblock['required']:
            if codeblock['required'] == 'cpu':
                inserted_codes_f = cpu_str
            elif codeblock['required'] == 'gpu':
                inserted_codes_f = gpu_str
        else:
            if RUN_ON_DEVICE == "cpu":
                inserted_codes_f = cpu_str
            elif RUN_ON_DEVICE == "gpu":
                inserted_codes_f = gpu_str
    inserted_codes_b = '\nprint("{}\'s sample code (name:{}, id:{}) is executed successfully!")'.format(
        apiname, codeblock['name'], codeblock['id'])

    cb = codeblock['codes']
    last_future_line_end = find_last_future_line_end(cb)
    if last_future_line_end:
        return cb[:last_future_line_end] + inserted_codes_f + cb[
            last_future_line_end:] + inserted_codes_b
    else:
        return inserted_codes_f + cb + inserted_codes_b


def sampcd_extract_to_file(srccom, name, htype="def", hname=""):
    """
    Extract sample codes from __doc__, and write them to files.

    Args:
        srccom(str): the source comment of some API whose
                     example codes will be extracted and run.
        name(str): the name of the API.
        htype(str): the type of hint banners, def/class/method.
        hname(str): the name of the hint  banners , e.t. def hname.

    Returns:
        sample_code_filenames(list of str)
    """
    global GPU_ID, RUN_ON_DEVICE, SAMPLECODE_TEMPDIR  # readonly
    global SUMMARY_INFO  # update

    codeblocks = extract_code_blocks_from_docstr(srccom)
    if len(codeblocks) == 0:
        SUMMARY_INFO['nocodes'].append(name)
        # detect sample codes using >>> to format and consider this situation as wrong
        logger.info(htype + " name:" + name)
        logger.info("-----------------------")
        if srccom.find("Examples:") != -1:
            logger.info("----example code check----")
            if srccom.find(">>>") != -1:
                logger.warning(r"""Deprecated sample code style:
    Examples:
        >>>codeline
        >>>codeline

Please use '.. code-block:: python' to format the sample code.""")
                return []
        else:
            logger.warning("Error: No sample code!")
            return []

    sample_code_filenames = []
    for y, cb in enumerate(codeblocks):
        matched = is_required_match(cb['required'], name)
        # matched has three states:
        # True - please execute it;
        # None - no sample code found;
        # False - it need other special equipment or environment.
        # so, the following conditional statements are intentionally arranged.
        if matched == True:
            tfname = os.path.join(SAMPLECODE_TEMPDIR, '{}_example{}'.format(
                name, '.py'
                if len(codeblocks) == 1 else '_{}.py'.format(y + 1)))
            with open(tfname, 'w') as tempf:
                sampcd = insert_codes_into_codeblock(cb, name)
                tempf.write(sampcd)
            sample_code_filenames.append(tfname)
        elif matched is None:
            logger.info('{}\' code block (name:{}, id:{}) is skipped.'.format(
                name, cb['name'], cb['id']))
            SUMMARY_INFO['skiptest'].append("{}-{}".format(name, cb['id']))
        elif matched == False:
            logger.info(
                '{}\' code block (name:{}, id:{}) required({}) not match capacity({}).'.
                format(name, cb['name'], cb['id'], cb['required'],
                       SAMPLE_CODE_TEST_CAPACITY))
            if cb['required'] not in SUMMARY_INFO:
                SUMMARY_INFO[cb['required']] = []
            SUMMARY_INFO[cb['required']].append("{}-{}".format(name, cb['id']))

    return sample_code_filenames


def execute_samplecode(tfname):
    """
    Execute a sample-code test

    Args:
        tfname: the filename of the sample code
    
    Returns:
        result: success or not
        tfname: same as the input argument
        msg: the stdout output of the sample code executing
        time: time consumed by sample code
    """
    result = True
    msg = None
    if platform.python_version()[0] in ["3"]:
        cmd = [sys.executable, tfname]
    else:
        logger.error("Error: fail to parse python version!")
        result = False
        exit(1)

    logger.info("----example code check----")
    logger.info("executing sample code: %s", tfname)
    start_time = time.time()
    subprc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = subprc.communicate()
    msg = "".join(output.decode(encoding='utf-8'))
    err = "".join(error.decode(encoding='utf-8'))
    end_time = time.time()

    if subprc.returncode != 0:
        with open(tfname, 'r') as f:
            logger.warning("""Sample code error found in %s:
-----------------------
%s
-----------------------
subprocess return code: %d
Error Raised from Sample Code:
stderr: %s
stdout: %s
""", tfname, f.read(), subprc.returncode, err, msg)
        logger.info("----example code check failed----")
        result = False
    else:
        logger.info("----example code check success----")

    # msg is the returned code execution report
    return result, tfname, msg, end_time - start_time


def get_filenames(full_test=False):
    '''
    this function will get the sample code files that pending for check.

    Args:
        full_test: the full apis or the increment

    Returns:

        dict: the sample code files pending for check .

    '''
    global whl_error
    import paddle
    whl_error = []
    if full_test:
        get_full_api_from_pr_spec()
    else:
        get_incrementapi()
    all_sample_code_filenames = {}
    with open(API_DIFF_SPEC_FN) as f:
        for line in f.readlines():
            api = line.replace('\n', '')
            try:
                api_obj = eval(api)
            except AttributeError:
                whl_error.append(api)
                continue
            except SyntaxError:
                logger.warning('line:%s, api:%s', line, api)
                # paddle.Tensor.<lambda>
                continue
            if hasattr(api_obj, '__doc__') and api_obj.__doc__:
                sample_code_filenames = sampcd_extract_to_file(api_obj.__doc__,
                                                               api)
                for tfname in sample_code_filenames:
                    all_sample_code_filenames[tfname] = api
    return all_sample_code_filenames


def get_api_md5(path):
    """
    read the api spec file, and scratch the md5sum value of every api's docstring.

    Args:
        path: the api spec file. ATTENTION the path relative
    
    Returns:
        api_md5(dict): key is the api's real fullname, value is the md5sum.
    """
    api_md5 = {}
    API_spec = os.path.abspath(os.path.join(os.getcwd(), "..", path))
    if not os.path.isfile(API_spec):
        return api_md5
    pat = re.compile(r'\((paddle[^,]+)\W*document\W*([0-9a-z]{32})')
    patArgSpec = re.compile(
        r'^(paddle[^,]+)\s+\(ArgSpec.*document\W*([0-9a-z]{32})')
    with open(API_spec) as f:
        for line in f.readlines():
            mo = pat.search(line)
            if not mo:
                mo = patArgSpec.search(line)
            if mo:
                api_md5[mo.group(1)] = mo.group(2)
    return api_md5


def get_full_api():
    """
    get all the apis
    """
    global API_DIFF_SPEC_FN  ## readonly
    from print_signatures import get_all_api_from_modulelist
    member_dict = get_all_api_from_modulelist()
    with open(API_DIFF_SPEC_FN, 'w') as f:
        f.write("\n".join(member_dict.keys()))


def get_full_api_by_walk():
    """
    get all the apis
    """
    global API_DIFF_SPEC_FN  ## readonly
    from print_signatures import get_all_api
    apilist = get_all_api()
    with open(API_DIFF_SPEC_FN, 'w') as f:
        f.write("\n".join([ai[0] for ai in apilist]))


def get_full_api_from_pr_spec():
    """
    get all the apis
    """
    global API_PR_SPEC_FN, API_DIFF_SPEC_FN  ## readonly
    pr_api = get_api_md5(API_PR_SPEC_FN)
    if len(pr_api):
        with open(API_DIFF_SPEC_FN, 'w') as f:
            f.write("\n".join(pr_api.keys()))
    else:
        get_full_api_by_walk()


def get_incrementapi():
    '''
    this function will get the apis that difference between API_DEV.spec and API_PR.spec.
    '''
    global API_DEV_SPEC_FN, API_PR_SPEC_FN, API_DIFF_SPEC_FN  ## readonly
    dev_api = get_api_md5(API_DEV_SPEC_FN)
    pr_api = get_api_md5(API_PR_SPEC_FN)
    with open(API_DIFF_SPEC_FN, 'w') as f:
        for key in pr_api:
            if key in dev_api:
                if dev_api[key] != pr_api[key]:
                    logger.debug("%s in dev is %s, different from pr's %s", key,
                                 dev_api[key], pr_api[key])
                    f.write(key)
                    f.write('\n')
            else:
                logger.debug("%s is not in dev", key)
                f.write(key)
                f.write('\n')


arguments = [
    # flags, dest, type, default, help
    ['--gpu_id', 'gpu_id', int, 0, 'GPU device id to use [0]'],
    ['--logf', 'logf', str, None, 'file for logging'],
    ['--threads', 'threads', int, 0, 'sub processes number'],
]


def parse_args():
    """
    Parse input arguments
    """
    global arguments
    parser = argparse.ArgumentParser(description='run Sample Code Test')
    # parser.add_argument('--cpu', dest='cpu_mode', action="store_true",
    #                     help='Use CPU mode (overrides --gpu)')
    # parser.add_argument('--gpu', dest='gpu_mode', action="store_true")
    parser.add_argument('--debug', dest='debug', action="store_true")
    parser.add_argument('--full-test', dest='full_test', action="store_true")
    parser.add_argument('mode', type=str, help='run on device', default='cpu')
    for item in arguments:
        parser.add_argument(
            item[0], dest=item[1], help=item[4], type=item[2], default=item[3])

    if len(sys.argv) == 1:
        args = parser.parse_args(['cpu'])
        return args
    #    parser.print_help()
    #    sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    if args.logf:
        logfHandler = logging.FileHandler(args.logf)
        logfHandler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(funcName)s:%(lineno)d - %(levelname)s - %(message)s"
            ))
        logger.addHandler(logfHandler)

    if args.mode == "gpu":
        GPU_ID = args.gpu_id
        logger.info("using GPU_ID %d", GPU_ID)
    elif args.mode != "cpu":
        logger.error("Unrecognized argument:%s, 'cpu' or 'gpu' is desired.",
                     args.mode)
        sys.exit("Invalid arguments")
    RUN_ON_DEVICE = args.mode
    get_test_capacity()
    logger.info("API check -- Example Code")
    logger.info("sample_test running under python %s",
                platform.python_version())

    if os.path.exists(SAMPLECODE_TEMPDIR):
        if not os.path.isdir(SAMPLECODE_TEMPDIR):
            os.remove(SAMPLECODE_TEMPDIR)
            os.mkdir(SAMPLECODE_TEMPDIR)
    else:
        os.mkdir(SAMPLECODE_TEMPDIR)

    filenames = get_filenames(args.full_test)
    if len(filenames) == 0 and len(whl_error) == 0:
        logger.info("-----API_PR.spec is the same as API_DEV.spec-----")
        exit(0)
    logger.info("API_PR is diff from API_DEV: %s", filenames)

    threads = multiprocessing.cpu_count()
    if args.threads:
        threads = args.threads
    po = multiprocessing.Pool(threads)
    results = po.map_async(execute_samplecode, filenames.keys())
    po.close()
    po.join()

    result = results.get()

    # delete temp files
    if not args.debug:
        shutil.rmtree(SAMPLECODE_TEMPDIR)

    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    logger.addHandler(stdout_handler)
    logger.info("----------------End of the Check--------------------")
    if len(whl_error) != 0:
        logger.info("%s is not in whl.", whl_error)
        logger.info("")
        logger.info("Please check the whl package and API_PR.spec!")
        logger.info("You can follow these steps in order to generate API.spec:")
        logger.info("1. cd ${paddle_path}, compile paddle;")
        logger.info("2. pip install build/python/dist/(build whl package);")
        logger.info(
            "3. run 'python tools/print_signatures.py paddle > paddle/fluid/API.spec'."
        )
        for temp in result:
            if not temp[0]:
                logger.info("In addition, mistakes found in sample codes: %s",
                            temp[1])
        logger.info("----------------------------------------------------")
        exit(1)
    else:
        timeovered_test = {}
        for temp in result:
            if not temp[0]:
                logger.info("In addition, mistakes found in sample codes: %s",
                            temp[1])
                SUMMARY_INFO['failed'].append(temp[1])
            else:
                SUMMARY_INFO['success'].append(temp[1])
            if temp[3] > 10:
                timeovered_test[temp[1]] = temp[3]

        if len(timeovered_test):
            logger.info("%d sample codes ran time over 10s",
                        len(timeovered_test))
            if args.debug:
                for k, v in timeovered_test.items():
                    logger.info('{} - {}s'.format(k, v))
        if len(SUMMARY_INFO['success']):
            logger.info("%d sample codes ran success",
                        len(SUMMARY_INFO['success']))
        for k, v in SUMMARY_INFO.items():
            if k not in ['success', 'failed', 'skiptest', 'nocodes']:
                logger.info("%d sample codes required not match for %s",
                            len(v), k)
        if len(SUMMARY_INFO['skiptest']):
            logger.info("%d sample codes skipped",
                        len(SUMMARY_INFO['skiptest']))
            if args.debug:
                logger.info('\n'.join(SUMMARY_INFO['skiptest']))
        if len(SUMMARY_INFO['nocodes']):
            logger.info("%d apis don't have sample codes",
                        len(SUMMARY_INFO['nocodes']))
            if args.debug:
                logger.info('\n'.join(SUMMARY_INFO['nocodes']))
        if len(SUMMARY_INFO['failed']):
            logger.info("%d sample codes ran failed",
                        len(SUMMARY_INFO['failed']))
            logger.info('\n'.join(SUMMARY_INFO['failed']))
            logger.info(
                "Mistakes found in sample codes. Please recheck the sample codes."
            )
            exit(1)

    logger.info("Sample code check is successful!")
