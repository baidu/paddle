# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import os
import json
import logging
import argparse


def check_path_exists(path):
    """Assert whether file/directory exists.
    """
    assert os.path.exists(path), "%s does not exist." % path


def parse_case_name(log_file_name):
    """Parse case name.
    """
    case_id, case_info = log_file_name.split("-")
    # case_info_list -> [framework, device, task, direction]
    case_info_list = case_info.split(".")[0].split("_")

    return "%s (%s on %s)" % (case_id, case_info_list[3], case_info_list[1])


def parse_log_file(log_file):
    """Load one case result from log file.
    """
    check_path_exists(log_file)

    result = None
    with open(log_file) as f:
        for line in f.read().strip().split('\n')[::-1]:
            try:
                result = json.loads(line)
                # maybe get a string
                if not isinstance(result, dict):
                    continue
                if result.get("disabled", False) == True:
                    return None
                return result
            except ValueError:
                pass  # do nothing

    if result is None:
        logging.warning("Parse %s fail!" % log_file)

    return result


def load_benchmark_result_from_logs_dir(logs_dir):
    """Load benchmark result from logs directory.
    """
    check_path_exists(logs_dir)

    log_file_path = lambda log_file: os.path.join(logs_dir, log_file)
    result_lambda = lambda log_file: (log_file, parse_log_file(log_file_path(log_file)))

    return dict(map(result_lambda, os.listdir(logs_dir)))


def check_speed_result(case_name, develop_data, pr_data, pr_result):
    """Check speed differences between develop and pr.
    """
    pr_gpu_time = pr_data.get("gpu_time")
    develop_gpu_time = develop_data.get("gpu_time")
    gpu_time_diff = (pr_gpu_time - develop_gpu_time) / develop_gpu_time

    pr_total_time = pr_data.get("total")
    develop_total_time = develop_data.get("total")
    total_time_diff = (pr_total_time - develop_total_time) / develop_total_time

    logging.info("------ OP: %s ------" % case_name)
    logging.info("GPU time change: %.5f%% (develop: %.7f -> PR: %.7f)" %
                 (gpu_time_diff * 100, develop_gpu_time, pr_gpu_time))
    logging.info("Total time change: %.5f%% (develop: %.7f -> PR: %.7f)" %
                 (total_time_diff * 100, develop_total_time, pr_total_time))
    logging.info("backward: %s" % pr_result.get("backward"))
    logging.info("parameters:")
    for line in pr_result.get("parameters").strip().split("\n"):
        logging.info("\t%s" % line)

    return gpu_time_diff > 0.05


def check_accuracy_result(case_name, pr_result):
    """Check accuracy result.
    """
    logging.info("------ OP: %s ------" % case_name)
    logging.info("Accuracy diff: %s" % pr_result.get("diff"))
    logging.info("backward: %s" % pr_result.get("backward"))
    logging.info("parameters:")
    for line in pr_result.get("parameters").strip().split("\n"):
        logging.info("\t%s" % line)

    return not pr_result.get("consistent")


def compare_benchmark_result(case_name, develop_result, pr_result,
                             check_results):
    """Compare the differences between develop and pr.
    """
    develop_speed = develop_result.get("speed")
    pr_speed = pr_result.get("speed")

    assert type(develop_speed) == type(
        pr_speed), "The types of comparison results need to be consistent."

    if isinstance(develop_speed, dict) and isinstance(pr_speed, dict):
        if check_speed_result(case_name, develop_speed, pr_speed, pr_result):
            check_results["speed"].append(case_name)
    else:
        if check_accuracy_result(case_name, pr_result):
            check_results["accuracy"].append(case_name)


def update_api_info_file(fail_case_list, api_info_file):
    """Update api info file to auto retry benchmark test.
    """
    check_path_exists(api_info_file)

    # set of case names for performance check failures
    parse_case_id_f = lambda x: x.split()[0].rsplit('_', 1)
    fail_case_dict = dict(map(parse_case_id_f, fail_case_list))

    # list of api infos for performance check failures
    api_info_list = list()
    with open(api_info_file) as f:
        for line in f:
            line_list = line.split(',')
            case = line_list[0].split(':')[0]
            if case in fail_case_dict:
                line_list[0] = "%s:%s" % (case, fail_case_dict[case])
                api_info_list.append(','.join(line_list))

    # update api info file
    with open(api_info_file, 'w') as f:
        for api_info_line in api_info_list:
            f.write(api_info_line)


def summary_results(check_results, api_info_file):
    """Summary results and return exit code.
    """
    for case_name in check_results["speed"]:
        logging.error("Check speed result with case \"%s\" failed." % case_name)

    for case_name in check_results["accuracy"]:
        logging.error("Check accuracy result with case \"%s\" failed." %
                      case_name)

    if len(check_results["speed"]) and api_info_file:
        update_api_info_file(check_results["speed"], api_info_file)

    if len(check_results["speed"]) or len(check_results["accuracy"]):
        return 8
    else:
        return 0


if __name__ == "__main__":
    """Load result from log directories and compare the differences.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="[%(filename)s:%(lineno)d] [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--develop_logs_dir",
        type=str,
        required=True,
        help="Specify the benchmark result directory of develop branch.")
    parser.add_argument(
        "--pr_logs_dir",
        type=str,
        required=True,
        help="Specify the benchmark result directory of PR branch.")
    parser.add_argument(
        "--api_info_file",
        type=str,
        required=False,
        help="Specify the api info to run benchmark test.")
    args = parser.parse_args()

    check_results = dict(accuracy=list(), speed=list())

    develop_result_dict = load_benchmark_result_from_logs_dir(
        args.develop_logs_dir)

    check_path_exists(args.pr_logs_dir)
    for log_file in os.listdir(args.pr_logs_dir):
        develop_result = develop_result_dict.get(log_file)
        pr_result = parse_log_file(os.path.join(args.pr_logs_dir, log_file))
        if develop_result is None or pr_result is None:
            continue
        case_name = parse_case_name(log_file)
        compare_benchmark_result(case_name, develop_result, pr_result,
                                 check_results)

    exit(summary_results(check_results, args.api_info_file))
