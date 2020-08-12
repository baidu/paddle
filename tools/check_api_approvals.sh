#!/bin/bash

if [ -z ${BRANCH} ]; then
    BRANCH="develop"
fi

PADDLE_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")/../" && pwd )"
approval_line=`curl -H "Authorization: token ${GITHUB_API_TOKEN}" https://api.github.com/repos/PaddlePaddle/Paddle/pulls/${GIT_PR_ID}/reviews?per_page=10000`
git_files=`git diff --numstat upstream/$BRANCH| wc -l`
git_count=`git diff --numstat upstream/$BRANCH| awk '{sum+=$1}END{print sum}'`
failed_num=0
echo_list=()


function check_approval(){
    person_num=`echo $@|awk '{for (i=2;i<=NF;i++)print $i}'`
    APPROVALS=`echo ${approval_line}|python ${PADDLE_ROOT}/tools/check_pr_approval.py $1 $person_num`
    if [[ "${APPROVALS}" == "FALSE" && "${echo_line}" != "" ]]; then
        add_failed "${failed_num}. ${echo_line}"
    fi
}

function add_failed(){
    failed_num=`expr $failed_num + 1`
    echo_list="${echo_list[@]}$1"
}

api_spec_diff=`python ${PADDLE_ROOT}/tools/diff_api.py ${PADDLE_ROOT}/paddle/fluid/API_DEV.spec.api  ${PADDLE_ROOT}/paddle/fluid/API_PR.spec.api` 
if [ "$api_spec_diff" != "" ]; then
    echo_line="You must have one RD (XiaoguangHu01 or lanxianghit) and one TPM (saxon-zh or jzhang533 or swtkiwi or Heeenrrry or TCChenlong) approval for the api change for the management reason of API interface.\n"
    check_approval 1 46782768 47554610
    echo_line=""
    check_approval 1 2870059 29231 27208573 28379894 11935832
fi

api_doc_spec_diff=`python ${PADDLE_ROOT}/tools/diff_api.py ${PADDLE_ROOT}/paddle/fluid/API_DEV.spec.doc  ${PADDLE_ROOT}/paddle/fluid/API_PR.spec.doc` 
if [ "$api_doc_spec_diff" != "" ]; then
    echo_line="You must have one TPM (saxon-zh or jzhang533 or swtkiwi or Heeenrrry or TCChenlong) approval for the api change for the management reason of API document.\n"
    check_approval 1 2870059 29231 27208573 28379894 11935832
fi

api_spec_diff=`python ${PADDLE_ROOT}/tools/check_api_source_without_core_ops.py ${PADDLE_ROOT}/paddle/fluid/API_DEV.source.md5  ${PADDLE_ROOT}/paddle/fluid/API_PR.source.md5` 
if [ "$api_spec_diff" != "" ]; then
    echo_line="You must have one RD (zhiqiu (Recommend) or phlrain) approval for the api change for the opreator-related api without 'core.ops'.\n"
    echo_line="${echo_line}For more details, please click [https://github.com/PaddlePaddle/Paddle/wiki/paddle_api_development_manual.md]\n"
    echo_line="${echo_line}Related APIs: ${api_spec_diff}\n"
    check_approval 1 6888866 43953930
fi

op_type_spec_diff=`python ${PADDLE_ROOT}/tools/check_op_register_type.py ${PADDLE_ROOT}/paddle/fluid/OP_TYPE_DEV.spec  ${PADDLE_ROOT}/paddle/fluid/OP_TYPE_PR.spec`
if [ "$op_type_spec_diff" != "" ]; then
    echo_line="You must have one RD (Aurelius84 (Recommend) or liym27 or zhhsplendid)approval for the data_type registration of new operator. More data_type of new operator should be registered in your PR. Please make sure that both float/double (or int/int64_t) have been registered.\n For more details, please click [https://github.com/PaddlePaddle/Paddle/wiki/Data-types-of-generic-Op-must-be-fully-registered].\n"
    check_approval 1 9j301846 33742067 7913861
fi

op_desc_diff=`python ${PADDLE_ROOT}/tools/check_op_desc.py ${PADDLE_ROOT}/paddle/fluid/OP_DESC_DEV.spec  ${PADDLE_ROOT}/paddle/fluid/OP_DESC_PR.spec`
if [ "$op_desc_diff" != "" ]; then
    echo_line="You must have one RD (liym27 (Recommend), zhhsplendid, Aurelius84, lanxianghit or phlrain) approval for the changes of Inputs/Output/Attrs of OPs. The changes of OPs will cause that the new version inference fails to load model trained by the old version. Please modify your code. \n For more details, please click [https://github.com/PaddlePaddle/Paddle/wiki/OP-Input-Output-Attribute-Compatibility-Modification].\n${op_desc_diff}\n"
    check_approval 1 33742067 7913861 9301846 47554610 43953930
fi

DEV_OP_USE_DEFAULT_GRAD_MAKER_SPEC=${PADDLE_ROOT}/paddle/fluid/op_use_default_grad_maker_DEV.spec
PR_OP_USE_DEFAULT_GRAD_MAKER_SPEC=${PADDLE_ROOT}/paddle/fluid/op_use_default_grad_maker_PR.spec
ADDED_OP_USE_DEFAULT_GRAD_MAKER=`python ${PADDLE_ROOT}/tools/diff_use_default_grad_op_maker.py ${DEV_OP_USE_DEFAULT_GRAD_MAKER_SPEC} ${PR_OP_USE_DEFAULT_GRAD_MAKER_SPEC}` 
if [ "${ADDED_OP_USE_DEFAULT_GRAD_MAKER}" != "" ]; then
  echo_line="You must have one RD (sneaxiy (Recommend) or luotao1) approval because you use DefaultGradOpMaker for ${ADDED_OP_USE_DEFAULT_GRAD_MAKER}, which manages the grad_op memory optimization.\n" 
  check_approval 1 32832641 6836917
fi

if [ -n "${echo_list}" ];then
  echo "****************"
  echo -e "${echo_list[@]}"
  echo "There are ${failed_num} approved errors."
  echo "****************"
fi

python ${PADDLE_ROOT}/tools/diff_api.py ${PADDLE_ROOT}/paddle/fluid/API_DEV.spec  ${PADDLE_ROOT}/paddle/fluid/API_PR.spec
python ${PADDLE_ROOT}/tools/check_op_register_type.py ${PADDLE_ROOT}/paddle/fluid/OP_TYPE_DEV.spec  ${PADDLE_ROOT}/paddle/fluid/OP_TYPE_PR.spec
if [ -n "${echo_list}" ]; then
  exit 1
fi
