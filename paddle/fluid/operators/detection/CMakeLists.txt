set(LOCAL_DETECTION_LIBS)

function(detection_library TARGET_NAME)
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS)
    set(options "")
    set(common_deps op_registry)
    set(pybind_flag 0)
    cmake_parse_arguments(detection_library "${options}" "${oneValueArgs}"
            "${multiValueArgs}" ${ARGN})
    op_library(${TARGET_NAME} SRCS ${detection_library_SRCS} DEPS ${common_deps} ${detection_library_DEPS})
    set(LOCAL_DETECTION_LIBS
            ${TARGET_NAME}
            ${LOCAL_DETECTION_LIBS}
        PARENT_SCOPE)
endfunction()

detection_library(bipartite_match_op SRCS bipartite_match_op.cc)
detection_library(box_coder_op SRCS box_coder_op.cc box_coder_op.cu)
detection_library(iou_similarity_op SRCS iou_similarity_op.cc
iou_similarity_op.cu)
detection_library(mine_hard_examples_op SRCS mine_hard_examples_op.cc)
detection_library(multiclass_nms_op SRCS multiclass_nms_op.cc poly_util.cc gpc.cc)
detection_library(prior_box_op SRCS prior_box_op.cc prior_box_op.cu)
detection_library(anchor_generator_op SRCS anchor_generator_op.cc
anchor_generator_op.cu)
detection_library(target_assign_op SRCS target_assign_op.cc
target_assign_op.cu)
detection_library(polygon_box_transform_op SRCS polygon_box_transform_op.cc
polygon_box_transform_op.cu)
detection_library(rpn_target_assign_op SRCS rpn_target_assign_op.cc)
detection_library(generate_proposal_labels_op SRCS generate_proposal_labels_op.cc)

if(WITH_GPU)
  detection_library(generate_proposals_op SRCS generate_proposals_op.cc generate_proposals_op.cu DEPS memory cub)
else()
  detection_library(generate_proposals_op SRCS generate_proposals_op.cc)
endif()

detection_library(roi_perspective_transform_op SRCS roi_perspective_transform_op.cc roi_perspective_transform_op.cu)
#Export local libraries to parent
set(DETECTION_LIBRARY ${LOCAL_DETECTION_LIBS} PARENT_SCOPE)
