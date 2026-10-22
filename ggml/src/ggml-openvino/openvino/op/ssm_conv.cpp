#అన్తర్భావ "../node_context.h"
#అన్తర్భావ "../op_table.h"
#అన్తర్భావ "../utils.h"

#అన్తర్భావ <openvino/op/constant.hpp>
#అన్తర్భావ <openvino/op/group_conv.hpp>
#అన్తర్భావ <openvino/op/reshape.hpp>
#అన్తర్భావ <openvino/op/squeeze.hpp>
#అన్తర్భావ <openvino/op/transpose.hpp>
#అన్తర్భావ <openvino/op/unsqueeze.hpp>

నామాకాశ ov {
నామాకాశ frontend {
నామాకాశ ggml {
నామాకాశ op {

Output_సదిశ translate_ssm_conv(స్థిరరాశి కణుపు_Context & context) {
    అంకె_inputs_check(context, 2, 2);

    స్వయంచాలక sx = context.get_ఆదానం(0);                         // conv state + input: OV shape [1, n_s, d_inner, ncs]
    స్వయంచాలక c = context.get_ఆదానం(1);                          // conv1d weight:      OV shape [1, 1, d_inner, d_conv]

    స్వయంచాలక sx_shape = context.get_ఆదానం_shape(0).గమ్యం_shape();  // [1, n_s, d_inner, ncs]
    స్వయంచాలక c_shape = context.get_ఆదానం_shape(1).గమ్యం_shape();   // [1, 1, d_inner, d_conv]

    // int64_t n_s  = sx_shape[1];
    int64_t d_inner = sx_shape[2];
    // int64_t ncs  = sx_shape[3];  // d_conv - 1 + n_t
    int64_t d_conv  = c_shape[3];
    // int64_t n_t  = ncs - d_conv + 1;

    // Reshape sx from [1, n_s, d_inner, ncs] to [n_s, d_inner, ncs] for 1D GroupConvolution
    స్వయంచాలక sx_reshaped =
        std::make_shared<ov::op::v0::Squeeze>(sx, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));

    // Reshape c from [1, 1, d_inner, d_conv] to [d_inner, 1, 1, d_conv]
    // GroupConvolution filter: [groups, out_channels/groups, in_channels/groups, kernel_size]
    స్వయంచాలక c_new_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{d_inner, 1, 1, d_conv});
    స్వయంచాలక c_reshaped = std::make_shared<ov::op::v1::Reshape>(c, c_new_shape, అబద్ధం);

    // Depthwise 1D convolution: groups=d_inner, stride=1, no padding, no dilation
    // Input: [n_s, d_inner, ncs], Filter: [d_inner, 1, 1, d_conv]
    // Output: [n_s, d_inner, n_t]
    స్వయంచాలక conv = std::make_shared<ov::op::v1::సమూహం_Convolution>(
        sx_reshaped, c_reshaped, ov::Strides{1}, ov::CoordinateDiff{0}, ov::CoordinateDiff{0}, ov::Strides{1});

    // Transpose from [n_s, d_inner, n_t] to [n_s, n_t, d_inner]
    స్వయంచాలక perm = ov::op::v0::Constant::create(ov::element::i64, {3}, std::vector<int64_t>{0, 2, 1});
    స్వయంచాలక transposed = std::make_shared<ov::op::v1::Transpose>(conv, perm);

    // Reshape to output shape [1, n_s, n_t, d_inner]
    స్వయంచాలక res =
        std::make_shared<ov::op::v0::Unsqueeze>(transposed, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));

    తిరిగి rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
