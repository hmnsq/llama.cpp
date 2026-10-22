#ਅਨ੍ਤਰ੍ਭਾਵ "../node_context.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../op_table.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../utils.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/constant.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/group_conv.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/reshape.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/squeeze.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/transpose.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/unsqueeze.hpp>

ਨਾਮਾਕਾਸ਼ ov {
ਨਾਮਾਕਾਸ਼ frontend {
ਨਾਮਾਕਾਸ਼ ggml {
ਨਾਮਾਕਾਸ਼ op {

Output_ਸਦਿਸ਼ translate_ssm_conv(ਅਚਲ ਗੰਢ_Context & context) {
    ਅੰਕ_inputs_check(context, 2, 2);

    ਸਵੈਚਾਲਿਤ sx = context.get_ਨਿਵੇਸ਼(0);                         // conv state + input: OV shape [1, n_s, d_inner, ncs]
    ਸਵੈਚਾਲਿਤ c = context.get_ਨਿਵੇਸ਼(1);                          // conv1d weight:      OV shape [1, 1, d_inner, d_conv]

    ਸਵੈਚਾਲਿਤ sx_shape = context.get_ਨਿਵੇਸ਼_shape(0).ਮੰਜ਼ਿਲ_shape();  // [1, n_s, d_inner, ncs]
    ਸਵੈਚਾਲਿਤ c_shape = context.get_ਨਿਵੇਸ਼_shape(1).ਮੰਜ਼ਿਲ_shape();   // [1, 1, d_inner, d_conv]

    // int64_t n_s  = sx_shape[1];
    int64_t d_inner = sx_shape[2];
    // int64_t ncs  = sx_shape[3];  // d_conv - 1 + n_t
    int64_t d_conv  = c_shape[3];
    // int64_t n_t  = ncs - d_conv + 1;

    // Reshape sx from [1, n_s, d_inner, ncs] to [n_s, d_inner, ncs] for 1D GroupConvolution
    ਸਵੈਚਾਲਿਤ sx_reshaped =
        std::make_shared<ov::op::v0::Squeeze>(sx, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));

    // Reshape c from [1, 1, d_inner, d_conv] to [d_inner, 1, 1, d_conv]
    // GroupConvolution filter: [groups, out_channels/groups, in_channels/groups, kernel_size]
    ਸਵੈਚਾਲਿਤ c_new_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{d_inner, 1, 1, d_conv});
    ਸਵੈਚਾਲਿਤ c_reshaped = std::make_shared<ov::op::v1::Reshape>(c, c_new_shape, ਝੂਠ);

    // Depthwise 1D convolution: groups=d_inner, stride=1, no padding, no dilation
    // Input: [n_s, d_inner, ncs], Filter: [d_inner, 1, 1, d_conv]
    // Output: [n_s, d_inner, n_t]
    ਸਵੈਚਾਲਿਤ conv = std::make_shared<ov::op::v1::ਸਮੂਹ_Convolution>(
        sx_reshaped, c_reshaped, ov::Strides{1}, ov::CoordinateDiff{0}, ov::CoordinateDiff{0}, ov::Strides{1});

    // Transpose from [n_s, d_inner, n_t] to [n_s, n_t, d_inner]
    ਸਵੈਚਾਲਿਤ perm = ov::op::v0::Constant::create(ov::element::i64, {3}, std::vector<int64_t>{0, 2, 1});
    ਸਵੈਚਾਲਿਤ transposed = std::make_shared<ov::op::v1::Transpose>(conv, perm);

    // Reshape to output shape [1, n_s, n_t, d_inner]
    ਸਵੈਚਾਲਿਤ res =
        std::make_shared<ov::op::v0::Unsqueeze>(transposed, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));

    ਵਾਪਸ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
