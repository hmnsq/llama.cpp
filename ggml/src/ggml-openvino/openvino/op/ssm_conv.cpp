#अन्तर्भाव "../node_context.h"
#अन्तर्भाव "../op_table.h"
#अन्तर्भाव "../utils.h"

#अन्तर्भाव <openvino/op/constant.hpp>
#अन्तर्भाव <openvino/op/group_conv.hpp>
#अन्तर्भाव <openvino/op/reshape.hpp>
#अन्तर्भाव <openvino/op/squeeze.hpp>
#अन्तर्भाव <openvino/op/transpose.hpp>
#अन्तर्भाव <openvino/op/unsqueeze.hpp>

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {
नामाकाश op {

Output_सदिशम् translate_ssm_conv(ध्रुव ग्रन्थिः_Context & context) {
    अङ्कः_inputs_check(context, 2, 2);

    स्वचालित sx = context.get_निवेशः(0);                         // conv state + input: OV shape [1, n_s, d_inner, ncs]
    स्वचालित c = context.get_निवेशः(1);                          // conv1d weight:      OV shape [1, 1, d_inner, d_conv]

    स्वचालित sx_shape = context.get_निवेशः_shape(0).गन्तव्यम्_shape();  // [1, n_s, d_inner, ncs]
    स्वचालित c_shape = context.get_निवेशः_shape(1).गन्तव्यम्_shape();   // [1, 1, d_inner, d_conv]

    // int64_t n_s  = sx_shape[1];
    int64_t d_inner = sx_shape[2];
    // int64_t ncs  = sx_shape[3];  // d_conv - 1 + n_t
    int64_t d_conv  = c_shape[3];
    // int64_t n_t  = ncs - d_conv + 1;

    // Reshape sx from [1, n_s, d_inner, ncs] to [n_s, d_inner, ncs] for 1D GroupConvolution
    स्वचालित sx_reshaped =
        std::make_shared<ov::op::v0::Squeeze>(sx, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));

    // Reshape c from [1, 1, d_inner, d_conv] to [d_inner, 1, 1, d_conv]
    // GroupConvolution filter: [groups, out_channels/groups, in_channels/groups, kernel_size]
    स्वचालित c_new_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{d_inner, 1, 1, d_conv});
    स्वचालित c_reshaped = std::make_shared<ov::op::v1::Reshape>(c, c_new_shape, असत्य);

    // Depthwise 1D convolution: groups=d_inner, stride=1, no padding, no dilation
    // Input: [n_s, d_inner, ncs], Filter: [d_inner, 1, 1, d_conv]
    // Output: [n_s, d_inner, n_t]
    स्वचालित conv = std::make_shared<ov::op::v1::गणः_Convolution>(
        sx_reshaped, c_reshaped, ov::Strides{1}, ov::CoordinateDiff{0}, ov::CoordinateDiff{0}, ov::Strides{1});

    // Transpose from [n_s, d_inner, n_t] to [n_s, n_t, d_inner]
    स्वचालित perm = ov::op::v0::Constant::create(ov::element::i64, {3}, std::vector<int64_t>{0, 2, 1});
    स्वचालित transposed = std::make_shared<ov::op::v1::Transpose>(conv, perm);

    // Reshape to output shape [1, n_s, n_t, d_inner]
    स्वचालित res =
        std::make_shared<ov::op::v0::Unsqueeze>(transposed, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));

    प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
