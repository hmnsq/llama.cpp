#ضمن "../node_context.h"
#ضمن "../op_table.h"
#ضمن "../utils.h"

#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/group_conv.hpp>
#ضمن <openvino/op/reshape.hpp>
#ضمن <openvino/op/squeeze.hpp>
#ضمن <openvino/op/transpose.hpp>
#ضمن <openvino/op/unsqueeze.hpp>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق op {

Output_وکتور translate_ssm_conv(ثابت گره_Context & context) {
    رقم_inputs_check(context, 2, 2);

    خودکار sx = context.get_درون_داد(0);                         // conv state + input: OV shape [1, n_s, d_inner, ncs]
    خودکار c = context.get_درون_داد(1);                          // conv1d weight:      OV shape [1, 1, d_inner, d_conv]

    خودکار sx_shape = context.get_درون_داد_shape(0).مقصد_shape();  // [1, n_s, d_inner, ncs]
    خودکار c_shape = context.get_درون_داد_shape(1).مقصد_shape();   // [1, 1, d_inner, d_conv]

    // int64_t n_s  = sx_shape[1];
    int64_t d_inner = sx_shape[2];
    // int64_t ncs  = sx_shape[3];  // d_conv - 1 + n_t
    int64_t d_conv  = c_shape[3];
    // int64_t n_t  = ncs - d_conv + 1;

    // Reshape sx from [1, n_s, d_inner, ncs] to [n_s, d_inner, ncs] for 1D GroupConvolution
    خودکار sx_reshaped =
        std::make_shared<ov::op::v0::Squeeze>(sx, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));

    // Reshape c from [1, 1, d_inner, d_conv] to [d_inner, 1, 1, d_conv]
    // GroupConvolution filter: [groups, out_channels/groups, in_channels/groups, kernel_size]
    خودکار c_new_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{d_inner, 1, 1, d_conv});
    خودکار c_reshaped = std::make_shared<ov::op::v1::Reshape>(c, c_new_shape, خطأ);

    // Depthwise 1D convolution: groups=d_inner, stride=1, no padding, no dilation
    // Input: [n_s, d_inner, ncs], Filter: [d_inner, 1, 1, d_conv]
    // Output: [n_s, d_inner, n_t]
    خودکار conv = std::make_shared<ov::op::v1::گروه_Convolution>(
        sx_reshaped, c_reshaped, ov::Strides{1}, ov::CoordinateDiff{0}, ov::CoordinateDiff{0}, ov::Strides{1});

    // Transpose from [n_s, d_inner, n_t] to [n_s, n_t, d_inner]
    خودکار perm = ov::op::v0::Constant::create(ov::element::i64, {3}, std::vector<int64_t>{0, 2, 1});
    خودکار transposed = std::make_shared<ov::op::v1::Transpose>(conv, perm);

    // Reshape to output shape [1, n_s, n_t, d_inner]
    خودکار res =
        std::make_shared<ov::op::v0::Unsqueeze>(transposed, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
