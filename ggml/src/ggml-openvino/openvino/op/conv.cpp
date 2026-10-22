#অন্তর্ভাব "../node_context.h"
#অন্তর্ভাব "../op_table.h"
#অন্তর্ভাব "../utils.h"

#অন্তর্ভাব <openvino/op/constant.hpp>
#অন্তর্ভাব <openvino/op/convert.hpp>
#অন্তর্ভাব <openvino/op/convolution.hpp>
#অন্তর্ভাব <openvino/op/group_conv.hpp>
#অন্তর্ভাব <openvino/op/reshape.hpp>
#অন্তর্ভাব <openvino/op/unsqueeze.hpp>

নামাকাশ ov {
নামাকাশ frontend {
নামাকাশ ggml {
নামাকাশ op {

OutputVector translate_conv_2d(ধ্রুবক গ্রন্থি_Context & context) {
    অঙ্ক_inputs_check(context, 2, 2);

    ov::Output<Node> kernel = process_view_প্রবেশ_new(context, 0);
    ov::Output<Node> প্রবেশ  = process_view_প্রবেশ_new(context, 1);

    যদি (kernel.get_element_type() != প্রবেশ.get_element_type()) {
        kernel = std::make_shared<ov::op::v0::Convert>(kernel, প্রবেশ.get_element_type());
    }

    ধ্রুবক int32_t * params = context.get_output_op_params();
    ধ্রুবক পূর্ণসংখ্যা s0 = params[0];
    ধ্রুবক পূর্ণসংখ্যা s1 = params[1];
    ধ্রুবক পূর্ণসংখ্যা p0 = params[2];
    ধ্রুবক পূর্ণসংখ্যা p1 = params[3];
    ধ্রুবক পূর্ণসংখ্যা d0 = params[4];
    ধ্রুবক পূর্ণসংখ্যা d1 = params[5];

    ov::Strides strides{static_cast<size_t>(s1), static_cast<size_t>(s0)};
    ov::CoordinateDiff pads_begin{static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::CoordinateDiff pads_end{static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::Strides dilations{static_cast<size_t>(d1), static_cast<size_t>(d0)};

    ov::Output<Node> res = std::make_shared<ov::op::v1::Convolution>(
        প্রবেশ, kernel, strides, pads_begin, pads_end, dilations, ov::op::PadType::EXPLICIT);

    ধ্রুবক auto output_type = context.get_output_type();
    যদি (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    ফেরত rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_conv_2d_dw(ধ্রুবক গ্রন্থি_Context & context) {
    অঙ্ক_inputs_check(context, 2, 2);

    ov::Output<Node> kernel = process_view_প্রবেশ_new(context, 0);
    ov::Output<Node> প্রবেশ  = process_view_প্রবেশ_new(context, 1);

    যদি (kernel.get_element_type() != প্রবেশ.get_element_type()) {
        kernel = std::make_shared<ov::op::v0::Convert>(kernel, প্রবেশ.get_element_type());
    }

    ধ্রুবক int32_t * params = context.get_output_op_params();
    ধ্রুবক পূর্ণসংখ্যা s0 = params[0];
    ধ্রুবক পূর্ণসংখ্যা s1 = params[1];
    ধ্রুবক পূর্ণসংখ্যা p0 = params[2];
    ধ্রুবক পূর্ণসংখ্যা p1 = params[3];
    ধ্রুবক পূর্ণসংখ্যা d0 = params[4];
    ধ্রুবক পূর্ণসংখ্যা d1 = params[5];

    // Reshape kernel from [C, 1, KH, KW] to [C, 1, 1, KH, KW] for 2D GroupConvolution
    auto unsqueeze_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto kernel_5d = std::make_shared<ov::op::v0::Unsqueeze>(kernel, unsqueeze_axis);

    ov::Strides strides{static_cast<size_t>(s1), static_cast<size_t>(s0)};
    ov::CoordinateDiff pads_begin{static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::CoordinateDiff pads_end{static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::Strides dilations{static_cast<size_t>(d1), static_cast<size_t>(d0)};

    ov::Output<Node> res = std::make_shared<ov::op::v1::GroupConvolution>(
        প্রবেশ, kernel_5d, strides, pads_begin, pads_end, dilations, ov::op::PadType::EXPLICIT);

    ধ্রুবক auto output_type = context.get_output_type();
    যদি (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    ফেরত rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_conv_transpose_1d(ধ্রুবক গ্রন্থি_Context & context) {
    অঙ্ক_inputs_check(context, 2, 2);

    ov::Output<Node> kernel = process_view_প্রবেশ_new(context, 0);
    ov::Output<Node> প্রবেশ  = process_view_প্রবেশ_new(context, 1);

    যদি (kernel.get_element_type() != প্রবেশ.get_element_type()) {
        kernel = std::make_shared<ov::op::v0::Convert>(kernel, প্রবেশ.get_element_type());
    }

    ধ্রুবক int32_t * params = context.get_output_op_params();
    ধ্রুবক পূর্ণসংখ্যা s0 = params[0];
    ধ্রুবক পূর্ণসংখ্যা p0 = params[1];
    ধ্রুবক পূর্ণসংখ্যা d0 = params[2];

    ধ্রুবক auto kernel_shape = context.get_প্রবেশ_shape(0).to_shape();  // [1, Cin, Cout, K]
    ধ্রুবক int64_t Cin = kernel_shape[1];
    ধ্রুবক int64_t Cout = kernel_shape[2];
    ধ্রুবক int64_t K = kernel_shape[3];

    ধ্রুবক auto প্রবেশ_shape = context.get_প্রবেশ_shape(1).to_shape();   // [1, N, Cin, L]
    ধ্রুবক int64_t N = প্রবেশ_shape[0] * প্রবেশ_shape[1];
    ধ্রুবক int64_t L = প্রবেশ_shape[3];

    auto kernel_3d = std::make_shared<ov::op::v1::Reshape>(
        kernel, ov::op::v0::Constant::create(ov::element::i64, {3}, {Cin, Cout, K}), মিথ্যা);
    auto প্রবেশ_3d = std::make_shared<ov::op::v1::Reshape>(
        প্রবেশ, ov::op::v0::Constant::create(ov::element::i64, {3}, {N, Cin, L}), মিথ্যা);

    ov::Strides strides{static_cast<size_t>(s0)};
    ov::CoordinateDiff pads_begin{static_cast<ptrdiff_t>(p0)};
    ov::CoordinateDiff pads_end{static_cast<ptrdiff_t>(p0)};
    ov::Strides dilations{static_cast<size_t>(d0)};

    auto conv_tr = std::make_shared<ov::op::v1::Convolution_Backprop_তথ্য>(
        প্রবেশ_3d, kernel_3d, strides, pads_begin, pads_end, dilations);

    ধ্রুবক auto out_shape = context.get_output_shape().to_shape();
    auto out_shape_const = ov::op::v0::Constant::create(
        ov::element::i64, {4}, {static_cast<int64_t>(out_shape[0]), static_cast<int64_t>(out_shape[1]),
                                static_cast<int64_t>(out_shape[2]), static_cast<int64_t>(out_shape[3])});
    ov::Output<Node> res = std::make_shared<ov::op::v1::Reshape>(conv_tr, out_shape_const, মিথ্যা);

    ধ্রুবক auto output_type = context.get_output_type();
    যদি (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    ফেরত rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_conv_transpose_2d(ধ্রুবক গ্রন্থি_Context & context) {
    অঙ্ক_inputs_check(context, 2, 2);

    ov::Output<Node> kernel = process_view_প্রবেশ_new(context, 0);
    ov::Output<Node> প্রবেশ  = process_view_প্রবেশ_new(context, 1);

    যদি (kernel.get_element_type() != প্রবেশ.get_element_type()) {
        kernel = std::make_shared<ov::op::v0::Convert>(kernel, প্রবেশ.get_element_type());
    }

    ধ্রুবক int32_t * params = context.get_output_op_params();
    ধ্রুবক পূর্ণসংখ্যা stride = params[0];

    ov::Strides strides{static_cast<size_t>(stride), static_cast<size_t>(stride)};
    ov::CoordinateDiff pads_begin{0, 0};
    ov::CoordinateDiff pads_end{0, 0};
    ov::Strides dilations{1, 1};

    ov::Output<Node> res = std::make_shared<ov::op::v1::Convolution_Backprop_তথ্য>(
        প্রবেশ, kernel, strides, pads_begin, pads_end, dilations);

    ধ্রুবক auto output_type = context.get_output_type();
    যদি (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    ফেরত rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_conv_3d(ধ্রুবক গ্রন্থি_Context & context) {
    অঙ্ক_inputs_check(context, 2, 2);

    ov::Output<Node> kernel = process_view_প্রবেশ_new(context, 0);
    ov::Output<Node> প্রবেশ  = process_view_প্রবেশ_new(context, 1);

    যদি (kernel.get_element_type() != প্রবেশ.get_element_type()) {
        kernel = std::make_shared<ov::op::v0::Convert>(kernel, প্রবেশ.get_element_type());
    }

    ধ্রুবক int32_t * params = context.get_output_op_params();
    ধ্রুবক পূর্ণসংখ্যা s0 = params[0];
    ধ্রুবক পূর্ণসংখ্যা s1 = params[1];
    ধ্রুবক পূর্ণসংখ্যা s2 = params[2];
    ধ্রুবক পূর্ণসংখ্যা p0 = params[3];
    ধ্রুবক পূর্ণসংখ্যা p1 = params[4];
    ধ্রুবক পূর্ণসংখ্যা p2 = params[5];
    ধ্রুবক পূর্ণসংখ্যা d0 = params[6];
    ধ্রুবক পূর্ণসংখ্যা d1 = params[7];
    ধ্রুবক পূর্ণসংখ্যা d2 = params[8];
    ধ্রুবক পূর্ণসংখ্যা c  = params[9];
    ধ্রুবক পূর্ণসংখ্যা n  = params[10];
    ধ্রুবক পূর্ণসংখ্যা oc = params[11];

    ধ্রুবক auto kshape = context.get_প্রবেশ_shape(0).to_shape();  // [c*oc, KD, KH, KW]
    ধ্রুবক int64_t KD = kshape[1];
    ধ্রুবক int64_t KH = kshape[2];
    ধ্রুবক int64_t KW = kshape[3];

    ধ্রুবক auto inshape = context.get_প্রবেশ_shape(1).to_shape(); // [c*n, ID, IH, IW]
    ধ্রুবক int64_t ID = inshape[1];
    ধ্রুবক int64_t IH = inshape[2];
    ধ্রুবক int64_t IW = inshape[3];

    auto kernel_5d = std::make_shared<ov::op::v1::Reshape>(
        kernel, ov::op::v0::Constant::create(ov::element::i64, {5}, {static_cast<int64_t>(oc), static_cast<int64_t>(c), KD, KH, KW}), মিথ্যা);
    auto প্রবেশ_5d = std::make_shared<ov::op::v1::Reshape>(
        প্রবেশ, ov::op::v0::Constant::create(ov::element::i64, {5}, {static_cast<int64_t>(n), static_cast<int64_t>(c), ID, IH, IW}), মিথ্যা);

    ov::Strides strides{static_cast<size_t>(s2), static_cast<size_t>(s1), static_cast<size_t>(s0)};
    ov::CoordinateDiff pads_begin{static_cast<ptrdiff_t>(p2), static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::CoordinateDiff pads_end{static_cast<ptrdiff_t>(p2), static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::Strides dilations{static_cast<size_t>(d2), static_cast<size_t>(d1), static_cast<size_t>(d0)};

    auto conv = std::make_shared<ov::op::v1::Convolution>(
        প্রবেশ_5d, kernel_5d, strides, pads_begin, pads_end, dilations, ov::op::PadType::EXPLICIT);

    ধ্রুবক auto out_shape = context.get_output_shape().to_shape(); // [oc*n, OD, OH, OW]
    auto out_shape_const = ov::op::v0::Constant::create(
        ov::element::i64, {4}, {static_cast<int64_t>(out_shape[0]), static_cast<int64_t>(out_shape[1]),
                                static_cast<int64_t>(out_shape[2]), static_cast<int64_t>(out_shape[3])});
    ov::Output<Node> res = std::make_shared<ov::op::v1::Reshape>(conv, out_shape_const, মিথ্যা);

    ধ্রুবক auto output_type = context.get_output_type();
    যদি (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    ফেরত rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
