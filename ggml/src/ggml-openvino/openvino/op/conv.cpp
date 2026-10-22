#شامل "../node_context.h"
#شامل "../op_table.h"
#شامل "../utils.h"

#شامل <openvino/op/constant.hpp>
#شامل <openvino/op/convert.hpp>
#شامل <openvino/op/convolution.hpp>
#شامل <openvino/op/group_conv.hpp>
#شامل <openvino/op/reshape.hpp>
#شامل <openvino/op/unsqueeze.hpp>

فضاینام ov {
فضاینام frontend {
فضاینام ggml {
فضاینام op {

OutputVector translate_conv_2d(ثابت گره_Context & context) {
    رقم_inputs_check(context, 2, 2);

    ov::Output<Node> kernel = process_view_درون_داد_new(context, 0);
    ov::Output<Node> درون_داد  = process_view_درون_داد_new(context, 1);

    اگر (kernel.get_element_type() != درون_داد.get_element_type()) {
        kernel = std::make_shared<ov::op::v0::Convert>(kernel, درون_داد.get_element_type());
    }

    ثابت int32_t * params = context.get_output_op_params();
    ثابت صحیح s0 = params[0];
    ثابت صحیح s1 = params[1];
    ثابت صحیح p0 = params[2];
    ثابت صحیح p1 = params[3];
    ثابت صحیح d0 = params[4];
    ثابت صحیح d1 = params[5];

    ov::Strides strides{static_cast<size_t>(s1), static_cast<size_t>(s0)};
    ov::CoordinateDiff pads_begin{static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::CoordinateDiff pads_end{static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::Strides dilations{static_cast<size_t>(d1), static_cast<size_t>(d0)};

    ov::Output<Node> res = std::make_shared<ov::op::v1::Convolution>(
        درون_داد, kernel, strides, pads_begin, pads_end, dilations, ov::op::PadType::EXPLICIT);

    ثابت auto output_type = context.get_output_type();
    اگر (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    بازگشت rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_conv_2d_dw(ثابت گره_Context & context) {
    رقم_inputs_check(context, 2, 2);

    ov::Output<Node> kernel = process_view_درون_داد_new(context, 0);
    ov::Output<Node> درون_داد  = process_view_درون_داد_new(context, 1);

    اگر (kernel.get_element_type() != درون_داد.get_element_type()) {
        kernel = std::make_shared<ov::op::v0::Convert>(kernel, درون_داد.get_element_type());
    }

    ثابت int32_t * params = context.get_output_op_params();
    ثابت صحیح s0 = params[0];
    ثابت صحیح s1 = params[1];
    ثابت صحیح p0 = params[2];
    ثابت صحیح p1 = params[3];
    ثابت صحیح d0 = params[4];
    ثابت صحیح d1 = params[5];

    // Reshape kernel from [C, 1, KH, KW] to [C, 1, 1, KH, KW] for 2D GroupConvolution
    auto unsqueeze_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto kernel_5d = std::make_shared<ov::op::v0::Unsqueeze>(kernel, unsqueeze_axis);

    ov::Strides strides{static_cast<size_t>(s1), static_cast<size_t>(s0)};
    ov::CoordinateDiff pads_begin{static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::CoordinateDiff pads_end{static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::Strides dilations{static_cast<size_t>(d1), static_cast<size_t>(d0)};

    ov::Output<Node> res = std::make_shared<ov::op::v1::GroupConvolution>(
        درون_داد, kernel_5d, strides, pads_begin, pads_end, dilations, ov::op::PadType::EXPLICIT);

    ثابت auto output_type = context.get_output_type();
    اگر (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    بازگشت rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_conv_transpose_1d(ثابت گره_Context & context) {
    رقم_inputs_check(context, 2, 2);

    ov::Output<Node> kernel = process_view_درون_داد_new(context, 0);
    ov::Output<Node> درون_داد  = process_view_درون_داد_new(context, 1);

    اگر (kernel.get_element_type() != درون_داد.get_element_type()) {
        kernel = std::make_shared<ov::op::v0::Convert>(kernel, درون_داد.get_element_type());
    }

    ثابت int32_t * params = context.get_output_op_params();
    ثابت صحیح s0 = params[0];
    ثابت صحیح p0 = params[1];
    ثابت صحیح d0 = params[2];

    ثابت auto kernel_shape = context.get_درون_داد_shape(0).to_shape();  // [1, Cin, Cout, K]
    ثابت int64_t Cin = kernel_shape[1];
    ثابت int64_t Cout = kernel_shape[2];
    ثابت int64_t K = kernel_shape[3];

    ثابت auto درون_داد_shape = context.get_درون_داد_shape(1).to_shape();   // [1, N, Cin, L]
    ثابت int64_t N = درون_داد_shape[0] * درون_داد_shape[1];
    ثابت int64_t L = درون_داد_shape[3];

    auto kernel_3d = std::make_shared<ov::op::v1::Reshape>(
        kernel, ov::op::v0::Constant::create(ov::element::i64, {3}, {Cin, Cout, K}), نادرست);
    auto درون_داد_3d = std::make_shared<ov::op::v1::Reshape>(
        درون_داد, ov::op::v0::Constant::create(ov::element::i64, {3}, {N, Cin, L}), نادرست);

    ov::Strides strides{static_cast<size_t>(s0)};
    ov::CoordinateDiff pads_begin{static_cast<ptrdiff_t>(p0)};
    ov::CoordinateDiff pads_end{static_cast<ptrdiff_t>(p0)};
    ov::Strides dilations{static_cast<size_t>(d0)};

    auto conv_tr = std::make_shared<ov::op::v1::ConvolutionBackpropData>(
        درون_داد_3d, kernel_3d, strides, pads_begin, pads_end, dilations);

    ثابت auto out_shape = context.get_output_shape().to_shape();
    auto out_shape_const = ov::op::v0::Constant::create(
        ov::element::i64, {4}, {static_cast<int64_t>(out_shape[0]), static_cast<int64_t>(out_shape[1]),
                                static_cast<int64_t>(out_shape[2]), static_cast<int64_t>(out_shape[3])});
    ov::Output<Node> res = std::make_shared<ov::op::v1::Reshape>(conv_tr, out_shape_const, نادرست);

    ثابت auto output_type = context.get_output_type();
    اگر (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    بازگشت rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_conv_transpose_2d(ثابت گره_Context & context) {
    رقم_inputs_check(context, 2, 2);

    ov::Output<Node> kernel = process_view_درون_داد_new(context, 0);
    ov::Output<Node> درون_داد  = process_view_درون_داد_new(context, 1);

    اگر (kernel.get_element_type() != درون_داد.get_element_type()) {
        kernel = std::make_shared<ov::op::v0::Convert>(kernel, درون_داد.get_element_type());
    }

    ثابت int32_t * params = context.get_output_op_params();
    ثابت صحیح stride = params[0];

    ov::Strides strides{static_cast<size_t>(stride), static_cast<size_t>(stride)};
    ov::CoordinateDiff pads_begin{0, 0};
    ov::CoordinateDiff pads_end{0, 0};
    ov::Strides dilations{1, 1};

    ov::Output<Node> res = std::make_shared<ov::op::v1::ConvolutionBackpropData>(
        درون_داد, kernel, strides, pads_begin, pads_end, dilations);

    ثابت auto output_type = context.get_output_type();
    اگر (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    بازگشت rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_conv_3d(ثابت گره_Context & context) {
    رقم_inputs_check(context, 2, 2);

    ov::Output<Node> kernel = process_view_درون_داد_new(context, 0);
    ov::Output<Node> درون_داد  = process_view_درون_داد_new(context, 1);

    اگر (kernel.get_element_type() != درون_داد.get_element_type()) {
        kernel = std::make_shared<ov::op::v0::Convert>(kernel, درون_داد.get_element_type());
    }

    ثابت int32_t * params = context.get_output_op_params();
    ثابت صحیح s0 = params[0];
    ثابت صحیح s1 = params[1];
    ثابت صحیح s2 = params[2];
    ثابت صحیح p0 = params[3];
    ثابت صحیح p1 = params[4];
    ثابت صحیح p2 = params[5];
    ثابت صحیح d0 = params[6];
    ثابت صحیح d1 = params[7];
    ثابت صحیح d2 = params[8];
    ثابت صحیح c  = params[9];
    ثابت صحیح n  = params[10];
    ثابت صحیح oc = params[11];

    ثابت auto kshape = context.get_درون_داد_shape(0).to_shape();  // [c*oc, KD, KH, KW]
    ثابت int64_t KD = kshape[1];
    ثابت int64_t KH = kshape[2];
    ثابت int64_t KW = kshape[3];

    ثابت auto inshape = context.get_درون_داد_shape(1).to_shape(); // [c*n, ID, IH, IW]
    ثابت int64_t ID = inshape[1];
    ثابت int64_t IH = inshape[2];
    ثابت int64_t IW = inshape[3];

    auto kernel_5d = std::make_shared<ov::op::v1::Reshape>(
        kernel, ov::op::v0::Constant::create(ov::element::i64, {5}, {static_cast<int64_t>(oc), static_cast<int64_t>(c), KD, KH, KW}), نادرست);
    auto درون_داد_5d = std::make_shared<ov::op::v1::Reshape>(
        درون_داد, ov::op::v0::Constant::create(ov::element::i64, {5}, {static_cast<int64_t>(n), static_cast<int64_t>(c), ID, IH, IW}), نادرست);

    ov::Strides strides{static_cast<size_t>(s2), static_cast<size_t>(s1), static_cast<size_t>(s0)};
    ov::CoordinateDiff pads_begin{static_cast<ptrdiff_t>(p2), static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::CoordinateDiff pads_end{static_cast<ptrdiff_t>(p2), static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::Strides dilations{static_cast<size_t>(d2), static_cast<size_t>(d1), static_cast<size_t>(d0)};

    auto conv = std::make_shared<ov::op::v1::Convolution>(
        درون_داد_5d, kernel_5d, strides, pads_begin, pads_end, dilations, ov::op::PadType::EXPLICIT);

    ثابت auto out_shape = context.get_output_shape().to_shape(); // [oc*n, OD, OH, OW]
    auto out_shape_const = ov::op::v0::Constant::create(
        ov::element::i64, {4}, {static_cast<int64_t>(out_shape[0]), static_cast<int64_t>(out_shape[1]),
                                static_cast<int64_t>(out_shape[2]), static_cast<int64_t>(out_shape[3])});
    ov::Output<Node> res = std::make_shared<ov::op::v1::Reshape>(conv, out_shape_const, نادرست);

    ثابت auto output_type = context.get_output_type();
    اگر (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    بازگشت rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
