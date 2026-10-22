#અન્તર્ભાવ "../node_context.h"
#અન્તર્ભાવ "../op_table.h"
#અન્તર્ભાવ "../utils.h"

#અન્તર્ભાવ <openvino/op/constant.hpp>
#અન્તર્ભાવ <openvino/op/convert.hpp>
#અન્તર્ભાવ <openvino/op/convolution.hpp>
#અન્તર્ભાવ <openvino/op/group_conv.hpp>
#અન્તર્ભાવ <openvino/op/reshape.hpp>
#અન્તર્ભાવ <openvino/op/unsqueeze.hpp>

નામાકાશ ov {
નામાકાશ frontend {
નામાકાશ ggml {
નામાકાશ op {

OutputVector translate_conv_2d(અચલ ગાંઠ_Context & context) {
    અંક_inputs_check(context, 2, 2);

    ov::Output<Node> kernel = process_view_આગત_new(context, 0);
    ov::Output<Node> આગત  = process_view_આગત_new(context, 1);

    જો (kernel.get_element_type() != આગત.get_element_type()) {
        kernel = std::make_shared<ov::op::v0::Convert>(kernel, આગત.get_element_type());
    }

    અચલ int32_t * params = context.get_output_op_params();
    અચલ પૂર્ણાંક s0 = params[0];
    અચલ પૂર્ણાંક s1 = params[1];
    અચલ પૂર્ણાંક p0 = params[2];
    અચલ પૂર્ણાંક p1 = params[3];
    અચલ પૂર્ણાંક d0 = params[4];
    અચલ પૂર્ણાંક d1 = params[5];

    ov::Strides strides{static_cast<size_t>(s1), static_cast<size_t>(s0)};
    ov::CoordinateDiff pads_begin{static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::CoordinateDiff pads_end{static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::Strides dilations{static_cast<size_t>(d1), static_cast<size_t>(d0)};

    ov::Output<Node> res = std::make_shared<ov::op::v1::Convolution>(
        આગત, kernel, strides, pads_begin, pads_end, dilations, ov::op::PadType::EXPLICIT);

    અચલ auto output_type = context.get_output_type();
    જો (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    પાછા rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_conv_2d_dw(અચલ ગાંઠ_Context & context) {
    અંક_inputs_check(context, 2, 2);

    ov::Output<Node> kernel = process_view_આગત_new(context, 0);
    ov::Output<Node> આગત  = process_view_આગત_new(context, 1);

    જો (kernel.get_element_type() != આગત.get_element_type()) {
        kernel = std::make_shared<ov::op::v0::Convert>(kernel, આગત.get_element_type());
    }

    અચલ int32_t * params = context.get_output_op_params();
    અચલ પૂર્ણાંક s0 = params[0];
    અચલ પૂર્ણાંક s1 = params[1];
    અચલ પૂર્ણાંક p0 = params[2];
    અચલ પૂર્ણાંક p1 = params[3];
    અચલ પૂર્ણાંક d0 = params[4];
    અચલ પૂર્ણાંક d1 = params[5];

    // Reshape kernel from [C, 1, KH, KW] to [C, 1, 1, KH, KW] for 2D GroupConvolution
    auto unsqueeze_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto kernel_5d = std::make_shared<ov::op::v0::Unsqueeze>(kernel, unsqueeze_axis);

    ov::Strides strides{static_cast<size_t>(s1), static_cast<size_t>(s0)};
    ov::CoordinateDiff pads_begin{static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::CoordinateDiff pads_end{static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::Strides dilations{static_cast<size_t>(d1), static_cast<size_t>(d0)};

    ov::Output<Node> res = std::make_shared<ov::op::v1::GroupConvolution>(
        આગત, kernel_5d, strides, pads_begin, pads_end, dilations, ov::op::PadType::EXPLICIT);

    અચલ auto output_type = context.get_output_type();
    જો (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    પાછા rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_conv_transpose_1d(અચલ ગાંઠ_Context & context) {
    અંક_inputs_check(context, 2, 2);

    ov::Output<Node> kernel = process_view_આગત_new(context, 0);
    ov::Output<Node> આગત  = process_view_આગત_new(context, 1);

    જો (kernel.get_element_type() != આગત.get_element_type()) {
        kernel = std::make_shared<ov::op::v0::Convert>(kernel, આગત.get_element_type());
    }

    અચલ int32_t * params = context.get_output_op_params();
    અચલ પૂર્ણાંક s0 = params[0];
    અચલ પૂર્ણાંક p0 = params[1];
    અચલ પૂર્ણાંક d0 = params[2];

    અચલ auto kernel_shape = context.get_આગત_shape(0).to_shape();  // [1, Cin, Cout, K]
    અચલ int64_t Cin = kernel_shape[1];
    અચલ int64_t Cout = kernel_shape[2];
    અચલ int64_t K = kernel_shape[3];

    અચલ auto આગત_shape = context.get_આગત_shape(1).to_shape();   // [1, N, Cin, L]
    અચલ int64_t N = આગત_shape[0] * આગત_shape[1];
    અચલ int64_t L = આગત_shape[3];

    auto kernel_3d = std::make_shared<ov::op::v1::Reshape>(
        kernel, ov::op::v0::Constant::create(ov::element::i64, {3}, {Cin, Cout, K}), ખોટું);
    auto આગત_3d = std::make_shared<ov::op::v1::Reshape>(
        આગત, ov::op::v0::Constant::create(ov::element::i64, {3}, {N, Cin, L}), ખોટું);

    ov::Strides strides{static_cast<size_t>(s0)};
    ov::CoordinateDiff pads_begin{static_cast<ptrdiff_t>(p0)};
    ov::CoordinateDiff pads_end{static_cast<ptrdiff_t>(p0)};
    ov::Strides dilations{static_cast<size_t>(d0)};

    auto conv_tr = std::make_shared<ov::op::v1::ConvolutionBackpropData>(
        આગત_3d, kernel_3d, strides, pads_begin, pads_end, dilations);

    અચલ auto out_shape = context.get_output_shape().to_shape();
    auto out_shape_const = ov::op::v0::Constant::create(
        ov::element::i64, {4}, {static_cast<int64_t>(out_shape[0]), static_cast<int64_t>(out_shape[1]),
                                static_cast<int64_t>(out_shape[2]), static_cast<int64_t>(out_shape[3])});
    ov::Output<Node> res = std::make_shared<ov::op::v1::Reshape>(conv_tr, out_shape_const, ખોટું);

    અચલ auto output_type = context.get_output_type();
    જો (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    પાછા rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_conv_transpose_2d(અચલ ગાંઠ_Context & context) {
    અંક_inputs_check(context, 2, 2);

    ov::Output<Node> kernel = process_view_આગત_new(context, 0);
    ov::Output<Node> આગત  = process_view_આગત_new(context, 1);

    જો (kernel.get_element_type() != આગત.get_element_type()) {
        kernel = std::make_shared<ov::op::v0::Convert>(kernel, આગત.get_element_type());
    }

    અચલ int32_t * params = context.get_output_op_params();
    અચલ પૂર્ણાંક stride = params[0];

    ov::Strides strides{static_cast<size_t>(stride), static_cast<size_t>(stride)};
    ov::CoordinateDiff pads_begin{0, 0};
    ov::CoordinateDiff pads_end{0, 0};
    ov::Strides dilations{1, 1};

    ov::Output<Node> res = std::make_shared<ov::op::v1::ConvolutionBackpropData>(
        આગત, kernel, strides, pads_begin, pads_end, dilations);

    અચલ auto output_type = context.get_output_type();
    જો (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    પાછા rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_conv_3d(અચલ ગાંઠ_Context & context) {
    અંક_inputs_check(context, 2, 2);

    ov::Output<Node> kernel = process_view_આગત_new(context, 0);
    ov::Output<Node> આગત  = process_view_આગત_new(context, 1);

    જો (kernel.get_element_type() != આગત.get_element_type()) {
        kernel = std::make_shared<ov::op::v0::Convert>(kernel, આગત.get_element_type());
    }

    અચલ int32_t * params = context.get_output_op_params();
    અચલ પૂર્ણાંક s0 = params[0];
    અચલ પૂર્ણાંક s1 = params[1];
    અચલ પૂર્ણાંક s2 = params[2];
    અચલ પૂર્ણાંક p0 = params[3];
    અચલ પૂર્ણાંક p1 = params[4];
    અચલ પૂર્ણાંક p2 = params[5];
    અચલ પૂર્ણાંક d0 = params[6];
    અચલ પૂર્ણાંક d1 = params[7];
    અચલ પૂર્ણાંક d2 = params[8];
    અચલ પૂર્ણાંક c  = params[9];
    અચલ પૂર્ણાંક n  = params[10];
    અચલ પૂર્ણાંક oc = params[11];

    અચલ auto kshape = context.get_આગત_shape(0).to_shape();  // [c*oc, KD, KH, KW]
    અચલ int64_t KD = kshape[1];
    અચલ int64_t KH = kshape[2];
    અચલ int64_t KW = kshape[3];

    અચલ auto inshape = context.get_આગત_shape(1).to_shape(); // [c*n, ID, IH, IW]
    અચલ int64_t ID = inshape[1];
    અચલ int64_t IH = inshape[2];
    અચલ int64_t IW = inshape[3];

    auto kernel_5d = std::make_shared<ov::op::v1::Reshape>(
        kernel, ov::op::v0::Constant::create(ov::element::i64, {5}, {static_cast<int64_t>(oc), static_cast<int64_t>(c), KD, KH, KW}), ખોટું);
    auto આગત_5d = std::make_shared<ov::op::v1::Reshape>(
        આગત, ov::op::v0::Constant::create(ov::element::i64, {5}, {static_cast<int64_t>(n), static_cast<int64_t>(c), ID, IH, IW}), ખોટું);

    ov::Strides strides{static_cast<size_t>(s2), static_cast<size_t>(s1), static_cast<size_t>(s0)};
    ov::CoordinateDiff pads_begin{static_cast<ptrdiff_t>(p2), static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::CoordinateDiff pads_end{static_cast<ptrdiff_t>(p2), static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::Strides dilations{static_cast<size_t>(d2), static_cast<size_t>(d1), static_cast<size_t>(d0)};

    auto conv = std::make_shared<ov::op::v1::Convolution>(
        આગત_5d, kernel_5d, strides, pads_begin, pads_end, dilations, ov::op::PadType::EXPLICIT);

    અચલ auto out_shape = context.get_output_shape().to_shape(); // [oc*n, OD, OH, OW]
    auto out_shape_const = ov::op::v0::Constant::create(
        ov::element::i64, {4}, {static_cast<int64_t>(out_shape[0]), static_cast<int64_t>(out_shape[1]),
                                static_cast<int64_t>(out_shape[2]), static_cast<int64_t>(out_shape[3])});
    ov::Output<Node> res = std::make_shared<ov::op::v1::Reshape>(conv, out_shape_const, ખોટું);

    અચલ auto output_type = context.get_output_type();
    જો (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    પાછા rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
