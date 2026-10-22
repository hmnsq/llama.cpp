#अन्तर्भाव "../node_context.h"
#अन्तर्भाव "../op_table.h"
#अन्तर्भाव "../utils.h"

#अन्तर्भाव <openvino/op/constant.hpp>
#अन्तर्भाव <openvino/op/convert.hpp>
#अन्तर्भाव <openvino/op/convolution.hpp>
#अन्तर्भाव <openvino/op/group_conv.hpp>
#अन्तर्भाव <openvino/op/reshape.hpp>
#अन्तर्भाव <openvino/op/unsqueeze.hpp>

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {
नामाकाश op {

OutputVector translate_conv_2d(ध्रुव गाँठ_Context & context) {
    अंक_inputs_check(context, 2, 2);

    ov::Output<Node> kernel = process_view_निवेश_new(context, 0);
    ov::Output<Node> निवेश  = process_view_निवेश_new(context, 1);

    यदि (kernel.get_element_type() != निवेश.get_element_type()) {
        kernel = std::make_shared<ov::op::v0::Convert>(kernel, निवेश.get_element_type());
    }

    ध्रुव int32_t * params = context.get_output_op_params();
    ध्रुव पूर्णाङ्क s0 = params[0];
    ध्रुव पूर्णाङ्क s1 = params[1];
    ध्रुव पूर्णाङ्क p0 = params[2];
    ध्रुव पूर्णाङ्क p1 = params[3];
    ध्रुव पूर्णाङ्क d0 = params[4];
    ध्रुव पूर्णाङ्क d1 = params[5];

    ov::Strides strides{static_cast<size_t>(s1), static_cast<size_t>(s0)};
    ov::CoordinateDiff pads_begin{static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::CoordinateDiff pads_end{static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::Strides dilations{static_cast<size_t>(d1), static_cast<size_t>(d0)};

    ov::Output<Node> res = std::make_shared<ov::op::v1::Convolution>(
        निवेश, kernel, strides, pads_begin, pads_end, dilations, ov::op::PadType::EXPLICIT);

    ध्रुव auto output_type = context.get_output_type();
    यदि (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_conv_2d_dw(ध्रुव गाँठ_Context & context) {
    अंक_inputs_check(context, 2, 2);

    ov::Output<Node> kernel = process_view_निवेश_new(context, 0);
    ov::Output<Node> निवेश  = process_view_निवेश_new(context, 1);

    यदि (kernel.get_element_type() != निवेश.get_element_type()) {
        kernel = std::make_shared<ov::op::v0::Convert>(kernel, निवेश.get_element_type());
    }

    ध्रुव int32_t * params = context.get_output_op_params();
    ध्रुव पूर्णाङ्क s0 = params[0];
    ध्रुव पूर्णाङ्क s1 = params[1];
    ध्रुव पूर्णाङ्क p0 = params[2];
    ध्रुव पूर्णाङ्क p1 = params[3];
    ध्रुव पूर्णाङ्क d0 = params[4];
    ध्रुव पूर्णाङ्क d1 = params[5];

    // Reshape kernel from [C, 1, KH, KW] to [C, 1, 1, KH, KW] for 2D GroupConvolution
    auto unsqueeze_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto kernel_5d = std::make_shared<ov::op::v0::Unsqueeze>(kernel, unsqueeze_axis);

    ov::Strides strides{static_cast<size_t>(s1), static_cast<size_t>(s0)};
    ov::CoordinateDiff pads_begin{static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::CoordinateDiff pads_end{static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::Strides dilations{static_cast<size_t>(d1), static_cast<size_t>(d0)};

    ov::Output<Node> res = std::make_shared<ov::op::v1::GroupConvolution>(
        निवेश, kernel_5d, strides, pads_begin, pads_end, dilations, ov::op::PadType::EXPLICIT);

    ध्रुव auto output_type = context.get_output_type();
    यदि (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_conv_transpose_1d(ध्रुव गाँठ_Context & context) {
    अंक_inputs_check(context, 2, 2);

    ov::Output<Node> kernel = process_view_निवेश_new(context, 0);
    ov::Output<Node> निवेश  = process_view_निवेश_new(context, 1);

    यदि (kernel.get_element_type() != निवेश.get_element_type()) {
        kernel = std::make_shared<ov::op::v0::Convert>(kernel, निवेश.get_element_type());
    }

    ध्रुव int32_t * params = context.get_output_op_params();
    ध्रुव पूर्णाङ्क s0 = params[0];
    ध्रुव पूर्णाङ्क p0 = params[1];
    ध्रुव पूर्णाङ्क d0 = params[2];

    ध्रुव auto kernel_shape = context.get_निवेश_shape(0).to_shape();  // [1, Cin, Cout, K]
    ध्रुव int64_t Cin = kernel_shape[1];
    ध्रुव int64_t Cout = kernel_shape[2];
    ध्रुव int64_t K = kernel_shape[3];

    ध्रुव auto निवेश_shape = context.get_निवेश_shape(1).to_shape();   // [1, N, Cin, L]
    ध्रुव int64_t N = निवेश_shape[0] * निवेश_shape[1];
    ध्रुव int64_t L = निवेश_shape[3];

    auto kernel_3d = std::make_shared<ov::op::v1::Reshape>(
        kernel, ov::op::v0::Constant::create(ov::element::i64, {3}, {Cin, Cout, K}), असत्य);
    auto निवेश_3d = std::make_shared<ov::op::v1::Reshape>(
        निवेश, ov::op::v0::Constant::create(ov::element::i64, {3}, {N, Cin, L}), असत्य);

    ov::Strides strides{static_cast<size_t>(s0)};
    ov::CoordinateDiff pads_begin{static_cast<ptrdiff_t>(p0)};
    ov::CoordinateDiff pads_end{static_cast<ptrdiff_t>(p0)};
    ov::Strides dilations{static_cast<size_t>(d0)};

    auto conv_tr = std::make_shared<ov::op::v1::ConvolutionBackpropData>(
        निवेश_3d, kernel_3d, strides, pads_begin, pads_end, dilations);

    ध्रुव auto out_shape = context.get_output_shape().to_shape();
    auto out_shape_const = ov::op::v0::Constant::create(
        ov::element::i64, {4}, {static_cast<int64_t>(out_shape[0]), static_cast<int64_t>(out_shape[1]),
                                static_cast<int64_t>(out_shape[2]), static_cast<int64_t>(out_shape[3])});
    ov::Output<Node> res = std::make_shared<ov::op::v1::Reshape>(conv_tr, out_shape_const, असत्य);

    ध्रुव auto output_type = context.get_output_type();
    यदि (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_conv_transpose_2d(ध्रुव गाँठ_Context & context) {
    अंक_inputs_check(context, 2, 2);

    ov::Output<Node> kernel = process_view_निवेश_new(context, 0);
    ov::Output<Node> निवेश  = process_view_निवेश_new(context, 1);

    यदि (kernel.get_element_type() != निवेश.get_element_type()) {
        kernel = std::make_shared<ov::op::v0::Convert>(kernel, निवेश.get_element_type());
    }

    ध्रुव int32_t * params = context.get_output_op_params();
    ध्रुव पूर्णाङ्क stride = params[0];

    ov::Strides strides{static_cast<size_t>(stride), static_cast<size_t>(stride)};
    ov::CoordinateDiff pads_begin{0, 0};
    ov::CoordinateDiff pads_end{0, 0};
    ov::Strides dilations{1, 1};

    ov::Output<Node> res = std::make_shared<ov::op::v1::ConvolutionBackpropData>(
        निवेश, kernel, strides, pads_begin, pads_end, dilations);

    ध्रुव auto output_type = context.get_output_type();
    यदि (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_conv_3d(ध्रुव गाँठ_Context & context) {
    अंक_inputs_check(context, 2, 2);

    ov::Output<Node> kernel = process_view_निवेश_new(context, 0);
    ov::Output<Node> निवेश  = process_view_निवेश_new(context, 1);

    यदि (kernel.get_element_type() != निवेश.get_element_type()) {
        kernel = std::make_shared<ov::op::v0::Convert>(kernel, निवेश.get_element_type());
    }

    ध्रुव int32_t * params = context.get_output_op_params();
    ध्रुव पूर्णाङ्क s0 = params[0];
    ध्रुव पूर्णाङ्क s1 = params[1];
    ध्रुव पूर्णाङ्क s2 = params[2];
    ध्रुव पूर्णाङ्क p0 = params[3];
    ध्रुव पूर्णाङ्क p1 = params[4];
    ध्रुव पूर्णाङ्क p2 = params[5];
    ध्रुव पूर्णाङ्क d0 = params[6];
    ध्रुव पूर्णाङ्क d1 = params[7];
    ध्रुव पूर्णाङ्क d2 = params[8];
    ध्रुव पूर्णाङ्क c  = params[9];
    ध्रुव पूर्णाङ्क n  = params[10];
    ध्रुव पूर्णाङ्क oc = params[11];

    ध्रुव auto kshape = context.get_निवेश_shape(0).to_shape();  // [c*oc, KD, KH, KW]
    ध्रुव int64_t KD = kshape[1];
    ध्रुव int64_t KH = kshape[2];
    ध्रुव int64_t KW = kshape[3];

    ध्रुव auto inshape = context.get_निवेश_shape(1).to_shape(); // [c*n, ID, IH, IW]
    ध्रुव int64_t ID = inshape[1];
    ध्रुव int64_t IH = inshape[2];
    ध्रुव int64_t IW = inshape[3];

    auto kernel_5d = std::make_shared<ov::op::v1::Reshape>(
        kernel, ov::op::v0::Constant::create(ov::element::i64, {5}, {static_cast<int64_t>(oc), static_cast<int64_t>(c), KD, KH, KW}), असत्य);
    auto निवेश_5d = std::make_shared<ov::op::v1::Reshape>(
        निवेश, ov::op::v0::Constant::create(ov::element::i64, {5}, {static_cast<int64_t>(n), static_cast<int64_t>(c), ID, IH, IW}), असत्य);

    ov::Strides strides{static_cast<size_t>(s2), static_cast<size_t>(s1), static_cast<size_t>(s0)};
    ov::CoordinateDiff pads_begin{static_cast<ptrdiff_t>(p2), static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::CoordinateDiff pads_end{static_cast<ptrdiff_t>(p2), static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::Strides dilations{static_cast<size_t>(d2), static_cast<size_t>(d1), static_cast<size_t>(d0)};

    auto conv = std::make_shared<ov::op::v1::Convolution>(
        निवेश_5d, kernel_5d, strides, pads_begin, pads_end, dilations, ov::op::PadType::EXPLICIT);

    ध्रुव auto out_shape = context.get_output_shape().to_shape(); // [oc*n, OD, OH, OW]
    auto out_shape_const = ov::op::v0::Constant::create(
        ov::element::i64, {4}, {static_cast<int64_t>(out_shape[0]), static_cast<int64_t>(out_shape[1]),
                                static_cast<int64_t>(out_shape[2]), static_cast<int64_t>(out_shape[3])});
    ov::Output<Node> res = std::make_shared<ov::op::v1::Reshape>(conv, out_shape_const, असत्य);

    ध्रुव auto output_type = context.get_output_type();
    यदि (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
