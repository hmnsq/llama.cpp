#ضمن "../node_context.h"
#ضمن "../op_table.h"
#ضمن "../utils.h"

#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/convert.hpp>
#ضمن <openvino/op/convolution.hpp>
#ضمن <openvino/op/group_conv.hpp>
#ضمن <openvino/op/reshape.hpp>
#ضمن <openvino/op/unsqueeze.hpp>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق op {

OutputVector translate_conv_2d(ثابت NodeContext & context) {
    num_inputs_check(context, 2, 2);

    ov::Output<Node> kernel = process_view_input_new(context, 0);
    ov::Output<Node> input  = process_view_input_new(context, 1);

    𐤀𐤌 (kernel.get_element_type() != input.get_element_type()) {
        kernel = std::make_shared<ov::op::v0::Convert>(kernel, input.get_element_type());
    }

    ثابت int32_t * params = context.get_output_op_params();
    ثابت صحيح s0 = params[0];
    ثابت صحيح s1 = params[1];
    ثابت صحيح p0 = params[2];
    ثابت صحيح p1 = params[3];
    ثابت صحيح d0 = params[4];
    ثابت صحيح d1 = params[5];

    ov::Strides strides{static_cast<size_t>(s1), static_cast<size_t>(s0)};
    ov::CoordinateDiff pads_begin{static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::CoordinateDiff pads_end{static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::Strides dilations{static_cast<size_t>(d1), static_cast<size_t>(d0)};

    ov::Output<Node> res = std::make_shared<ov::op::v1::Convolution>(
        input, kernel, strides, pads_begin, pads_end, dilations, ov::op::PadType::EXPLICIT);

    ثابت auto output_type = context.get_output_type();
    𐤀𐤌 (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_conv_2d_dw(ثابت NodeContext & context) {
    num_inputs_check(context, 2, 2);

    ov::Output<Node> kernel = process_view_input_new(context, 0);
    ov::Output<Node> input  = process_view_input_new(context, 1);

    𐤀𐤌 (kernel.get_element_type() != input.get_element_type()) {
        kernel = std::make_shared<ov::op::v0::Convert>(kernel, input.get_element_type());
    }

    ثابت int32_t * params = context.get_output_op_params();
    ثابت صحيح s0 = params[0];
    ثابت صحيح s1 = params[1];
    ثابت صحيح p0 = params[2];
    ثابت صحيح p1 = params[3];
    ثابت صحيح d0 = params[4];
    ثابت صحيح d1 = params[5];

    // Reshape kernel from [C, 1, KH, KW] to [C, 1, 1, KH, KW] for 2D GroupConvolution
    auto unsqueeze_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto kernel_5d = std::make_shared<ov::op::v0::Unsqueeze>(kernel, unsqueeze_axis);

    ov::Strides strides{static_cast<size_t>(s1), static_cast<size_t>(s0)};
    ov::CoordinateDiff pads_begin{static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::CoordinateDiff pads_end{static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::Strides dilations{static_cast<size_t>(d1), static_cast<size_t>(d0)};

    ov::Output<Node> res = std::make_shared<ov::op::v1::GroupConvolution>(
        input, kernel_5d, strides, pads_begin, pads_end, dilations, ov::op::PadType::EXPLICIT);

    ثابت auto output_type = context.get_output_type();
    𐤀𐤌 (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_conv_transpose_1d(ثابت NodeContext & context) {
    num_inputs_check(context, 2, 2);

    ov::Output<Node> kernel = process_view_input_new(context, 0);
    ov::Output<Node> input  = process_view_input_new(context, 1);

    𐤀𐤌 (kernel.get_element_type() != input.get_element_type()) {
        kernel = std::make_shared<ov::op::v0::Convert>(kernel, input.get_element_type());
    }

    ثابت int32_t * params = context.get_output_op_params();
    ثابت صحيح s0 = params[0];
    ثابت صحيح p0 = params[1];
    ثابت صحيح d0 = params[2];

    ثابت auto kernel_shape = context.get_input_shape(0).to_shape();  // [1, Cin, Cout, K]
    ثابت int64_t Cin = kernel_shape[1];
    ثابت int64_t Cout = kernel_shape[2];
    ثابت int64_t K = kernel_shape[3];

    ثابت auto input_shape = context.get_input_shape(1).to_shape();   // [1, N, Cin, L]
    ثابت int64_t N = input_shape[0] * input_shape[1];
    ثابت int64_t L = input_shape[3];

    auto kernel_3d = std::make_shared<ov::op::v1::Reshape>(
        kernel, ov::op::v0::Constant::create(ov::element::i64, {3}, {Cin, Cout, K}), خطأ);
    auto input_3d = std::make_shared<ov::op::v1::Reshape>(
        input, ov::op::v0::Constant::create(ov::element::i64, {3}, {N, Cin, L}), خطأ);

    ov::Strides strides{static_cast<size_t>(s0)};
    ov::CoordinateDiff pads_begin{static_cast<ptrdiff_t>(p0)};
    ov::CoordinateDiff pads_end{static_cast<ptrdiff_t>(p0)};
    ov::Strides dilations{static_cast<size_t>(d0)};

    auto conv_tr = std::make_shared<ov::op::v1::ConvolutionBackpropData>(
        input_3d, kernel_3d, strides, pads_begin, pads_end, dilations);

    ثابت auto out_shape = context.get_output_shape().to_shape();
    auto out_shape_const = ov::op::v0::Constant::create(
        ov::element::i64, {4}, {static_cast<int64_t>(out_shape[0]), static_cast<int64_t>(out_shape[1]),
                                static_cast<int64_t>(out_shape[2]), static_cast<int64_t>(out_shape[3])});
    ov::Output<Node> res = std::make_shared<ov::op::v1::Reshape>(conv_tr, out_shape_const, خطأ);

    ثابت auto output_type = context.get_output_type();
    𐤀𐤌 (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_conv_transpose_2d(ثابت NodeContext & context) {
    num_inputs_check(context, 2, 2);

    ov::Output<Node> kernel = process_view_input_new(context, 0);
    ov::Output<Node> input  = process_view_input_new(context, 1);

    𐤀𐤌 (kernel.get_element_type() != input.get_element_type()) {
        kernel = std::make_shared<ov::op::v0::Convert>(kernel, input.get_element_type());
    }

    ثابت int32_t * params = context.get_output_op_params();
    ثابت صحيح stride = params[0];

    ov::Strides strides{static_cast<size_t>(stride), static_cast<size_t>(stride)};
    ov::CoordinateDiff pads_begin{0, 0};
    ov::CoordinateDiff pads_end{0, 0};
    ov::Strides dilations{1, 1};

    ov::Output<Node> res = std::make_shared<ov::op::v1::ConvolutionBackpropData>(
        input, kernel, strides, pads_begin, pads_end, dilations);

    ثابت auto output_type = context.get_output_type();
    𐤀𐤌 (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_conv_3d(ثابت NodeContext & context) {
    num_inputs_check(context, 2, 2);

    ov::Output<Node> kernel = process_view_input_new(context, 0);
    ov::Output<Node> input  = process_view_input_new(context, 1);

    𐤀𐤌 (kernel.get_element_type() != input.get_element_type()) {
        kernel = std::make_shared<ov::op::v0::Convert>(kernel, input.get_element_type());
    }

    ثابت int32_t * params = context.get_output_op_params();
    ثابت صحيح s0 = params[0];
    ثابت صحيح s1 = params[1];
    ثابت صحيح s2 = params[2];
    ثابت صحيح p0 = params[3];
    ثابت صحيح p1 = params[4];
    ثابت صحيح p2 = params[5];
    ثابت صحيح d0 = params[6];
    ثابت صحيح d1 = params[7];
    ثابت صحيح d2 = params[8];
    ثابت صحيح c  = params[9];
    ثابت صحيح n  = params[10];
    ثابت صحيح oc = params[11];

    ثابت auto kshape = context.get_input_shape(0).to_shape();  // [c*oc, KD, KH, KW]
    ثابت int64_t KD = kshape[1];
    ثابت int64_t KH = kshape[2];
    ثابت int64_t KW = kshape[3];

    ثابت auto inshape = context.get_input_shape(1).to_shape(); // [c*n, ID, IH, IW]
    ثابت int64_t ID = inshape[1];
    ثابت int64_t IH = inshape[2];
    ثابت int64_t IW = inshape[3];

    auto kernel_5d = std::make_shared<ov::op::v1::Reshape>(
        kernel, ov::op::v0::Constant::create(ov::element::i64, {5}, {static_cast<int64_t>(oc), static_cast<int64_t>(c), KD, KH, KW}), خطأ);
    auto input_5d = std::make_shared<ov::op::v1::Reshape>(
        input, ov::op::v0::Constant::create(ov::element::i64, {5}, {static_cast<int64_t>(n), static_cast<int64_t>(c), ID, IH, IW}), خطأ);

    ov::Strides strides{static_cast<size_t>(s2), static_cast<size_t>(s1), static_cast<size_t>(s0)};
    ov::CoordinateDiff pads_begin{static_cast<ptrdiff_t>(p2), static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::CoordinateDiff pads_end{static_cast<ptrdiff_t>(p2), static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::Strides dilations{static_cast<size_t>(d2), static_cast<size_t>(d1), static_cast<size_t>(d0)};

    auto conv = std::make_shared<ov::op::v1::Convolution>(
        input_5d, kernel_5d, strides, pads_begin, pads_end, dilations, ov::op::PadType::EXPLICIT);

    ثابت auto out_shape = context.get_output_shape().to_shape(); // [oc*n, OD, OH, OW]
    auto out_shape_const = ov::op::v0::Constant::create(
        ov::element::i64, {4}, {static_cast<int64_t>(out_shape[0]), static_cast<int64_t>(out_shape[1]),
                                static_cast<int64_t>(out_shape[2]), static_cast<int64_t>(out_shape[3])});
    ov::Output<Node> res = std::make_shared<ov::op::v1::Reshape>(conv, out_shape_const, خطأ);

    ثابت auto output_type = context.get_output_type();
    𐤀𐤌 (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
