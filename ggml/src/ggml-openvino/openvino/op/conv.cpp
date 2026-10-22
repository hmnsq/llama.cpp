#അന്തര്ഭാവ "../node_context.h"
#അന്തര്ഭാവ "../op_table.h"
#അന്തര്ഭാവ "../utils.h"

#അന്തര്ഭാവ <openvino/op/constant.hpp>
#അന്തര്ഭാവ <openvino/op/convert.hpp>
#അന്തര്ഭാവ <openvino/op/convolution.hpp>
#അന്തര്ഭാവ <openvino/op/group_conv.hpp>
#അന്തര്ഭാവ <openvino/op/reshape.hpp>
#അന്തര്ഭാവ <openvino/op/unsqueeze.hpp>

നാമാകാശ ov {
നാമാകാശ frontend {
നാമാകാശ ggml {
നാമാകാശ op {

OutputVector translate_conv_2d(സ്ഥിരം കണ്ണി_Context & context) {
    അക്കം_inputs_check(context, 2, 2);

    ov::Output<Node> kernel = process_view_നിവേശം_new(context, 0);
    ov::Output<Node> നിവേശം  = process_view_നിവേശം_new(context, 1);

    എങ്കിൽ (kernel.get_element_type() != നിവേശം.get_element_type()) {
        kernel = std::make_shared<ov::op::v0::Convert>(kernel, നിവേശം.get_element_type());
    }

    സ്ഥിരം int32_t * params = context.get_output_op_params();
    സ്ഥിരം പൂർണ്ണസംഖ്യ s0 = params[0];
    സ്ഥിരം പൂർണ്ണസംഖ്യ s1 = params[1];
    സ്ഥിരം പൂർണ്ണസംഖ്യ p0 = params[2];
    സ്ഥിരം പൂർണ്ണസംഖ്യ p1 = params[3];
    സ്ഥിരം പൂർണ്ണസംഖ്യ d0 = params[4];
    സ്ഥിരം പൂർണ്ണസംഖ്യ d1 = params[5];

    ov::Strides strides{static_cast<size_t>(s1), static_cast<size_t>(s0)};
    ov::CoordinateDiff pads_begin{static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::CoordinateDiff pads_end{static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::Strides dilations{static_cast<size_t>(d1), static_cast<size_t>(d0)};

    ov::Output<Node> res = std::make_shared<ov::op::v1::Convolution>(
        നിവേശം, kernel, strides, pads_begin, pads_end, dilations, ov::op::PadType::EXPLICIT);

    സ്ഥിരം auto output_type = context.get_output_type();
    എങ്കിൽ (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    തിരികെ rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_conv_2d_dw(സ്ഥിരം കണ്ണി_Context & context) {
    അക്കം_inputs_check(context, 2, 2);

    ov::Output<Node> kernel = process_view_നിവേശം_new(context, 0);
    ov::Output<Node> നിവേശം  = process_view_നിവേശം_new(context, 1);

    എങ്കിൽ (kernel.get_element_type() != നിവേശം.get_element_type()) {
        kernel = std::make_shared<ov::op::v0::Convert>(kernel, നിവേശം.get_element_type());
    }

    സ്ഥിരം int32_t * params = context.get_output_op_params();
    സ്ഥിരം പൂർണ്ണസംഖ്യ s0 = params[0];
    സ്ഥിരം പൂർണ്ണസംഖ്യ s1 = params[1];
    സ്ഥിരം പൂർണ്ണസംഖ്യ p0 = params[2];
    സ്ഥിരം പൂർണ്ണസംഖ്യ p1 = params[3];
    സ്ഥിരം പൂർണ്ണസംഖ്യ d0 = params[4];
    സ്ഥിരം പൂർണ്ണസംഖ്യ d1 = params[5];

    // Reshape kernel from [C, 1, KH, KW] to [C, 1, 1, KH, KW] for 2D GroupConvolution
    auto unsqueeze_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto kernel_5d = std::make_shared<ov::op::v0::Unsqueeze>(kernel, unsqueeze_axis);

    ov::Strides strides{static_cast<size_t>(s1), static_cast<size_t>(s0)};
    ov::CoordinateDiff pads_begin{static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::CoordinateDiff pads_end{static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::Strides dilations{static_cast<size_t>(d1), static_cast<size_t>(d0)};

    ov::Output<Node> res = std::make_shared<ov::op::v1::GroupConvolution>(
        നിവേശം, kernel_5d, strides, pads_begin, pads_end, dilations, ov::op::PadType::EXPLICIT);

    സ്ഥിരം auto output_type = context.get_output_type();
    എങ്കിൽ (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    തിരികെ rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_conv_transpose_1d(സ്ഥിരം കണ്ണി_Context & context) {
    അക്കം_inputs_check(context, 2, 2);

    ov::Output<Node> kernel = process_view_നിവേശം_new(context, 0);
    ov::Output<Node> നിവേശം  = process_view_നിവേശം_new(context, 1);

    എങ്കിൽ (kernel.get_element_type() != നിവേശം.get_element_type()) {
        kernel = std::make_shared<ov::op::v0::Convert>(kernel, നിവേശം.get_element_type());
    }

    സ്ഥിരം int32_t * params = context.get_output_op_params();
    സ്ഥിരം പൂർണ്ണസംഖ്യ s0 = params[0];
    സ്ഥിരം പൂർണ്ണസംഖ്യ p0 = params[1];
    സ്ഥിരം പൂർണ്ണസംഖ്യ d0 = params[2];

    സ്ഥിരം auto kernel_shape = context.get_നിവേശം_shape(0).to_shape();  // [1, Cin, Cout, K]
    സ്ഥിരം int64_t Cin = kernel_shape[1];
    സ്ഥിരം int64_t Cout = kernel_shape[2];
    സ്ഥിരം int64_t K = kernel_shape[3];

    സ്ഥിരം auto നിവേശം_shape = context.get_നിവേശം_shape(1).to_shape();   // [1, N, Cin, L]
    സ്ഥിരം int64_t N = നിവേശം_shape[0] * നിവേശം_shape[1];
    സ്ഥിരം int64_t L = നിവേശം_shape[3];

    auto kernel_3d = std::make_shared<ov::op::v1::Reshape>(
        kernel, ov::op::v0::Constant::create(ov::element::i64, {3}, {Cin, Cout, K}), അസത്യം);
    auto നിവേശം_3d = std::make_shared<ov::op::v1::Reshape>(
        നിവേശം, ov::op::v0::Constant::create(ov::element::i64, {3}, {N, Cin, L}), അസത്യം);

    ov::Strides strides{static_cast<size_t>(s0)};
    ov::CoordinateDiff pads_begin{static_cast<ptrdiff_t>(p0)};
    ov::CoordinateDiff pads_end{static_cast<ptrdiff_t>(p0)};
    ov::Strides dilations{static_cast<size_t>(d0)};

    auto conv_tr = std::make_shared<ov::op::v1::ConvolutionBackpropData>(
        നിവേശം_3d, kernel_3d, strides, pads_begin, pads_end, dilations);

    സ്ഥിരം auto out_shape = context.get_output_shape().to_shape();
    auto out_shape_const = ov::op::v0::Constant::create(
        ov::element::i64, {4}, {static_cast<int64_t>(out_shape[0]), static_cast<int64_t>(out_shape[1]),
                                static_cast<int64_t>(out_shape[2]), static_cast<int64_t>(out_shape[3])});
    ov::Output<Node> res = std::make_shared<ov::op::v1::Reshape>(conv_tr, out_shape_const, അസത്യം);

    സ്ഥിരം auto output_type = context.get_output_type();
    എങ്കിൽ (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    തിരികെ rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_conv_transpose_2d(സ്ഥിരം കണ്ണി_Context & context) {
    അക്കം_inputs_check(context, 2, 2);

    ov::Output<Node> kernel = process_view_നിവേശം_new(context, 0);
    ov::Output<Node> നിവേശം  = process_view_നിവേശം_new(context, 1);

    എങ്കിൽ (kernel.get_element_type() != നിവേശം.get_element_type()) {
        kernel = std::make_shared<ov::op::v0::Convert>(kernel, നിവേശം.get_element_type());
    }

    സ്ഥിരം int32_t * params = context.get_output_op_params();
    സ്ഥിരം പൂർണ്ണസംഖ്യ stride = params[0];

    ov::Strides strides{static_cast<size_t>(stride), static_cast<size_t>(stride)};
    ov::CoordinateDiff pads_begin{0, 0};
    ov::CoordinateDiff pads_end{0, 0};
    ov::Strides dilations{1, 1};

    ov::Output<Node> res = std::make_shared<ov::op::v1::ConvolutionBackpropData>(
        നിവേശം, kernel, strides, pads_begin, pads_end, dilations);

    സ്ഥിരം auto output_type = context.get_output_type();
    എങ്കിൽ (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    തിരികെ rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_conv_3d(സ്ഥിരം കണ്ണി_Context & context) {
    അക്കം_inputs_check(context, 2, 2);

    ov::Output<Node> kernel = process_view_നിവേശം_new(context, 0);
    ov::Output<Node> നിവേശം  = process_view_നിവേശം_new(context, 1);

    എങ്കിൽ (kernel.get_element_type() != നിവേശം.get_element_type()) {
        kernel = std::make_shared<ov::op::v0::Convert>(kernel, നിവേശം.get_element_type());
    }

    സ്ഥിരം int32_t * params = context.get_output_op_params();
    സ്ഥിരം പൂർണ്ണസംഖ്യ s0 = params[0];
    സ്ഥിരം പൂർണ്ണസംഖ്യ s1 = params[1];
    സ്ഥിരം പൂർണ്ണസംഖ്യ s2 = params[2];
    സ്ഥിരം പൂർണ്ണസംഖ്യ p0 = params[3];
    സ്ഥിരം പൂർണ്ണസംഖ്യ p1 = params[4];
    സ്ഥിരം പൂർണ്ണസംഖ്യ p2 = params[5];
    സ്ഥിരം പൂർണ്ണസംഖ്യ d0 = params[6];
    സ്ഥിരം പൂർണ്ണസംഖ്യ d1 = params[7];
    സ്ഥിരം പൂർണ്ണസംഖ്യ d2 = params[8];
    സ്ഥിരം പൂർണ്ണസംഖ്യ c  = params[9];
    സ്ഥിരം പൂർണ്ണസംഖ്യ n  = params[10];
    സ്ഥിരം പൂർണ്ണസംഖ്യ oc = params[11];

    സ്ഥിരം auto kshape = context.get_നിവേശം_shape(0).to_shape();  // [c*oc, KD, KH, KW]
    സ്ഥിരം int64_t KD = kshape[1];
    സ്ഥിരം int64_t KH = kshape[2];
    സ്ഥിരം int64_t KW = kshape[3];

    സ്ഥിരം auto inshape = context.get_നിവേശം_shape(1).to_shape(); // [c*n, ID, IH, IW]
    സ്ഥിരം int64_t ID = inshape[1];
    സ്ഥിരം int64_t IH = inshape[2];
    സ്ഥിരം int64_t IW = inshape[3];

    auto kernel_5d = std::make_shared<ov::op::v1::Reshape>(
        kernel, ov::op::v0::Constant::create(ov::element::i64, {5}, {static_cast<int64_t>(oc), static_cast<int64_t>(c), KD, KH, KW}), അസത്യം);
    auto നിവേശം_5d = std::make_shared<ov::op::v1::Reshape>(
        നിവേശം, ov::op::v0::Constant::create(ov::element::i64, {5}, {static_cast<int64_t>(n), static_cast<int64_t>(c), ID, IH, IW}), അസത്യം);

    ov::Strides strides{static_cast<size_t>(s2), static_cast<size_t>(s1), static_cast<size_t>(s0)};
    ov::CoordinateDiff pads_begin{static_cast<ptrdiff_t>(p2), static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::CoordinateDiff pads_end{static_cast<ptrdiff_t>(p2), static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::Strides dilations{static_cast<size_t>(d2), static_cast<size_t>(d1), static_cast<size_t>(d0)};

    auto conv = std::make_shared<ov::op::v1::Convolution>(
        നിവേശം_5d, kernel_5d, strides, pads_begin, pads_end, dilations, ov::op::PadType::EXPLICIT);

    സ്ഥിരം auto out_shape = context.get_output_shape().to_shape(); // [oc*n, OD, OH, OW]
    auto out_shape_const = ov::op::v0::Constant::create(
        ov::element::i64, {4}, {static_cast<int64_t>(out_shape[0]), static_cast<int64_t>(out_shape[1]),
                                static_cast<int64_t>(out_shape[2]), static_cast<int64_t>(out_shape[3])});
    ov::Output<Node> res = std::make_shared<ov::op::v1::Reshape>(conv, out_shape_const, അസത്യം);

    സ്ഥിരം auto output_type = context.get_output_type();
    എങ്കിൽ (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    തിരികെ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
