#ଅନ୍ତର୍ଭାଵ "../node_context.h"
#ଅନ୍ତର୍ଭାଵ "../op_table.h"
#ଅନ୍ତର୍ଭାଵ "../utils.h"

#ଅନ୍ତର୍ଭାଵ <openvino/op/constant.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/convert.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/convolution.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/group_conv.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/reshape.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/unsqueeze.hpp>

ନାମାକାଶ ov {
ନାମାକାଶ frontend {
ନାମାକାଶ ggml {
ନାମାକାଶ op {

OutputVector translate_conv_2d(ଅଚଳ ଗଣ୍ଠି_Context & context) {
    ଅଙ୍କ_inputs_check(context, 2, 2);

    ov::Output<Node> kernel = process_view_ନିବେଶ_new(context, 0);
    ov::Output<Node> ନିବେଶ  = process_view_ନିବେଶ_new(context, 1);

    ଯଦି (kernel.get_element_type() != ନିବେଶ.get_element_type()) {
        kernel = std::make_shared<ov::op::v0::Convert>(kernel, ନିବେଶ.get_element_type());
    }

    ଅଚଳ int32_t * params = context.get_output_op_params();
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s0 = params[0];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s1 = params[1];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା p0 = params[2];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା p1 = params[3];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା d0 = params[4];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା d1 = params[5];

    ov::Strides strides{static_cast<size_t>(s1), static_cast<size_t>(s0)};
    ov::CoordinateDiff pads_begin{static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::CoordinateDiff pads_end{static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::Strides dilations{static_cast<size_t>(d1), static_cast<size_t>(d0)};

    ov::Output<Node> res = std::make_shared<ov::op::v1::Convolution>(
        ନିବେଶ, kernel, strides, pads_begin, pads_end, dilations, ov::op::PadType::EXPLICIT);

    ଅଚଳ auto output_type = context.get_output_type();
    ଯଦି (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    ଫେରନ୍ତୁ rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_conv_2d_dw(ଅଚଳ ଗଣ୍ଠି_Context & context) {
    ଅଙ୍କ_inputs_check(context, 2, 2);

    ov::Output<Node> kernel = process_view_ନିବେଶ_new(context, 0);
    ov::Output<Node> ନିବେଶ  = process_view_ନିବେଶ_new(context, 1);

    ଯଦି (kernel.get_element_type() != ନିବେଶ.get_element_type()) {
        kernel = std::make_shared<ov::op::v0::Convert>(kernel, ନିବେଶ.get_element_type());
    }

    ଅଚଳ int32_t * params = context.get_output_op_params();
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s0 = params[0];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s1 = params[1];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା p0 = params[2];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା p1 = params[3];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା d0 = params[4];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା d1 = params[5];

    // Reshape kernel from [C, 1, KH, KW] to [C, 1, 1, KH, KW] for 2D GroupConvolution
    auto unsqueeze_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto kernel_5d = std::make_shared<ov::op::v0::Unsqueeze>(kernel, unsqueeze_axis);

    ov::Strides strides{static_cast<size_t>(s1), static_cast<size_t>(s0)};
    ov::CoordinateDiff pads_begin{static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::CoordinateDiff pads_end{static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::Strides dilations{static_cast<size_t>(d1), static_cast<size_t>(d0)};

    ov::Output<Node> res = std::make_shared<ov::op::v1::GroupConvolution>(
        ନିବେଶ, kernel_5d, strides, pads_begin, pads_end, dilations, ov::op::PadType::EXPLICIT);

    ଅଚଳ auto output_type = context.get_output_type();
    ଯଦି (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    ଫେରନ୍ତୁ rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_conv_transpose_1d(ଅଚଳ ଗଣ୍ଠି_Context & context) {
    ଅଙ୍କ_inputs_check(context, 2, 2);

    ov::Output<Node> kernel = process_view_ନିବେଶ_new(context, 0);
    ov::Output<Node> ନିବେଶ  = process_view_ନିବେଶ_new(context, 1);

    ଯଦି (kernel.get_element_type() != ନିବେଶ.get_element_type()) {
        kernel = std::make_shared<ov::op::v0::Convert>(kernel, ନିବେଶ.get_element_type());
    }

    ଅଚଳ int32_t * params = context.get_output_op_params();
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s0 = params[0];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା p0 = params[1];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା d0 = params[2];

    ଅଚଳ auto kernel_shape = context.get_ନିବେଶ_shape(0).to_shape();  // [1, Cin, Cout, K]
    ଅଚଳ int64_t Cin = kernel_shape[1];
    ଅଚଳ int64_t Cout = kernel_shape[2];
    ଅଚଳ int64_t K = kernel_shape[3];

    ଅଚଳ auto ନିବେଶ_shape = context.get_ନିବେଶ_shape(1).to_shape();   // [1, N, Cin, L]
    ଅଚଳ int64_t N = ନିବେଶ_shape[0] * ନିବେଶ_shape[1];
    ଅଚଳ int64_t L = ନିବେଶ_shape[3];

    auto kernel_3d = std::make_shared<ov::op::v1::Reshape>(
        kernel, ov::op::v0::Constant::create(ov::element::i64, {3}, {Cin, Cout, K}), ମିଥ୍ୟା);
    auto ନିବେଶ_3d = std::make_shared<ov::op::v1::Reshape>(
        ନିବେଶ, ov::op::v0::Constant::create(ov::element::i64, {3}, {N, Cin, L}), ମିଥ୍ୟା);

    ov::Strides strides{static_cast<size_t>(s0)};
    ov::CoordinateDiff pads_begin{static_cast<ptrdiff_t>(p0)};
    ov::CoordinateDiff pads_end{static_cast<ptrdiff_t>(p0)};
    ov::Strides dilations{static_cast<size_t>(d0)};

    auto conv_tr = std::make_shared<ov::op::v1::ConvolutionBackpropData>(
        ନିବେଶ_3d, kernel_3d, strides, pads_begin, pads_end, dilations);

    ଅଚଳ auto out_shape = context.get_output_shape().to_shape();
    auto out_shape_const = ov::op::v0::Constant::create(
        ov::element::i64, {4}, {static_cast<int64_t>(out_shape[0]), static_cast<int64_t>(out_shape[1]),
                                static_cast<int64_t>(out_shape[2]), static_cast<int64_t>(out_shape[3])});
    ov::Output<Node> res = std::make_shared<ov::op::v1::Reshape>(conv_tr, out_shape_const, ମିଥ୍ୟା);

    ଅଚଳ auto output_type = context.get_output_type();
    ଯଦି (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    ଫେରନ୍ତୁ rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_conv_transpose_2d(ଅଚଳ ଗଣ୍ଠି_Context & context) {
    ଅଙ୍କ_inputs_check(context, 2, 2);

    ov::Output<Node> kernel = process_view_ନିବେଶ_new(context, 0);
    ov::Output<Node> ନିବେଶ  = process_view_ନିବେଶ_new(context, 1);

    ଯଦି (kernel.get_element_type() != ନିବେଶ.get_element_type()) {
        kernel = std::make_shared<ov::op::v0::Convert>(kernel, ନିବେଶ.get_element_type());
    }

    ଅଚଳ int32_t * params = context.get_output_op_params();
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride = params[0];

    ov::Strides strides{static_cast<size_t>(stride), static_cast<size_t>(stride)};
    ov::CoordinateDiff pads_begin{0, 0};
    ov::CoordinateDiff pads_end{0, 0};
    ov::Strides dilations{1, 1};

    ov::Output<Node> res = std::make_shared<ov::op::v1::ConvolutionBackpropData>(
        ନିବେଶ, kernel, strides, pads_begin, pads_end, dilations);

    ଅଚଳ auto output_type = context.get_output_type();
    ଯଦି (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    ଫେରନ୍ତୁ rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_conv_3d(ଅଚଳ ଗଣ୍ଠି_Context & context) {
    ଅଙ୍କ_inputs_check(context, 2, 2);

    ov::Output<Node> kernel = process_view_ନିବେଶ_new(context, 0);
    ov::Output<Node> ନିବେଶ  = process_view_ନିବେଶ_new(context, 1);

    ଯଦି (kernel.get_element_type() != ନିବେଶ.get_element_type()) {
        kernel = std::make_shared<ov::op::v0::Convert>(kernel, ନିବେଶ.get_element_type());
    }

    ଅଚଳ int32_t * params = context.get_output_op_params();
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s0 = params[0];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s1 = params[1];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s2 = params[2];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା p0 = params[3];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା p1 = params[4];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା p2 = params[5];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା d0 = params[6];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା d1 = params[7];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା d2 = params[8];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା c  = params[9];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n  = params[10];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା oc = params[11];

    ଅଚଳ auto kshape = context.get_ନିବେଶ_shape(0).to_shape();  // [c*oc, KD, KH, KW]
    ଅଚଳ int64_t KD = kshape[1];
    ଅଚଳ int64_t KH = kshape[2];
    ଅଚଳ int64_t KW = kshape[3];

    ଅଚଳ auto inshape = context.get_ନିବେଶ_shape(1).to_shape(); // [c*n, ID, IH, IW]
    ଅଚଳ int64_t ID = inshape[1];
    ଅଚଳ int64_t IH = inshape[2];
    ଅଚଳ int64_t IW = inshape[3];

    auto kernel_5d = std::make_shared<ov::op::v1::Reshape>(
        kernel, ov::op::v0::Constant::create(ov::element::i64, {5}, {static_cast<int64_t>(oc), static_cast<int64_t>(c), KD, KH, KW}), ମିଥ୍ୟା);
    auto ନିବେଶ_5d = std::make_shared<ov::op::v1::Reshape>(
        ନିବେଶ, ov::op::v0::Constant::create(ov::element::i64, {5}, {static_cast<int64_t>(n), static_cast<int64_t>(c), ID, IH, IW}), ମିଥ୍ୟା);

    ov::Strides strides{static_cast<size_t>(s2), static_cast<size_t>(s1), static_cast<size_t>(s0)};
    ov::CoordinateDiff pads_begin{static_cast<ptrdiff_t>(p2), static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::CoordinateDiff pads_end{static_cast<ptrdiff_t>(p2), static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::Strides dilations{static_cast<size_t>(d2), static_cast<size_t>(d1), static_cast<size_t>(d0)};

    auto conv = std::make_shared<ov::op::v1::Convolution>(
        ନିବେଶ_5d, kernel_5d, strides, pads_begin, pads_end, dilations, ov::op::PadType::EXPLICIT);

    ଅଚଳ auto out_shape = context.get_output_shape().to_shape(); // [oc*n, OD, OH, OW]
    auto out_shape_const = ov::op::v0::Constant::create(
        ov::element::i64, {4}, {static_cast<int64_t>(out_shape[0]), static_cast<int64_t>(out_shape[1]),
                                static_cast<int64_t>(out_shape[2]), static_cast<int64_t>(out_shape[3])});
    ov::Output<Node> res = std::make_shared<ov::op::v1::Reshape>(conv, out_shape_const, ମିଥ୍ୟା);

    ଅଚଳ auto output_type = context.get_output_type();
    ଯଦି (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    ଫେରନ୍ତୁ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
