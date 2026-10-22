#அந்தர்பாவ "../node_context.h"
#அந்தர்பாவ "../op_table.h"
#அந்தர்பாவ "../utils.h"

#அந்தர்பாவ <openvino/op/constant.hpp>
#அந்தர்பாவ <openvino/op/convert.hpp>
#அந்தர்பாவ <openvino/op/convolution.hpp>
#அந்தர்பாவ <openvino/op/group_conv.hpp>
#அந்தர்பாவ <openvino/op/reshape.hpp>
#அந்தர்பாவ <openvino/op/unsqueeze.hpp>

நாமாகாஶ ov {
நாமாகாஶ frontend {
நாமாகாஶ ggml {
நாமாகாஶ op {

OutputVector translate_conv_2d(மாறிலி முனை_Context & context) {
    இலக்கம்_inputs_check(context, 2, 2);

    ov::Output<Node> kernel = process_view_உள்ளீடு_new(context, 0);
    ov::Output<Node> உள்ளீடு  = process_view_உள்ளீடு_new(context, 1);

    எனில் (kernel.get_element_type() != உள்ளீடு.get_element_type()) {
        kernel = std::make_shared<ov::op::v0::Convert>(kernel, உள்ளீடு.get_element_type());
    }

    மாறிலி int32_t * params = context.get_output_op_params();
    மாறிலி முழு s0 = params[0];
    மாறிலி முழு s1 = params[1];
    மாறிலி முழு p0 = params[2];
    மாறிலி முழு p1 = params[3];
    மாறிலி முழு d0 = params[4];
    மாறிலி முழு d1 = params[5];

    ov::Strides strides{static_cast<size_t>(s1), static_cast<size_t>(s0)};
    ov::CoordinateDiff pads_begin{static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::CoordinateDiff pads_end{static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::Strides dilations{static_cast<size_t>(d1), static_cast<size_t>(d0)};

    ov::Output<Node> res = std::make_shared<ov::op::v1::Convolution>(
        உள்ளீடு, kernel, strides, pads_begin, pads_end, dilations, ov::op::PadType::EXPLICIT);

    மாறிலி auto output_type = context.get_output_type();
    எனில் (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    திரும்பு rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_conv_2d_dw(மாறிலி முனை_Context & context) {
    இலக்கம்_inputs_check(context, 2, 2);

    ov::Output<Node> kernel = process_view_உள்ளீடு_new(context, 0);
    ov::Output<Node> உள்ளீடு  = process_view_உள்ளீடு_new(context, 1);

    எனில் (kernel.get_element_type() != உள்ளீடு.get_element_type()) {
        kernel = std::make_shared<ov::op::v0::Convert>(kernel, உள்ளீடு.get_element_type());
    }

    மாறிலி int32_t * params = context.get_output_op_params();
    மாறிலி முழு s0 = params[0];
    மாறிலி முழு s1 = params[1];
    மாறிலி முழு p0 = params[2];
    மாறிலி முழு p1 = params[3];
    மாறிலி முழு d0 = params[4];
    மாறிலி முழு d1 = params[5];

    // Reshape kernel from [C, 1, KH, KW] to [C, 1, 1, KH, KW] for 2D GroupConvolution
    auto unsqueeze_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto kernel_5d = std::make_shared<ov::op::v0::Unsqueeze>(kernel, unsqueeze_axis);

    ov::Strides strides{static_cast<size_t>(s1), static_cast<size_t>(s0)};
    ov::CoordinateDiff pads_begin{static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::CoordinateDiff pads_end{static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::Strides dilations{static_cast<size_t>(d1), static_cast<size_t>(d0)};

    ov::Output<Node> res = std::make_shared<ov::op::v1::GroupConvolution>(
        உள்ளீடு, kernel_5d, strides, pads_begin, pads_end, dilations, ov::op::PadType::EXPLICIT);

    மாறிலி auto output_type = context.get_output_type();
    எனில் (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    திரும்பு rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_conv_transpose_1d(மாறிலி முனை_Context & context) {
    இலக்கம்_inputs_check(context, 2, 2);

    ov::Output<Node> kernel = process_view_உள்ளீடு_new(context, 0);
    ov::Output<Node> உள்ளீடு  = process_view_உள்ளீடு_new(context, 1);

    எனில் (kernel.get_element_type() != உள்ளீடு.get_element_type()) {
        kernel = std::make_shared<ov::op::v0::Convert>(kernel, உள்ளீடு.get_element_type());
    }

    மாறிலி int32_t * params = context.get_output_op_params();
    மாறிலி முழு s0 = params[0];
    மாறிலி முழு p0 = params[1];
    மாறிலி முழு d0 = params[2];

    மாறிலி auto kernel_shape = context.get_உள்ளீடு_shape(0).to_shape();  // [1, Cin, Cout, K]
    மாறிலி int64_t Cin = kernel_shape[1];
    மாறிலி int64_t Cout = kernel_shape[2];
    மாறிலி int64_t K = kernel_shape[3];

    மாறிலி auto உள்ளீடு_shape = context.get_உள்ளீடு_shape(1).to_shape();   // [1, N, Cin, L]
    மாறிலி int64_t N = உள்ளீடு_shape[0] * உள்ளீடு_shape[1];
    மாறிலி int64_t L = உள்ளீடு_shape[3];

    auto kernel_3d = std::make_shared<ov::op::v1::Reshape>(
        kernel, ov::op::v0::Constant::create(ov::element::i64, {3}, {Cin, Cout, K}), பொய்);
    auto உள்ளீடு_3d = std::make_shared<ov::op::v1::Reshape>(
        உள்ளீடு, ov::op::v0::Constant::create(ov::element::i64, {3}, {N, Cin, L}), பொய்);

    ov::Strides strides{static_cast<size_t>(s0)};
    ov::CoordinateDiff pads_begin{static_cast<ptrdiff_t>(p0)};
    ov::CoordinateDiff pads_end{static_cast<ptrdiff_t>(p0)};
    ov::Strides dilations{static_cast<size_t>(d0)};

    auto conv_tr = std::make_shared<ov::op::v1::ConvolutionBackpropData>(
        உள்ளீடு_3d, kernel_3d, strides, pads_begin, pads_end, dilations);

    மாறிலி auto out_shape = context.get_output_shape().to_shape();
    auto out_shape_const = ov::op::v0::Constant::create(
        ov::element::i64, {4}, {static_cast<int64_t>(out_shape[0]), static_cast<int64_t>(out_shape[1]),
                                static_cast<int64_t>(out_shape[2]), static_cast<int64_t>(out_shape[3])});
    ov::Output<Node> res = std::make_shared<ov::op::v1::Reshape>(conv_tr, out_shape_const, பொய்);

    மாறிலி auto output_type = context.get_output_type();
    எனில் (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    திரும்பு rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_conv_transpose_2d(மாறிலி முனை_Context & context) {
    இலக்கம்_inputs_check(context, 2, 2);

    ov::Output<Node> kernel = process_view_உள்ளீடு_new(context, 0);
    ov::Output<Node> உள்ளீடு  = process_view_உள்ளீடு_new(context, 1);

    எனில் (kernel.get_element_type() != உள்ளீடு.get_element_type()) {
        kernel = std::make_shared<ov::op::v0::Convert>(kernel, உள்ளீடு.get_element_type());
    }

    மாறிலி int32_t * params = context.get_output_op_params();
    மாறிலி முழு stride = params[0];

    ov::Strides strides{static_cast<size_t>(stride), static_cast<size_t>(stride)};
    ov::CoordinateDiff pads_begin{0, 0};
    ov::CoordinateDiff pads_end{0, 0};
    ov::Strides dilations{1, 1};

    ov::Output<Node> res = std::make_shared<ov::op::v1::ConvolutionBackpropData>(
        உள்ளீடு, kernel, strides, pads_begin, pads_end, dilations);

    மாறிலி auto output_type = context.get_output_type();
    எனில் (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    திரும்பு rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_conv_3d(மாறிலி முனை_Context & context) {
    இலக்கம்_inputs_check(context, 2, 2);

    ov::Output<Node> kernel = process_view_உள்ளீடு_new(context, 0);
    ov::Output<Node> உள்ளீடு  = process_view_உள்ளீடு_new(context, 1);

    எனில் (kernel.get_element_type() != உள்ளீடு.get_element_type()) {
        kernel = std::make_shared<ov::op::v0::Convert>(kernel, உள்ளீடு.get_element_type());
    }

    மாறிலி int32_t * params = context.get_output_op_params();
    மாறிலி முழு s0 = params[0];
    மாறிலி முழு s1 = params[1];
    மாறிலி முழு s2 = params[2];
    மாறிலி முழு p0 = params[3];
    மாறிலி முழு p1 = params[4];
    மாறிலி முழு p2 = params[5];
    மாறிலி முழு d0 = params[6];
    மாறிலி முழு d1 = params[7];
    மாறிலி முழு d2 = params[8];
    மாறிலி முழு c  = params[9];
    மாறிலி முழு n  = params[10];
    மாறிலி முழு oc = params[11];

    மாறிலி auto kshape = context.get_உள்ளீடு_shape(0).to_shape();  // [c*oc, KD, KH, KW]
    மாறிலி int64_t KD = kshape[1];
    மாறிலி int64_t KH = kshape[2];
    மாறிலி int64_t KW = kshape[3];

    மாறிலி auto inshape = context.get_உள்ளீடு_shape(1).to_shape(); // [c*n, ID, IH, IW]
    மாறிலி int64_t ID = inshape[1];
    மாறிலி int64_t IH = inshape[2];
    மாறிலி int64_t IW = inshape[3];

    auto kernel_5d = std::make_shared<ov::op::v1::Reshape>(
        kernel, ov::op::v0::Constant::create(ov::element::i64, {5}, {static_cast<int64_t>(oc), static_cast<int64_t>(c), KD, KH, KW}), பொய்);
    auto உள்ளீடு_5d = std::make_shared<ov::op::v1::Reshape>(
        உள்ளீடு, ov::op::v0::Constant::create(ov::element::i64, {5}, {static_cast<int64_t>(n), static_cast<int64_t>(c), ID, IH, IW}), பொய்);

    ov::Strides strides{static_cast<size_t>(s2), static_cast<size_t>(s1), static_cast<size_t>(s0)};
    ov::CoordinateDiff pads_begin{static_cast<ptrdiff_t>(p2), static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::CoordinateDiff pads_end{static_cast<ptrdiff_t>(p2), static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::Strides dilations{static_cast<size_t>(d2), static_cast<size_t>(d1), static_cast<size_t>(d0)};

    auto conv = std::make_shared<ov::op::v1::Convolution>(
        உள்ளீடு_5d, kernel_5d, strides, pads_begin, pads_end, dilations, ov::op::PadType::EXPLICIT);

    மாறிலி auto out_shape = context.get_output_shape().to_shape(); // [oc*n, OD, OH, OW]
    auto out_shape_const = ov::op::v0::Constant::create(
        ov::element::i64, {4}, {static_cast<int64_t>(out_shape[0]), static_cast<int64_t>(out_shape[1]),
                                static_cast<int64_t>(out_shape[2]), static_cast<int64_t>(out_shape[3])});
    ov::Output<Node> res = std::make_shared<ov::op::v1::Reshape>(conv, out_shape_const, பொய்);

    மாறிலி auto output_type = context.get_output_type();
    எனில் (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    திரும்பு rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
