#అన్తర్భావ "../node_context.h"
#అన్తర్భావ "../op_table.h"
#అన్తర్భావ "../utils.h"

#అన్తర్భావ <openvino/op/constant.hpp>
#అన్తర్భావ <openvino/op/convert.hpp>
#అన్తర్భావ <openvino/op/convolution.hpp>
#అన్తర్భావ <openvino/op/group_conv.hpp>
#అన్తర్భావ <openvino/op/reshape.hpp>
#అన్తర్భావ <openvino/op/unsqueeze.hpp>

నామాకాశ ov {
నామాకాశ frontend {
నామాకాశ ggml {
నామాకాశ op {

Output_సదిశ translate_conv_2d(స్థిరరాశి కణుపు_Context & context) {
    అంకె_inputs_check(context, 2, 2);

    ov::Output<Node> kernel = process_view_ఆదానం_new(context, 0);
    ov::Output<Node> ఆదానం  = process_view_ఆదానం_new(context, 1);

    ఒకవేళ (kernel.get_element_type() != ఆదానం.get_element_type()) {
        kernel = std::make_shared<ov::op::v0::Convert>(kernel, ఆదానం.get_element_type());
    }

    స్థిరరాశి int32_t * params = context.get_output_op_params();
    స్థిరరాశి పూర్ణాంకం s0 = params[0];
    స్థిరరాశి పూర్ణాంకం s1 = params[1];
    స్థిరరాశి పూర్ణాంకం p0 = params[2];
    స్థిరరాశి పూర్ణాంకం p1 = params[3];
    స్థిరరాశి పూర్ణాంకం d0 = params[4];
    స్థిరరాశి పూర్ణాంకం d1 = params[5];

    ov::Strides strides{static_cast<size_t>(s1), static_cast<size_t>(s0)};
    ov::CoordinateDiff pads_begin{static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::CoordinateDiff pads_చివర{static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::Strides dilations{static_cast<size_t>(d1), static_cast<size_t>(d0)};

    ov::Output<Node> res = std::make_shared<ov::op::v1::Convolution>(
        ఆదానం, kernel, strides, pads_begin, pads_చివర, dilations, ov::op::PadType::EXPLICIT);

    స్థిరరాశి స్వయంచాలక output_type = context.get_output_type();
    ఒకవేళ (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    తిరిగి rename_outputs_with_suffix({res}, context.get_name());
}

Output_సదిశ translate_conv_2d_dw(స్థిరరాశి కణుపు_Context & context) {
    అంకె_inputs_check(context, 2, 2);

    ov::Output<Node> kernel = process_view_ఆదానం_new(context, 0);
    ov::Output<Node> ఆదానం  = process_view_ఆదానం_new(context, 1);

    ఒకవేళ (kernel.get_element_type() != ఆదానం.get_element_type()) {
        kernel = std::make_shared<ov::op::v0::Convert>(kernel, ఆదానం.get_element_type());
    }

    స్థిరరాశి int32_t * params = context.get_output_op_params();
    స్థిరరాశి పూర్ణాంకం s0 = params[0];
    స్థిరరాశి పూర్ణాంకం s1 = params[1];
    స్థిరరాశి పూర్ణాంకం p0 = params[2];
    స్థిరరాశి పూర్ణాంకం p1 = params[3];
    స్థిరరాశి పూర్ణాంకం d0 = params[4];
    స్థిరరాశి పూర్ణాంకం d1 = params[5];

    // Reshape kernel from [C, 1, KH, KW] to [C, 1, 1, KH, KW] for 2D GroupConvolution
    స్వయంచాలక unsqueeze_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    స్వయంచాలక kernel_5d = std::make_shared<ov::op::v0::Unsqueeze>(kernel, unsqueeze_axis);

    ov::Strides strides{static_cast<size_t>(s1), static_cast<size_t>(s0)};
    ov::CoordinateDiff pads_begin{static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::CoordinateDiff pads_చివర{static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::Strides dilations{static_cast<size_t>(d1), static_cast<size_t>(d0)};

    ov::Output<Node> res = std::make_shared<ov::op::v1::సమూహం_Convolution>(
        ఆదానం, kernel_5d, strides, pads_begin, pads_చివర, dilations, ov::op::PadType::EXPLICIT);

    స్థిరరాశి స్వయంచాలక output_type = context.get_output_type();
    ఒకవేళ (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    తిరిగి rename_outputs_with_suffix({res}, context.get_name());
}

Output_సదిశ translate_conv_transpose_1d(స్థిరరాశి కణుపు_Context & context) {
    అంకె_inputs_check(context, 2, 2);

    ov::Output<Node> kernel = process_view_ఆదానం_new(context, 0);
    ov::Output<Node> ఆదానం  = process_view_ఆదానం_new(context, 1);

    ఒకవేళ (kernel.get_element_type() != ఆదానం.get_element_type()) {
        kernel = std::make_shared<ov::op::v0::Convert>(kernel, ఆదానం.get_element_type());
    }

    స్థిరరాశి int32_t * params = context.get_output_op_params();
    స్థిరరాశి పూర్ణాంకం s0 = params[0];
    స్థిరరాశి పూర్ణాంకం p0 = params[1];
    స్థిరరాశి పూర్ణాంకం d0 = params[2];

    స్థిరరాశి స్వయంచాలక kernel_shape = context.get_ఆదానం_shape(0).గమ్యం_shape();  // [1, Cin, Cout, K]
    స్థిరరాశి int64_t Cin = kernel_shape[1];
    స్థిరరాశి int64_t Cout = kernel_shape[2];
    స్థిరరాశి int64_t K = kernel_shape[3];

    స్థిరరాశి స్వయంచాలక ఆదానం_shape = context.get_ఆదానం_shape(1).గమ్యం_shape();   // [1, N, Cin, L]
    స్థిరరాశి int64_t N = ఆదానం_shape[0] * ఆదానం_shape[1];
    స్థిరరాశి int64_t L = ఆదానం_shape[3];

    స్వయంచాలక kernel_3d = std::make_shared<ov::op::v1::Reshape>(
        kernel, ov::op::v0::Constant::create(ov::element::i64, {3}, {Cin, Cout, K}), అబద్ధం);
    స్వయంచాలక ఆదానం_3d = std::make_shared<ov::op::v1::Reshape>(
        ఆదానం, ov::op::v0::Constant::create(ov::element::i64, {3}, {N, Cin, L}), అబద్ధం);

    ov::Strides strides{static_cast<size_t>(s0)};
    ov::CoordinateDiff pads_begin{static_cast<ptrdiff_t>(p0)};
    ov::CoordinateDiff pads_చివర{static_cast<ptrdiff_t>(p0)};
    ov::Strides dilations{static_cast<size_t>(d0)};

    స్వయంచాలక conv_tr = std::make_shared<ov::op::v1::Convolution_Backprop_దత్తాంశం>(
        ఆదానం_3d, kernel_3d, strides, pads_begin, pads_చివర, dilations);

    స్థిరరాశి స్వయంచాలక బహిర్గతం_shape = context.get_output_shape().గమ్యం_shape();
    స్వయంచాలక బహిర్గతం_shape_const = ov::op::v0::Constant::create(
        ov::element::i64, {4}, {static_cast<int64_t>(బహిర్గతం_shape[0]), static_cast<int64_t>(బహిర్గతం_shape[1]),
                                static_cast<int64_t>(బహిర్గతం_shape[2]), static_cast<int64_t>(బహిర్గతం_shape[3])});
    ov::Output<Node> res = std::make_shared<ov::op::v1::Reshape>(conv_tr, బహిర్గతం_shape_const, అబద్ధం);

    స్థిరరాశి స్వయంచాలక output_type = context.get_output_type();
    ఒకవేళ (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    తిరిగి rename_outputs_with_suffix({res}, context.get_name());
}

Output_సదిశ translate_conv_transpose_2d(స్థిరరాశి కణుపు_Context & context) {
    అంకె_inputs_check(context, 2, 2);

    ov::Output<Node> kernel = process_view_ఆదానం_new(context, 0);
    ov::Output<Node> ఆదానం  = process_view_ఆదానం_new(context, 1);

    ఒకవేళ (kernel.get_element_type() != ఆదానం.get_element_type()) {
        kernel = std::make_shared<ov::op::v0::Convert>(kernel, ఆదానం.get_element_type());
    }

    స్థిరరాశి int32_t * params = context.get_output_op_params();
    స్థిరరాశి పూర్ణాంకం stride = params[0];

    ov::Strides strides{static_cast<size_t>(stride), static_cast<size_t>(stride)};
    ov::CoordinateDiff pads_begin{0, 0};
    ov::CoordinateDiff pads_చివర{0, 0};
    ov::Strides dilations{1, 1};

    ov::Output<Node> res = std::make_shared<ov::op::v1::Convolution_Backprop_దత్తాంశం>(
        ఆదానం, kernel, strides, pads_begin, pads_చివర, dilations);

    స్థిరరాశి స్వయంచాలక output_type = context.get_output_type();
    ఒకవేళ (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    తిరిగి rename_outputs_with_suffix({res}, context.get_name());
}

Output_సదిశ translate_conv_3d(స్థిరరాశి కణుపు_Context & context) {
    అంకె_inputs_check(context, 2, 2);

    ov::Output<Node> kernel = process_view_ఆదానం_new(context, 0);
    ov::Output<Node> ఆదానం  = process_view_ఆదానం_new(context, 1);

    ఒకవేళ (kernel.get_element_type() != ఆదానం.get_element_type()) {
        kernel = std::make_shared<ov::op::v0::Convert>(kernel, ఆదానం.get_element_type());
    }

    స్థిరరాశి int32_t * params = context.get_output_op_params();
    స్థిరరాశి పూర్ణాంకం s0 = params[0];
    స్థిరరాశి పూర్ణాంకం s1 = params[1];
    స్థిరరాశి పూర్ణాంకం s2 = params[2];
    స్థిరరాశి పూర్ణాంకం p0 = params[3];
    స్థిరరాశి పూర్ణాంకం p1 = params[4];
    స్థిరరాశి పూర్ణాంకం p2 = params[5];
    స్థిరరాశి పూర్ణాంకం d0 = params[6];
    స్థిరరాశి పూర్ణాంకం d1 = params[7];
    స్థిరరాశి పూర్ణాంకం d2 = params[8];
    స్థిరరాశి పూర్ణాంకం c  = params[9];
    స్థిరరాశి పూర్ణాంకం n  = params[10];
    స్థిరరాశి పూర్ణాంకం oc = params[11];

    స్థిరరాశి స్వయంచాలక kshape = context.get_ఆదానం_shape(0).గమ్యం_shape();  // [c*oc, KD, KH, KW]
    స్థిరరాశి int64_t KD = kshape[1];
    స్థిరరాశి int64_t KH = kshape[2];
    స్థిరరాశి int64_t KW = kshape[3];

    స్థిరరాశి స్వయంచాలక inshape = context.get_ఆదానం_shape(1).గమ్యం_shape(); // [c*n, ID, IH, IW]
    స్థిరరాశి int64_t ID = inshape[1];
    స్థిరరాశి int64_t IH = inshape[2];
    స్థిరరాశి int64_t IW = inshape[3];

    స్వయంచాలక kernel_5d = std::make_shared<ov::op::v1::Reshape>(
        kernel, ov::op::v0::Constant::create(ov::element::i64, {5}, {static_cast<int64_t>(oc), static_cast<int64_t>(c), KD, KH, KW}), అబద్ధం);
    స్వయంచాలక ఆదానం_5d = std::make_shared<ov::op::v1::Reshape>(
        ఆదానం, ov::op::v0::Constant::create(ov::element::i64, {5}, {static_cast<int64_t>(n), static_cast<int64_t>(c), ID, IH, IW}), అబద్ధం);

    ov::Strides strides{static_cast<size_t>(s2), static_cast<size_t>(s1), static_cast<size_t>(s0)};
    ov::CoordinateDiff pads_begin{static_cast<ptrdiff_t>(p2), static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::CoordinateDiff pads_చివర{static_cast<ptrdiff_t>(p2), static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::Strides dilations{static_cast<size_t>(d2), static_cast<size_t>(d1), static_cast<size_t>(d0)};

    స్వయంచాలక conv = std::make_shared<ov::op::v1::Convolution>(
        ఆదానం_5d, kernel_5d, strides, pads_begin, pads_చివర, dilations, ov::op::PadType::EXPLICIT);

    స్థిరరాశి స్వయంచాలక బహిర్గతం_shape = context.get_output_shape().గమ్యం_shape(); // [oc*n, OD, OH, OW]
    స్వయంచాలక బహిర్గతం_shape_const = ov::op::v0::Constant::create(
        ov::element::i64, {4}, {static_cast<int64_t>(బహిర్గతం_shape[0]), static_cast<int64_t>(బహిర్గతం_shape[1]),
                                static_cast<int64_t>(బహిర్గతం_shape[2]), static_cast<int64_t>(బహిర్గతం_shape[3])});
    ov::Output<Node> res = std::make_shared<ov::op::v1::Reshape>(conv, బహిర్గతం_shape_const, అబద్ధం);

    స్థిరరాశి స్వయంచాలక output_type = context.get_output_type();
    ఒకవేళ (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    తిరిగి rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
