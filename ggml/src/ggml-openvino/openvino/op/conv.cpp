#समावेश "../node_context.h"
#समावेश "../op_table.h"
#समावेश "../utils.h"

#समावेश <openvino/op/constant.hpp>
#समावेश <openvino/op/convert.hpp>
#समावेश <openvino/op/convolution.hpp>
#समावेश <openvino/op/group_conv.hpp>
#समावेश <openvino/op/reshape.hpp>
#समावेश <openvino/op/unsqueeze.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_conv_2d(स्थिर गाँठ_Context & context) {
    अंक_inputs_check(context, 2, 2);

    ov::Output<Node> kernel = process_view_निवेश_new(context, 0);
    ov::Output<Node> निवेश  = process_view_निवेश_new(context, 1);

    अगर (kernel.get_element_type() != निवेश.get_element_type()) {
        kernel = std::make_shared<ov::op::v0::Convert>(kernel, निवेश.get_element_type());
    }

    स्थिर int32_t * params = context.get_output_op_params();
    स्थिर पूर्णांक s0 = params[0];
    स्थिर पूर्णांक s1 = params[1];
    स्थिर पूर्णांक p0 = params[2];
    स्थिर पूर्णांक p1 = params[3];
    स्थिर पूर्णांक d0 = params[4];
    स्थिर पूर्णांक d1 = params[5];

    ov::Strides strides{static_cast<माप_प्रकार>(s1), static_cast<माप_प्रकार>(s0)};
    ov::CoordinateDiff pads_begin{static_cast<सूचक_भेद_प्रकार>(p1), static_cast<सूचक_भेद_प्रकार>(p0)};
    ov::CoordinateDiff pads_end{static_cast<सूचक_भेद_प्रकार>(p1), static_cast<सूचक_भेद_प्रकार>(p0)};
    ov::Strides dilations{static_cast<माप_प्रकार>(d1), static_cast<माप_प्रकार>(d0)};

    ov::Output<Node> res = std::make_shared<ov::op::v1::Convolution>(
        निवेश, kernel, strides, pads_begin, pads_end, dilations, ov::op::PadType::EXPLICIT);

    स्थिर स्वतः output_type = context.get_output_type();
    अगर (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    वापस rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_conv_2d_dw(स्थिर गाँठ_Context & context) {
    अंक_inputs_check(context, 2, 2);

    ov::Output<Node> kernel = process_view_निवेश_new(context, 0);
    ov::Output<Node> निवेश  = process_view_निवेश_new(context, 1);

    अगर (kernel.get_element_type() != निवेश.get_element_type()) {
        kernel = std::make_shared<ov::op::v0::Convert>(kernel, निवेश.get_element_type());
    }

    स्थिर int32_t * params = context.get_output_op_params();
    स्थिर पूर्णांक s0 = params[0];
    स्थिर पूर्णांक s1 = params[1];
    स्थिर पूर्णांक p0 = params[2];
    स्थिर पूर्णांक p1 = params[3];
    स्थिर पूर्णांक d0 = params[4];
    स्थिर पूर्णांक d1 = params[5];

    // Reshape kernel from [C, 1, KH, KW] to [C, 1, 1, KH, KW] for 2D GroupConvolution
    स्वतः unsqueeze_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    स्वतः kernel_5d = std::make_shared<ov::op::v0::Unsqueeze>(kernel, unsqueeze_axis);

    ov::Strides strides{static_cast<माप_प्रकार>(s1), static_cast<माप_प्रकार>(s0)};
    ov::CoordinateDiff pads_begin{static_cast<सूचक_भेद_प्रकार>(p1), static_cast<सूचक_भेद_प्रकार>(p0)};
    ov::CoordinateDiff pads_end{static_cast<सूचक_भेद_प्रकार>(p1), static_cast<सूचक_भेद_प्रकार>(p0)};
    ov::Strides dilations{static_cast<माप_प्रकार>(d1), static_cast<माप_प्रकार>(d0)};

    ov::Output<Node> res = std::make_shared<ov::op::v1::GroupConvolution>(
        निवेश, kernel_5d, strides, pads_begin, pads_end, dilations, ov::op::PadType::EXPLICIT);

    स्थिर स्वतः output_type = context.get_output_type();
    अगर (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    वापस rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_conv_transpose_1d(स्थिर गाँठ_Context & context) {
    अंक_inputs_check(context, 2, 2);

    ov::Output<Node> kernel = process_view_निवेश_new(context, 0);
    ov::Output<Node> निवेश  = process_view_निवेश_new(context, 1);

    अगर (kernel.get_element_type() != निवेश.get_element_type()) {
        kernel = std::make_shared<ov::op::v0::Convert>(kernel, निवेश.get_element_type());
    }

    स्थिर int32_t * params = context.get_output_op_params();
    स्थिर पूर्णांक s0 = params[0];
    स्थिर पूर्णांक p0 = params[1];
    स्थिर पूर्णांक d0 = params[2];

    स्थिर स्वतः kernel_shape = context.get_निवेश_shape(0).to_shape();  // [1, Cin, Cout, K]
    स्थिर int64_t Cin = kernel_shape[1];
    स्थिर int64_t Cout = kernel_shape[2];
    स्थिर int64_t K = kernel_shape[3];

    स्थिर स्वतः निवेश_shape = context.get_निवेश_shape(1).to_shape();   // [1, N, Cin, L]
    स्थिर int64_t N = निवेश_shape[0] * निवेश_shape[1];
    स्थिर int64_t L = निवेश_shape[3];

    स्वतः kernel_3d = std::make_shared<ov::op::v1::Reshape>(
        kernel, ov::op::v0::Constant::create(ov::element::i64, {3}, {Cin, Cout, K}), false);
    स्वतः निवेश_3d = std::make_shared<ov::op::v1::Reshape>(
        निवेश, ov::op::v0::Constant::create(ov::element::i64, {3}, {N, Cin, L}), false);

    ov::Strides strides{static_cast<माप_प्रकार>(s0)};
    ov::CoordinateDiff pads_begin{static_cast<सूचक_भेद_प्रकार>(p0)};
    ov::CoordinateDiff pads_end{static_cast<सूचक_भेद_प्रकार>(p0)};
    ov::Strides dilations{static_cast<माप_प्रकार>(d0)};

    स्वतः conv_tr = std::make_shared<ov::op::v1::Convolution_Backprop_तथ्य>(
        निवेश_3d, kernel_3d, strides, pads_begin, pads_end, dilations);

    स्थिर स्वतः out_shape = context.get_output_shape().to_shape();
    स्वतः out_shape_const = ov::op::v0::Constant::create(
        ov::element::i64, {4}, {static_cast<int64_t>(out_shape[0]), static_cast<int64_t>(out_shape[1]),
                                static_cast<int64_t>(out_shape[2]), static_cast<int64_t>(out_shape[3])});
    ov::Output<Node> res = std::make_shared<ov::op::v1::Reshape>(conv_tr, out_shape_const, false);

    स्थिर स्वतः output_type = context.get_output_type();
    अगर (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    वापस rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_conv_transpose_2d(स्थिर गाँठ_Context & context) {
    अंक_inputs_check(context, 2, 2);

    ov::Output<Node> kernel = process_view_निवेश_new(context, 0);
    ov::Output<Node> निवेश  = process_view_निवेश_new(context, 1);

    अगर (kernel.get_element_type() != निवेश.get_element_type()) {
        kernel = std::make_shared<ov::op::v0::Convert>(kernel, निवेश.get_element_type());
    }

    स्थिर int32_t * params = context.get_output_op_params();
    स्थिर पूर्णांक stride = params[0];

    ov::Strides strides{static_cast<माप_प्रकार>(stride), static_cast<माप_प्रकार>(stride)};
    ov::CoordinateDiff pads_begin{0, 0};
    ov::CoordinateDiff pads_end{0, 0};
    ov::Strides dilations{1, 1};

    ov::Output<Node> res = std::make_shared<ov::op::v1::Convolution_Backprop_तथ्य>(
        निवेश, kernel, strides, pads_begin, pads_end, dilations);

    स्थिर स्वतः output_type = context.get_output_type();
    अगर (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    वापस rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_conv_3d(स्थिर गाँठ_Context & context) {
    अंक_inputs_check(context, 2, 2);

    ov::Output<Node> kernel = process_view_निवेश_new(context, 0);
    ov::Output<Node> निवेश  = process_view_निवेश_new(context, 1);

    अगर (kernel.get_element_type() != निवेश.get_element_type()) {
        kernel = std::make_shared<ov::op::v0::Convert>(kernel, निवेश.get_element_type());
    }

    स्थिर int32_t * params = context.get_output_op_params();
    स्थिर पूर्णांक s0 = params[0];
    स्थिर पूर्णांक s1 = params[1];
    स्थिर पूर्णांक s2 = params[2];
    स्थिर पूर्णांक p0 = params[3];
    स्थिर पूर्णांक p1 = params[4];
    स्थिर पूर्णांक p2 = params[5];
    स्थिर पूर्णांक d0 = params[6];
    स्थिर पूर्णांक d1 = params[7];
    स्थिर पूर्णांक d2 = params[8];
    स्थिर पूर्णांक c  = params[9];
    स्थिर पूर्णांक n  = params[10];
    स्थिर पूर्णांक oc = params[11];

    स्थिर स्वतः kshape = context.get_निवेश_shape(0).to_shape();  // [c*oc, KD, KH, KW]
    स्थिर int64_t KD = kshape[1];
    स्थिर int64_t KH = kshape[2];
    स्थिर int64_t KW = kshape[3];

    स्थिर स्वतः inshape = context.get_निवेश_shape(1).to_shape(); // [c*n, ID, IH, IW]
    स्थिर int64_t ID = inshape[1];
    स्थिर int64_t IH = inshape[2];
    स्थिर int64_t IW = inshape[3];

    स्वतः kernel_5d = std::make_shared<ov::op::v1::Reshape>(
        kernel, ov::op::v0::Constant::create(ov::element::i64, {5}, {static_cast<int64_t>(oc), static_cast<int64_t>(c), KD, KH, KW}), false);
    स्वतः निवेश_5d = std::make_shared<ov::op::v1::Reshape>(
        निवेश, ov::op::v0::Constant::create(ov::element::i64, {5}, {static_cast<int64_t>(n), static_cast<int64_t>(c), ID, IH, IW}), false);

    ov::Strides strides{static_cast<माप_प्रकार>(s2), static_cast<माप_प्रकार>(s1), static_cast<माप_प्रकार>(s0)};
    ov::CoordinateDiff pads_begin{static_cast<सूचक_भेद_प्रकार>(p2), static_cast<सूचक_भेद_प्रकार>(p1), static_cast<सूचक_भेद_प्रकार>(p0)};
    ov::CoordinateDiff pads_end{static_cast<सूचक_भेद_प्रकार>(p2), static_cast<सूचक_भेद_प्रकार>(p1), static_cast<सूचक_भेद_प्रकार>(p0)};
    ov::Strides dilations{static_cast<माप_प्रकार>(d2), static_cast<माप_प्रकार>(d1), static_cast<माप_प्रकार>(d0)};

    स्वतः conv = std::make_shared<ov::op::v1::Convolution>(
        निवेश_5d, kernel_5d, strides, pads_begin, pads_end, dilations, ov::op::PadType::EXPLICIT);

    स्थिर स्वतः out_shape = context.get_output_shape().to_shape(); // [oc*n, OD, OH, OW]
    स्वतः out_shape_const = ov::op::v0::Constant::create(
        ov::element::i64, {4}, {static_cast<int64_t>(out_shape[0]), static_cast<int64_t>(out_shape[1]),
                                static_cast<int64_t>(out_shape[2]), static_cast<int64_t>(out_shape[3])});
    ov::Output<Node> res = std::make_shared<ov::op::v1::Reshape>(conv, out_shape_const, false);

    स्थिर स्वतः output_type = context.get_output_type();
    अगर (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    वापस rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
