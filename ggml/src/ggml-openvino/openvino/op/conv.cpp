#ਅਨ੍ਤਰ੍ਭਾਵ "../node_context.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../op_table.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../utils.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/constant.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/convert.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/convolution.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/group_conv.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/reshape.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/unsqueeze.hpp>

ਨਾਮਾਕਾਸ਼ ov {
ਨਾਮਾਕਾਸ਼ frontend {
ਨਾਮਾਕਾਸ਼ ggml {
ਨਾਮਾਕਾਸ਼ op {

Output_ਸਦਿਸ਼ translate_conv_2d(ਅਚਲ ਗੰਢ_Context & context) {
    ਅੰਕ_inputs_check(context, 2, 2);

    ov::Output<Node> kernel = process_view_ਨਿਵੇਸ਼_new(context, 0);
    ov::Output<Node> ਨਿਵੇਸ਼  = process_view_ਨਿਵੇਸ਼_new(context, 1);

    ਜੇ (kernel.get_element_type() != ਨਿਵੇਸ਼.get_element_type()) {
        kernel = std::make_shared<ov::op::v0::Convert>(kernel, ਨਿਵੇਸ਼.get_element_type());
    }

    ਅਚਲ int32_t * params = context.get_output_op_params();
    ਅਚਲ ਪੂਰਨ s0 = params[0];
    ਅਚਲ ਪੂਰਨ s1 = params[1];
    ਅਚਲ ਪੂਰਨ p0 = params[2];
    ਅਚਲ ਪੂਰਨ p1 = params[3];
    ਅਚਲ ਪੂਰਨ d0 = params[4];
    ਅਚਲ ਪੂਰਨ d1 = params[5];

    ov::Strides strides{static_cast<size_t>(s1), static_cast<size_t>(s0)};
    ov::CoordinateDiff pads_begin{static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::CoordinateDiff pads_ਸਿਰਾ{static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::Strides dilations{static_cast<size_t>(d1), static_cast<size_t>(d0)};

    ov::Output<Node> res = std::make_shared<ov::op::v1::Convolution>(
        ਨਿਵੇਸ਼, kernel, strides, pads_begin, pads_ਸਿਰਾ, dilations, ov::op::PadType::EXPLICIT);

    ਅਚਲ ਸਵੈਚਾਲਿਤ output_type = context.get_output_type();
    ਜੇ (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    ਵਾਪਸ rename_outputs_with_suffix({res}, context.get_name());
}

Output_ਸਦਿਸ਼ translate_conv_2d_dw(ਅਚਲ ਗੰਢ_Context & context) {
    ਅੰਕ_inputs_check(context, 2, 2);

    ov::Output<Node> kernel = process_view_ਨਿਵੇਸ਼_new(context, 0);
    ov::Output<Node> ਨਿਵੇਸ਼  = process_view_ਨਿਵੇਸ਼_new(context, 1);

    ਜੇ (kernel.get_element_type() != ਨਿਵੇਸ਼.get_element_type()) {
        kernel = std::make_shared<ov::op::v0::Convert>(kernel, ਨਿਵੇਸ਼.get_element_type());
    }

    ਅਚਲ int32_t * params = context.get_output_op_params();
    ਅਚਲ ਪੂਰਨ s0 = params[0];
    ਅਚਲ ਪੂਰਨ s1 = params[1];
    ਅਚਲ ਪੂਰਨ p0 = params[2];
    ਅਚਲ ਪੂਰਨ p1 = params[3];
    ਅਚਲ ਪੂਰਨ d0 = params[4];
    ਅਚਲ ਪੂਰਨ d1 = params[5];

    // Reshape kernel from [C, 1, KH, KW] to [C, 1, 1, KH, KW] for 2D GroupConvolution
    ਸਵੈਚਾਲਿਤ unsqueeze_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    ਸਵੈਚਾਲਿਤ kernel_5d = std::make_shared<ov::op::v0::Unsqueeze>(kernel, unsqueeze_axis);

    ov::Strides strides{static_cast<size_t>(s1), static_cast<size_t>(s0)};
    ov::CoordinateDiff pads_begin{static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::CoordinateDiff pads_ਸਿਰਾ{static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::Strides dilations{static_cast<size_t>(d1), static_cast<size_t>(d0)};

    ov::Output<Node> res = std::make_shared<ov::op::v1::ਸਮੂਹ_Convolution>(
        ਨਿਵੇਸ਼, kernel_5d, strides, pads_begin, pads_ਸਿਰਾ, dilations, ov::op::PadType::EXPLICIT);

    ਅਚਲ ਸਵੈਚਾਲਿਤ output_type = context.get_output_type();
    ਜੇ (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    ਵਾਪਸ rename_outputs_with_suffix({res}, context.get_name());
}

Output_ਸਦਿਸ਼ translate_conv_transpose_1d(ਅਚਲ ਗੰਢ_Context & context) {
    ਅੰਕ_inputs_check(context, 2, 2);

    ov::Output<Node> kernel = process_view_ਨਿਵੇਸ਼_new(context, 0);
    ov::Output<Node> ਨਿਵੇਸ਼  = process_view_ਨਿਵੇਸ਼_new(context, 1);

    ਜੇ (kernel.get_element_type() != ਨਿਵੇਸ਼.get_element_type()) {
        kernel = std::make_shared<ov::op::v0::Convert>(kernel, ਨਿਵੇਸ਼.get_element_type());
    }

    ਅਚਲ int32_t * params = context.get_output_op_params();
    ਅਚਲ ਪੂਰਨ s0 = params[0];
    ਅਚਲ ਪੂਰਨ p0 = params[1];
    ਅਚਲ ਪੂਰਨ d0 = params[2];

    ਅਚਲ ਸਵੈਚਾਲਿਤ kernel_shape = context.get_ਨਿਵੇਸ਼_shape(0).ਮੰਜ਼ਿਲ_shape();  // [1, Cin, Cout, K]
    ਅਚਲ int64_t Cin = kernel_shape[1];
    ਅਚਲ int64_t Cout = kernel_shape[2];
    ਅਚਲ int64_t K = kernel_shape[3];

    ਅਚਲ ਸਵੈਚਾਲਿਤ ਨਿਵੇਸ਼_shape = context.get_ਨਿਵੇਸ਼_shape(1).ਮੰਜ਼ਿਲ_shape();   // [1, N, Cin, L]
    ਅਚਲ int64_t N = ਨਿਵੇਸ਼_shape[0] * ਨਿਵੇਸ਼_shape[1];
    ਅਚਲ int64_t L = ਨਿਵੇਸ਼_shape[3];

    ਸਵੈਚਾਲਿਤ kernel_3d = std::make_shared<ov::op::v1::Reshape>(
        kernel, ov::op::v0::Constant::create(ov::element::i64, {3}, {Cin, Cout, K}), ਝੂਠ);
    ਸਵੈਚਾਲਿਤ ਨਿਵੇਸ਼_3d = std::make_shared<ov::op::v1::Reshape>(
        ਨਿਵੇਸ਼, ov::op::v0::Constant::create(ov::element::i64, {3}, {N, Cin, L}), ਝੂਠ);

    ov::Strides strides{static_cast<size_t>(s0)};
    ov::CoordinateDiff pads_begin{static_cast<ptrdiff_t>(p0)};
    ov::CoordinateDiff pads_ਸਿਰਾ{static_cast<ptrdiff_t>(p0)};
    ov::Strides dilations{static_cast<size_t>(d0)};

    ਸਵੈਚਾਲਿਤ conv_tr = std::make_shared<ov::op::v1::Convolution_Backprop_ਅੰਕੜੇ>(
        ਨਿਵੇਸ਼_3d, kernel_3d, strides, pads_begin, pads_ਸਿਰਾ, dilations);

    ਅਚਲ ਸਵੈਚਾਲਿਤ ਬਾਹਰਲਾ_shape = context.get_output_shape().ਮੰਜ਼ਿਲ_shape();
    ਸਵੈਚਾਲਿਤ ਬਾਹਰਲਾ_shape_const = ov::op::v0::Constant::create(
        ov::element::i64, {4}, {static_cast<int64_t>(ਬਾਹਰਲਾ_shape[0]), static_cast<int64_t>(ਬਾਹਰਲਾ_shape[1]),
                                static_cast<int64_t>(ਬਾਹਰਲਾ_shape[2]), static_cast<int64_t>(ਬਾਹਰਲਾ_shape[3])});
    ov::Output<Node> res = std::make_shared<ov::op::v1::Reshape>(conv_tr, ਬਾਹਰਲਾ_shape_const, ਝੂਠ);

    ਅਚਲ ਸਵੈਚਾਲਿਤ output_type = context.get_output_type();
    ਜੇ (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    ਵਾਪਸ rename_outputs_with_suffix({res}, context.get_name());
}

Output_ਸਦਿਸ਼ translate_conv_transpose_2d(ਅਚਲ ਗੰਢ_Context & context) {
    ਅੰਕ_inputs_check(context, 2, 2);

    ov::Output<Node> kernel = process_view_ਨਿਵੇਸ਼_new(context, 0);
    ov::Output<Node> ਨਿਵੇਸ਼  = process_view_ਨਿਵੇਸ਼_new(context, 1);

    ਜੇ (kernel.get_element_type() != ਨਿਵੇਸ਼.get_element_type()) {
        kernel = std::make_shared<ov::op::v0::Convert>(kernel, ਨਿਵੇਸ਼.get_element_type());
    }

    ਅਚਲ int32_t * params = context.get_output_op_params();
    ਅਚਲ ਪੂਰਨ stride = params[0];

    ov::Strides strides{static_cast<size_t>(stride), static_cast<size_t>(stride)};
    ov::CoordinateDiff pads_begin{0, 0};
    ov::CoordinateDiff pads_ਸਿਰਾ{0, 0};
    ov::Strides dilations{1, 1};

    ov::Output<Node> res = std::make_shared<ov::op::v1::Convolution_Backprop_ਅੰਕੜੇ>(
        ਨਿਵੇਸ਼, kernel, strides, pads_begin, pads_ਸਿਰਾ, dilations);

    ਅਚਲ ਸਵੈਚਾਲਿਤ output_type = context.get_output_type();
    ਜੇ (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    ਵਾਪਸ rename_outputs_with_suffix({res}, context.get_name());
}

Output_ਸਦਿਸ਼ translate_conv_3d(ਅਚਲ ਗੰਢ_Context & context) {
    ਅੰਕ_inputs_check(context, 2, 2);

    ov::Output<Node> kernel = process_view_ਨਿਵੇਸ਼_new(context, 0);
    ov::Output<Node> ਨਿਵੇਸ਼  = process_view_ਨਿਵੇਸ਼_new(context, 1);

    ਜੇ (kernel.get_element_type() != ਨਿਵੇਸ਼.get_element_type()) {
        kernel = std::make_shared<ov::op::v0::Convert>(kernel, ਨਿਵੇਸ਼.get_element_type());
    }

    ਅਚਲ int32_t * params = context.get_output_op_params();
    ਅਚਲ ਪੂਰਨ s0 = params[0];
    ਅਚਲ ਪੂਰਨ s1 = params[1];
    ਅਚਲ ਪੂਰਨ s2 = params[2];
    ਅਚਲ ਪੂਰਨ p0 = params[3];
    ਅਚਲ ਪੂਰਨ p1 = params[4];
    ਅਚਲ ਪੂਰਨ p2 = params[5];
    ਅਚਲ ਪੂਰਨ d0 = params[6];
    ਅਚਲ ਪੂਰਨ d1 = params[7];
    ਅਚਲ ਪੂਰਨ d2 = params[8];
    ਅਚਲ ਪੂਰਨ c  = params[9];
    ਅਚਲ ਪੂਰਨ n  = params[10];
    ਅਚਲ ਪੂਰਨ oc = params[11];

    ਅਚਲ ਸਵੈਚਾਲਿਤ kshape = context.get_ਨਿਵੇਸ਼_shape(0).ਮੰਜ਼ਿਲ_shape();  // [c*oc, KD, KH, KW]
    ਅਚਲ int64_t KD = kshape[1];
    ਅਚਲ int64_t KH = kshape[2];
    ਅਚਲ int64_t KW = kshape[3];

    ਅਚਲ ਸਵੈਚਾਲਿਤ inshape = context.get_ਨਿਵੇਸ਼_shape(1).ਮੰਜ਼ਿਲ_shape(); // [c*n, ID, IH, IW]
    ਅਚਲ int64_t ID = inshape[1];
    ਅਚਲ int64_t IH = inshape[2];
    ਅਚਲ int64_t IW = inshape[3];

    ਸਵੈਚਾਲਿਤ kernel_5d = std::make_shared<ov::op::v1::Reshape>(
        kernel, ov::op::v0::Constant::create(ov::element::i64, {5}, {static_cast<int64_t>(oc), static_cast<int64_t>(c), KD, KH, KW}), ਝੂਠ);
    ਸਵੈਚਾਲਿਤ ਨਿਵੇਸ਼_5d = std::make_shared<ov::op::v1::Reshape>(
        ਨਿਵੇਸ਼, ov::op::v0::Constant::create(ov::element::i64, {5}, {static_cast<int64_t>(n), static_cast<int64_t>(c), ID, IH, IW}), ਝੂਠ);

    ov::Strides strides{static_cast<size_t>(s2), static_cast<size_t>(s1), static_cast<size_t>(s0)};
    ov::CoordinateDiff pads_begin{static_cast<ptrdiff_t>(p2), static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::CoordinateDiff pads_ਸਿਰਾ{static_cast<ptrdiff_t>(p2), static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::Strides dilations{static_cast<size_t>(d2), static_cast<size_t>(d1), static_cast<size_t>(d0)};

    ਸਵੈਚਾਲਿਤ conv = std::make_shared<ov::op::v1::Convolution>(
        ਨਿਵੇਸ਼_5d, kernel_5d, strides, pads_begin, pads_ਸਿਰਾ, dilations, ov::op::PadType::EXPLICIT);

    ਅਚਲ ਸਵੈਚਾਲਿਤ ਬਾਹਰਲਾ_shape = context.get_output_shape().ਮੰਜ਼ਿਲ_shape(); // [oc*n, OD, OH, OW]
    ਸਵੈਚਾਲਿਤ ਬਾਹਰਲਾ_shape_const = ov::op::v0::Constant::create(
        ov::element::i64, {4}, {static_cast<int64_t>(ਬਾਹਰਲਾ_shape[0]), static_cast<int64_t>(ਬਾਹਰਲਾ_shape[1]),
                                static_cast<int64_t>(ਬਾਹਰਲਾ_shape[2]), static_cast<int64_t>(ਬਾਹਰਲਾ_shape[3])});
    ov::Output<Node> res = std::make_shared<ov::op::v1::Reshape>(conv, ਬਾਹਰਲਾ_shape_const, ਝੂਠ);

    ਅਚਲ ਸਵੈਚਾਲਿਤ output_type = context.get_output_type();
    ਜੇ (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    ਵਾਪਸ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
