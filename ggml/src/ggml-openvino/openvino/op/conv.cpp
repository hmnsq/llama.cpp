#ಅನ್ತರ್ಭಾವ "../node_context.h"
#ಅನ್ತರ್ಭಾವ "../op_table.h"
#ಅನ್ತರ್ಭಾವ "../utils.h"

#ಅನ್ತರ್ಭಾವ <openvino/op/constant.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/convert.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/convolution.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/group_conv.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/reshape.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/unsqueeze.hpp>

ನಾಮಾಕಾಶ ov {
ನಾಮಾಕಾಶ frontend {
ನಾಮಾಕಾಶ ggml {
ನಾಮಾಕಾಶ op {

Output_ಸದಿಶ translate_conv_2d(ಸ್ಥಿರಾಂಕ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 2, 2);

    ov::Output<Node> kernel = process_view_ಒಳಹರಿವು_new(context, 0);
    ov::Output<Node> ಒಳಹರಿವು  = process_view_ಒಳಹರಿವು_new(context, 1);

    ಆದರೆ (kernel.get_element_type() != ಒಳಹರಿವು.get_element_type()) {
        kernel = std::make_shared<ov::op::v0::Convert>(kernel, ಒಳಹರಿವು.get_element_type());
    }

    ಸ್ಥಿರಾಂಕ int32_t * params = context.get_output_op_params();
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s0 = params[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s1 = params[1];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ p0 = params[2];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ p1 = params[3];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ d0 = params[4];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ d1 = params[5];

    ov::Strides strides{static_cast<size_t>(s1), static_cast<size_t>(s0)};
    ov::CoordinateDiff pads_begin{static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::CoordinateDiff pads_ಕೊನೆ{static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::Strides dilations{static_cast<size_t>(d1), static_cast<size_t>(d0)};

    ov::Output<Node> res = std::make_shared<ov::op::v1::Convolution>(
        ಒಳಹರಿವು, kernel, strides, pads_begin, pads_ಕೊನೆ, dilations, ov::op::PadType::EXPLICIT);

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ output_type = context.get_output_type();
    ಆದರೆ (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    ಹಿಂತಿರುಗಿ rename_outputs_with_suffix({res}, context.get_name());
}

Output_ಸದಿಶ translate_conv_2d_dw(ಸ್ಥಿರಾಂಕ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 2, 2);

    ov::Output<Node> kernel = process_view_ಒಳಹರಿವು_new(context, 0);
    ov::Output<Node> ಒಳಹರಿವು  = process_view_ಒಳಹರಿವು_new(context, 1);

    ಆದರೆ (kernel.get_element_type() != ಒಳಹರಿವು.get_element_type()) {
        kernel = std::make_shared<ov::op::v0::Convert>(kernel, ಒಳಹರಿವು.get_element_type());
    }

    ಸ್ಥಿರಾಂಕ int32_t * params = context.get_output_op_params();
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s0 = params[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s1 = params[1];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ p0 = params[2];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ p1 = params[3];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ d0 = params[4];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ d1 = params[5];

    // Reshape kernel from [C, 1, KH, KW] to [C, 1, 1, KH, KW] for 2D GroupConvolution
    ಸ್ವಯಂಚಾಲಿತ unsqueeze_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    ಸ್ವಯಂಚಾಲಿತ kernel_5d = std::make_shared<ov::op::v0::Unsqueeze>(kernel, unsqueeze_axis);

    ov::Strides strides{static_cast<size_t>(s1), static_cast<size_t>(s0)};
    ov::CoordinateDiff pads_begin{static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::CoordinateDiff pads_ಕೊನೆ{static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::Strides dilations{static_cast<size_t>(d1), static_cast<size_t>(d0)};

    ov::Output<Node> res = std::make_shared<ov::op::v1::ಗುಂಪು_Convolution>(
        ಒಳಹರಿವು, kernel_5d, strides, pads_begin, pads_ಕೊನೆ, dilations, ov::op::PadType::EXPLICIT);

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ output_type = context.get_output_type();
    ಆದರೆ (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    ಹಿಂತಿರುಗಿ rename_outputs_with_suffix({res}, context.get_name());
}

Output_ಸದಿಶ translate_conv_transpose_1d(ಸ್ಥಿರಾಂಕ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 2, 2);

    ov::Output<Node> kernel = process_view_ಒಳಹರಿವು_new(context, 0);
    ov::Output<Node> ಒಳಹರಿವು  = process_view_ಒಳಹರಿವು_new(context, 1);

    ಆದರೆ (kernel.get_element_type() != ಒಳಹರಿವು.get_element_type()) {
        kernel = std::make_shared<ov::op::v0::Convert>(kernel, ಒಳಹರಿವು.get_element_type());
    }

    ಸ್ಥಿರಾಂಕ int32_t * params = context.get_output_op_params();
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s0 = params[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ p0 = params[1];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ d0 = params[2];

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ kernel_shape = context.get_ಒಳಹರಿವು_shape(0).ಗಮ್ಯ_shape();  // [1, Cin, Cout, K]
    ಸ್ಥಿರಾಂಕ int64_t Cin = kernel_shape[1];
    ಸ್ಥಿರಾಂಕ int64_t Cout = kernel_shape[2];
    ಸ್ಥಿರಾಂಕ int64_t K = kernel_shape[3];

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ ಒಳಹರಿವು_shape = context.get_ಒಳಹರಿವು_shape(1).ಗಮ್ಯ_shape();   // [1, N, Cin, L]
    ಸ್ಥಿರಾಂಕ int64_t N = ಒಳಹರಿವು_shape[0] * ಒಳಹರಿವು_shape[1];
    ಸ್ಥಿರಾಂಕ int64_t L = ಒಳಹರಿವು_shape[3];

    ಸ್ವಯಂಚಾಲಿತ kernel_3d = std::make_shared<ov::op::v1::Reshape>(
        kernel, ov::op::v0::Constant::create(ov::element::i64, {3}, {Cin, Cout, K}), ಸುಳ್ಳು);
    ಸ್ವಯಂಚಾಲಿತ ಒಳಹರಿವು_3d = std::make_shared<ov::op::v1::Reshape>(
        ಒಳಹರಿವು, ov::op::v0::Constant::create(ov::element::i64, {3}, {N, Cin, L}), ಸುಳ್ಳು);

    ov::Strides strides{static_cast<size_t>(s0)};
    ov::CoordinateDiff pads_begin{static_cast<ptrdiff_t>(p0)};
    ov::CoordinateDiff pads_ಕೊನೆ{static_cast<ptrdiff_t>(p0)};
    ov::Strides dilations{static_cast<size_t>(d0)};

    ಸ್ವಯಂಚಾಲಿತ conv_tr = std::make_shared<ov::op::v1::Convolution_Backprop_ದತ್ತಾಂಶ>(
        ಒಳಹರಿವು_3d, kernel_3d, strides, pads_begin, pads_ಕೊನೆ, dilations);

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ ಹೊರಹೋಗು_shape = context.get_output_shape().ಗಮ್ಯ_shape();
    ಸ್ವಯಂಚಾಲಿತ ಹೊರಹೋಗು_shape_const = ov::op::v0::Constant::create(
        ov::element::i64, {4}, {static_cast<int64_t>(ಹೊರಹೋಗು_shape[0]), static_cast<int64_t>(ಹೊರಹೋಗು_shape[1]),
                                static_cast<int64_t>(ಹೊರಹೋಗು_shape[2]), static_cast<int64_t>(ಹೊರಹೋಗು_shape[3])});
    ov::Output<Node> res = std::make_shared<ov::op::v1::Reshape>(conv_tr, ಹೊರಹೋಗು_shape_const, ಸುಳ್ಳು);

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ output_type = context.get_output_type();
    ಆದರೆ (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    ಹಿಂತಿರುಗಿ rename_outputs_with_suffix({res}, context.get_name());
}

Output_ಸದಿಶ translate_conv_transpose_2d(ಸ್ಥಿರಾಂಕ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 2, 2);

    ov::Output<Node> kernel = process_view_ಒಳಹರಿವು_new(context, 0);
    ov::Output<Node> ಒಳಹರಿವು  = process_view_ಒಳಹರಿವು_new(context, 1);

    ಆದರೆ (kernel.get_element_type() != ಒಳಹರಿವು.get_element_type()) {
        kernel = std::make_shared<ov::op::v0::Convert>(kernel, ಒಳಹರಿವು.get_element_type());
    }

    ಸ್ಥಿರಾಂಕ int32_t * params = context.get_output_op_params();
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride = params[0];

    ov::Strides strides{static_cast<size_t>(stride), static_cast<size_t>(stride)};
    ov::CoordinateDiff pads_begin{0, 0};
    ov::CoordinateDiff pads_ಕೊನೆ{0, 0};
    ov::Strides dilations{1, 1};

    ov::Output<Node> res = std::make_shared<ov::op::v1::Convolution_Backprop_ದತ್ತಾಂಶ>(
        ಒಳಹರಿವು, kernel, strides, pads_begin, pads_ಕೊನೆ, dilations);

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ output_type = context.get_output_type();
    ಆದರೆ (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    ಹಿಂತಿರುಗಿ rename_outputs_with_suffix({res}, context.get_name());
}

Output_ಸದಿಶ translate_conv_3d(ಸ್ಥಿರಾಂಕ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 2, 2);

    ov::Output<Node> kernel = process_view_ಒಳಹರಿವು_new(context, 0);
    ov::Output<Node> ಒಳಹರಿವು  = process_view_ಒಳಹರಿವು_new(context, 1);

    ಆದರೆ (kernel.get_element_type() != ಒಳಹರಿವು.get_element_type()) {
        kernel = std::make_shared<ov::op::v0::Convert>(kernel, ಒಳಹರಿವು.get_element_type());
    }

    ಸ್ಥಿರಾಂಕ int32_t * params = context.get_output_op_params();
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s0 = params[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s1 = params[1];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s2 = params[2];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ p0 = params[3];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ p1 = params[4];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ p2 = params[5];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ d0 = params[6];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ d1 = params[7];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ d2 = params[8];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ c  = params[9];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n  = params[10];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ oc = params[11];

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ kshape = context.get_ಒಳಹರಿವು_shape(0).ಗಮ್ಯ_shape();  // [c*oc, KD, KH, KW]
    ಸ್ಥಿರಾಂಕ int64_t KD = kshape[1];
    ಸ್ಥಿರಾಂಕ int64_t KH = kshape[2];
    ಸ್ಥಿರಾಂಕ int64_t KW = kshape[3];

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ inshape = context.get_ಒಳಹರಿವು_shape(1).ಗಮ್ಯ_shape(); // [c*n, ID, IH, IW]
    ಸ್ಥಿರಾಂಕ int64_t ID = inshape[1];
    ಸ್ಥಿರಾಂಕ int64_t IH = inshape[2];
    ಸ್ಥಿರಾಂಕ int64_t IW = inshape[3];

    ಸ್ವಯಂಚಾಲಿತ kernel_5d = std::make_shared<ov::op::v1::Reshape>(
        kernel, ov::op::v0::Constant::create(ov::element::i64, {5}, {static_cast<int64_t>(oc), static_cast<int64_t>(c), KD, KH, KW}), ಸುಳ್ಳು);
    ಸ್ವಯಂಚಾಲಿತ ಒಳಹರಿವು_5d = std::make_shared<ov::op::v1::Reshape>(
        ಒಳಹರಿವು, ov::op::v0::Constant::create(ov::element::i64, {5}, {static_cast<int64_t>(n), static_cast<int64_t>(c), ID, IH, IW}), ಸುಳ್ಳು);

    ov::Strides strides{static_cast<size_t>(s2), static_cast<size_t>(s1), static_cast<size_t>(s0)};
    ov::CoordinateDiff pads_begin{static_cast<ptrdiff_t>(p2), static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::CoordinateDiff pads_ಕೊನೆ{static_cast<ptrdiff_t>(p2), static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::Strides dilations{static_cast<size_t>(d2), static_cast<size_t>(d1), static_cast<size_t>(d0)};

    ಸ್ವಯಂಚಾಲಿತ conv = std::make_shared<ov::op::v1::Convolution>(
        ಒಳಹರಿವು_5d, kernel_5d, strides, pads_begin, pads_ಕೊನೆ, dilations, ov::op::PadType::EXPLICIT);

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ ಹೊರಹೋಗು_shape = context.get_output_shape().ಗಮ್ಯ_shape(); // [oc*n, OD, OH, OW]
    ಸ್ವಯಂಚಾಲಿತ ಹೊರಹೋಗು_shape_const = ov::op::v0::Constant::create(
        ov::element::i64, {4}, {static_cast<int64_t>(ಹೊರಹೋಗು_shape[0]), static_cast<int64_t>(ಹೊರಹೋಗು_shape[1]),
                                static_cast<int64_t>(ಹೊರಹೋಗು_shape[2]), static_cast<int64_t>(ಹೊರಹೋಗು_shape[3])});
    ov::Output<Node> res = std::make_shared<ov::op::v1::Reshape>(conv, ಹೊರಹೋಗು_shape_const, ಸುಳ್ಳು);

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ output_type = context.get_output_type();
    ಆದರೆ (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    ಹಿಂತಿರುಗಿ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
