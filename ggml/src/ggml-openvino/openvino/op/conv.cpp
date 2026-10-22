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

OutputVector translate_conv_2d(ಧ್ರುವ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 2, 2);

    ov::Output<Node> kernel = process_view_ಒಳಹರಿವು_new(context, 0);
    ov::Output<Node> ಒಳಹರಿವು  = process_view_ಒಳಹರಿವು_new(context, 1);

    ಯದಿ (kernel.get_element_type() != ಒಳಹರಿವು.get_element_type()) {
        kernel = std::make_shared<ov::op::v0::Convert>(kernel, ಒಳಹರಿವು.get_element_type());
    }

    ಧ್ರುವ int32_t * params = context.get_output_op_params();
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s0 = params[0];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s1 = params[1];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ p0 = params[2];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ p1 = params[3];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ d0 = params[4];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ d1 = params[5];

    ov::Strides strides{static_cast<size_t>(s1), static_cast<size_t>(s0)};
    ov::CoordinateDiff pads_begin{static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::CoordinateDiff pads_end{static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::Strides dilations{static_cast<size_t>(d1), static_cast<size_t>(d0)};

    ov::Output<Node> res = std::make_shared<ov::op::v1::Convolution>(
        ಒಳಹರಿವು, kernel, strides, pads_begin, pads_end, dilations, ov::op::PadType::EXPLICIT);

    ಧ್ರುವ auto output_type = context.get_output_type();
    ಯದಿ (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    ಪ್ರತ್ಯಾಗಮ rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_conv_2d_dw(ಧ್ರುವ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 2, 2);

    ov::Output<Node> kernel = process_view_ಒಳಹರಿವು_new(context, 0);
    ov::Output<Node> ಒಳಹರಿವು  = process_view_ಒಳಹರಿವು_new(context, 1);

    ಯದಿ (kernel.get_element_type() != ಒಳಹರಿವು.get_element_type()) {
        kernel = std::make_shared<ov::op::v0::Convert>(kernel, ಒಳಹರಿವು.get_element_type());
    }

    ಧ್ರುವ int32_t * params = context.get_output_op_params();
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s0 = params[0];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s1 = params[1];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ p0 = params[2];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ p1 = params[3];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ d0 = params[4];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ d1 = params[5];

    // Reshape kernel from [C, 1, KH, KW] to [C, 1, 1, KH, KW] for 2D GroupConvolution
    auto unsqueeze_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto kernel_5d = std::make_shared<ov::op::v0::Unsqueeze>(kernel, unsqueeze_axis);

    ov::Strides strides{static_cast<size_t>(s1), static_cast<size_t>(s0)};
    ov::CoordinateDiff pads_begin{static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::CoordinateDiff pads_end{static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::Strides dilations{static_cast<size_t>(d1), static_cast<size_t>(d0)};

    ov::Output<Node> res = std::make_shared<ov::op::v1::GroupConvolution>(
        ಒಳಹರಿವು, kernel_5d, strides, pads_begin, pads_end, dilations, ov::op::PadType::EXPLICIT);

    ಧ್ರುವ auto output_type = context.get_output_type();
    ಯದಿ (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    ಪ್ರತ್ಯಾಗಮ rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_conv_transpose_1d(ಧ್ರುವ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 2, 2);

    ov::Output<Node> kernel = process_view_ಒಳಹರಿವು_new(context, 0);
    ov::Output<Node> ಒಳಹರಿವು  = process_view_ಒಳಹರಿವು_new(context, 1);

    ಯದಿ (kernel.get_element_type() != ಒಳಹರಿವು.get_element_type()) {
        kernel = std::make_shared<ov::op::v0::Convert>(kernel, ಒಳಹರಿವು.get_element_type());
    }

    ಧ್ರುವ int32_t * params = context.get_output_op_params();
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s0 = params[0];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ p0 = params[1];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ d0 = params[2];

    ಧ್ರುವ auto kernel_shape = context.get_ಒಳಹರಿವು_shape(0).to_shape();  // [1, Cin, Cout, K]
    ಧ್ರುವ int64_t Cin = kernel_shape[1];
    ಧ್ರುವ int64_t Cout = kernel_shape[2];
    ಧ್ರುವ int64_t K = kernel_shape[3];

    ಧ್ರುವ auto ಒಳಹರಿವು_shape = context.get_ಒಳಹರಿವು_shape(1).to_shape();   // [1, N, Cin, L]
    ಧ್ರುವ int64_t N = ಒಳಹರಿವು_shape[0] * ಒಳಹರಿವು_shape[1];
    ಧ್ರುವ int64_t L = ಒಳಹರಿವು_shape[3];

    auto kernel_3d = std::make_shared<ov::op::v1::Reshape>(
        kernel, ov::op::v0::Constant::create(ov::element::i64, {3}, {Cin, Cout, K}), ಅಸತ್ಯ);
    auto ಒಳಹರಿವು_3d = std::make_shared<ov::op::v1::Reshape>(
        ಒಳಹರಿವು, ov::op::v0::Constant::create(ov::element::i64, {3}, {N, Cin, L}), ಅಸತ್ಯ);

    ov::Strides strides{static_cast<size_t>(s0)};
    ov::CoordinateDiff pads_begin{static_cast<ptrdiff_t>(p0)};
    ov::CoordinateDiff pads_end{static_cast<ptrdiff_t>(p0)};
    ov::Strides dilations{static_cast<size_t>(d0)};

    auto conv_tr = std::make_shared<ov::op::v1::ConvolutionBackpropData>(
        ಒಳಹರಿವು_3d, kernel_3d, strides, pads_begin, pads_end, dilations);

    ಧ್ರುವ auto out_shape = context.get_output_shape().to_shape();
    auto out_shape_const = ov::op::v0::Constant::create(
        ov::element::i64, {4}, {static_cast<int64_t>(out_shape[0]), static_cast<int64_t>(out_shape[1]),
                                static_cast<int64_t>(out_shape[2]), static_cast<int64_t>(out_shape[3])});
    ov::Output<Node> res = std::make_shared<ov::op::v1::Reshape>(conv_tr, out_shape_const, ಅಸತ್ಯ);

    ಧ್ರುವ auto output_type = context.get_output_type();
    ಯದಿ (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    ಪ್ರತ್ಯಾಗಮ rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_conv_transpose_2d(ಧ್ರುವ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 2, 2);

    ov::Output<Node> kernel = process_view_ಒಳಹರಿವು_new(context, 0);
    ov::Output<Node> ಒಳಹರಿವು  = process_view_ಒಳಹರಿವು_new(context, 1);

    ಯದಿ (kernel.get_element_type() != ಒಳಹರಿವು.get_element_type()) {
        kernel = std::make_shared<ov::op::v0::Convert>(kernel, ಒಳಹರಿವು.get_element_type());
    }

    ಧ್ರುವ int32_t * params = context.get_output_op_params();
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ stride = params[0];

    ov::Strides strides{static_cast<size_t>(stride), static_cast<size_t>(stride)};
    ov::CoordinateDiff pads_begin{0, 0};
    ov::CoordinateDiff pads_end{0, 0};
    ov::Strides dilations{1, 1};

    ov::Output<Node> res = std::make_shared<ov::op::v1::ConvolutionBackpropData>(
        ಒಳಹರಿವು, kernel, strides, pads_begin, pads_end, dilations);

    ಧ್ರುವ auto output_type = context.get_output_type();
    ಯದಿ (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    ಪ್ರತ್ಯಾಗಮ rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_conv_3d(ಧ್ರುವ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 2, 2);

    ov::Output<Node> kernel = process_view_ಒಳಹರಿವು_new(context, 0);
    ov::Output<Node> ಒಳಹರಿವು  = process_view_ಒಳಹರಿವು_new(context, 1);

    ಯದಿ (kernel.get_element_type() != ಒಳಹರಿವು.get_element_type()) {
        kernel = std::make_shared<ov::op::v0::Convert>(kernel, ಒಳಹರಿವು.get_element_type());
    }

    ಧ್ರುವ int32_t * params = context.get_output_op_params();
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s0 = params[0];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s1 = params[1];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s2 = params[2];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ p0 = params[3];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ p1 = params[4];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ p2 = params[5];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ d0 = params[6];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ d1 = params[7];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ d2 = params[8];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ c  = params[9];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n  = params[10];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ oc = params[11];

    ಧ್ರುವ auto kshape = context.get_ಒಳಹರಿವು_shape(0).to_shape();  // [c*oc, KD, KH, KW]
    ಧ್ರುವ int64_t KD = kshape[1];
    ಧ್ರುವ int64_t KH = kshape[2];
    ಧ್ರುವ int64_t KW = kshape[3];

    ಧ್ರುವ auto inshape = context.get_ಒಳಹರಿವು_shape(1).to_shape(); // [c*n, ID, IH, IW]
    ಧ್ರುವ int64_t ID = inshape[1];
    ಧ್ರುವ int64_t IH = inshape[2];
    ಧ್ರುವ int64_t IW = inshape[3];

    auto kernel_5d = std::make_shared<ov::op::v1::Reshape>(
        kernel, ov::op::v0::Constant::create(ov::element::i64, {5}, {static_cast<int64_t>(oc), static_cast<int64_t>(c), KD, KH, KW}), ಅಸತ್ಯ);
    auto ಒಳಹರಿವು_5d = std::make_shared<ov::op::v1::Reshape>(
        ಒಳಹರಿವು, ov::op::v0::Constant::create(ov::element::i64, {5}, {static_cast<int64_t>(n), static_cast<int64_t>(c), ID, IH, IW}), ಅಸತ್ಯ);

    ov::Strides strides{static_cast<size_t>(s2), static_cast<size_t>(s1), static_cast<size_t>(s0)};
    ov::CoordinateDiff pads_begin{static_cast<ptrdiff_t>(p2), static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::CoordinateDiff pads_end{static_cast<ptrdiff_t>(p2), static_cast<ptrdiff_t>(p1), static_cast<ptrdiff_t>(p0)};
    ov::Strides dilations{static_cast<size_t>(d2), static_cast<size_t>(d1), static_cast<size_t>(d0)};

    auto conv = std::make_shared<ov::op::v1::Convolution>(
        ಒಳಹರಿವು_5d, kernel_5d, strides, pads_begin, pads_end, dilations, ov::op::PadType::EXPLICIT);

    ಧ್ರುವ auto out_shape = context.get_output_shape().to_shape(); // [oc*n, OD, OH, OW]
    auto out_shape_const = ov::op::v0::Constant::create(
        ov::element::i64, {4}, {static_cast<int64_t>(out_shape[0]), static_cast<int64_t>(out_shape[1]),
                                static_cast<int64_t>(out_shape[2]), static_cast<int64_t>(out_shape[3])});
    ov::Output<Node> res = std::make_shared<ov::op::v1::Reshape>(conv, out_shape_const, ಅಸತ್ಯ);

    ಧ್ರುವ auto output_type = context.get_output_type();
    ಯದಿ (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    ಪ್ರತ್ಯಾಗಮ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
