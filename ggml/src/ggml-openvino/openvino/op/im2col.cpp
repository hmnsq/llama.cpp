#अन्तर्भाव "../node_context.h"
#अन्तर्भाव "../op_table.h"
#अन्तर्भाव "../utils.h"

#अन्तर्भाव <cstddef>
#अन्तर्भाव <memory>
#अन्तर्भाव <openvino/core/shape.hpp>
#अन्तर्भाव <openvino/core/strides.hpp>
#अन्तर्भाव <openvino/op/concat.hpp>
#अन्तर्भाव <openvino/op/constant.hpp>
#अन्तर्भाव <openvino/op/convert.hpp>
#अन्तर्भाव <openvino/op/extractimagepatches.hpp>
#अन्तर्भाव <openvino/op/pad.hpp>
#अन्तर्भाव <openvino/op/reshape.hpp>
#अन्तर्भाव <openvino/op/slice.hpp>
#अन्तर्भाव <openvino/op/transpose.hpp>
#अन्तर्भाव <openvino/op/util/attr_types.hpp>

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {
नामाकाश op {

OutputVector translate_im2col(अचर गाँठो_Context & context) {
    अङ्क_inputs_check(context, 2, 2);
    अचर int32_t * params = context.get_output_op_params();
    int32_t s0 = params[0];
    int32_t s1 = params[1];
    int32_t p0 = params[2];
    int32_t p1 = params[3];
    int32_t d0 = params[4];
    int32_t d1 = params[5];
    तार्किक is_2D = params[6] == 1;
    ov::Output<Node> res;

    ov::Output<Node> image = context.get_आगत(1);
    अचर ov::Shape kernel_shape = context.get_आगत(0).get_shape();

    अचर माप_प्रकार IC = is_2D ? kernel_shape[1] : kernel_shape[2];
    अचर माप_प्रकार KH = is_2D ? kernel_shape[2] : 1;
    अचर माप_प्रकार KW = kernel_shape[3];

    int32_t stride_w = s0;
    int32_t stride_h = is_2D ? s1 : 1;
    int32_t pad_w = p0;
    int32_t pad_h = is_2D ? p1 : 0;
    int32_t dil_w = d0;
    int32_t dil_h = is_2D ? d1 : 1;

    यदि (!is_2D) {
        // GGML input shape: [IW, IC, N, 1]
        // OpenVINO input shape: [1, N, IC, IW]
        // Reshape image to: [N, IC, 1, IW]
        अचर ov::Shape image_shape = image.get_shape();
        अचर माप_प्रकार N = image_shape[1];
        अचर माप_प्रकार IW = image_shape[3];
        स्वतः image_reshape_shape = ov::op::v0::Constant::create(
            ov::element::i64, ov::Shape{4},
            std::vector<int64_t>{static_cast<int64_t>(N), static_cast<int64_t>(IC), 1, static_cast<int64_t>(IW)});
        image = std::make_shared<ov::op::v1::Reshape>(image, image_reshape_shape, असत्य);
    }

    अचर ov::Shape patch_sizes = {KH, KW};
    अचर ov::Strides strides = {static_cast<माप_प्रकार>(stride_h), static_cast<माप_प्रकार>(stride_w)};
    अचर ov::Shape rates = {static_cast<माप_प्रकार>(dil_h), static_cast<माप_प्रकार>(dil_w)};

    स्वतः pads_begin =
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 0, pad_h, pad_w});
    स्वतः pads_end =
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 0, pad_h, pad_w});

    स्वतः pad = std::make_shared<ov::op::v1::Pad>(image, pads_begin, pads_end, ov::op::PadMode::CONSTANT);
    स्वतः patches =
        std::make_shared<ov::op::v3::ExtractImagePatches>(pad, patch_sizes, strides, rates, ov::op::PadType::VALID);

    // [N, KH*KW*IC, OH, OW] → [N, OH, OW, KH*KW*IC]
    स्वतः perm1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 2, 3, 1});
    स्वतः t1 = std::make_shared<ov::op::v1::Transpose>(patches, perm1);

    // [N, OH, OW, KH*KW*IC] → [N, OH, OW, KH*KW, IC]
    अचर ov::Shape out_shape = t1->get_output_shape(0);
    अचर माप_प्रकार N = out_shape[0];
    अचर माप_प्रकार OH = out_shape[1];
    अचर माप_प्रकार OW = out_shape[2];
    स्वतः reshape1_shape = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{5},
        std::vector<int64_t>{static_cast<int64_t>(N), static_cast<int64_t>(OH), static_cast<int64_t>(OW),
                             static_cast<int64_t>(KH * KW), static_cast<int64_t>(IC)});
    स्वतः r1 = std::make_shared<ov::op::v1::Reshape>(t1, reshape1_shape, असत्य);

    // [N, OH, OW, KH*KW, IC] → [N, OH, OW, IC, KH*KW]
    स्वतः perm2 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{5}, std::vector<int64_t>{0, 1, 2, 4, 3});
    स्वतः t2 = std::make_shared<ov::op::v1::Transpose>(r1, perm2);

    // flatten back to [N, OH, OW, IC*KH*KW]
    स्वतः r2_shape = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{4},
        std::vector<int64_t>{static_cast<int64_t>(N), static_cast<int64_t>(OH), static_cast<int64_t>(OW),
                             static_cast<int64_t>(IC * KH * KW)});
    res = std::make_shared<ov::op::v1::Reshape>(t2, r2_shape, असत्य);

    यदि (!is_2D) {
        // [N, 1, OW, IC * KW] -> [1, N, OW, IC * KW]
        स्वतः final_reshape_shape = ov::op::v0::Constant::create(
            ov::element::i64, ov::Shape{4},
            std::vector<int64_t>{1, static_cast<int64_t>(N), static_cast<int64_t>(OW), static_cast<int64_t>(IC * KW)});
        res = std::make_shared<ov::op::v1::Reshape>(res, final_reshape_shape, असत्य);
    }

    स्वतः output_type = context.get_output_type();
    यदि (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    फर्काऊ rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_im2col_3d(अचर गाँठो_Context & context) {
    अङ्क_inputs_check(context, 2, 2);
    अचर int32_t * params = context.get_output_op_params();
    int32_t s0 = params[0];
    int32_t s1 = params[1];
    int32_t s2 = params[2];
    int32_t p0 = params[3];
    int32_t p1 = params[4];
    int32_t p2 = params[5];
    int32_t d0 = params[6];
    int32_t d1 = params[7];
    int32_t d2 = params[8];
    int32_t IC = params[9];

    ov::Output<Node> image = process_view_आगत_new(context, 1);
    अचर ov::Shape kernel_shape = context.get_आगत(0).get_shape();
    अचर ov::Shape image_shape = image.get_shape();
    अचर ov::Shape out_shape = context.get_output_shape().to_shape();

    अचर माप_प्रकार KD = kernel_shape[1];
    अचर माप_प्रकार KH = kernel_shape[2];
    अचर माप_प्रकार KW = kernel_shape[3];

    अचर माप_प्रकार N = image_shape[0] / static_cast<माप_प्रकार>(IC);
    अचर माप_प्रकार ID = image_shape[1];
    अचर माप_प्रकार IH = image_shape[2];
    अचर माप_प्रकार IW = image_shape[3];

    अचर माप_प्रकार OD = (ID + 2 * p2 - d2 * (KD - 1) - 1) / s2 + 1;
    अचर माप_प्रकार OH = (IH + 2 * p1 - d1 * (KH - 1) - 1) / s1 + 1;
    अचर माप_प्रकार OW = (IW + 2 * p0 - d0 * (KW - 1) - 1) / s0 + 1;

    यदि (N == 0 || OD == 0 || OH == 0 || OW == 0) {
        स्वतः output_type = context.get_output_type();
        ov::Output<Node> res = ov::op::v0::Constant::create(
            output_type, ov::Shape{N * OD, OH, OW, static_cast<माप_प्रकार>(IC * KD * KH * KW)}, {});
        फर्काऊ rename_outputs_with_suffix({res}, context.get_name());
    }

    अचर माप_प्रकार IH_pad = IH + 2 * p1;
    अचर माप_प्रकार IW_pad = IW + 2 * p0;

    स्वतः image_5d_shape = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{5},
        std::vector<int64_t>{static_cast<int64_t>(N), static_cast<int64_t>(IC), static_cast<int64_t>(ID),
                             static_cast<int64_t>(IH), static_cast<int64_t>(IW)});
    स्वतः image_5d = std::make_shared<ov::op::v1::Reshape>(image, image_5d_shape, असत्य);

    स्वतः pads_begin = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{5}, std::vector<int64_t>{0, 0, p2, p1, p0});
    स्वतः pads_end = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{5}, std::vector<int64_t>{0, 0, p2, p1, p0});
    स्वतः pad_3d = std::make_shared<ov::op::v1::Pad>(image_5d, pads_begin, pads_end, ov::op::PadMode::CONSTANT);

    अचर ov::Shape patch_sizes = {KH, KW};
    अचर ov::Strides strides = {static_cast<माप_प्रकार>(s1), static_cast<माप_प्रकार>(s0)};
    अचर ov::Shape rates = {static_cast<माप_प्रकार>(d1), static_cast<माप_प्रकार>(d0)};

    ov::OutputVector kd_slices;
    kd_slices.reserve(KD);

    स्वतः perm_nod = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{5}, {0, 2, 1, 3, 4});
    स्वतः reshape_4d_shape = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{4},
        std::vector<int64_t>{static_cast<int64_t>(N * OD), static_cast<int64_t>(IC),
                             static_cast<int64_t>(IH_pad), static_cast<int64_t>(IW_pad)});
    स्वतः perm1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 3, 1});
    स्वतः r1_shape = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{5},
        std::vector<int64_t>{static_cast<int64_t>(N * OD), static_cast<int64_t>(OH), static_cast<int64_t>(OW),
                             static_cast<int64_t>(KH * KW), static_cast<int64_t>(IC)});
    स्वतः perm2 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{5}, {0, 1, 2, 4, 3});
    स्वतः r2_shape = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{6},
        std::vector<int64_t>{static_cast<int64_t>(N * OD), static_cast<int64_t>(OH), static_cast<int64_t>(OW),
                             static_cast<int64_t>(IC), 1, static_cast<int64_t>(KH * KW)});
    स्वतः step_c = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {static_cast<int64_t>(s2)});
    स्वतः axes_c = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});

    लागि (माप_प्रकार ikd = 0; ikd < KD; ++ikd) {
        स्वतः start_c = ov::op::v0::Constant::create(
            ov::element::i64, ov::Shape{1}, {static_cast<int64_t>(ikd * d2)});
        स्वतः stop_c = ov::op::v0::Constant::create(
            ov::element::i64, ov::Shape{1}, {static_cast<int64_t>(ikd * d2 + OD * s2)});
        स्वतः depth_slice = std::make_shared<ov::op::v8::Slice>(pad_3d, start_c, stop_c, step_c, axes_c);
        स्वतः depth_slice_trans = std::make_shared<ov::op::v1::Transpose>(depth_slice, perm_nod);
        स्वतः depth_slice_4d = std::make_shared<ov::op::v1::Reshape>(depth_slice_trans, reshape_4d_shape, असत्य);

        स्वतः patches = std::make_shared<ov::op::v3::ExtractImagePatches>(
            depth_slice_4d, patch_sizes, strides, rates, ov::op::PadType::VALID);
        स्वतः t1 = std::make_shared<ov::op::v1::Transpose>(patches, perm1);
        स्वतः r1 = std::make_shared<ov::op::v1::Reshape>(t1, r1_shape, असत्य);
        स्वतः t2 = std::make_shared<ov::op::v1::Transpose>(r1, perm2);
        स्वतः r2 = std::make_shared<ov::op::v1::Reshape>(t2, r2_shape, असत्य);
        kd_slices.push_back(r2);
    }

    ov::Output<Node> res;
    यदि (KD == 1) {
        res = kd_slices[0];
    } अन्यथा {
        res = std::make_shared<ov::op::v0::Concat>(kd_slices, 4);
    }

    स्वतः final_shape = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{4},
        std::vector<int64_t>{static_cast<int64_t>(N * OD), static_cast<int64_t>(OH), static_cast<int64_t>(OW),
                             static_cast<int64_t>(IC * KD * KH * KW)});
    res = std::make_shared<ov::op::v1::Reshape>(res, final_shape, असत्य);

    स्वतः output_type = context.get_output_type();
    यदि (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    फर्काऊ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
