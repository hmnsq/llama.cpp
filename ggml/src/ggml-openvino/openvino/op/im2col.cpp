#ಅನ್ತರ್ಭಾವ "../node_context.h"
#ಅನ್ತರ್ಭಾವ "../op_table.h"
#ಅನ್ತರ್ಭಾವ "../utils.h"

#ಅನ್ತರ್ಭಾವ <cstddef>
#ಅನ್ತರ್ಭಾವ <memory>
#ಅನ್ತರ್ಭಾವ <openvino/core/shape.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/core/strides.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/concat.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/constant.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/convert.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/extractimagepatches.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/pad.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/reshape.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/slice.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/transpose.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/util/attr_types.hpp>

ನಾಮಾಕಾಶ ov {
ನಾಮಾಕಾಶ frontend {
ನಾಮಾಕಾಶ ggml {
ನಾಮಾಕಾಶ op {

Output_ಸದಿಶ translate_im2col(ಸ್ಥಿರಾಂಕ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 2, 2);
    ಸ್ಥಿರಾಂಕ int32_t * params = context.get_output_op_params();
    int32_t s0 = params[0];
    int32_t s1 = params[1];
    int32_t p0 = params[2];
    int32_t p1 = params[3];
    int32_t d0 = params[4];
    int32_t d1 = params[5];
    ತಾರ್ಕಿಕ is_2D = params[6] == 1;
    ov::Output<Node> res;

    ov::Output<Node> image = context.get_ಒಳಹರಿವು(1);
    ಸ್ಥಿರಾಂಕ ov::Shape kernel_shape = context.get_ಒಳಹರಿವು(0).get_shape();

    ಸ್ಥಿರಾಂಕ size_t IC = is_2D ? kernel_shape[1] : kernel_shape[2];
    ಸ್ಥಿರಾಂಕ size_t KH = is_2D ? kernel_shape[2] : 1;
    ಸ್ಥಿರಾಂಕ size_t KW = kernel_shape[3];

    int32_t stride_w = s0;
    int32_t stride_h = is_2D ? s1 : 1;
    int32_t pad_w = p0;
    int32_t pad_h = is_2D ? p1 : 0;
    int32_t dil_w = d0;
    int32_t dil_h = is_2D ? d1 : 1;

    ಆದರೆ (!is_2D) {
        // GGML input shape: [IW, IC, N, 1]
        // OpenVINO input shape: [1, N, IC, IW]
        // Reshape image to: [N, IC, 1, IW]
        ಸ್ಥಿರಾಂಕ ov::Shape image_shape = image.get_shape();
        ಸ್ಥಿರಾಂಕ size_t N = image_shape[1];
        ಸ್ಥಿರಾಂಕ size_t IW = image_shape[3];
        ಸ್ವಯಂಚಾಲಿತ image_reshape_shape = ov::op::v0::Constant::create(
            ov::element::i64, ov::Shape{4},
            std::vector<int64_t>{static_cast<int64_t>(N), static_cast<int64_t>(IC), 1, static_cast<int64_t>(IW)});
        image = std::make_shared<ov::op::v1::Reshape>(image, image_reshape_shape, ಸುಳ್ಳು);
    }

    ಸ್ಥಿರಾಂಕ ov::Shape patch_sizes = {KH, KW};
    ಸ್ಥಿರಾಂಕ ov::Strides strides = {static_cast<size_t>(stride_h), static_cast<size_t>(stride_w)};
    ಸ್ಥಿರಾಂಕ ov::Shape rates = {static_cast<size_t>(dil_h), static_cast<size_t>(dil_w)};

    ಸ್ವಯಂಚಾಲಿತ pads_begin =
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 0, pad_h, pad_w});
    ಸ್ವಯಂಚಾಲಿತ pads_ಕೊನೆ =
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 0, pad_h, pad_w});

    ಸ್ವಯಂಚಾಲಿತ pad = std::make_shared<ov::op::v1::Pad>(image, pads_begin, pads_ಕೊನೆ, ov::op::Pad_ಬಹುಲಕ::CONSTANT);
    ಸ್ವಯಂಚಾಲಿತ patches =
        std::make_shared<ov::op::v3::ಹೊರತೆಗೆ_Image_Patches>(pad, patch_sizes, strides, rates, ov::op::PadType::VALID);

    // [N, KH*KW*IC, OH, OW] → [N, OH, OW, KH*KW*IC]
    ಸ್ವಯಂಚಾಲಿತ perm1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 2, 3, 1});
    ಸ್ವಯಂಚಾಲಿತ t1 = std::make_shared<ov::op::v1::Transpose>(patches, perm1);

    // [N, OH, OW, KH*KW*IC] → [N, OH, OW, KH*KW, IC]
    ಸ್ಥಿರಾಂಕ ov::Shape ಹೊರಹೋಗು_shape = t1->get_output_shape(0);
    ಸ್ಥಿರಾಂಕ size_t N = ಹೊರಹೋಗು_shape[0];
    ಸ್ಥಿರಾಂಕ size_t OH = ಹೊರಹೋಗು_shape[1];
    ಸ್ಥಿರಾಂಕ size_t OW = ಹೊರಹೋಗು_shape[2];
    ಸ್ವಯಂಚಾಲಿತ reshape1_shape = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{5},
        std::vector<int64_t>{static_cast<int64_t>(N), static_cast<int64_t>(OH), static_cast<int64_t>(OW),
                             static_cast<int64_t>(KH * KW), static_cast<int64_t>(IC)});
    ಸ್ವಯಂಚಾಲಿತ r1 = std::make_shared<ov::op::v1::Reshape>(t1, reshape1_shape, ಸುಳ್ಳು);

    // [N, OH, OW, KH*KW, IC] → [N, OH, OW, IC, KH*KW]
    ಸ್ವಯಂಚಾಲಿತ perm2 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{5}, std::vector<int64_t>{0, 1, 2, 4, 3});
    ಸ್ವಯಂಚಾಲಿತ t2 = std::make_shared<ov::op::v1::Transpose>(r1, perm2);

    // flatten back to [N, OH, OW, IC*KH*KW]
    ಸ್ವಯಂಚಾಲಿತ r2_shape = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{4},
        std::vector<int64_t>{static_cast<int64_t>(N), static_cast<int64_t>(OH), static_cast<int64_t>(OW),
                             static_cast<int64_t>(IC * KH * KW)});
    res = std::make_shared<ov::op::v1::Reshape>(t2, r2_shape, ಸುಳ್ಳು);

    ಆದರೆ (!is_2D) {
        // [N, 1, OW, IC * KW] -> [1, N, OW, IC * KW]
        ಸ್ವಯಂಚಾಲಿತ final_reshape_shape = ov::op::v0::Constant::create(
            ov::element::i64, ov::Shape{4},
            std::vector<int64_t>{1, static_cast<int64_t>(N), static_cast<int64_t>(OW), static_cast<int64_t>(IC * KW)});
        res = std::make_shared<ov::op::v1::Reshape>(res, final_reshape_shape, ಸುಳ್ಳು);
    }

    ಸ್ವಯಂಚಾಲಿತ output_type = context.get_output_type();
    ಆದರೆ (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    ಹಿಂತಿರುಗಿ rename_outputs_with_suffix({res}, context.get_name());
}

Output_ಸದಿಶ translate_im2col_3d(ಸ್ಥಿರಾಂಕ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 2, 2);
    ಸ್ಥಿರಾಂಕ int32_t * params = context.get_output_op_params();
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

    ov::Output<Node> image = process_view_ಒಳಹರಿವು_new(context, 1);
    ಸ್ಥಿರಾಂಕ ov::Shape kernel_shape = context.get_ಒಳಹರಿವು(0).get_shape();
    ಸ್ಥಿರಾಂಕ ov::Shape image_shape = image.get_shape();
    ಸ್ಥಿರಾಂಕ ov::Shape ಹೊರಹೋಗು_shape = context.get_output_shape().ಗಮ್ಯ_shape();

    ಸ್ಥಿರಾಂಕ size_t KD = kernel_shape[1];
    ಸ್ಥಿರಾಂಕ size_t KH = kernel_shape[2];
    ಸ್ಥಿರಾಂಕ size_t KW = kernel_shape[3];

    ಸ್ಥಿರಾಂಕ size_t N = image_shape[0] / static_cast<size_t>(IC);
    ಸ್ಥಿರಾಂಕ size_t ID = image_shape[1];
    ಸ್ಥಿರಾಂಕ size_t IH = image_shape[2];
    ಸ್ಥಿರಾಂಕ size_t IW = image_shape[3];

    ಸ್ಥಿರಾಂಕ size_t OD = (ID + 2 * p2 - d2 * (KD - 1) - 1) / s2 + 1;
    ಸ್ಥಿರಾಂಕ size_t OH = (IH + 2 * p1 - d1 * (KH - 1) - 1) / s1 + 1;
    ಸ್ಥಿರಾಂಕ size_t OW = (IW + 2 * p0 - d0 * (KW - 1) - 1) / s0 + 1;

    ಆದರೆ (N == 0 || OD == 0 || OH == 0 || OW == 0) {
        ಸ್ವಯಂಚಾಲಿತ output_type = context.get_output_type();
        ov::Output<Node> res = ov::op::v0::Constant::create(
            output_type, ov::Shape{N * OD, OH, OW, static_cast<size_t>(IC * KD * KH * KW)}, {});
        ಹಿಂತಿರುಗಿ rename_outputs_with_suffix({res}, context.get_name());
    }

    ಸ್ಥಿರಾಂಕ size_t IH_pad = IH + 2 * p1;
    ಸ್ಥಿರಾಂಕ size_t IW_pad = IW + 2 * p0;

    ಸ್ವಯಂಚಾಲಿತ image_5d_shape = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{5},
        std::vector<int64_t>{static_cast<int64_t>(N), static_cast<int64_t>(IC), static_cast<int64_t>(ID),
                             static_cast<int64_t>(IH), static_cast<int64_t>(IW)});
    ಸ್ವಯಂಚಾಲಿತ image_5d = std::make_shared<ov::op::v1::Reshape>(image, image_5d_shape, ಸುಳ್ಳು);

    ಸ್ವಯಂಚಾಲಿತ pads_begin = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{5}, std::vector<int64_t>{0, 0, p2, p1, p0});
    ಸ್ವಯಂಚಾಲಿತ pads_ಕೊನೆ = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{5}, std::vector<int64_t>{0, 0, p2, p1, p0});
    ಸ್ವಯಂಚಾಲಿತ pad_3d = std::make_shared<ov::op::v1::Pad>(image_5d, pads_begin, pads_ಕೊನೆ, ov::op::Pad_ಬಹುಲಕ::CONSTANT);

    ಸ್ಥಿರಾಂಕ ov::Shape patch_sizes = {KH, KW};
    ಸ್ಥಿರಾಂಕ ov::Strides strides = {static_cast<size_t>(s1), static_cast<size_t>(s0)};
    ಸ್ಥಿರಾಂಕ ov::Shape rates = {static_cast<size_t>(d1), static_cast<size_t>(d0)};

    ov::Output_ಸದಿಶ kd_slices;
    kd_slices.reserve(KD);

    ಸ್ವಯಂಚಾಲಿತ perm_nod = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{5}, {0, 2, 1, 3, 4});
    ಸ್ವಯಂಚಾಲಿತ reshape_4d_shape = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{4},
        std::vector<int64_t>{static_cast<int64_t>(N * OD), static_cast<int64_t>(IC),
                             static_cast<int64_t>(IH_pad), static_cast<int64_t>(IW_pad)});
    ಸ್ವಯಂಚಾಲಿತ perm1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 3, 1});
    ಸ್ವಯಂಚಾಲಿತ r1_shape = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{5},
        std::vector<int64_t>{static_cast<int64_t>(N * OD), static_cast<int64_t>(OH), static_cast<int64_t>(OW),
                             static_cast<int64_t>(KH * KW), static_cast<int64_t>(IC)});
    ಸ್ವಯಂಚಾಲಿತ perm2 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{5}, {0, 1, 2, 4, 3});
    ಸ್ವಯಂಚಾಲಿತ r2_shape = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{6},
        std::vector<int64_t>{static_cast<int64_t>(N * OD), static_cast<int64_t>(OH), static_cast<int64_t>(OW),
                             static_cast<int64_t>(IC), 1, static_cast<int64_t>(KH * KW)});
    ಸ್ವಯಂಚಾಲಿತ ಹೆಜ್ಜೆ_c = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {static_cast<int64_t>(s2)});
    ಸ್ವಯಂಚಾಲಿತ axes_c = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});

    ಗಾಗಿ (size_t ikd = 0; ikd < KD; ++ikd) {
        ಸ್ವಯಂಚಾಲಿತ ಆರಂಭ_c = ov::op::v0::Constant::create(
            ov::element::i64, ov::Shape{1}, {static_cast<int64_t>(ikd * d2)});
        ಸ್ವಯಂಚಾಲಿತ stop_c = ov::op::v0::Constant::create(
            ov::element::i64, ov::Shape{1}, {static_cast<int64_t>(ikd * d2 + OD * s2)});
        ಸ್ವಯಂಚಾಲಿತ ಆಳ_ತುಂಡು = std::make_shared<ov::op::v8::ತುಂಡು>(pad_3d, ಆರಂಭ_c, stop_c, ಹೆಜ್ಜೆ_c, axes_c);
        ಸ್ವಯಂಚಾಲಿತ ಆಳ_ತುಂಡು_trans = std::make_shared<ov::op::v1::Transpose>(ಆಳ_ತುಂಡು, perm_nod);
        ಸ್ವಯಂಚಾಲಿತ ಆಳ_ತುಂಡು_4d = std::make_shared<ov::op::v1::Reshape>(ಆಳ_ತುಂಡು_trans, reshape_4d_shape, ಸುಳ್ಳು);

        ಸ್ವಯಂಚಾಲಿತ patches = std::make_shared<ov::op::v3::ಹೊರತೆಗೆ_Image_Patches>(
            ಆಳ_ತುಂಡು_4d, patch_sizes, strides, rates, ov::op::PadType::VALID);
        ಸ್ವಯಂಚಾಲಿತ t1 = std::make_shared<ov::op::v1::Transpose>(patches, perm1);
        ಸ್ವಯಂಚಾಲಿತ r1 = std::make_shared<ov::op::v1::Reshape>(t1, r1_shape, ಸುಳ್ಳು);
        ಸ್ವಯಂಚಾಲಿತ t2 = std::make_shared<ov::op::v1::Transpose>(r1, perm2);
        ಸ್ವಯಂಚಾಲಿತ r2 = std::make_shared<ov::op::v1::Reshape>(t2, r2_shape, ಸುಳ್ಳು);
        kd_slices.push_back(r2);
    }

    ov::Output<Node> res;
    ಆದರೆ (KD == 1) {
        res = kd_slices[0];
    } ಇಲ್ಲದಿದ್ದರೆ {
        res = std::make_shared<ov::op::v0::Concat>(kd_slices, 4);
    }

    ಸ್ವಯಂಚಾಲಿತ final_shape = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{4},
        std::vector<int64_t>{static_cast<int64_t>(N * OD), static_cast<int64_t>(OH), static_cast<int64_t>(OW),
                             static_cast<int64_t>(IC * KD * KH * KW)});
    res = std::make_shared<ov::op::v1::Reshape>(res, final_shape, ಸುಳ್ಳು);

    ಸ್ವಯಂಚಾಲಿತ output_type = context.get_output_type();
    ಆದರೆ (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    ಹಿಂತಿರುಗಿ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
