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

OutputVector translate_im2col(ಧ್ರುವ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 2, 2);
    ಧ್ರುವ int32_t * params = context.get_output_op_params();
    int32_t s0 = params[0];
    int32_t s1 = params[1];
    int32_t p0 = params[2];
    int32_t p1 = params[3];
    int32_t d0 = params[4];
    int32_t d1 = params[5];
    ತಾರ್ಕಿಕ is_2D = params[6] == 1;
    ov::Output<Node> res;

    ov::Output<Node> image = context.get_ಒಳಹರಿವು(1);
    ಧ್ರುವ ov::Shape kernel_shape = context.get_ಒಳಹರಿವು(0).get_shape();

    ಧ್ರುವ size_t IC = is_2D ? kernel_shape[1] : kernel_shape[2];
    ಧ್ರುವ size_t KH = is_2D ? kernel_shape[2] : 1;
    ಧ್ರುವ size_t KW = kernel_shape[3];

    int32_t stride_w = s0;
    int32_t stride_h = is_2D ? s1 : 1;
    int32_t pad_w = p0;
    int32_t pad_h = is_2D ? p1 : 0;
    int32_t dil_w = d0;
    int32_t dil_h = is_2D ? d1 : 1;

    ಯದಿ (!is_2D) {
        // GGML input shape: [IW, IC, N, 1]
        // OpenVINO input shape: [1, N, IC, IW]
        // Reshape image to: [N, IC, 1, IW]
        ಧ್ರುವ ov::Shape image_shape = image.get_shape();
        ಧ್ರುವ size_t N = image_shape[1];
        ಧ್ರುವ size_t IW = image_shape[3];
        auto image_reshape_shape = ov::op::v0::Constant::create(
            ov::element::i64, ov::Shape{4},
            std::vector<int64_t>{static_cast<int64_t>(N), static_cast<int64_t>(IC), 1, static_cast<int64_t>(IW)});
        image = std::make_shared<ov::op::v1::Reshape>(image, image_reshape_shape, ಅಸತ್ಯ);
    }

    ಧ್ರುವ ov::Shape patch_sizes = {KH, KW};
    ಧ್ರುವ ov::Strides strides = {static_cast<size_t>(stride_h), static_cast<size_t>(stride_w)};
    ಧ್ರುವ ov::Shape rates = {static_cast<size_t>(dil_h), static_cast<size_t>(dil_w)};

    auto pads_begin =
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 0, pad_h, pad_w});
    auto pads_end =
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 0, pad_h, pad_w});

    auto pad = std::make_shared<ov::op::v1::Pad>(image, pads_begin, pads_end, ov::op::PadMode::CONSTANT);
    auto patches =
        std::make_shared<ov::op::v3::ExtractImagePatches>(pad, patch_sizes, strides, rates, ov::op::PadType::VALID);

    // [N, KH*KW*IC, OH, OW] → [N, OH, OW, KH*KW*IC]
    auto perm1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 2, 3, 1});
    auto t1 = std::make_shared<ov::op::v1::Transpose>(patches, perm1);

    // [N, OH, OW, KH*KW*IC] → [N, OH, OW, KH*KW, IC]
    ಧ್ರುವ ov::Shape out_shape = t1->get_output_shape(0);
    ಧ್ರುವ size_t N = out_shape[0];
    ಧ್ರುವ size_t OH = out_shape[1];
    ಧ್ರುವ size_t OW = out_shape[2];
    auto reshape1_shape = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{5},
        std::vector<int64_t>{static_cast<int64_t>(N), static_cast<int64_t>(OH), static_cast<int64_t>(OW),
                             static_cast<int64_t>(KH * KW), static_cast<int64_t>(IC)});
    auto r1 = std::make_shared<ov::op::v1::Reshape>(t1, reshape1_shape, ಅಸತ್ಯ);

    // [N, OH, OW, KH*KW, IC] → [N, OH, OW, IC, KH*KW]
    auto perm2 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{5}, std::vector<int64_t>{0, 1, 2, 4, 3});
    auto t2 = std::make_shared<ov::op::v1::Transpose>(r1, perm2);

    // flatten back to [N, OH, OW, IC*KH*KW]
    auto r2_shape = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{4},
        std::vector<int64_t>{static_cast<int64_t>(N), static_cast<int64_t>(OH), static_cast<int64_t>(OW),
                             static_cast<int64_t>(IC * KH * KW)});
    res = std::make_shared<ov::op::v1::Reshape>(t2, r2_shape, ಅಸತ್ಯ);

    ಯದಿ (!is_2D) {
        // [N, 1, OW, IC * KW] -> [1, N, OW, IC * KW]
        auto final_reshape_shape = ov::op::v0::Constant::create(
            ov::element::i64, ov::Shape{4},
            std::vector<int64_t>{1, static_cast<int64_t>(N), static_cast<int64_t>(OW), static_cast<int64_t>(IC * KW)});
        res = std::make_shared<ov::op::v1::Reshape>(res, final_reshape_shape, ಅಸತ್ಯ);
    }

    auto output_type = context.get_output_type();
    ಯದಿ (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    ಪ್ರತ್ಯಾಗಮ rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_im2col_3d(ಧ್ರುವ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 2, 2);
    ಧ್ರುವ int32_t * params = context.get_output_op_params();
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
    ಧ್ರುವ ov::Shape kernel_shape = context.get_ಒಳಹರಿವು(0).get_shape();
    ಧ್ರುವ ov::Shape image_shape = image.get_shape();
    ಧ್ರುವ ov::Shape out_shape = context.get_output_shape().to_shape();

    ಧ್ರುವ size_t KD = kernel_shape[1];
    ಧ್ರುವ size_t KH = kernel_shape[2];
    ಧ್ರುವ size_t KW = kernel_shape[3];

    ಧ್ರುವ size_t N = image_shape[0] / static_cast<size_t>(IC);
    ಧ್ರುವ size_t ID = image_shape[1];
    ಧ್ರುವ size_t IH = image_shape[2];
    ಧ್ರುವ size_t IW = image_shape[3];

    ಧ್ರುವ size_t OD = (ID + 2 * p2 - d2 * (KD - 1) - 1) / s2 + 1;
    ಧ್ರುವ size_t OH = (IH + 2 * p1 - d1 * (KH - 1) - 1) / s1 + 1;
    ಧ್ರುವ size_t OW = (IW + 2 * p0 - d0 * (KW - 1) - 1) / s0 + 1;

    ಯದಿ (N == 0 || OD == 0 || OH == 0 || OW == 0) {
        auto output_type = context.get_output_type();
        ov::Output<Node> res = ov::op::v0::Constant::create(
            output_type, ov::Shape{N * OD, OH, OW, static_cast<size_t>(IC * KD * KH * KW)}, {});
        ಪ್ರತ್ಯಾಗಮ rename_outputs_with_suffix({res}, context.get_name());
    }

    ಧ್ರುವ size_t IH_pad = IH + 2 * p1;
    ಧ್ರುವ size_t IW_pad = IW + 2 * p0;

    auto image_5d_shape = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{5},
        std::vector<int64_t>{static_cast<int64_t>(N), static_cast<int64_t>(IC), static_cast<int64_t>(ID),
                             static_cast<int64_t>(IH), static_cast<int64_t>(IW)});
    auto image_5d = std::make_shared<ov::op::v1::Reshape>(image, image_5d_shape, ಅಸತ್ಯ);

    auto pads_begin = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{5}, std::vector<int64_t>{0, 0, p2, p1, p0});
    auto pads_end = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{5}, std::vector<int64_t>{0, 0, p2, p1, p0});
    auto pad_3d = std::make_shared<ov::op::v1::Pad>(image_5d, pads_begin, pads_end, ov::op::PadMode::CONSTANT);

    ಧ್ರುವ ov::Shape patch_sizes = {KH, KW};
    ಧ್ರುವ ov::Strides strides = {static_cast<size_t>(s1), static_cast<size_t>(s0)};
    ಧ್ರುವ ov::Shape rates = {static_cast<size_t>(d1), static_cast<size_t>(d0)};

    ov::OutputVector kd_slices;
    kd_slices.reserve(KD);

    auto perm_nod = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{5}, {0, 2, 1, 3, 4});
    auto reshape_4d_shape = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{4},
        std::vector<int64_t>{static_cast<int64_t>(N * OD), static_cast<int64_t>(IC),
                             static_cast<int64_t>(IH_pad), static_cast<int64_t>(IW_pad)});
    auto perm1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 3, 1});
    auto r1_shape = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{5},
        std::vector<int64_t>{static_cast<int64_t>(N * OD), static_cast<int64_t>(OH), static_cast<int64_t>(OW),
                             static_cast<int64_t>(KH * KW), static_cast<int64_t>(IC)});
    auto perm2 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{5}, {0, 1, 2, 4, 3});
    auto r2_shape = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{6},
        std::vector<int64_t>{static_cast<int64_t>(N * OD), static_cast<int64_t>(OH), static_cast<int64_t>(OW),
                             static_cast<int64_t>(IC), 1, static_cast<int64_t>(KH * KW)});
    auto step_c = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {static_cast<int64_t>(s2)});
    auto axes_c = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});

    ಪ್ರತಿ (size_t ikd = 0; ikd < KD; ++ikd) {
        auto start_c = ov::op::v0::Constant::create(
            ov::element::i64, ov::Shape{1}, {static_cast<int64_t>(ikd * d2)});
        auto stop_c = ov::op::v0::Constant::create(
            ov::element::i64, ov::Shape{1}, {static_cast<int64_t>(ikd * d2 + OD * s2)});
        auto depth_slice = std::make_shared<ov::op::v8::Slice>(pad_3d, start_c, stop_c, step_c, axes_c);
        auto depth_slice_trans = std::make_shared<ov::op::v1::Transpose>(depth_slice, perm_nod);
        auto depth_slice_4d = std::make_shared<ov::op::v1::Reshape>(depth_slice_trans, reshape_4d_shape, ಅಸತ್ಯ);

        auto patches = std::make_shared<ov::op::v3::ExtractImagePatches>(
            depth_slice_4d, patch_sizes, strides, rates, ov::op::PadType::VALID);
        auto t1 = std::make_shared<ov::op::v1::Transpose>(patches, perm1);
        auto r1 = std::make_shared<ov::op::v1::Reshape>(t1, r1_shape, ಅಸತ್ಯ);
        auto t2 = std::make_shared<ov::op::v1::Transpose>(r1, perm2);
        auto r2 = std::make_shared<ov::op::v1::Reshape>(t2, r2_shape, ಅಸತ್ಯ);
        kd_slices.push_back(r2);
    }

    ov::Output<Node> res;
    ಯದಿ (KD == 1) {
        res = kd_slices[0];
    } ಅನ್ಯಥಾ {
        res = std::make_shared<ov::op::v0::Concat>(kd_slices, 4);
    }

    auto final_shape = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{4},
        std::vector<int64_t>{static_cast<int64_t>(N * OD), static_cast<int64_t>(OH), static_cast<int64_t>(OW),
                             static_cast<int64_t>(IC * KD * KH * KW)});
    res = std::make_shared<ov::op::v1::Reshape>(res, final_shape, ಅಸತ್ಯ);

    auto output_type = context.get_output_type();
    ಯದಿ (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    ಪ್ರತ್ಯಾಗಮ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
