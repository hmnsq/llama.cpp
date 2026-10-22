#ضمن "../node_context.h"
#ضمن "../op_table.h"
#ضمن "../utils.h"

#ضمن <cstddef>
#ضمن <memory>
#ضمن <openvino/core/shape.hpp>
#ضمن <openvino/core/strides.hpp>
#ضمن <openvino/op/concat.hpp>
#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/convert.hpp>
#ضمن <openvino/op/extractimagepatches.hpp>
#ضمن <openvino/op/pad.hpp>
#ضمن <openvino/op/reshape.hpp>
#ضمن <openvino/op/slice.hpp>
#ضمن <openvino/op/transpose.hpp>
#ضمن <openvino/op/util/attr_types.hpp>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق op {

OutputVector translate_im2col(ثابت گرہ_Context & context) {
    عدد_inputs_check(context, 2, 2);
    ثابت int32_t * params = context.get_output_op_params();
    int32_t s0 = params[0];
    int32_t s1 = params[1];
    int32_t p0 = params[2];
    int32_t p1 = params[3];
    int32_t d0 = params[4];
    int32_t d1 = params[5];
    منطقي is_2D = params[6] == 1;
    ov::Output<Node> res;

    ov::Output<Node> image = context.get_داخلہ(1);
    ثابت ov::Shape kernel_shape = context.get_داخلہ(0).get_shape();

    ثابت size_t IC = is_2D ? kernel_shape[1] : kernel_shape[2];
    ثابت size_t KH = is_2D ? kernel_shape[2] : 1;
    ثابت size_t KW = kernel_shape[3];

    int32_t stride_w = s0;
    int32_t stride_h = is_2D ? s1 : 1;
    int32_t pad_w = p0;
    int32_t pad_h = is_2D ? p1 : 0;
    int32_t dil_w = d0;
    int32_t dil_h = is_2D ? d1 : 1;

    إذا (!is_2D) {
        // GGML input shape: [IW, IC, N, 1]
        // OpenVINO input shape: [1, N, IC, IW]
        // Reshape image to: [N, IC, 1, IW]
        ثابت ov::Shape image_shape = image.get_shape();
        ثابت size_t N = image_shape[1];
        ثابت size_t IW = image_shape[3];
        auto image_reshape_shape = ov::op::v0::Constant::create(
            ov::element::i64, ov::Shape{4},
            std::vector<int64_t>{static_cast<int64_t>(N), static_cast<int64_t>(IC), 1, static_cast<int64_t>(IW)});
        image = std::make_shared<ov::op::v1::Reshape>(image, image_reshape_shape, خطأ);
    }

    ثابت ov::Shape patch_sizes = {KH, KW};
    ثابت ov::Strides strides = {static_cast<size_t>(stride_h), static_cast<size_t>(stride_w)};
    ثابت ov::Shape rates = {static_cast<size_t>(dil_h), static_cast<size_t>(dil_w)};

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
    ثابت ov::Shape out_shape = t1->get_output_shape(0);
    ثابت size_t N = out_shape[0];
    ثابت size_t OH = out_shape[1];
    ثابت size_t OW = out_shape[2];
    auto reshape1_shape = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{5},
        std::vector<int64_t>{static_cast<int64_t>(N), static_cast<int64_t>(OH), static_cast<int64_t>(OW),
                             static_cast<int64_t>(KH * KW), static_cast<int64_t>(IC)});
    auto r1 = std::make_shared<ov::op::v1::Reshape>(t1, reshape1_shape, خطأ);

    // [N, OH, OW, KH*KW, IC] → [N, OH, OW, IC, KH*KW]
    auto perm2 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{5}, std::vector<int64_t>{0, 1, 2, 4, 3});
    auto t2 = std::make_shared<ov::op::v1::Transpose>(r1, perm2);

    // flatten back to [N, OH, OW, IC*KH*KW]
    auto r2_shape = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{4},
        std::vector<int64_t>{static_cast<int64_t>(N), static_cast<int64_t>(OH), static_cast<int64_t>(OW),
                             static_cast<int64_t>(IC * KH * KW)});
    res = std::make_shared<ov::op::v1::Reshape>(t2, r2_shape, خطأ);

    إذا (!is_2D) {
        // [N, 1, OW, IC * KW] -> [1, N, OW, IC * KW]
        auto final_reshape_shape = ov::op::v0::Constant::create(
            ov::element::i64, ov::Shape{4},
            std::vector<int64_t>{1, static_cast<int64_t>(N), static_cast<int64_t>(OW), static_cast<int64_t>(IC * KW)});
        res = std::make_shared<ov::op::v1::Reshape>(res, final_reshape_shape, خطأ);
    }

    auto output_type = context.get_output_type();
    إذا (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_im2col_3d(ثابت گرہ_Context & context) {
    عدد_inputs_check(context, 2, 2);
    ثابت int32_t * params = context.get_output_op_params();
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

    ov::Output<Node> image = process_view_داخلہ_new(context, 1);
    ثابت ov::Shape kernel_shape = context.get_داخلہ(0).get_shape();
    ثابت ov::Shape image_shape = image.get_shape();
    ثابت ov::Shape out_shape = context.get_output_shape().to_shape();

    ثابت size_t KD = kernel_shape[1];
    ثابت size_t KH = kernel_shape[2];
    ثابت size_t KW = kernel_shape[3];

    ثابت size_t N = image_shape[0] / static_cast<size_t>(IC);
    ثابت size_t ID = image_shape[1];
    ثابت size_t IH = image_shape[2];
    ثابت size_t IW = image_shape[3];

    ثابت size_t OD = (ID + 2 * p2 - d2 * (KD - 1) - 1) / s2 + 1;
    ثابت size_t OH = (IH + 2 * p1 - d1 * (KH - 1) - 1) / s1 + 1;
    ثابت size_t OW = (IW + 2 * p0 - d0 * (KW - 1) - 1) / s0 + 1;

    إذا (N == 0 || OD == 0 || OH == 0 || OW == 0) {
        auto output_type = context.get_output_type();
        ov::Output<Node> res = ov::op::v0::Constant::create(
            output_type, ov::Shape{N * OD, OH, OW, static_cast<size_t>(IC * KD * KH * KW)}, {});
        أرجع rename_outputs_with_suffix({res}, context.get_name());
    }

    ثابت size_t IH_pad = IH + 2 * p1;
    ثابت size_t IW_pad = IW + 2 * p0;

    auto image_5d_shape = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{5},
        std::vector<int64_t>{static_cast<int64_t>(N), static_cast<int64_t>(IC), static_cast<int64_t>(ID),
                             static_cast<int64_t>(IH), static_cast<int64_t>(IW)});
    auto image_5d = std::make_shared<ov::op::v1::Reshape>(image, image_5d_shape, خطأ);

    auto pads_begin = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{5}, std::vector<int64_t>{0, 0, p2, p1, p0});
    auto pads_end = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{5}, std::vector<int64_t>{0, 0, p2, p1, p0});
    auto pad_3d = std::make_shared<ov::op::v1::Pad>(image_5d, pads_begin, pads_end, ov::op::PadMode::CONSTANT);

    ثابت ov::Shape patch_sizes = {KH, KW};
    ثابت ov::Strides strides = {static_cast<size_t>(s1), static_cast<size_t>(s0)};
    ثابت ov::Shape rates = {static_cast<size_t>(d1), static_cast<size_t>(d0)};

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

    لكل (size_t ikd = 0; ikd < KD; ++ikd) {
        auto start_c = ov::op::v0::Constant::create(
            ov::element::i64, ov::Shape{1}, {static_cast<int64_t>(ikd * d2)});
        auto stop_c = ov::op::v0::Constant::create(
            ov::element::i64, ov::Shape{1}, {static_cast<int64_t>(ikd * d2 + OD * s2)});
        auto depth_slice = std::make_shared<ov::op::v8::Slice>(pad_3d, start_c, stop_c, step_c, axes_c);
        auto depth_slice_trans = std::make_shared<ov::op::v1::Transpose>(depth_slice, perm_nod);
        auto depth_slice_4d = std::make_shared<ov::op::v1::Reshape>(depth_slice_trans, reshape_4d_shape, خطأ);

        auto patches = std::make_shared<ov::op::v3::ExtractImagePatches>(
            depth_slice_4d, patch_sizes, strides, rates, ov::op::PadType::VALID);
        auto t1 = std::make_shared<ov::op::v1::Transpose>(patches, perm1);
        auto r1 = std::make_shared<ov::op::v1::Reshape>(t1, r1_shape, خطأ);
        auto t2 = std::make_shared<ov::op::v1::Transpose>(r1, perm2);
        auto r2 = std::make_shared<ov::op::v1::Reshape>(t2, r2_shape, خطأ);
        kd_slices.push_back(r2);
    }

    ov::Output<Node> res;
    إذا (KD == 1) {
        res = kd_slices[0];
    } وإلا {
        res = std::make_shared<ov::op::v0::Concat>(kd_slices, 4);
    }

    auto final_shape = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{4},
        std::vector<int64_t>{static_cast<int64_t>(N * OD), static_cast<int64_t>(OH), static_cast<int64_t>(OW),
                             static_cast<int64_t>(IC * KD * KH * KW)});
    res = std::make_shared<ov::op::v1::Reshape>(res, final_shape, خطأ);

    auto output_type = context.get_output_type();
    إذا (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
