#ਅਨ੍ਤਰ੍ਭਾਵ "../node_context.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../op_table.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../utils.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <cstddef>
#ਅਨ੍ਤਰ੍ਭਾਵ <memory>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/core/shape.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/core/strides.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/concat.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/constant.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/convert.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/extractimagepatches.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/pad.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/reshape.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/slice.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/transpose.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/util/attr_types.hpp>

ਨਾਮਾਕਾਸ਼ ov {
ਨਾਮਾਕਾਸ਼ frontend {
ਨਾਮਾਕਾਸ਼ ggml {
ਨਾਮਾਕਾਸ਼ op {

Output_ਸਦਿਸ਼ translate_im2col(ਅਚਲ ਗੰਢ_Context & context) {
    ਅੰਕ_inputs_check(context, 2, 2);
    ਅਚਲ int32_t * params = context.get_output_op_params();
    int32_t s0 = params[0];
    int32_t s1 = params[1];
    int32_t p0 = params[2];
    int32_t p1 = params[3];
    int32_t d0 = params[4];
    int32_t d1 = params[5];
    ਤਾਰਕਿਕ is_2D = params[6] == 1;
    ov::Output<Node> res;

    ov::Output<Node> image = context.get_ਨਿਵੇਸ਼(1);
    ਅਚਲ ov::Shape kernel_shape = context.get_ਨਿਵੇਸ਼(0).get_shape();

    ਅਚਲ size_t IC = is_2D ? kernel_shape[1] : kernel_shape[2];
    ਅਚਲ size_t KH = is_2D ? kernel_shape[2] : 1;
    ਅਚਲ size_t KW = kernel_shape[3];

    int32_t stride_w = s0;
    int32_t stride_h = is_2D ? s1 : 1;
    int32_t pad_w = p0;
    int32_t pad_h = is_2D ? p1 : 0;
    int32_t dil_w = d0;
    int32_t dil_h = is_2D ? d1 : 1;

    ਜੇ (!is_2D) {
        // GGML input shape: [IW, IC, N, 1]
        // OpenVINO input shape: [1, N, IC, IW]
        // Reshape image to: [N, IC, 1, IW]
        ਅਚਲ ov::Shape image_shape = image.get_shape();
        ਅਚਲ size_t N = image_shape[1];
        ਅਚਲ size_t IW = image_shape[3];
        ਸਵੈਚਾਲਿਤ image_reshape_shape = ov::op::v0::Constant::create(
            ov::element::i64, ov::Shape{4},
            std::vector<int64_t>{static_cast<int64_t>(N), static_cast<int64_t>(IC), 1, static_cast<int64_t>(IW)});
        image = std::make_shared<ov::op::v1::Reshape>(image, image_reshape_shape, ਝੂਠ);
    }

    ਅਚਲ ov::Shape patch_sizes = {KH, KW};
    ਅਚਲ ov::Strides strides = {static_cast<size_t>(stride_h), static_cast<size_t>(stride_w)};
    ਅਚਲ ov::Shape rates = {static_cast<size_t>(dil_h), static_cast<size_t>(dil_w)};

    ਸਵੈਚਾਲਿਤ pads_begin =
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 0, pad_h, pad_w});
    ਸਵੈਚਾਲਿਤ pads_ਸਿਰਾ =
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 0, pad_h, pad_w});

    ਸਵੈਚਾਲਿਤ pad = std::make_shared<ov::op::v1::Pad>(image, pads_begin, pads_ਸਿਰਾ, ov::op::Pad_ਬਹੁਲਕ::CONSTANT);
    ਸਵੈਚਾਲਿਤ patches =
        std::make_shared<ov::op::v3::ਕੱਢੋ_Image_Patches>(pad, patch_sizes, strides, rates, ov::op::PadType::VALID);

    // [N, KH*KW*IC, OH, OW] → [N, OH, OW, KH*KW*IC]
    ਸਵੈਚਾਲਿਤ perm1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 2, 3, 1});
    ਸਵੈਚਾਲਿਤ t1 = std::make_shared<ov::op::v1::Transpose>(patches, perm1);

    // [N, OH, OW, KH*KW*IC] → [N, OH, OW, KH*KW, IC]
    ਅਚਲ ov::Shape ਬਾਹਰਲਾ_shape = t1->get_output_shape(0);
    ਅਚਲ size_t N = ਬਾਹਰਲਾ_shape[0];
    ਅਚਲ size_t OH = ਬਾਹਰਲਾ_shape[1];
    ਅਚਲ size_t OW = ਬਾਹਰਲਾ_shape[2];
    ਸਵੈਚਾਲਿਤ reshape1_shape = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{5},
        std::vector<int64_t>{static_cast<int64_t>(N), static_cast<int64_t>(OH), static_cast<int64_t>(OW),
                             static_cast<int64_t>(KH * KW), static_cast<int64_t>(IC)});
    ਸਵੈਚਾਲਿਤ r1 = std::make_shared<ov::op::v1::Reshape>(t1, reshape1_shape, ਝੂਠ);

    // [N, OH, OW, KH*KW, IC] → [N, OH, OW, IC, KH*KW]
    ਸਵੈਚਾਲਿਤ perm2 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{5}, std::vector<int64_t>{0, 1, 2, 4, 3});
    ਸਵੈਚਾਲਿਤ t2 = std::make_shared<ov::op::v1::Transpose>(r1, perm2);

    // flatten back to [N, OH, OW, IC*KH*KW]
    ਸਵੈਚਾਲਿਤ r2_shape = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{4},
        std::vector<int64_t>{static_cast<int64_t>(N), static_cast<int64_t>(OH), static_cast<int64_t>(OW),
                             static_cast<int64_t>(IC * KH * KW)});
    res = std::make_shared<ov::op::v1::Reshape>(t2, r2_shape, ਝੂਠ);

    ਜੇ (!is_2D) {
        // [N, 1, OW, IC * KW] -> [1, N, OW, IC * KW]
        ਸਵੈਚਾਲਿਤ final_reshape_shape = ov::op::v0::Constant::create(
            ov::element::i64, ov::Shape{4},
            std::vector<int64_t>{1, static_cast<int64_t>(N), static_cast<int64_t>(OW), static_cast<int64_t>(IC * KW)});
        res = std::make_shared<ov::op::v1::Reshape>(res, final_reshape_shape, ਝੂਠ);
    }

    ਸਵੈਚਾਲਿਤ output_type = context.get_output_type();
    ਜੇ (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    ਵਾਪਸ rename_outputs_with_suffix({res}, context.get_name());
}

Output_ਸਦਿਸ਼ translate_im2col_3d(ਅਚਲ ਗੰਢ_Context & context) {
    ਅੰਕ_inputs_check(context, 2, 2);
    ਅਚਲ int32_t * params = context.get_output_op_params();
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

    ov::Output<Node> image = process_view_ਨਿਵੇਸ਼_new(context, 1);
    ਅਚਲ ov::Shape kernel_shape = context.get_ਨਿਵੇਸ਼(0).get_shape();
    ਅਚਲ ov::Shape image_shape = image.get_shape();
    ਅਚਲ ov::Shape ਬਾਹਰਲਾ_shape = context.get_output_shape().ਮੰਜ਼ਿਲ_shape();

    ਅਚਲ size_t KD = kernel_shape[1];
    ਅਚਲ size_t KH = kernel_shape[2];
    ਅਚਲ size_t KW = kernel_shape[3];

    ਅਚਲ size_t N = image_shape[0] / static_cast<size_t>(IC);
    ਅਚਲ size_t ID = image_shape[1];
    ਅਚਲ size_t IH = image_shape[2];
    ਅਚਲ size_t IW = image_shape[3];

    ਅਚਲ size_t OD = (ID + 2 * p2 - d2 * (KD - 1) - 1) / s2 + 1;
    ਅਚਲ size_t OH = (IH + 2 * p1 - d1 * (KH - 1) - 1) / s1 + 1;
    ਅਚਲ size_t OW = (IW + 2 * p0 - d0 * (KW - 1) - 1) / s0 + 1;

    ਜੇ (N == 0 || OD == 0 || OH == 0 || OW == 0) {
        ਸਵੈਚਾਲਿਤ output_type = context.get_output_type();
        ov::Output<Node> res = ov::op::v0::Constant::create(
            output_type, ov::Shape{N * OD, OH, OW, static_cast<size_t>(IC * KD * KH * KW)}, {});
        ਵਾਪਸ rename_outputs_with_suffix({res}, context.get_name());
    }

    ਅਚਲ size_t IH_pad = IH + 2 * p1;
    ਅਚਲ size_t IW_pad = IW + 2 * p0;

    ਸਵੈਚਾਲਿਤ image_5d_shape = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{5},
        std::vector<int64_t>{static_cast<int64_t>(N), static_cast<int64_t>(IC), static_cast<int64_t>(ID),
                             static_cast<int64_t>(IH), static_cast<int64_t>(IW)});
    ਸਵੈਚਾਲਿਤ image_5d = std::make_shared<ov::op::v1::Reshape>(image, image_5d_shape, ਝੂਠ);

    ਸਵੈਚਾਲਿਤ pads_begin = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{5}, std::vector<int64_t>{0, 0, p2, p1, p0});
    ਸਵੈਚਾਲਿਤ pads_ਸਿਰਾ = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{5}, std::vector<int64_t>{0, 0, p2, p1, p0});
    ਸਵੈਚਾਲਿਤ pad_3d = std::make_shared<ov::op::v1::Pad>(image_5d, pads_begin, pads_ਸਿਰਾ, ov::op::Pad_ਬਹੁਲਕ::CONSTANT);

    ਅਚਲ ov::Shape patch_sizes = {KH, KW};
    ਅਚਲ ov::Strides strides = {static_cast<size_t>(s1), static_cast<size_t>(s0)};
    ਅਚਲ ov::Shape rates = {static_cast<size_t>(d1), static_cast<size_t>(d0)};

    ov::Output_ਸਦਿਸ਼ kd_slices;
    kd_slices.reserve(KD);

    ਸਵੈਚਾਲਿਤ perm_nod = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{5}, {0, 2, 1, 3, 4});
    ਸਵੈਚਾਲਿਤ reshape_4d_shape = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{4},
        std::vector<int64_t>{static_cast<int64_t>(N * OD), static_cast<int64_t>(IC),
                             static_cast<int64_t>(IH_pad), static_cast<int64_t>(IW_pad)});
    ਸਵੈਚਾਲਿਤ perm1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 3, 1});
    ਸਵੈਚਾਲਿਤ r1_shape = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{5},
        std::vector<int64_t>{static_cast<int64_t>(N * OD), static_cast<int64_t>(OH), static_cast<int64_t>(OW),
                             static_cast<int64_t>(KH * KW), static_cast<int64_t>(IC)});
    ਸਵੈਚਾਲਿਤ perm2 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{5}, {0, 1, 2, 4, 3});
    ਸਵੈਚਾਲਿਤ r2_shape = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{6},
        std::vector<int64_t>{static_cast<int64_t>(N * OD), static_cast<int64_t>(OH), static_cast<int64_t>(OW),
                             static_cast<int64_t>(IC), 1, static_cast<int64_t>(KH * KW)});
    ਸਵੈਚਾਲਿਤ ਕਦਮ_c = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {static_cast<int64_t>(s2)});
    ਸਵੈਚਾਲਿਤ axes_c = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});

    ਲਈ (size_t ikd = 0; ikd < KD; ++ikd) {
        ਸਵੈਚਾਲਿਤ ਸ਼ੁਰੂ_c = ov::op::v0::Constant::create(
            ov::element::i64, ov::Shape{1}, {static_cast<int64_t>(ikd * d2)});
        ਸਵੈਚਾਲਿਤ stop_c = ov::op::v0::Constant::create(
            ov::element::i64, ov::Shape{1}, {static_cast<int64_t>(ikd * d2 + OD * s2)});
        ਸਵੈਚਾਲਿਤ ਡੂੰਘਾਈ_ਟੁਕੜਾ = std::make_shared<ov::op::v8::ਟੁਕੜਾ>(pad_3d, ਸ਼ੁਰੂ_c, stop_c, ਕਦਮ_c, axes_c);
        ਸਵੈਚਾਲਿਤ ਡੂੰਘਾਈ_ਟੁਕੜਾ_trans = std::make_shared<ov::op::v1::Transpose>(ਡੂੰਘਾਈ_ਟੁਕੜਾ, perm_nod);
        ਸਵੈਚਾਲਿਤ ਡੂੰਘਾਈ_ਟੁਕੜਾ_4d = std::make_shared<ov::op::v1::Reshape>(ਡੂੰਘਾਈ_ਟੁਕੜਾ_trans, reshape_4d_shape, ਝੂਠ);

        ਸਵੈਚਾਲਿਤ patches = std::make_shared<ov::op::v3::ਕੱਢੋ_Image_Patches>(
            ਡੂੰਘਾਈ_ਟੁਕੜਾ_4d, patch_sizes, strides, rates, ov::op::PadType::VALID);
        ਸਵੈਚਾਲਿਤ t1 = std::make_shared<ov::op::v1::Transpose>(patches, perm1);
        ਸਵੈਚਾਲਿਤ r1 = std::make_shared<ov::op::v1::Reshape>(t1, r1_shape, ਝੂਠ);
        ਸਵੈਚਾਲਿਤ t2 = std::make_shared<ov::op::v1::Transpose>(r1, perm2);
        ਸਵੈਚਾਲਿਤ r2 = std::make_shared<ov::op::v1::Reshape>(t2, r2_shape, ਝੂਠ);
        kd_slices.push_back(r2);
    }

    ov::Output<Node> res;
    ਜੇ (KD == 1) {
        res = kd_slices[0];
    } ਨਹੀਂਤਾਂ {
        res = std::make_shared<ov::op::v0::Concat>(kd_slices, 4);
    }

    ਸਵੈਚਾਲਿਤ final_shape = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{4},
        std::vector<int64_t>{static_cast<int64_t>(N * OD), static_cast<int64_t>(OH), static_cast<int64_t>(OW),
                             static_cast<int64_t>(IC * KD * KH * KW)});
    res = std::make_shared<ov::op::v1::Reshape>(res, final_shape, ਝੂਠ);

    ਸਵੈਚਾਲਿਤ output_type = context.get_output_type();
    ਜੇ (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    ਵਾਪਸ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
