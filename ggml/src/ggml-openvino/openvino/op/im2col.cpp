#شامل "../node_context.h"
#شامل "../op_table.h"
#شامل "../utils.h"

#شامل <cstddef>
#شامل <memory>
#شامل <openvino/core/shape.hpp>
#شامل <openvino/core/strides.hpp>
#شامل <openvino/op/concat.hpp>
#شامل <openvino/op/constant.hpp>
#شامل <openvino/op/convert.hpp>
#شامل <openvino/op/extractimagepatches.hpp>
#شامل <openvino/op/pad.hpp>
#شامل <openvino/op/reshape.hpp>
#شامل <openvino/op/slice.hpp>
#شامل <openvino/op/transpose.hpp>
#شامل <openvino/op/util/attr_types.hpp>

نامخلا ov {
نامخلا frontend {
نامخلا ggml {
نامخلا op {

Output_سمتیہ translate_im2col(ثابت گرہ_Context & context) {
    عدد_inputs_check(context, 2, 2);
    ثابت int32_t * params = context.get_output_op_params();
    int32_t s0 = params[0];
    int32_t s1 = params[1];
    int32_t p0 = params[2];
    int32_t p1 = params[3];
    int32_t d0 = params[4];
    int32_t d1 = params[5];
    منطقی is_2D = params[6] == 1;
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

    اگر (!is_2D) {
        // GGML input shape: [IW, IC, N, 1]
        // OpenVINO input shape: [1, N, IC, IW]
        // Reshape image to: [N, IC, 1, IW]
        ثابت ov::Shape image_shape = image.get_shape();
        ثابت size_t N = image_shape[1];
        ثابت size_t IW = image_shape[3];
        خودکار image_reshape_shape = ov::op::v0::Constant::create(
            ov::element::i64, ov::Shape{4},
            std::vector<int64_t>{static_cast<int64_t>(N), static_cast<int64_t>(IC), 1, static_cast<int64_t>(IW)});
        image = std::make_shared<ov::op::v1::Reshape>(image, image_reshape_shape, جھوٹ);
    }

    ثابت ov::Shape patch_sizes = {KH, KW};
    ثابت ov::Strides strides = {static_cast<size_t>(stride_h), static_cast<size_t>(stride_w)};
    ثابت ov::Shape rates = {static_cast<size_t>(dil_h), static_cast<size_t>(dil_w)};

    خودکار pads_begin =
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 0, pad_h, pad_w});
    خودکار pads_انتہا =
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 0, pad_h, pad_w});

    خودکار pad = std::make_shared<ov::op::v1::Pad>(image, pads_begin, pads_انتہا, ov::op::Pad_منوال::CONSTANT);
    خودکار patches =
        std::make_shared<ov::op::v3::نکالو_Image_Patches>(pad, patch_sizes, strides, rates, ov::op::PadType::VALID);

    // [N, KH*KW*IC, OH, OW] → [N, OH, OW, KH*KW*IC]
    خودکار perm1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 2, 3, 1});
    خودکار t1 = std::make_shared<ov::op::v1::Transpose>(patches, perm1);

    // [N, OH, OW, KH*KW*IC] → [N, OH, OW, KH*KW, IC]
    ثابت ov::Shape برآمد_shape = t1->get_output_shape(0);
    ثابت size_t N = برآمد_shape[0];
    ثابت size_t OH = برآمد_shape[1];
    ثابت size_t OW = برآمد_shape[2];
    خودکار reshape1_shape = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{5},
        std::vector<int64_t>{static_cast<int64_t>(N), static_cast<int64_t>(OH), static_cast<int64_t>(OW),
                             static_cast<int64_t>(KH * KW), static_cast<int64_t>(IC)});
    خودکار r1 = std::make_shared<ov::op::v1::Reshape>(t1, reshape1_shape, جھوٹ);

    // [N, OH, OW, KH*KW, IC] → [N, OH, OW, IC, KH*KW]
    خودکار perm2 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{5}, std::vector<int64_t>{0, 1, 2, 4, 3});
    خودکار t2 = std::make_shared<ov::op::v1::Transpose>(r1, perm2);

    // flatten back to [N, OH, OW, IC*KH*KW]
    خودکار r2_shape = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{4},
        std::vector<int64_t>{static_cast<int64_t>(N), static_cast<int64_t>(OH), static_cast<int64_t>(OW),
                             static_cast<int64_t>(IC * KH * KW)});
    res = std::make_shared<ov::op::v1::Reshape>(t2, r2_shape, جھوٹ);

    اگر (!is_2D) {
        // [N, 1, OW, IC * KW] -> [1, N, OW, IC * KW]
        خودکار final_reshape_shape = ov::op::v0::Constant::create(
            ov::element::i64, ov::Shape{4},
            std::vector<int64_t>{1, static_cast<int64_t>(N), static_cast<int64_t>(OW), static_cast<int64_t>(IC * KW)});
        res = std::make_shared<ov::op::v1::Reshape>(res, final_reshape_shape, جھوٹ);
    }

    خودکار output_type = context.get_output_type();
    اگر (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    واپس rename_outputs_with_suffix({res}, context.get_name());
}

Output_سمتیہ translate_im2col_3d(ثابت گرہ_Context & context) {
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
    ثابت ov::Shape برآمد_shape = context.get_output_shape().مقصد_shape();

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

    اگر (N == 0 || OD == 0 || OH == 0 || OW == 0) {
        خودکار output_type = context.get_output_type();
        ov::Output<Node> res = ov::op::v0::Constant::create(
            output_type, ov::Shape{N * OD, OH, OW, static_cast<size_t>(IC * KD * KH * KW)}, {});
        واپس rename_outputs_with_suffix({res}, context.get_name());
    }

    ثابت size_t IH_pad = IH + 2 * p1;
    ثابت size_t IW_pad = IW + 2 * p0;

    خودکار image_5d_shape = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{5},
        std::vector<int64_t>{static_cast<int64_t>(N), static_cast<int64_t>(IC), static_cast<int64_t>(ID),
                             static_cast<int64_t>(IH), static_cast<int64_t>(IW)});
    خودکار image_5d = std::make_shared<ov::op::v1::Reshape>(image, image_5d_shape, جھوٹ);

    خودکار pads_begin = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{5}, std::vector<int64_t>{0, 0, p2, p1, p0});
    خودکار pads_انتہا = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{5}, std::vector<int64_t>{0, 0, p2, p1, p0});
    خودکار pad_3d = std::make_shared<ov::op::v1::Pad>(image_5d, pads_begin, pads_انتہا, ov::op::Pad_منوال::CONSTANT);

    ثابت ov::Shape patch_sizes = {KH, KW};
    ثابت ov::Strides strides = {static_cast<size_t>(s1), static_cast<size_t>(s0)};
    ثابت ov::Shape rates = {static_cast<size_t>(d1), static_cast<size_t>(d0)};

    ov::Output_سمتیہ kd_slices;
    kd_slices.reserve(KD);

    خودکار perm_nod = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{5}, {0, 2, 1, 3, 4});
    خودکار reshape_4d_shape = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{4},
        std::vector<int64_t>{static_cast<int64_t>(N * OD), static_cast<int64_t>(IC),
                             static_cast<int64_t>(IH_pad), static_cast<int64_t>(IW_pad)});
    خودکار perm1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 3, 1});
    خودکار r1_shape = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{5},
        std::vector<int64_t>{static_cast<int64_t>(N * OD), static_cast<int64_t>(OH), static_cast<int64_t>(OW),
                             static_cast<int64_t>(KH * KW), static_cast<int64_t>(IC)});
    خودکار perm2 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{5}, {0, 1, 2, 4, 3});
    خودکار r2_shape = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{6},
        std::vector<int64_t>{static_cast<int64_t>(N * OD), static_cast<int64_t>(OH), static_cast<int64_t>(OW),
                             static_cast<int64_t>(IC), 1, static_cast<int64_t>(KH * KW)});
    خودکار قدم_c = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {static_cast<int64_t>(s2)});
    خودکار axes_c = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});

    برائے (size_t ikd = 0; ikd < KD; ++ikd) {
        خودکار آغاز_c = ov::op::v0::Constant::create(
            ov::element::i64, ov::Shape{1}, {static_cast<int64_t>(ikd * d2)});
        خودکار stop_c = ov::op::v0::Constant::create(
            ov::element::i64, ov::Shape{1}, {static_cast<int64_t>(ikd * d2 + OD * s2)});
        خودکار گہرائی_قاش = std::make_shared<ov::op::v8::قاش>(pad_3d, آغاز_c, stop_c, قدم_c, axes_c);
        خودکار گہرائی_قاش_trans = std::make_shared<ov::op::v1::Transpose>(گہرائی_قاش, perm_nod);
        خودکار گہرائی_قاش_4d = std::make_shared<ov::op::v1::Reshape>(گہرائی_قاش_trans, reshape_4d_shape, جھوٹ);

        خودکار patches = std::make_shared<ov::op::v3::نکالو_Image_Patches>(
            گہرائی_قاش_4d, patch_sizes, strides, rates, ov::op::PadType::VALID);
        خودکار t1 = std::make_shared<ov::op::v1::Transpose>(patches, perm1);
        خودکار r1 = std::make_shared<ov::op::v1::Reshape>(t1, r1_shape, جھوٹ);
        خودکار t2 = std::make_shared<ov::op::v1::Transpose>(r1, perm2);
        خودکار r2 = std::make_shared<ov::op::v1::Reshape>(t2, r2_shape, جھوٹ);
        kd_slices.push_back(r2);
    }

    ov::Output<Node> res;
    اگر (KD == 1) {
        res = kd_slices[0];
    } ورنہ {
        res = std::make_shared<ov::op::v0::Concat>(kd_slices, 4);
    }

    خودکار final_shape = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{4},
        std::vector<int64_t>{static_cast<int64_t>(N * OD), static_cast<int64_t>(OH), static_cast<int64_t>(OW),
                             static_cast<int64_t>(IC * KD * KH * KW)});
    res = std::make_shared<ov::op::v1::Reshape>(res, final_shape, جھوٹ);

    خودکار output_type = context.get_output_type();
    اگر (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    واپس rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
