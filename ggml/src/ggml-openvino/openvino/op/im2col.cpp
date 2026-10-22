#అన్తర్భావ "../node_context.h"
#అన్తర్భావ "../op_table.h"
#అన్తర్భావ "../utils.h"

#అన్తర్భావ <cstddef>
#అన్తర్భావ <memory>
#అన్తర్భావ <openvino/core/shape.hpp>
#అన్తర్భావ <openvino/core/strides.hpp>
#అన్తర్భావ <openvino/op/concat.hpp>
#అన్తర్భావ <openvino/op/constant.hpp>
#అన్తర్భావ <openvino/op/convert.hpp>
#అన్తర్భావ <openvino/op/extractimagepatches.hpp>
#అన్తర్భావ <openvino/op/pad.hpp>
#అన్తర్భావ <openvino/op/reshape.hpp>
#అన్తర్భావ <openvino/op/slice.hpp>
#అన్తర్భావ <openvino/op/transpose.hpp>
#అన్తర్భావ <openvino/op/util/attr_types.hpp>

నామాకాశ ov {
నామాకాశ frontend {
నామాకాశ ggml {
నామాకాశ op {

Output_సదిశ translate_im2col(స్థిరరాశి కణుపు_Context & context) {
    అంకె_inputs_check(context, 2, 2);
    స్థిరరాశి int32_t * params = context.get_output_op_params();
    int32_t s0 = params[0];
    int32_t s1 = params[1];
    int32_t p0 = params[2];
    int32_t p1 = params[3];
    int32_t d0 = params[4];
    int32_t d1 = params[5];
    తార్కిక is_2D = params[6] == 1;
    ov::Output<Node> res;

    ov::Output<Node> image = context.get_ఆదానం(1);
    స్థిరరాశి ov::Shape kernel_shape = context.get_ఆదానం(0).get_shape();

    స్థిరరాశి size_t IC = is_2D ? kernel_shape[1] : kernel_shape[2];
    స్థిరరాశి size_t KH = is_2D ? kernel_shape[2] : 1;
    స్థిరరాశి size_t KW = kernel_shape[3];

    int32_t stride_w = s0;
    int32_t stride_h = is_2D ? s1 : 1;
    int32_t pad_w = p0;
    int32_t pad_h = is_2D ? p1 : 0;
    int32_t dil_w = d0;
    int32_t dil_h = is_2D ? d1 : 1;

    ఒకవేళ (!is_2D) {
        // GGML input shape: [IW, IC, N, 1]
        // OpenVINO input shape: [1, N, IC, IW]
        // Reshape image to: [N, IC, 1, IW]
        స్థిరరాశి ov::Shape image_shape = image.get_shape();
        స్థిరరాశి size_t N = image_shape[1];
        స్థిరరాశి size_t IW = image_shape[3];
        స్వయంచాలక image_reshape_shape = ov::op::v0::Constant::create(
            ov::element::i64, ov::Shape{4},
            std::vector<int64_t>{static_cast<int64_t>(N), static_cast<int64_t>(IC), 1, static_cast<int64_t>(IW)});
        image = std::make_shared<ov::op::v1::Reshape>(image, image_reshape_shape, అబద్ధం);
    }

    స్థిరరాశి ov::Shape patch_sizes = {KH, KW};
    స్థిరరాశి ov::Strides strides = {static_cast<size_t>(stride_h), static_cast<size_t>(stride_w)};
    స్థిరరాశి ov::Shape rates = {static_cast<size_t>(dil_h), static_cast<size_t>(dil_w)};

    స్వయంచాలక pads_begin =
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 0, pad_h, pad_w});
    స్వయంచాలక pads_చివర =
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 0, pad_h, pad_w});

    స్వయంచాలక pad = std::make_shared<ov::op::v1::Pad>(image, pads_begin, pads_చివర, ov::op::Pad_బహుళకం::CONSTANT);
    స్వయంచాలక patches =
        std::make_shared<ov::op::v3::వెలికితీయి_Image_Patches>(pad, patch_sizes, strides, rates, ov::op::PadType::VALID);

    // [N, KH*KW*IC, OH, OW] → [N, OH, OW, KH*KW*IC]
    స్వయంచాలక perm1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 2, 3, 1});
    స్వయంచాలక t1 = std::make_shared<ov::op::v1::Transpose>(patches, perm1);

    // [N, OH, OW, KH*KW*IC] → [N, OH, OW, KH*KW, IC]
    స్థిరరాశి ov::Shape బహిర్గతం_shape = t1->get_output_shape(0);
    స్థిరరాశి size_t N = బహిర్గతం_shape[0];
    స్థిరరాశి size_t OH = బహిర్గతం_shape[1];
    స్థిరరాశి size_t OW = బహిర్గతం_shape[2];
    స్వయంచాలక reshape1_shape = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{5},
        std::vector<int64_t>{static_cast<int64_t>(N), static_cast<int64_t>(OH), static_cast<int64_t>(OW),
                             static_cast<int64_t>(KH * KW), static_cast<int64_t>(IC)});
    స్వయంచాలక r1 = std::make_shared<ov::op::v1::Reshape>(t1, reshape1_shape, అబద్ధం);

    // [N, OH, OW, KH*KW, IC] → [N, OH, OW, IC, KH*KW]
    స్వయంచాలక perm2 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{5}, std::vector<int64_t>{0, 1, 2, 4, 3});
    స్వయంచాలక t2 = std::make_shared<ov::op::v1::Transpose>(r1, perm2);

    // flatten back to [N, OH, OW, IC*KH*KW]
    స్వయంచాలక r2_shape = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{4},
        std::vector<int64_t>{static_cast<int64_t>(N), static_cast<int64_t>(OH), static_cast<int64_t>(OW),
                             static_cast<int64_t>(IC * KH * KW)});
    res = std::make_shared<ov::op::v1::Reshape>(t2, r2_shape, అబద్ధం);

    ఒకవేళ (!is_2D) {
        // [N, 1, OW, IC * KW] -> [1, N, OW, IC * KW]
        స్వయంచాలక final_reshape_shape = ov::op::v0::Constant::create(
            ov::element::i64, ov::Shape{4},
            std::vector<int64_t>{1, static_cast<int64_t>(N), static_cast<int64_t>(OW), static_cast<int64_t>(IC * KW)});
        res = std::make_shared<ov::op::v1::Reshape>(res, final_reshape_shape, అబద్ధం);
    }

    స్వయంచాలక output_type = context.get_output_type();
    ఒకవేళ (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    తిరిగి rename_outputs_with_suffix({res}, context.get_name());
}

Output_సదిశ translate_im2col_3d(స్థిరరాశి కణుపు_Context & context) {
    అంకె_inputs_check(context, 2, 2);
    స్థిరరాశి int32_t * params = context.get_output_op_params();
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

    ov::Output<Node> image = process_view_ఆదానం_new(context, 1);
    స్థిరరాశి ov::Shape kernel_shape = context.get_ఆదానం(0).get_shape();
    స్థిరరాశి ov::Shape image_shape = image.get_shape();
    స్థిరరాశి ov::Shape బహిర్గతం_shape = context.get_output_shape().గమ్యం_shape();

    స్థిరరాశి size_t KD = kernel_shape[1];
    స్థిరరాశి size_t KH = kernel_shape[2];
    స్థిరరాశి size_t KW = kernel_shape[3];

    స్థిరరాశి size_t N = image_shape[0] / static_cast<size_t>(IC);
    స్థిరరాశి size_t ID = image_shape[1];
    స్థిరరాశి size_t IH = image_shape[2];
    స్థిరరాశి size_t IW = image_shape[3];

    స్థిరరాశి size_t OD = (ID + 2 * p2 - d2 * (KD - 1) - 1) / s2 + 1;
    స్థిరరాశి size_t OH = (IH + 2 * p1 - d1 * (KH - 1) - 1) / s1 + 1;
    స్థిరరాశి size_t OW = (IW + 2 * p0 - d0 * (KW - 1) - 1) / s0 + 1;

    ఒకవేళ (N == 0 || OD == 0 || OH == 0 || OW == 0) {
        స్వయంచాలక output_type = context.get_output_type();
        ov::Output<Node> res = ov::op::v0::Constant::create(
            output_type, ov::Shape{N * OD, OH, OW, static_cast<size_t>(IC * KD * KH * KW)}, {});
        తిరిగి rename_outputs_with_suffix({res}, context.get_name());
    }

    స్థిరరాశి size_t IH_pad = IH + 2 * p1;
    స్థిరరాశి size_t IW_pad = IW + 2 * p0;

    స్వయంచాలక image_5d_shape = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{5},
        std::vector<int64_t>{static_cast<int64_t>(N), static_cast<int64_t>(IC), static_cast<int64_t>(ID),
                             static_cast<int64_t>(IH), static_cast<int64_t>(IW)});
    స్వయంచాలక image_5d = std::make_shared<ov::op::v1::Reshape>(image, image_5d_shape, అబద్ధం);

    స్వయంచాలక pads_begin = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{5}, std::vector<int64_t>{0, 0, p2, p1, p0});
    స్వయంచాలక pads_చివర = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{5}, std::vector<int64_t>{0, 0, p2, p1, p0});
    స్వయంచాలక pad_3d = std::make_shared<ov::op::v1::Pad>(image_5d, pads_begin, pads_చివర, ov::op::Pad_బహుళకం::CONSTANT);

    స్థిరరాశి ov::Shape patch_sizes = {KH, KW};
    స్థిరరాశి ov::Strides strides = {static_cast<size_t>(s1), static_cast<size_t>(s0)};
    స్థిరరాశి ov::Shape rates = {static_cast<size_t>(d1), static_cast<size_t>(d0)};

    ov::Output_సదిశ kd_slices;
    kd_slices.reserve(KD);

    స్వయంచాలక perm_nod = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{5}, {0, 2, 1, 3, 4});
    స్వయంచాలక reshape_4d_shape = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{4},
        std::vector<int64_t>{static_cast<int64_t>(N * OD), static_cast<int64_t>(IC),
                             static_cast<int64_t>(IH_pad), static_cast<int64_t>(IW_pad)});
    స్వయంచాలక perm1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 3, 1});
    స్వయంచాలక r1_shape = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{5},
        std::vector<int64_t>{static_cast<int64_t>(N * OD), static_cast<int64_t>(OH), static_cast<int64_t>(OW),
                             static_cast<int64_t>(KH * KW), static_cast<int64_t>(IC)});
    స్వయంచాలక perm2 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{5}, {0, 1, 2, 4, 3});
    స్వయంచాలక r2_shape = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{6},
        std::vector<int64_t>{static_cast<int64_t>(N * OD), static_cast<int64_t>(OH), static_cast<int64_t>(OW),
                             static_cast<int64_t>(IC), 1, static_cast<int64_t>(KH * KW)});
    స్వయంచాలక అడుగు_c = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {static_cast<int64_t>(s2)});
    స్వయంచాలక axes_c = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});

    కొరకు (size_t ikd = 0; ikd < KD; ++ikd) {
        స్వయంచాలక ప్రారంభం_c = ov::op::v0::Constant::create(
            ov::element::i64, ov::Shape{1}, {static_cast<int64_t>(ikd * d2)});
        స్వయంచాలక stop_c = ov::op::v0::Constant::create(
            ov::element::i64, ov::Shape{1}, {static_cast<int64_t>(ikd * d2 + OD * s2)});
        స్వయంచాలక లోతు_ముక్క = std::make_shared<ov::op::v8::ముక్క>(pad_3d, ప్రారంభం_c, stop_c, అడుగు_c, axes_c);
        స్వయంచాలక లోతు_ముక్క_trans = std::make_shared<ov::op::v1::Transpose>(లోతు_ముక్క, perm_nod);
        స్వయంచాలక లోతు_ముక్క_4d = std::make_shared<ov::op::v1::Reshape>(లోతు_ముక్క_trans, reshape_4d_shape, అబద్ధం);

        స్వయంచాలక patches = std::make_shared<ov::op::v3::వెలికితీయి_Image_Patches>(
            లోతు_ముక్క_4d, patch_sizes, strides, rates, ov::op::PadType::VALID);
        స్వయంచాలక t1 = std::make_shared<ov::op::v1::Transpose>(patches, perm1);
        స్వయంచాలక r1 = std::make_shared<ov::op::v1::Reshape>(t1, r1_shape, అబద్ధం);
        స్వయంచాలక t2 = std::make_shared<ov::op::v1::Transpose>(r1, perm2);
        స్వయంచాలక r2 = std::make_shared<ov::op::v1::Reshape>(t2, r2_shape, అబద్ధం);
        kd_slices.push_back(r2);
    }

    ov::Output<Node> res;
    ఒకవేళ (KD == 1) {
        res = kd_slices[0];
    } లేకపోతే {
        res = std::make_shared<ov::op::v0::Concat>(kd_slices, 4);
    }

    స్వయంచాలక final_shape = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{4},
        std::vector<int64_t>{static_cast<int64_t>(N * OD), static_cast<int64_t>(OH), static_cast<int64_t>(OW),
                             static_cast<int64_t>(IC * KD * KH * KW)});
    res = std::make_shared<ov::op::v1::Reshape>(res, final_shape, అబద్ధం);

    స్వయంచాలక output_type = context.get_output_type();
    ఒకవేళ (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    తిరిగి rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
