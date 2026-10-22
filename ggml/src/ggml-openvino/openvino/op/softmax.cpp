#అన్తర్భావ "../node_context.h"
#అన్తర్భావ "../op_table.h"
#అన్తర్భావ "../utils.h"

#అన్తర్భావ <cmath>
#అన్తర్భావ <cstdint>
#అన్తర్భావ <cstring>
#అన్తర్భావ <memory>
#అన్తర్భావ <openvino/op/broadcast.hpp>
#అన్తర్భావ <openvino/frontend/exception.hpp>
#అన్తర్భావ <openvino/op/add.hpp>
#అన్తర్భావ <openvino/op/concat.hpp>
#అన్తర్భావ <openvino/op/constant.hpp>
#అన్తర్భావ <openvino/op/convert.hpp>
#అన్తర్భావ <openvino/op/multiply.hpp>
#అన్తర్భావ <openvino/op/reshape.hpp>
#అన్తర్భావ <openvino/op/shape_of.hpp>
#అన్తర్భావ <openvino/op/slice.hpp>
#అన్తర్భావ <openvino/op/softmax.hpp>
#అన్తర్భావ <vector>

నామాకాశ ov {
నామాకాశ frontend {
నామాకాశ ggml {
నామాకాశ op {

స్థిర తార్కిక is_static_one(స్థిరరాశి ov::Dimension & dim) {
    తిరిగి dim.is_static() && dim.get_నిడివి() == 1;
}

స్థిర తార్కిక same_static_dim(స్థిరరాశి ov::Dimension & lhs, స్థిరరాశి ov::Dimension & rhs) {
    తిరిగి lhs.is_static() && rhs.is_static() && lhs.get_నిడివి() == rhs.get_నిడివి();
}

స్థిర తార్కిక is_attention_sinks_ఆదానం_shape(స్థిరరాశి ov::PartialShape & candidate, స్థిరరాశి ov::PartialShape & logits_shape) {
    ఒకవేళ (candidate.rank().is_dynamic() || logits_shape.rank().is_dynamic() || candidate.rank().get_నిడివి() != 4 ||
        logits_shape.rank().get_నిడివి() != 4) {
        తిరిగి అబద్ధం;
    }

    తిరిగి is_static_one(candidate[0]) && is_static_one(candidate[1]) && is_static_one(candidate[2]) &&
           same_static_dim(candidate[3], logits_shape[1]);
}

// Reimplementation of GGML_OP_SOFT_MAX semantics for OpenVINO backend:
// 1) logits = src0 * scale
// 2) logits += mask (if provided)
// 3) append attention sinks as hidden logits (if provided)
// 4) softmax over the last dimension and remove the hidden sink column
Output_సదిశ translate_soft_max(స్థిరరాశి కణుపు_Context & context) {
    అంకె_inputs_check(context, 1, 3);

    దశాంశం scale = 1.0f;
    దశాంశం max_bias = 0.0f;
    జ్ఞాపకం_నకలు(&scale, (దశాంశం *) context.get_output_op_params() + 0, పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&max_bias, (దశాంశం *) context.get_output_op_params() + 1, పరిమాణం(దశాంశం));

    ov::Output<ov::Node> logits = context.get_ఆదానం(0);
    స్థిరరాశి తార్కిక రెండవ_ఆదానం_is_sinks =
        context.get_ఆదానం_కొలత() == 2 && is_attention_sinks_ఆదానం_shape(context.get_ఆదానం_shape(1), context.get_output_shape());
    స్థిరరాశి తార్కిక has_ముసుగు = context.get_ఆదానం_కొలత() > 1 && !రెండవ_ఆదానం_is_sinks;
    స్థిరరాశి తార్కిక has_sinks = రెండవ_ఆదానం_is_sinks || context.get_ఆదానం_కొలత() > 2;
    స్థిరరాశి size_t sinks_ఆదానం_idx = రెండవ_ఆదానం_is_sinks ? 1 : 2;

    // Apply scale first: logits = src0 * scale
    ఒకవేళ (scale != 1.0f) {
        స్వయంచాలక scale_const =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<దశాంశం>{scale});
        logits = std::make_shared<ov::op::v1::Multiply>(logits, scale_const);
    }

    FRONT_చివర_CHECK_IMPLEMENTED(!(max_bias > 0.0f && !has_ముసుగు),
                                "OpenVINO softmax ALiBi path requires mask input");

    // Optional mask add: logits += mask
    // For max_bias > 0 (ALiBi), apply per-head slope to mask before adding.
    ఒకవేళ (has_ముసుగు) {
        ov::Output<ov::Node> ముసుగు = context.get_ఆదానం(1);

        // For stateful
        std::string ముసుగు_name = "KQ_mask_sliced";
        ఒకవేళ (context.get_ఆదానం_పేర్లు()[1].find("swa") != std::string::npos) {
            ముసుగు_name = "KQ_mask_swa_sliced";
        }
        ఒకవేళ (context.has_ఆదానం(ముసుగు_name)) {
            ముసుగు = context.get_ఆదానం(ముసుగు_name);
        }

        ఒకవేళ (ముసుగు.get_element_type() != logits.get_element_type()) {
            ముసుగు = std::make_shared<ov::op::v0::Convert>(ముసుగు, logits.get_element_type());
        }

        ఒకవేళ (max_bias > 0.0f) {
            స్వయంచాలక బహిర్గతం_shape = context.get_output_shape().గమ్యం_shape();
            FRONT_చివర_CHECK_IMPLEMENTED(బహిర్గతం_shape.size() == 4, "OpenVINO softmax ALiBi path expects rank-4 tensor");

            స్థిరరాశి uint32_t n_తల = static_cast<uint32_t>(బహిర్గతం_shape[1]);
            FRONT_చివర_CHECK_IMPLEMENTED(n_తల > 0, "OpenVINO softmax ALiBi path expects n_head > 0");

            స్థిరరాశి uint32_t n_తల_log2 = 1u << static_cast<uint32_t>(std::floor(std::log2(static_cast<దశాంశం>(n_తల))));
            స్థిరరాశి దశాంశం m0 = std::ఘాతం(2.0f, -(max_bias) / static_cast<దశాంశం>(n_తల_log2));
            స్థిరరాశి దశాంశం m1 = std::ఘాతం(2.0f, -(max_bias / 2.0f) / static_cast<దశాంశం>(n_తల_log2));

            std::vector<దశాంశం> slopes(n_తల);
            కొరకు (uint32_t h = 0; h < n_తల; ++h) {
                slopes[h] = h < n_తల_log2 ? std::ఘాతం(m0, static_cast<దశాంశం>(h + 1)) :
                                              std::ఘాతం(m1, static_cast<దశాంశం>(2 * (h - n_తల_log2) + 1));
            }

            ov::Output<ov::Node> వాలు_కణుపు =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{n_తల}, slopes);
            ఒకవేళ (వాలు_కణుపు.get_element_type() != ముసుగు.get_element_type()) {
                వాలు_కణుపు = std::make_shared<ov::op::v0::Convert>(వాలు_కణుపు, ముసుగు.get_element_type());
            }

            స్వయంచాలక వాలు_shape = std::make_shared<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4}, std::vector<int64_t>{1, static_cast<int64_t>(n_తల), 1, 1});
            స్వయంచాలక వాలు_4d = std::make_shared<ov::op::v1::Reshape>(వాలు_కణుపు, వాలు_shape, అబద్ధం);
            ముసుగు = std::make_shared<ov::op::v1::Multiply>(ముసుగు, వాలు_4d);
        }

        logits = std::make_shared<ov::op::v1::కూడు>(logits, ముసుగు);
    }

    ov::Output<ov::Node> softmax_ఆదానం = logits;
    ఒకవేళ (has_sinks) {
        ov::Output<ov::Node> sinks = context.get_ఆదానం(sinks_ఆదానం_idx);
        ఒకవేళ (sinks.get_element_type() != logits.get_element_type()) {
            sinks = std::make_shared<ov::op::v0::Convert>(sinks, logits.get_element_type());
        }

        స్వయంచాలక sink_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, {1, -1, 1, 1});
        స్వయంచాలక sinks_4d = std::make_shared<ov::op::v1::Reshape>(sinks, sink_shape, అబద్ధం);

        స్వయంచాలక logits_shape = std::make_shared<ov::op::v3::Shape_యొక్క>(logits, ov::element::i64);
        స్వయంచాలక zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        స్వయంచాలక one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        స్వయంచాలక three = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
        స్వయంచాలక four = ov::op::v0::Constant::create(ov::element::i64, {1}, {4});
        స్వయంచాలక shape_axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});

        స్వయంచాలక sink_ఉపసర్గ_shape = std::make_shared<ov::op::v8::ముక్క>(logits_shape, zero, three, one, shape_axis);
        స్వయంచాలక sink_చివరి_dim = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        స్వయంచాలక sink_ప్రసారం_shape = std::make_shared<ov::op::v0::Concat>(
            ov::Output_సదిశ{sink_ఉపసర్గ_shape, sink_చివరి_dim}, 0);
        స్వయంచాలక sink_స్తంభం = std::make_shared<ov::op::v3::ప్రసారం>(sinks_4d, sink_ప్రసారం_shape,
                                                                   ov::op::ప్రసారం_Type::BIDIRECTIONAL);
        softmax_ఆదానం = std::make_shared<ov::op::v0::Concat>(ov::Output_సదిశ{logits, sink_స్తంభం}, 3);

        స్వయంచాలక softmax_with_sink = std::make_shared<ov::op::v8::Softmax>(softmax_ఆదానం, -1);
        స్వయంచాలక original_చివరి_dim = std::make_shared<ov::op::v8::ముక్క>(logits_shape, three, four, one, shape_axis);
        స్వయంచాలక res = std::make_shared<ov::op::v8::ముక్క>(softmax_with_sink, zero, original_చివరి_dim, one, three);

        తిరిగి rename_outputs_with_suffix({res}, context.get_name());
    }

    // Softmax along last dimension (equivalent to ggml softmax over ne[0]).
    స్వయంచాలక res = std::make_shared<ov::op::v8::Softmax>(softmax_ఆదానం, -1);

    తిరిగి rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
