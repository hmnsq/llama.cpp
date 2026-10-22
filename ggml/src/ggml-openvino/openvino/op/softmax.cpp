#अन्तर्भाव "../node_context.h"
#अन्तर्भाव "../op_table.h"
#अन्तर्भाव "../utils.h"

#अन्तर्भाव <cmath>
#अन्तर्भाव <cstdint>
#अन्तर्भाव <cstring>
#अन्तर्भाव <memory>
#अन्तर्भाव <openvino/op/broadcast.hpp>
#अन्तर्भाव <openvino/frontend/exception.hpp>
#अन्तर्भाव <openvino/op/add.hpp>
#अन्तर्भाव <openvino/op/concat.hpp>
#अन्तर्भाव <openvino/op/constant.hpp>
#अन्तर्भाव <openvino/op/convert.hpp>
#अन्तर्भाव <openvino/op/multiply.hpp>
#अन्तर्भाव <openvino/op/reshape.hpp>
#अन्तर्भाव <openvino/op/shape_of.hpp>
#अन्तर्भाव <openvino/op/slice.hpp>
#अन्तर्भाव <openvino/op/softmax.hpp>
#अन्तर्भाव <vector>

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {
नामाकाश op {

स्थिर तार्किक is_static_one(ध्रुव ov::Dimension & dim) {
    प्रत्यागम dim.is_static() && dim.get_दीर्घता() == 1;
}

स्थिर तार्किक same_static_dim(ध्रुव ov::Dimension & lhs, ध्रुव ov::Dimension & rhs) {
    प्रत्यागम lhs.is_static() && rhs.is_static() && lhs.get_दीर्घता() == rhs.get_दीर्घता();
}

स्थिर तार्किक is_attention_sinks_निवेशः_shape(ध्रुव ov::PartialShape & candidate, ध्रुव ov::PartialShape & logits_shape) {
    यदि (candidate.rank().is_dynamic() || logits_shape.rank().is_dynamic() || candidate.rank().get_दीर्घता() != 4 ||
        logits_shape.rank().get_दीर्घता() != 4) {
        प्रत्यागम असत्य;
    }

    प्रत्यागम is_static_one(candidate[0]) && is_static_one(candidate[1]) && is_static_one(candidate[2]) &&
           same_static_dim(candidate[3], logits_shape[1]);
}

// Reimplementation of GGML_OP_SOFT_MAX semantics for OpenVINO backend:
// 1) logits = src0 * scale
// 2) logits += mask (if provided)
// 3) append attention sinks as hidden logits (if provided)
// 4) softmax over the last dimension and remove the hidden sink column
Output_सदिशम् translate_soft_max(ध्रुव ग्रन्थिः_Context & context) {
    अङ्कः_inputs_check(context, 1, 3);

    भिन्न scale = 1.0f;
    भिन्न max_bias = 0.0f;
    स्मृतिं_प्रतिलिख(&scale, (भिन्न *) context.get_output_op_params() + 0, परिमाण(भिन्न));
    स्मृतिं_प्रतिलिख(&max_bias, (भिन्न *) context.get_output_op_params() + 1, परिमाण(भिन्न));

    ov::Output<ov::Node> logits = context.get_निवेशः(0);
    ध्रुव तार्किक second_निवेशः_is_sinks =
        context.get_निवेशः_आकार() == 2 && is_attention_sinks_निवेशः_shape(context.get_निवेशः_shape(1), context.get_output_shape());
    ध्रुव तार्किक has_आवरणम् = context.get_निवेशः_आकार() > 1 && !second_निवेशः_is_sinks;
    ध्रुव तार्किक has_sinks = second_निवेशः_is_sinks || context.get_निवेशः_आकार() > 2;
    ध्रुव size_t sinks_निवेशः_idx = second_निवेशः_is_sinks ? 1 : 2;

    // Apply scale first: logits = src0 * scale
    यदि (scale != 1.0f) {
        स्वचालित scale_const =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<भिन्न>{scale});
        logits = std::make_shared<ov::op::v1::Multiply>(logits, scale_const);
    }

    FRONT_अन्तः_CHECK_IMPLEMENTED(!(max_bias > 0.0f && !has_आवरणम्),
                                "OpenVINO softmax ALiBi path requires mask input");

    // Optional mask add: logits += mask
    // For max_bias > 0 (ALiBi), apply per-head slope to mask before adding.
    यदि (has_आवरणम्) {
        ov::Output<ov::Node> आवरणम् = context.get_निवेशः(1);

        // For stateful
        std::string आवरणम्_name = "KQ_mask_sliced";
        यदि (context.get_निवेशः_नामानि()[1].find("swa") != std::string::npos) {
            आवरणम्_name = "KQ_mask_swa_sliced";
        }
        यदि (context.has_निवेशः(आवरणम्_name)) {
            आवरणम् = context.get_निवेशः(आवरणम्_name);
        }

        यदि (आवरणम्.get_element_type() != logits.get_element_type()) {
            आवरणम् = std::make_shared<ov::op::v0::Convert>(आवरणम्, logits.get_element_type());
        }

        यदि (max_bias > 0.0f) {
            स्वचालित बहिर्गमनम्_shape = context.get_output_shape().गन्तव्यम्_shape();
            FRONT_अन्तः_CHECK_IMPLEMENTED(बहिर्गमनम्_shape.size() == 4, "OpenVINO softmax ALiBi path expects rank-4 tensor");

            ध्रुव uint32_t n_शीर्ष = static_cast<uint32_t>(बहिर्गमनम्_shape[1]);
            FRONT_अन्तः_CHECK_IMPLEMENTED(n_शीर्ष > 0, "OpenVINO softmax ALiBi path expects n_head > 0");

            ध्रुव uint32_t n_शीर्ष_log2 = 1u << static_cast<uint32_t>(std::floor(std::log2(static_cast<भिन्न>(n_शीर्ष))));
            ध्रुव भिन्न m0 = std::घातः(2.0f, -(max_bias) / static_cast<भिन्न>(n_शीर्ष_log2));
            ध्रुव भिन्न m1 = std::घातः(2.0f, -(max_bias / 2.0f) / static_cast<भिन्न>(n_शीर्ष_log2));

            std::vector<भिन्न> slopes(n_शीर्ष);
            प्रति (uint32_t h = 0; h < n_शीर्ष; ++h) {
                slopes[h] = h < n_शीर्ष_log2 ? std::घातः(m0, static_cast<भिन्न>(h + 1)) :
                                              std::घातः(m1, static_cast<भिन्न>(2 * (h - n_शीर्ष_log2) + 1));
            }

            ov::Output<ov::Node> नतिः_ग्रन्थिः =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{n_शीर्ष}, slopes);
            यदि (नतिः_ग्रन्थिः.get_element_type() != आवरणम्.get_element_type()) {
                नतिः_ग्रन्थिः = std::make_shared<ov::op::v0::Convert>(नतिः_ग्रन्थिः, आवरणम्.get_element_type());
            }

            स्वचालित नतिः_shape = std::make_shared<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4}, std::vector<int64_t>{1, static_cast<int64_t>(n_शीर्ष), 1, 1});
            स्वचालित नतिः_4d = std::make_shared<ov::op::v1::Reshape>(नतिः_ग्रन्थिः, नतिः_shape, असत्य);
            आवरणम् = std::make_shared<ov::op::v1::Multiply>(आवरणम्, नतिः_4d);
        }

        logits = std::make_shared<ov::op::v1::सङ्कलय>(logits, आवरणम्);
    }

    ov::Output<ov::Node> softmax_निवेशः = logits;
    यदि (has_sinks) {
        ov::Output<ov::Node> sinks = context.get_निवेशः(sinks_निवेशः_idx);
        यदि (sinks.get_element_type() != logits.get_element_type()) {
            sinks = std::make_shared<ov::op::v0::Convert>(sinks, logits.get_element_type());
        }

        स्वचालित sink_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, {1, -1, 1, 1});
        स्वचालित sinks_4d = std::make_shared<ov::op::v1::Reshape>(sinks, sink_shape, असत्य);

        स्वचालित logits_shape = std::make_shared<ov::op::v3::Shape_सम्बन्धि>(logits, ov::element::i64);
        स्वचालित zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        स्वचालित one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        स्वचालित three = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
        स्वचालित four = ov::op::v0::Constant::create(ov::element::i64, {1}, {4});
        स्वचालित shape_axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});

        स्वचालित sink_उपसर्गः_shape = std::make_shared<ov::op::v8::खण्डिका>(logits_shape, zero, three, one, shape_axis);
        स्वचालित sink_चरमः_dim = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        स्वचालित sink_प्रसारणम्_shape = std::make_shared<ov::op::v0::Concat>(
            ov::Output_सदिशम्{sink_उपसर्गः_shape, sink_चरमः_dim}, 0);
        स्वचालित sink_column = std::make_shared<ov::op::v3::प्रसारणम्>(sinks_4d, sink_प्रसारणम्_shape,
                                                                   ov::op::प्रसारणम्_Type::BIDIRECTIONAL);
        softmax_निवेशः = std::make_shared<ov::op::v0::Concat>(ov::Output_सदिशम्{logits, sink_column}, 3);

        स्वचालित softmax_with_sink = std::make_shared<ov::op::v8::Softmax>(softmax_निवेशः, -1);
        स्वचालित original_चरमः_dim = std::make_shared<ov::op::v8::खण्डिका>(logits_shape, three, four, one, shape_axis);
        स्वचालित res = std::make_shared<ov::op::v8::खण्डिका>(softmax_with_sink, zero, original_चरमः_dim, one, three);

        प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
    }

    // Softmax along last dimension (equivalent to ggml softmax over ne[0]).
    स्वचालित res = std::make_shared<ov::op::v8::Softmax>(softmax_निवेशः, -1);

    प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
