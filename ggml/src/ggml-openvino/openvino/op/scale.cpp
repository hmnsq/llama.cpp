#अन्तर्भाव "../node_context.h"
#अन्तर्भाव "../op_table.h"
#अन्तर्भाव "../utils.h"

#अन्तर्भाव <openvino/core/except.hpp>
#अन्तर्भाव <openvino/op/add.hpp>
#अन्तर्भाव <openvino/op/concat.hpp>
#अन्तर्भाव <openvino/op/constant.hpp>
#अन्तर्भाव <openvino/op/convert.hpp>
#अन्तर्भाव <openvino/op/equal.hpp>
#अन्तर्भाव <openvino/op/gather.hpp>
#अन्तर्भाव <openvino/op/greater_eq.hpp>
#अन्तर्भाव <openvino/op/if.hpp>
#अन्तर्भाव <openvino/op/less.hpp>
#अन्तर्भाव <openvino/op/logical_or.hpp>
#अन्तर्भाव <openvino/op/multiply.hpp>
#अन्तर्भाव <openvino/op/range.hpp>
#अन्तर्भाव <openvino/op/reshape.hpp>
#अन्तर्भाव <openvino/op/shape_of.hpp>
#अन्तर्भाव <openvino/op/slice.hpp>
#अन्तर्भाव <openvino/op/squeeze.hpp>
#अन्तर्भाव <openvino/op/unsqueeze.hpp>
#अन्तर्भाव <vector>

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {
नामाकाश op {

Output_सदिशम् translate_scale(ध्रुव ग्रन्थिः_Context & context) {
    अङ्कः_inputs_check(context, 1, 1);

    भिन्न scale;
    भिन्न bias;
    स्मृतिं_प्रतिलिख(&scale, (भिन्न *) context.get_output_op_params() + 0, परिमाण(भिन्न));
    स्मृतिं_प्रतिलिख(&bias, (भिन्न *) context.get_output_op_params() + 1, परिमाण(भिन्न));

    स्वचालित scale_ग्रन्थिः = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<भिन्न>{scale});

    यदि (context.get_op_case() == 2) {
        प्रत्यागम {context.get_निवेशः(0)};
    }

    यदि (context.get_op_case() == 1 && context.has_निवेशः("cache_rs_reset_len")) {
        स्वचालित cache_rs_reset_idx = context.get_निवेशः("cache_rs_reset_idx");
        स्वचालित cache_rs_reset_len = context.get_निवेशः("cache_rs_reset_len");

        स्वचालित cache_rs = context.get_निवेशः(0);

        स्वचालित cache_shape = std::make_shared<ov::op::v3::Shape_सम्बन्धि>(cache_rs, ov::element::i64);
        स्वचालित n_slots_1d = std::make_shared<ov::op::v8::Gather>(
            cache_shape, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2}),
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0}));
        स्वचालित n_slots = std::make_shared<ov::op::v0::Squeeze>(n_slots_1d);

        स्वचालित iota = std::make_shared<ov::op::v4::Range>(
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0}), n_slots,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1}), ov::element::i64);

        स्वचालित idx_plus_len = std::make_shared<ov::op::v1::सङ्कलय>(cache_rs_reset_idx, cache_rs_reset_len);
        स्वचालित less_than_idx = std::make_shared<ov::op::v1::Less>(iota, cache_rs_reset_idx);
        स्वचालित greater_equal_idx_plus_len = std::make_shared<ov::op::v1::GreaterEqual>(iota, idx_plus_len);
        स्वचालित keep_आवरणम् = std::make_shared<ov::op::v1::LogicalOr>(less_than_idx, greater_equal_idx_plus_len);

        स्वचालित keep_आवरणम्_f32 = std::make_shared<ov::op::v0::Convert>(keep_आवरणम्, ov::element::f32);
        स्वचालित keep_आवरणम्_reshape = std::make_shared<ov::op::v0::Unsqueeze>(
            keep_आवरणम्_f32, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1}));

        स्वचालित cleared_cache_rs = std::make_shared<ov::op::v1::Multiply>(cache_rs, keep_आवरणम्_reshape);
        प्रत्यागम rename_outputs_with_suffix({cleared_cache_rs}, context.get_name());
    }

    स्वचालित scaled = std::make_shared<ov::op::v1::Multiply>(context.get_निवेशः(0), scale_ग्रन्थिः);

    std::shared_ptr<ov::Node> res;
    यदि (bias != 0.0f) {
        स्वचालित bias_ग्रन्थिः =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<भिन्न>{bias});
        res = std::make_shared<ov::op::v1::सङ्कलय>(scaled, bias_ग्रन्थिः);
    } अन्यथा {
        res = scaled;
    }

    प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
