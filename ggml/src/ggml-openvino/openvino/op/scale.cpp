#ਅਨ੍ਤਰ੍ਭਾਵ "../node_context.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../op_table.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../utils.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/core/except.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/add.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/concat.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/constant.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/convert.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/equal.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/gather.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/greater_eq.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/if.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/less.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/logical_or.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/multiply.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/range.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/reshape.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/shape_of.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/slice.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/squeeze.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/unsqueeze.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <vector>

ਨਾਮਾਕਾਸ਼ ov {
ਨਾਮਾਕਾਸ਼ frontend {
ਨਾਮਾਕਾਸ਼ ggml {
ਨਾਮਾਕਾਸ਼ op {

Output_ਸਦਿਸ਼ translate_scale(ਅਚਲ ਗੰਢ_Context & context) {
    ਅੰਕ_inputs_check(context, 1, 1);

    ਦਸ਼ਮਲਵ scale;
    ਦਸ਼ਮਲਵ bias;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&scale, (ਦਸ਼ਮਲਵ *) context.get_output_op_params() + 0, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&bias, (ਦਸ਼ਮਲਵ *) context.get_output_op_params() + 1, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਸਵੈਚਾਲਿਤ scale_ਗੰਢ = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<ਦਸ਼ਮਲਵ>{scale});

    ਜੇ (context.get_op_case() == 2) {
        ਵਾਪਸ {context.get_ਨਿਵੇਸ਼(0)};
    }

    ਜੇ (context.get_op_case() == 1 && context.has_ਨਿਵੇਸ਼("cache_rs_reset_len")) {
        ਸਵੈਚਾਲਿਤ cache_rs_reset_idx = context.get_ਨਿਵੇਸ਼("cache_rs_reset_idx");
        ਸਵੈਚਾਲਿਤ cache_rs_reset_len = context.get_ਨਿਵੇਸ਼("cache_rs_reset_len");

        ਸਵੈਚਾਲਿਤ cache_rs = context.get_ਨਿਵੇਸ਼(0);

        ਸਵੈਚਾਲਿਤ cache_shape = std::make_shared<ov::op::v3::Shape_ਦਾ>(cache_rs, ov::element::i64);
        ਸਵੈਚਾਲਿਤ n_slots_1d = std::make_shared<ov::op::v8::Gather>(
            cache_shape, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2}),
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0}));
        ਸਵੈਚਾਲਿਤ n_slots = std::make_shared<ov::op::v0::Squeeze>(n_slots_1d);

        ਸਵੈਚਾਲਿਤ iota = std::make_shared<ov::op::v4::Range>(
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0}), n_slots,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1}), ov::element::i64);

        ਸਵੈਚਾਲਿਤ idx_plus_len = std::make_shared<ov::op::v1::ਜੋੜੋ>(cache_rs_reset_idx, cache_rs_reset_len);
        ਸਵੈਚਾਲਿਤ less_than_idx = std::make_shared<ov::op::v1::Less>(iota, cache_rs_reset_idx);
        ਸਵੈਚਾਲਿਤ greater_equal_idx_plus_len = std::make_shared<ov::op::v1::GreaterEqual>(iota, idx_plus_len);
        ਸਵੈਚਾਲਿਤ keep_ਨਕਾਬ = std::make_shared<ov::op::v1::LogicalOr>(less_than_idx, greater_equal_idx_plus_len);

        ਸਵੈਚਾਲਿਤ keep_ਨਕਾਬ_f32 = std::make_shared<ov::op::v0::Convert>(keep_ਨਕਾਬ, ov::element::f32);
        ਸਵੈਚਾਲਿਤ keep_ਨਕਾਬ_reshape = std::make_shared<ov::op::v0::Unsqueeze>(
            keep_ਨਕਾਬ_f32, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1}));

        ਸਵੈਚਾਲਿਤ cleared_cache_rs = std::make_shared<ov::op::v1::Multiply>(cache_rs, keep_ਨਕਾਬ_reshape);
        ਵਾਪਸ rename_outputs_with_suffix({cleared_cache_rs}, context.get_name());
    }

    ਸਵੈਚਾਲਿਤ scaled = std::make_shared<ov::op::v1::Multiply>(context.get_ਨਿਵੇਸ਼(0), scale_ਗੰਢ);

    std::shared_ptr<ov::Node> res;
    ਜੇ (bias != 0.0f) {
        ਸਵੈਚਾਲਿਤ bias_ਗੰਢ =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<ਦਸ਼ਮਲਵ>{bias});
        res = std::make_shared<ov::op::v1::ਜੋੜੋ>(scaled, bias_ਗੰਢ);
    } ਨਹੀਂਤਾਂ {
        res = scaled;
    }

    ਵਾਪਸ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
