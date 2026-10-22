#ضمن "../node_context.h"
#ضمن "../op_table.h"
#ضمن "../utils.h"

#ضمن <openvino/core/except.hpp>
#ضمن <openvino/op/add.hpp>
#ضمن <openvino/op/concat.hpp>
#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/convert.hpp>
#ضمن <openvino/op/equal.hpp>
#ضمن <openvino/op/gather.hpp>
#ضمن <openvino/op/greater_eq.hpp>
#ضمن <openvino/op/if.hpp>
#ضمن <openvino/op/less.hpp>
#ضمن <openvino/op/logical_or.hpp>
#ضمن <openvino/op/multiply.hpp>
#ضمن <openvino/op/range.hpp>
#ضمن <openvino/op/reshape.hpp>
#ضمن <openvino/op/shape_of.hpp>
#ضمن <openvino/op/slice.hpp>
#ضمن <openvino/op/squeeze.hpp>
#ضمن <openvino/op/unsqueeze.hpp>
#ضمن <vector>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق op {

Output_سمتیہ translate_scale(ثابت گرہ_Context & context) {
    عدد_inputs_check(context, 1, 1);

    عائم scale;
    عائم bias;
    حافظہ_نقل(&scale, (عائم *) context.get_output_op_params() + 0, حجم(عائم));
    حافظہ_نقل(&bias, (عائم *) context.get_output_op_params() + 1, حجم(عائم));

    خودکار scale_گرہ = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<عائم>{scale});

    إذا (context.get_op_case() == 2) {
        أرجع {context.get_داخلہ(0)};
    }

    إذا (context.get_op_case() == 1 && context.has_داخلہ("cache_rs_reset_len")) {
        خودکار cache_rs_reset_idx = context.get_داخلہ("cache_rs_reset_idx");
        خودکار cache_rs_reset_len = context.get_داخلہ("cache_rs_reset_len");

        خودکار cache_rs = context.get_داخلہ(0);

        خودکار cache_shape = std::make_shared<ov::op::v3::Shape_دا>(cache_rs, ov::element::i64);
        خودکار n_slots_1d = std::make_shared<ov::op::v8::Gather>(
            cache_shape, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2}),
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0}));
        خودکار n_slots = std::make_shared<ov::op::v0::Squeeze>(n_slots_1d);

        خودکار iota = std::make_shared<ov::op::v4::Range>(
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0}), n_slots,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1}), ov::element::i64);

        خودکار idx_plus_len = std::make_shared<ov::op::v1::جوڑو>(cache_rs_reset_idx, cache_rs_reset_len);
        خودکار less_than_idx = std::make_shared<ov::op::v1::Less>(iota, cache_rs_reset_idx);
        خودکار greater_equal_idx_plus_len = std::make_shared<ov::op::v1::GreaterEqual>(iota, idx_plus_len);
        خودکار keep_نقاب = std::make_shared<ov::op::v1::LogicalOr>(less_than_idx, greater_equal_idx_plus_len);

        خودکار keep_نقاب_f32 = std::make_shared<ov::op::v0::Convert>(keep_نقاب, ov::element::f32);
        خودکار keep_نقاب_reshape = std::make_shared<ov::op::v0::Unsqueeze>(
            keep_نقاب_f32, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1}));

        خودکار cleared_cache_rs = std::make_shared<ov::op::v1::Multiply>(cache_rs, keep_نقاب_reshape);
        أرجع rename_outputs_with_suffix({cleared_cache_rs}, context.get_name());
    }

    خودکار scaled = std::make_shared<ov::op::v1::Multiply>(context.get_داخلہ(0), scale_گرہ);

    std::shared_ptr<ov::Node> res;
    إذا (bias != 0.0f) {
        خودکار bias_گرہ =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<عائم>{bias});
        res = std::make_shared<ov::op::v1::جوڑو>(scaled, bias_گرہ);
    } وإلا {
        res = scaled;
    }

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
