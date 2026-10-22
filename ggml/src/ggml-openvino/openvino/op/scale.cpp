#شامل "../node_context.h"
#شامل "../op_table.h"
#شامل "../utils.h"

#شامل <openvino/core/except.hpp>
#شامل <openvino/op/add.hpp>
#شامل <openvino/op/concat.hpp>
#شامل <openvino/op/constant.hpp>
#شامل <openvino/op/convert.hpp>
#شامل <openvino/op/equal.hpp>
#شامل <openvino/op/gather.hpp>
#شامل <openvino/op/greater_eq.hpp>
#شامل <openvino/op/if.hpp>
#شامل <openvino/op/less.hpp>
#شامل <openvino/op/logical_or.hpp>
#شامل <openvino/op/multiply.hpp>
#شامل <openvino/op/range.hpp>
#شامل <openvino/op/reshape.hpp>
#شامل <openvino/op/shape_of.hpp>
#شامل <openvino/op/slice.hpp>
#شامل <openvino/op/squeeze.hpp>
#شامل <openvino/op/unsqueeze.hpp>
#شامل <vector>

نامخلا ov {
نامخلا frontend {
نامخلا ggml {
نامخلا op {

Output_سمتیہ translate_scale(ثابت گرہ_Context & context) {
    عدد_inputs_check(context, 1, 1);

    اعشاری scale;
    اعشاری bias;
    حافظہ_نقل(&scale, (اعشاری *) context.get_output_op_params() + 0, حجم(اعشاری));
    حافظہ_نقل(&bias, (اعشاری *) context.get_output_op_params() + 1, حجم(اعشاری));

    خودکار scale_گرہ = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<اعشاری>{scale});

    اگر (context.get_op_case() == 2) {
        واپس {context.get_داخلہ(0)};
    }

    اگر (context.get_op_case() == 1 && context.has_داخلہ("cache_rs_reset_len")) {
        خودکار cache_rs_reset_idx = context.get_داخلہ("cache_rs_reset_idx");
        خودکار cache_rs_reset_len = context.get_داخلہ("cache_rs_reset_len");

        خودکار cache_rs = context.get_داخلہ(0);

        خودکار cache_shape = std::make_shared<ov::op::v3::Shape_کا>(cache_rs, ov::element::i64);
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
        واپس rename_outputs_with_suffix({cleared_cache_rs}, context.get_name());
    }

    خودکار scaled = std::make_shared<ov::op::v1::Multiply>(context.get_داخلہ(0), scale_گرہ);

    std::shared_ptr<ov::Node> res;
    اگر (bias != 0.0f) {
        خودکار bias_گرہ =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<اعشاری>{bias});
        res = std::make_shared<ov::op::v1::جوڑو>(scaled, bias_گرہ);
    } ورنہ {
        res = scaled;
    }

    واپس rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
