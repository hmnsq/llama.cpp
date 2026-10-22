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

فضاینام ov {
فضاینام frontend {
فضاینام ggml {
فضاینام op {

OutputVector translate_scale(ثابت گره_Context & context) {
    رقم_inputs_check(context, 1, 1);

    اعشاری scale;
    اعشاری bias;
    memcpy(&scale, (اعشاری *) context.get_output_op_params() + 0, اندازه(اعشاری));
    memcpy(&bias, (اعشاری *) context.get_output_op_params() + 1, اندازه(اعشاری));

    auto scale_گره = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<اعشاری>{scale});

    اگر (context.get_op_case() == 2) {
        بازگشت {context.get_درون_داد(0)};
    }

    اگر (context.get_op_case() == 1 && context.has_درون_داد("cache_rs_reset_len")) {
        auto cache_rs_reset_idx = context.get_درون_داد("cache_rs_reset_idx");
        auto cache_rs_reset_len = context.get_درون_داد("cache_rs_reset_len");

        auto cache_rs = context.get_درون_داد(0);

        auto cache_shape = std::make_shared<ov::op::v3::ShapeOf>(cache_rs, ov::element::i64);
        auto n_slots_1d = std::make_shared<ov::op::v8::Gather>(
            cache_shape, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2}),
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0}));
        auto n_slots = std::make_shared<ov::op::v0::Squeeze>(n_slots_1d);

        auto iota = std::make_shared<ov::op::v4::Range>(
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0}), n_slots,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1}), ov::element::i64);

        auto idx_plus_len = std::make_shared<ov::op::v1::Add>(cache_rs_reset_idx, cache_rs_reset_len);
        auto less_than_idx = std::make_shared<ov::op::v1::Less>(iota, cache_rs_reset_idx);
        auto greater_equal_idx_plus_len = std::make_shared<ov::op::v1::GreaterEqual>(iota, idx_plus_len);
        auto keep_mask = std::make_shared<ov::op::v1::LogicalOr>(less_than_idx, greater_equal_idx_plus_len);

        auto keep_mask_f32 = std::make_shared<ov::op::v0::Convert>(keep_mask, ov::element::f32);
        auto keep_mask_reshape = std::make_shared<ov::op::v0::Unsqueeze>(
            keep_mask_f32, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1}));

        auto cleared_cache_rs = std::make_shared<ov::op::v1::Multiply>(cache_rs, keep_mask_reshape);
        بازگشت rename_outputs_with_suffix({cleared_cache_rs}, context.get_name());
    }

    auto scaled = std::make_shared<ov::op::v1::Multiply>(context.get_درون_داد(0), scale_گره);

    std::shared_ptr<ov::Node> res;
    اگر (bias != 0.0f) {
        auto bias_گره =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<اعشاری>{bias});
        res = std::make_shared<ov::op::v1::Add>(scaled, bias_گره);
    } وگرنه {
        res = scaled;
    }

    بازگشت rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
