#અન્તર્ભાવ "../node_context.h"
#અન્તર્ભાવ "../op_table.h"
#અન્તર્ભાવ "../utils.h"

#અન્તર્ભાવ <openvino/core/except.hpp>
#અન્તર્ભાવ <openvino/op/add.hpp>
#અન્તર્ભાવ <openvino/op/concat.hpp>
#અન્તર્ભાવ <openvino/op/constant.hpp>
#અન્તર્ભાવ <openvino/op/convert.hpp>
#અન્તર્ભાવ <openvino/op/equal.hpp>
#અન્તર્ભાવ <openvino/op/gather.hpp>
#અન્તર્ભાવ <openvino/op/greater_eq.hpp>
#અન્તર્ભાવ <openvino/op/if.hpp>
#અન્તર્ભાવ <openvino/op/less.hpp>
#અન્તર્ભાવ <openvino/op/logical_or.hpp>
#અન્તર્ભાવ <openvino/op/multiply.hpp>
#અન્તર્ભાવ <openvino/op/range.hpp>
#અન્તર્ભાવ <openvino/op/reshape.hpp>
#અન્તર્ભાવ <openvino/op/shape_of.hpp>
#અન્તર્ભાવ <openvino/op/slice.hpp>
#અન્તર્ભાવ <openvino/op/squeeze.hpp>
#અન્તર્ભાવ <openvino/op/unsqueeze.hpp>
#અન્તર્ભાવ <vector>

નામાકાશ ov {
નામાકાશ frontend {
નામાકાશ ggml {
નામાકાશ op {

OutputVector translate_scale(અચલ ગાંઠ_Context & context) {
    અંક_inputs_check(context, 1, 1);

    દશાંશ scale;
    દશાંશ bias;
    memcpy(&scale, (દશાંશ *) context.get_output_op_params() + 0, કદ(દશાંશ));
    memcpy(&bias, (દશાંશ *) context.get_output_op_params() + 1, કદ(દશાંશ));

    auto scale_ગાંઠ = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<દશાંશ>{scale});

    જો (context.get_op_case() == 2) {
        પાછા {context.get_આગત(0)};
    }

    જો (context.get_op_case() == 1 && context.has_આગત("cache_rs_reset_len")) {
        auto cache_rs_reset_idx = context.get_આગત("cache_rs_reset_idx");
        auto cache_rs_reset_len = context.get_આગત("cache_rs_reset_len");

        auto cache_rs = context.get_આગત(0);

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
        પાછા rename_outputs_with_suffix({cleared_cache_rs}, context.get_name());
    }

    auto scaled = std::make_shared<ov::op::v1::Multiply>(context.get_આગત(0), scale_ગાંઠ);

    std::shared_ptr<ov::Node> res;
    જો (bias != 0.0f) {
        auto bias_ગાંઠ =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<દશાંશ>{bias});
        res = std::make_shared<ov::op::v1::Add>(scaled, bias_ગાંઠ);
    } નહીંતર {
        res = scaled;
    }

    પાછા rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
