#அந்தர்பாவ "../node_context.h"
#அந்தர்பாவ "../op_table.h"
#அந்தர்பாவ "../utils.h"

#அந்தர்பாவ <openvino/core/except.hpp>
#அந்தர்பாவ <openvino/op/add.hpp>
#அந்தர்பாவ <openvino/op/concat.hpp>
#அந்தர்பாவ <openvino/op/constant.hpp>
#அந்தர்பாவ <openvino/op/convert.hpp>
#அந்தர்பாவ <openvino/op/equal.hpp>
#அந்தர்பாவ <openvino/op/gather.hpp>
#அந்தர்பாவ <openvino/op/greater_eq.hpp>
#அந்தர்பாவ <openvino/op/if.hpp>
#அந்தர்பாவ <openvino/op/less.hpp>
#அந்தர்பாவ <openvino/op/logical_or.hpp>
#அந்தர்பாவ <openvino/op/multiply.hpp>
#அந்தர்பாவ <openvino/op/range.hpp>
#அந்தர்பாவ <openvino/op/reshape.hpp>
#அந்தர்பாவ <openvino/op/shape_of.hpp>
#அந்தர்பாவ <openvino/op/slice.hpp>
#அந்தர்பாவ <openvino/op/squeeze.hpp>
#அந்தர்பாவ <openvino/op/unsqueeze.hpp>
#அந்தர்பாவ <vector>

நாமாகாஶ ov {
நாமாகாஶ frontend {
நாமாகாஶ ggml {
நாமாகாஶ op {

OutputVector translate_scale(மாறிலி முனை_Context & context) {
    இலக்கம்_inputs_check(context, 1, 1);

    தசம scale;
    தசம bias;
    memcpy(&scale, (தசம *) context.get_output_op_params() + 0, அளவு(தசம));
    memcpy(&bias, (தசம *) context.get_output_op_params() + 1, அளவு(தசம));

    auto scale_முனை = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<தசம>{scale});

    எனில் (context.get_op_case() == 2) {
        திரும்பு {context.get_உள்ளீடு(0)};
    }

    எனில் (context.get_op_case() == 1 && context.has_உள்ளீடு("cache_rs_reset_len")) {
        auto cache_rs_reset_idx = context.get_உள்ளீடு("cache_rs_reset_idx");
        auto cache_rs_reset_len = context.get_உள்ளீடு("cache_rs_reset_len");

        auto cache_rs = context.get_உள்ளீடு(0);

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
        திரும்பு rename_outputs_with_suffix({cleared_cache_rs}, context.get_name());
    }

    auto scaled = std::make_shared<ov::op::v1::Multiply>(context.get_உள்ளீடு(0), scale_முனை);

    std::shared_ptr<ov::Node> res;
    எனில் (bias != 0.0f) {
        auto bias_முனை =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<தசம>{bias});
        res = std::make_shared<ov::op::v1::Add>(scaled, bias_முனை);
    } இல்லையெனில் {
        res = scaled;
    }

    திரும்பு rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
