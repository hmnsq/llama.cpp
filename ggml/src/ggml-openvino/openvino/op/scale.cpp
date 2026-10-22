#অন্তর্ভাৱ "../node_context.h"
#অন্তর্ভাৱ "../op_table.h"
#অন্তর্ভাৱ "../utils.h"

#অন্তর্ভাৱ <openvino/core/except.hpp>
#অন্তর্ভাৱ <openvino/op/add.hpp>
#অন্তর্ভাৱ <openvino/op/concat.hpp>
#অন্তর্ভাৱ <openvino/op/constant.hpp>
#অন্তর্ভাৱ <openvino/op/convert.hpp>
#অন্তর্ভাৱ <openvino/op/equal.hpp>
#অন্তর্ভাৱ <openvino/op/gather.hpp>
#অন্তর্ভাৱ <openvino/op/greater_eq.hpp>
#অন্তর্ভাৱ <openvino/op/if.hpp>
#অন্তর্ভাৱ <openvino/op/less.hpp>
#অন্তর্ভাৱ <openvino/op/logical_or.hpp>
#অন্তর্ভাৱ <openvino/op/multiply.hpp>
#অন্তর্ভাৱ <openvino/op/range.hpp>
#অন্তর্ভাৱ <openvino/op/reshape.hpp>
#অন্তর্ভাৱ <openvino/op/shape_of.hpp>
#অন্তর্ভাৱ <openvino/op/slice.hpp>
#অন্তর্ভাৱ <openvino/op/squeeze.hpp>
#অন্তর্ভাৱ <openvino/op/unsqueeze.hpp>
#অন্তর্ভাৱ <vector>

নামাকাশ ov {
নামাকাশ frontend {
নামাকাশ ggml {
নামাকাশ op {

OutputVector translate_scale(ধ্রুৱ গ্ৰন্থি_Context & context) {
    অঙ্ক_inputs_check(context, 1, 1);

    ভিন্ন scale;
    ভিন্ন bias;
    memcpy(&scale, (ভিন্ন *) context.get_output_op_params() + 0, পরিমাণ(ভিন্ন));
    memcpy(&bias, (ভিন্ন *) context.get_output_op_params() + 1, পরিমাণ(ভিন্ন));

    auto scale_গ্ৰন্থি = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<ভিন্ন>{scale});

    যদি (context.get_op_case() == 2) {
        ঘূৰাই {context.get_প্ৰবেশ(0)};
    }

    যদি (context.get_op_case() == 1 && context.has_প্ৰবেশ("cache_rs_reset_len")) {
        auto cache_rs_reset_idx = context.get_প্ৰবেশ("cache_rs_reset_idx");
        auto cache_rs_reset_len = context.get_প্ৰবেশ("cache_rs_reset_len");

        auto cache_rs = context.get_প্ৰবেশ(0);

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
        ঘূৰাই rename_outputs_with_suffix({cleared_cache_rs}, context.get_name());
    }

    auto scaled = std::make_shared<ov::op::v1::Multiply>(context.get_প্ৰবেশ(0), scale_গ্ৰন্থি);

    std::shared_ptr<ov::Node> res;
    যদি (bias != 0.0f) {
        auto bias_গ্ৰন্থি =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<ভিন্ন>{bias});
        res = std::make_shared<ov::op::v1::Add>(scaled, bias_গ্ৰন্থি);
    } অন্যথা {
        res = scaled;
    }

    ঘূৰাই rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
