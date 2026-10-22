#ଅନ୍ତର୍ଭାଵ "../node_context.h"
#ଅନ୍ତର୍ଭାଵ "../op_table.h"
#ଅନ୍ତର୍ଭାଵ "../utils.h"

#ଅନ୍ତର୍ଭାଵ <openvino/core/except.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/add.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/concat.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/constant.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/convert.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/equal.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/gather.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/greater_eq.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/if.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/less.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/logical_or.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/multiply.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/range.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/reshape.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/shape_of.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/slice.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/squeeze.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/unsqueeze.hpp>
#ଅନ୍ତର୍ଭାଵ <vector>

ନାମାକାଶ ov {
ନାମାକାଶ frontend {
ନାମାକାଶ ggml {
ନାମାକାଶ op {

OutputVector translate_scale(ଅଚଳ ଗଣ୍ଠି_Context & context) {
    ଅଙ୍କ_inputs_check(context, 1, 1);

    ଦଶମିକ scale;
    ଦଶମିକ bias;
    memcpy(&scale, (ଦଶମିକ *) context.get_output_op_params() + 0, ଆକାର(ଦଶମିକ));
    memcpy(&bias, (ଦଶମିକ *) context.get_output_op_params() + 1, ଆକାର(ଦଶମିକ));

    auto scale_ଗଣ୍ଠି = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<ଦଶମିକ>{scale});

    ଯଦି (context.get_op_case() == 2) {
        ଫେରନ୍ତୁ {context.get_ନିବେଶ(0)};
    }

    ଯଦି (context.get_op_case() == 1 && context.has_ନିବେଶ("cache_rs_reset_len")) {
        auto cache_rs_reset_idx = context.get_ନିବେଶ("cache_rs_reset_idx");
        auto cache_rs_reset_len = context.get_ନିବେଶ("cache_rs_reset_len");

        auto cache_rs = context.get_ନିବେଶ(0);

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
        ଫେରନ୍ତୁ rename_outputs_with_suffix({cleared_cache_rs}, context.get_name());
    }

    auto scaled = std::make_shared<ov::op::v1::Multiply>(context.get_ନିବେଶ(0), scale_ଗଣ୍ଠି);

    std::shared_ptr<ov::Node> res;
    ଯଦି (bias != 0.0f) {
        auto bias_ଗଣ୍ଠି =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<ଦଶମିକ>{bias});
        res = std::make_shared<ov::op::v1::Add>(scaled, bias_ଗଣ୍ଠି);
    } ଅନ୍ୟଥା {
        res = scaled;
    }

    ଫେରନ୍ତୁ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
