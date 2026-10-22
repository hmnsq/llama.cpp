#অন্তর্ভাৱ "../node_context.h"
#অন্তর্ভাৱ "../op_table.h"
#অন্তর্ভাৱ "../utils.h"
#অন্তর্ভাৱ "ggml.h"

#অন্তর্ভাৱ <openvino/frontend/exception.hpp>
#অন্তর্ভাৱ <openvino/op/constant.hpp>
#অন্তর্ভাৱ <openvino/op/squeeze.hpp>
#অন্তর্ভাৱ <openvino/op/topk.hpp>

নামাকাশ ov {
নামাকাশ frontend {
নামাকাশ ggml {
নামাকাশ op {

OutputVector translate_argsort(ধ্রুৱ গ্ৰন্থি_Context & context) {
    অঙ্ক_inputs_check(context, 1, 1);

    auto প্ৰবেশ = process_view_প্ৰবেশ_new(context, 0);

    ধ্রুৱ int32_t order = context.get_output_op_params()[0];

    ov::op::v11::শীৰ্ষ_K::Mode mode;
    ৱিকল্প (order) {
    পক্ষ GGML_SORT_ORDER_ASC:
        mode = ov::op::v11::শীৰ্ষ_K::Mode::MIN;
        ভঙ্গ;
    পক্ষ GGML_SORT_ORDER_DESC:
        mode = ov::op::v11::শীৰ্ষ_K::Mode::MAX;
        ভঙ্গ;
    স্ৱভাৱ:
        FRONT_END_OP_CONVERSION_CHECK(মিছা, "Unsupported GGML_OP_ARGSORT order: ", order);
    }

    // Stateful models drop the leading size-1 batch dim, so the expert axis is 2 there
    // instead of 3 (same rank-3-vs-rank-4 split as get_rows.cpp / process_view_input).
    ধ্রুৱ পূর্ণাঙ্ক axis = (context.is_stateful() && প্ৰবেশ.get_partial_shape().rank() == 3) ? 2 : 3;

    auto k = std::make_shared<ov::op::v0::Squeeze>(get_dimensions(প্ৰবেশ.get_গ্ৰন্থি_shared_ptr(), {axis}),
                                                   ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));

    auto topk = std::make_shared<ov::op::v11::শীৰ্ষ_K>(প্ৰবেশ, k, axis, mode, ov::op::v11::শীৰ্ষ_K::SortType::SORT_মানসমূহ,
                                                    context.get_output_type(), মিছা);

    ঘূৰাই rename_outputs_with_suffix({topk->output(1)}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
