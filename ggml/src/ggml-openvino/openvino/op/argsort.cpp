#অন্তর্ভাব "../node_context.h"
#অন্তর্ভাব "../op_table.h"
#অন্তর্ভাব "../utils.h"
#অন্তর্ভাব "ggml.h"

#অন্তর্ভাব <openvino/frontend/exception.hpp>
#অন্তর্ভাব <openvino/op/constant.hpp>
#অন্তর্ভাব <openvino/op/squeeze.hpp>
#অন্তর্ভাব <openvino/op/topk.hpp>

নামাকাশ ov {
নামাকাশ frontend {
নামাকাশ ggml {
নামাকাশ op {

OutputVector translate_argsort(ধ্রুব গ্রন্থি_Context & context) {
    অঙ্ক_inputs_check(context, 1, 1);

    auto প্রবেশ = process_view_প্রবেশ_new(context, 0);

    ধ্রুব int32_t order = context.get_output_op_params()[0];

    ov::op::v11::শীর্ষ_K::Mode mode;
    বিকল্প (order) {
    পক্ষ GGML_SORT_ORDER_ASC:
        mode = ov::op::v11::শীর্ষ_K::Mode::MIN;
        ভঙ্গ;
    পক্ষ GGML_SORT_ORDER_DESC:
        mode = ov::op::v11::শীর্ষ_K::Mode::MAX;
        ভঙ্গ;
    স্বভাব:
        FRONT_END_OP_CONVERSION_CHECK(অসত্য, "Unsupported GGML_OP_ARGSORT order: ", order);
    }

    // Stateful models drop the leading size-1 batch dim, so the expert axis is 2 there
    // instead of 3 (same rank-3-vs-rank-4 split as get_rows.cpp / process_view_input).
    ধ্রুব পূর্ণাঙ্ক axis = (context.is_stateful() && প্রবেশ.get_partial_shape().rank() == 3) ? 2 : 3;

    auto k = std::make_shared<ov::op::v0::Squeeze>(get_dimensions(প্রবেশ.get_গ্রন্থি_shared_ptr(), {axis}),
                                                   ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));

    auto topk = std::make_shared<ov::op::v11::শীর্ষ_K>(প্রবেশ, k, axis, mode, ov::op::v11::শীর্ষ_K::SortType::SORT_মানসমূহ,
                                                    context.get_output_type(), অসত্য);

    প্রত্যাগম rename_outputs_with_suffix({topk->output(1)}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
