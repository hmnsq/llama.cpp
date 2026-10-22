#ଅନ୍ତର୍ଭାଵ "../node_context.h"
#ଅନ୍ତର୍ଭାଵ "../op_table.h"
#ଅନ୍ତର୍ଭାଵ "../utils.h"
#ଅନ୍ତର୍ଭାଵ "ggml.h"

#ଅନ୍ତର୍ଭାଵ <openvino/frontend/exception.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/constant.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/squeeze.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/topk.hpp>

ନାମାକାଶ ov {
ନାମାକାଶ frontend {
ନାମାକାଶ ggml {
ନାମାକାଶ op {

OutputVector translate_argsort(ଅଚଳ ଗଣ୍ଠି_Context & context) {
    ଅଙ୍କ_inputs_check(context, 1, 1);

    auto ନିବେଶ = process_view_ନିବେଶ_new(context, 0);

    ଅଚଳ int32_t order = context.get_output_op_params()[0];

    ov::op::v11::ଶୀର୍ଷ_K::Mode mode;
    ବାଛ (order) {
    ସ୍ଥିତି GGML_SORT_ORDER_ASC:
        mode = ov::op::v11::ଶୀର୍ଷ_K::Mode::MIN;
        ଭାଙ୍ଗ;
    ସ୍ଥିତି GGML_SORT_ORDER_DESC:
        mode = ov::op::v11::ଶୀର୍ଷ_K::Mode::MAX;
        ଭାଙ୍ଗ;
    ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
        FRONT_END_OP_CONVERSION_CHECK(ମିଥ୍ୟା, "Unsupported GGML_OP_ARGSORT order: ", order);
    }

    // Stateful models drop the leading size-1 batch dim, so the expert axis is 2 there
    // instead of 3 (same rank-3-vs-rank-4 split as get_rows.cpp / process_view_input).
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା axis = (context.is_stateful() && ନିବେଶ.get_partial_shape().rank() == 3) ? 2 : 3;

    auto k = std::make_shared<ov::op::v0::Squeeze>(get_dimensions(ନିବେଶ.get_ଗଣ୍ଠି_shared_ptr(), {axis}),
                                                   ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));

    auto topk = std::make_shared<ov::op::v11::ଶୀର୍ଷ_K>(ନିବେଶ, k, axis, mode, ov::op::v11::ଶୀର୍ଷ_K::SortType::SORT_ମୂଲ୍ୟଗୁଡ଼ିକ,
                                                    context.get_output_type(), ମିଥ୍ୟା);

    ଫେରନ୍ତୁ rename_outputs_with_suffix({topk->output(1)}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
