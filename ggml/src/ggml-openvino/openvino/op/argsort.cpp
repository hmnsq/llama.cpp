#ضمن "../node_context.h"
#ضمن "../op_table.h"
#ضمن "../utils.h"
#ضمن "ggml.h"

#ضمن <openvino/frontend/exception.hpp>
#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/squeeze.hpp>
#ضمن <openvino/op/topk.hpp>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق op {

Output_ويڪٽر translate_argsort(ثابت ڳنڍ_Context & context) {
    انگ_inputs_check(context, 1, 1);

    خودڪار داخلا_2 = process_view_داخلا_new(context, 0);

    ثابت int32_t ترتيب = context.get_output_op_params()[0];

    ov::op::v11::چوٽي_K::گهڻائي_2 mode;
    بدل (ترتيب) {
    حالة GGML_ترتيب_ڏيو_ترتيب_ASC:
        mode = ov::op::v11::چوٽي_K::گهڻائي_2::MIN;
        اكسر;
    حالة GGML_ترتيب_ڏيو_ترتيب_DESC:
        mode = ov::op::v11::چوٽي_K::گهڻائي_2::MAX;
        اكسر;
    افتراضي:
        FRONT_پڇاڙي_OP_CONVERSION_CHECK(خطأ, "Unsupported GGML_OP_ARGSORT order: ", ترتيب);
    }

    // Stateful models drop the leading size-1 batch dim, so the expert axis is 2 there
    // instead of 3 (same rank-3-vs-rank-4 split as get_rows.cpp / process_view_input).
    ثابت صحيح axis = (context.is_stateful() && داخلا_2.get_partial_shape().rank() == 3) ? 2 : 3;

    خودڪار k = std::make_shared<ov::op::v0::Squeeze>(get_dimensions(داخلا_2.get_ڳنڍ_shared_ptr(), {axis}),
                                                   ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));

    خودڪار topk = std::make_shared<ov::op::v11::چوٽي_K>(داخلا_2, k, axis, mode, ov::op::v11::چوٽي_K::ترتيب_ڏيو_Type::ترتيب_ڏيو_قدرون,
                                                    context.get_output_type(), خطأ);

    أرجع rename_outputs_with_suffix({topk->output(1)}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
