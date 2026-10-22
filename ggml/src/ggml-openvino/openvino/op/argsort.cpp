#شامل "../node_context.h"
#شامل "../op_table.h"
#شامل "../utils.h"
#شامل "ggml.h"

#شامل <openvino/frontend/exception.hpp>
#شامل <openvino/op/constant.hpp>
#شامل <openvino/op/squeeze.hpp>
#شامل <openvino/op/topk.hpp>

نامخلا ov {
نامخلا frontend {
نامخلا ggml {
نامخلا op {

Output_سمتیہ translate_argsort(ثابت گرہ_Context & context) {
    عدد_inputs_check(context, 1, 1);

    خودکار داخلہ = process_view_داخلہ_new(context, 0);

    ثابت int32_t ترتیب = context.get_output_op_params()[0];

    ov::op::v11::چوٹی_K::منوال_2 mode;
    انتخاب (ترتیب) {
    حالت GGML_ترتیب_دو_ترتیب_ASC:
        mode = ov::op::v11::چوٹی_K::منوال_2::MIN;
        توڑو;
    حالت GGML_ترتیب_دو_ترتیب_DESC:
        mode = ov::op::v11::چوٹی_K::منوال_2::MAX;
        توڑو;
    طےشدہ:
        FRONT_انتہا_OP_CONVERSION_CHECK(جھوٹ, "Unsupported GGML_OP_ARGSORT order: ", ترتیب);
    }

    // Stateful models drop the leading size-1 batch dim, so the expert axis is 2 there
    // instead of 3 (same rank-3-vs-rank-4 split as get_rows.cpp / process_view_input).
    ثابت صحیح axis = (context.is_stateful() && داخلہ.get_partial_shape().rank() == 3) ? 2 : 3;

    خودکار k = std::make_shared<ov::op::v0::Squeeze>(get_dimensions(داخلہ.get_گرہ_shared_ptr(), {axis}),
                                                   ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));

    خودکار topk = std::make_shared<ov::op::v11::چوٹی_K>(داخلہ, k, axis, mode, ov::op::v11::چوٹی_K::ترتیب_دو_Type::ترتیب_دو_اقدار,
                                                    context.get_output_type(), جھوٹ);

    واپس rename_outputs_with_suffix({topk->output(1)}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
