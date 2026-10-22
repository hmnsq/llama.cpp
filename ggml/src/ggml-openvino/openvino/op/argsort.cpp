#अन्तर्भाव "../node_context.h"
#अन्तर्भाव "../op_table.h"
#अन्तर्भाव "../utils.h"
#अन्तर्भाव "ggml.h"

#अन्तर्भाव <openvino/frontend/exception.hpp>
#अन्तर्भाव <openvino/op/constant.hpp>
#अन्तर्भाव <openvino/op/squeeze.hpp>
#अन्तर्भाव <openvino/op/topk.hpp>

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {
नामाकाश op {

Output_सदिशम् translate_argsort(ध्रुव ग्रन्थिः_Context & context) {
    अङ्कः_inputs_check(context, 1, 1);

    स्वचालित निवेशः = process_view_निवेशः_new(context, 0);

    ध्रुव int32_t क्रमः = context.get_output_op_params()[0];

    ov::op::v11::शिखरम्_K::बहुलकम्_2 mode;
    विकल्प (क्रमः) {
    पक्ष GGML_क्रमय_क्रमः_ASC:
        mode = ov::op::v11::शिखरम्_K::बहुलकम्_2::MIN;
        भङ्ग;
    पक्ष GGML_क्रमय_क्रमः_DESC:
        mode = ov::op::v11::शिखरम्_K::बहुलकम्_2::MAX;
        भङ्ग;
    स्वभाव:
        FRONT_अन्तः_OP_CONVERSION_CHECK(असत्य, "Unsupported GGML_OP_ARGSORT order: ", क्रमः);
    }

    // Stateful models drop the leading size-1 batch dim, so the expert axis is 2 there
    // instead of 3 (same rank-3-vs-rank-4 split as get_rows.cpp / process_view_input).
    ध्रुव पूर्णाङ्क axis = (context.is_stateful() && निवेशः.get_partial_shape().rank() == 3) ? 2 : 3;

    स्वचालित k = std::make_shared<ov::op::v0::Squeeze>(get_dimensions(निवेशः.get_ग्रन्थिः_shared_ptr(), {axis}),
                                                   ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));

    स्वचालित topk = std::make_shared<ov::op::v11::शिखरम्_K>(निवेशः, k, axis, mode, ov::op::v11::शिखरम्_K::क्रमय_Type::क्रमय_मानानि,
                                                    context.get_output_type(), असत्य);

    प्रत्यागम rename_outputs_with_suffix({topk->output(1)}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
