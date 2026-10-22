#अन्तर्भाव "../node_context.h"
#अन्तर्भाव "../op_table.h"
#अन्तर्भाव "../utils.h"

#अन्तर्भाव <memory>
#अन्तर्भाव <openvino/frontend/exception.hpp>
#अन्तर्भाव <openvino/op/concat.hpp>
#अन्तर्भाव <openvino/op/convert.hpp>

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {
नामाकाश op {

Output_सदिशम् translate_concat(ध्रुव ग्रन्थिः_Context & context) {
    अङ्कः_inputs_check(context, 2, 2);

    ध्रुव int32_t * op_params = context.get_output_op_params();
    FRONT_अन्तः_CHECK_IMPLEMENTED(op_params != शून्यसूचकः, "CONCAT requires output op params");

    ध्रुव स्वचालित output_shape = context.get_output_shape();
    FRONT_अन्तः_CHECK_IMPLEMENTED(output_shape.rank().is_static(), "CONCAT requires static output rank");

    ध्रुव स्वचालित rank = output_shape.rank().get_दीर्घता();
    ध्रुव int32_t ggml_dim = op_params[0];
    FRONT_अन्तः_CHECK_IMPLEMENTED(ggml_dim >= 0 && ggml_dim < rank, "CONCAT axis is out of range");

    स्वचालित निवेशः_0 = process_view_निवेशः_new(context, 0);
    स्वचालित निवेशः_1 = process_view_निवेशः_new(context, 1);
    ध्रुव स्वचालित output_type = context.get_output_type();

    यदि (निवेशः_0.get_element_type() != output_type) {
        निवेशः_0 = std::make_shared<ov::op::v0::Convert>(निवेशः_0, output_type);
    }
    यदि (निवेशः_1.get_element_type() != output_type) {
        निवेशः_1 = std::make_shared<ov::op::v0::Convert>(निवेशः_1, output_type);
    }

    ध्रुव स्वचालित axis = static_cast<int64_t>(rank - 1 - ggml_dim);
    स्वचालित res = std::make_shared<ov::op::v0::Concat>(Output_सदिशम्{निवेशः_0, निवेशः_1}, axis);

    प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
