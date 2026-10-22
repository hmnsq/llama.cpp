#अन्तर्भाव "../node_context.h"
#अन्तर्भाव "../op_table.h"
#अन्तर्भाव "../utils.h"

#अन्तर्भाव <openvino/op/avg_pool.hpp>
#अन्तर्भाव <openvino/op/max_pool.hpp>
#अन्तर्भाव <openvino/op/convert.hpp>

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {
नामाकाश op {

OutputVector translate_pool_2d(ध्रुव गाँठ_Context & context) {
    अंक_inputs_check(context, 1, 1);
    ध्रुव int32_t * params = context.get_output_op_params();

    ध्रुव पूर्णाङ्क k0 = params[1];
    ध्रुव पूर्णाङ्क k1 = params[2];
    ध्रुव पूर्णाङ्क s0 = params[3];
    ध्रुव पूर्णाङ्क s1 = params[4];
    ध्रुव पूर्णाङ्क p0 = params[5];
    ध्रुव पूर्णाङ्क p1 = params[6];

    ध्रुव पूर्णाङ्क op_case = context.get_op_case();
    ov::Output<Node> निवेश = context.get_निवेश(0);
    ov::Strides strides{static_cast<माप_प्रकार>(s1), static_cast<माप_प्रकार>(s0)};
    ov::Shape pads_begin{static_cast<माप_प्रकार>(p1), static_cast<माप_प्रकार>(p0)};
    ov::Shape pads_end{static_cast<माप_प्रकार>(p1), static_cast<माप_प्रकार>(p0)};
    ov::Shape kernel{static_cast<माप_प्रकार>(k1), static_cast<माप_प्रकार>(k0)};
    ov::Output<Node> res;

    विकल्प (op_case) {
    पक्ष 1:  // GGML_OP_POOL_MAX
    {
        res = std::make_shared<ov::op::v1::MaxPool>(निवेश, strides, pads_begin, pads_end, kernel);
        भङ्ग;
    }
    पक्ष 2:  // GGML_OP_POOL_AVG
    {
        res = std::make_shared<ov::op::v1::AvgPool>(निवेश, strides, pads_begin, pads_end, kernel, असत्य);
        भङ्ग;
    }
    स्वभाव:
        भङ्ग;
    }
    प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
