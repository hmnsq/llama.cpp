#अन्तर्भाव "../node_context.h"
#अन्तर्भाव "../op_table.h"
#अन्तर्भाव "../utils.h"

#अन्तर्भाव <cstdint>
#अन्तर्भाव <memory>
#अन्तर्भाव <openvino/core/node.hpp>
#अन्तर्भाव <openvino/core/node_output.hpp>
#अन्तर्भाव <openvino/frontend/exception.hpp>
#अन्तर्भाव <openvino/op/concat.hpp>
#अन्तर्भाव <openvino/op/constant.hpp>
#अन्तर्भाव <openvino/op/reshape.hpp>
#अन्तर्भाव <vector>

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {
नामाकाश op {

Output_सदिशम् translate_reshape(ध्रुव ग्रन्थिः_Context & context) {
    अङ्कः_inputs_check(context, 1, 1);
    यदि (context.get_निवेशः(0).get_partial_shape().is_static() &&
        context.get_निवेशः_shape(0) == context.get_output_shape()) {
        प्रत्यागम {context.get_निवेशः(0)};
    }

    ध्रुव पूर्णाङ्क op_case = context.get_op_case();
    ध्रुव स्वचालित output_shape = context.get_output_shape().गन्तव्यम्_shape();
    std::vector<int64_t> shape(output_shape.begin(), output_shape.end());
    std::shared_ptr<ov::Node> new_shape_ग्रन्थिः;
    विकल्प (op_case) {
    पक्ष 0:
        भङ्ग;
    पक्ष 1:
    पक्ष 9:
        shape[1] = -1;
        यदि (context.is_stateful() && op_case == 1) {
            shape.erase(shape.begin());
        }
        भङ्ग;
    पक्ष 2:
    पक्ष 3:
        shape[2] = -1;
        यदि (op_case == 3) {
            shape[3] = 1;
        }
        भङ्ग;
    पक्ष 4:
        प्रत्यागम {context.get_निवेशः(0).get_ग्रन्थिः_shared_ptr()->निवेशः_मानम्(0)};
    पक्ष 5:
    पक्ष 7:
        shape = {1, 1, -1, shape[3]};
        यदि (context.is_stateful() && op_case == 5) {
            shape.erase(shape.begin());
        }
        भङ्ग;
    पक्ष 6:
        // Recurrent inputs keep the active sequence count separate from the token count.
        यदि (context.has_निवेशः("s_copy_active_slot_len")) {
            स्वचालित n_slot_सक्रिय_len = context.get_निवेशः("s_copy_active_slot_len");
            स्वचालित emb_आकार = ov::op::v0::Constant::create(ov::element::i64, {1}, {shape[3]});
            स्वचालित one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
            स्वचालित neg_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
            new_shape_ग्रन्थिः =
                std::make_shared<ov::op::v0::Concat>(ov::Output_सदिशम्{one, n_slot_सक्रिय_len, neg_one, emb_आकार}, 0);
        } अन्यथा {
            shape = {1, 1, -1, shape[3]};
        }
        भङ्ग;
    पक्ष 8:
        shape[0] = -1;
        भङ्ग;
    स्वभाव:
        FRONT_अन्तः_OP_CONVERSION_CHECK(असत्य, "Unsupported RESHAPE case: ", op_case);
    }
    यदि (!new_shape_ग्रन्थिः) {
        new_shape_ग्रन्थिः = ov::op::v0::Constant::create(ov::element::i64, {shape.size()}, shape);
    }
    स्वचालित res = std::make_shared<ov::op::v1::Reshape>(context.get_निवेशः(0), new_shape_ग्रन्थिः, असत्य);
    प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
