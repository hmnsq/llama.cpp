#समावेश "../node_context.h"
#समावेश "../op_table.h"
#समावेश "../utils.h"

#समावेश <cstdint>
#समावेश <memory>
#समावेश <openvino/core/node.hpp>
#समावेश <openvino/core/node_output.hpp>
#समावेश <openvino/frontend/exception.hpp>
#समावेश <openvino/op/concat.hpp>
#समावेश <openvino/op/constant.hpp>
#समावेश <openvino/op/reshape.hpp>
#समावेश <vector>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_reshape(स्थिर गाँठ_Context & context) {
    अंक_inputs_check(context, 1, 1);
    अगर (context.get_निवेश(0).get_partial_shape().is_static() &&
        context.get_निवेश_shape(0) == context.get_output_shape()) {
        वापस {context.get_निवेश(0)};
    }

    स्थिर पूर्णांक op_case = context.get_op_case();
    स्थिर स्वतः output_shape = context.get_output_shape().to_shape();
    std::vector<int64_t> shape(output_shape.begin(), output_shape.end());
    std::shared_ptr<ov::Node> new_shape_गाँठ;
    चयन (op_case) {
    हाल 0:
        अवरोध;
    हाल 1:
    हाल 9:
        shape[1] = -1;
        अगर (context.is_stateful() && op_case == 1) {
            shape.erase(shape.begin());
        }
        अवरोध;
    हाल 2:
    हाल 3:
        shape[2] = -1;
        अगर (op_case == 3) {
            shape[3] = 1;
        }
        अवरोध;
    हाल 4:
        वापस {context.get_निवेश(0).get_गाँठ_shared_ptr()->निवेश_मूल्य(0)};
    हाल 5:
    हाल 7:
        shape = {1, 1, -1, shape[3]};
        अगर (context.is_stateful() && op_case == 5) {
            shape.erase(shape.begin());
        }
        अवरोध;
    हाल 6:
        // Recurrent inputs keep the active sequence count separate from the token count.
        अगर (context.has_निवेश("s_copy_active_slot_len")) {
            स्वतः n_slot_active_len = context.get_निवेश("s_copy_active_slot_len");
            स्वतः emb_आकार = ov::op::v0::Constant::create(ov::element::i64, {1}, {shape[3]});
            स्वतः one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
            स्वतः neg_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
            new_shape_गाँठ =
                std::make_shared<ov::op::v0::Concat>(ov::OutputVector{one, n_slot_active_len, neg_one, emb_आकार}, 0);
        } अन्यथा {
            shape = {1, 1, -1, shape[3]};
        }
        अवरोध;
    हाल 8:
        shape[0] = -1;
        अवरोध;
    शेष:
        FRONT_END_OP_CONVERSION_CHECK(false, "Unsupported RESHAPE case: ", op_case);
    }
    अगर (!new_shape_गाँठ) {
        new_shape_गाँठ = ov::op::v0::Constant::create(ov::element::i64, {shape.size()}, shape);
    }
    स्वतः res = std::make_shared<ov::op::v1::Reshape>(context.get_निवेश(0), new_shape_गाँठ, false);
    वापस rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
