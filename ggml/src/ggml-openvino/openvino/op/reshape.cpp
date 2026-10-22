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

OutputVector translate_reshape(स्थिरांक गाठ_Context & context) {
    अंक_inputs_check(context, 1, 1);
    जर (context.get_आदान(0).get_partial_shape().is_static() &&
        context.get_आदान_shape(0) == context.get_output_shape()) {
        परत {context.get_आदान(0)};
    }

    स्थिरांक पूर्णांक op_case = context.get_op_case();
    स्थिरांक स्वतः output_shape = context.get_output_shape().to_shape();
    std::vector<int64_t> shape(output_shape.begin(), output_shape.end());
    std::shared_ptr<ov::Node> new_shape_गाठ;
    निवड (op_case) {
    स्थिती 0:
        थांबा;
    स्थिती 1:
    स्थिती 9:
        shape[1] = -1;
        जर (context.is_stateful() && op_case == 1) {
            shape.erase(shape.begin());
        }
        थांबा;
    स्थिती 2:
    स्थिती 3:
        shape[2] = -1;
        जर (op_case == 3) {
            shape[3] = 1;
        }
        थांबा;
    स्थिती 4:
        परत {context.get_आदान(0).get_गाठ_shared_ptr()->आदान_मूल्य(0)};
    स्थिती 5:
    स्थिती 7:
        shape = {1, 1, -1, shape[3]};
        जर (context.is_stateful() && op_case == 5) {
            shape.erase(shape.begin());
        }
        थांबा;
    स्थिती 6:
        // Recurrent inputs keep the active sequence count separate from the token count.
        जर (context.has_आदान("s_copy_active_slot_len")) {
            स्वतः n_slot_active_len = context.get_आदान("s_copy_active_slot_len");
            स्वतः emb_size = ov::op::v0::Constant::create(ov::element::i64, {1}, {shape[3]});
            स्वतः one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
            स्वतः neg_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
            new_shape_गाठ =
                std::make_shared<ov::op::v0::Concat>(ov::OutputVector{one, n_slot_active_len, neg_one, emb_size}, 0);
        } अन्यथा {
            shape = {1, 1, -1, shape[3]};
        }
        थांबा;
    स्थिती 8:
        shape[0] = -1;
        थांबा;
    पूर्वनिर्धारित:
        FRONT_END_OP_CONVERSION_CHECK(असत्य, "Unsupported RESHAPE case: ", op_case);
    }
    जर (!new_shape_गाठ) {
        new_shape_गाठ = ov::op::v0::Constant::create(ov::element::i64, {shape.size()}, shape);
    }
    स्वतः res = std::make_shared<ov::op::v1::Reshape>(context.get_आदान(0), new_shape_गाठ, असत्य);
    परत rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
