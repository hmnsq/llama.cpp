#شامل "../node_context.h"
#شامل "../op_table.h"
#شامل "../utils.h"

#شامل <cstdint>
#شامل <memory>
#شامل <openvino/core/node.hpp>
#شامل <openvino/core/node_output.hpp>
#شامل <openvino/frontend/exception.hpp>
#شامل <openvino/op/concat.hpp>
#شامل <openvino/op/constant.hpp>
#شامل <openvino/op/reshape.hpp>
#شامل <vector>

نامخلا ov {
نامخلا frontend {
نامخلا ggml {
نامخلا op {

Output_سمتیہ translate_reshape(ثابت گرہ_Context & context) {
    عدد_inputs_check(context, 1, 1);
    اگر (context.get_داخلہ(0).get_partial_shape().is_static() &&
        context.get_داخلہ_shape(0) == context.get_output_shape()) {
        واپس {context.get_داخلہ(0)};
    }

    ثابت صحیح op_case = context.get_op_case();
    ثابت خودکار output_shape = context.get_output_shape().مقصد_shape();
    std::vector<int64_t> shape(output_shape.begin(), output_shape.end());
    std::shared_ptr<ov::Node> new_shape_گرہ;
    انتخاب (op_case) {
    حالت 0:
        توڑو;
    حالت 1:
    حالت 9:
        shape[1] = -1;
        اگر (context.is_stateful() && op_case == 1) {
            shape.erase(shape.begin());
        }
        توڑو;
    حالت 2:
    حالت 3:
        shape[2] = -1;
        اگر (op_case == 3) {
            shape[3] = 1;
        }
        توڑو;
    حالت 4:
        واپس {context.get_داخلہ(0).get_گرہ_shared_ptr()->داخلہ_قدر(0)};
    حالت 5:
    حالت 7:
        shape = {1, 1, -1, shape[3]};
        اگر (context.is_stateful() && op_case == 5) {
            shape.erase(shape.begin());
        }
        توڑو;
    حالت 6:
        // Recurrent inputs keep the active sequence count separate from the token count.
        اگر (context.has_داخلہ("s_copy_active_slot_len")) {
            خودکار n_خانہ_فعال_len = context.get_داخلہ("s_copy_active_slot_len");
            خودکار emb_سائز = ov::op::v0::Constant::create(ov::element::i64, {1}, {shape[3]});
            خودکار one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
            خودکار neg_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
            new_shape_گرہ =
                std::make_shared<ov::op::v0::Concat>(ov::Output_سمتیہ{one, n_خانہ_فعال_len, neg_one, emb_سائز}, 0);
        } ورنہ {
            shape = {1, 1, -1, shape[3]};
        }
        توڑو;
    حالت 8:
        shape[0] = -1;
        توڑو;
    طےشدہ:
        FRONT_انتہا_OP_CONVERSION_CHECK(جھوٹ, "Unsupported RESHAPE case: ", op_case);
    }
    اگر (!new_shape_گرہ) {
        new_shape_گرہ = ov::op::v0::Constant::create(ov::element::i64, {shape.size()}, shape);
    }
    خودکار res = std::make_shared<ov::op::v1::Reshape>(context.get_داخلہ(0), new_shape_گرہ, جھوٹ);
    واپس rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
