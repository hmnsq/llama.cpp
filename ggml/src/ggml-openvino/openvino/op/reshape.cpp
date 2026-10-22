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

فضاینام ov {
فضاینام frontend {
فضاینام ggml {
فضاینام op {

OutputVector translate_reshape(ثابت گره_Context & context) {
    رقم_inputs_check(context, 1, 1);
    اگر (context.get_درون_داد(0).get_partial_shape().is_static() &&
        context.get_درون_داد_shape(0) == context.get_output_shape()) {
        بازگشت {context.get_درون_داد(0)};
    }

    ثابت صحیح op_case = context.get_op_case();
    ثابت auto output_shape = context.get_output_shape().to_shape();
    std::vector<int64_t> shape(output_shape.begin(), output_shape.end());
    std::shared_ptr<ov::Node> new_shape_گره;
    انتخاب (op_case) {
    حالت 0:
        بشکن;
    حالت 1:
    حالت 9:
        shape[1] = -1;
        اگر (context.is_stateful() && op_case == 1) {
            shape.erase(shape.begin());
        }
        بشکن;
    حالت 2:
    حالت 3:
        shape[2] = -1;
        اگر (op_case == 3) {
            shape[3] = 1;
        }
        بشکن;
    حالت 4:
        بازگشت {context.get_درون_داد(0).get_گره_shared_ptr()->درون_داد_مقدار(0)};
    حالت 5:
    حالت 7:
        shape = {1, 1, -1, shape[3]};
        اگر (context.is_stateful() && op_case == 5) {
            shape.erase(shape.begin());
        }
        بشکن;
    حالت 6:
        // Recurrent inputs keep the active sequence count separate from the token count.
        اگر (context.has_درون_داد("s_copy_active_slot_len")) {
            auto n_slot_active_len = context.get_درون_داد("s_copy_active_slot_len");
            auto emb_size = ov::op::v0::Constant::create(ov::element::i64, {1}, {shape[3]});
            auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
            auto neg_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
            new_shape_گره =
                std::make_shared<ov::op::v0::Concat>(ov::OutputVector{one, n_slot_active_len, neg_one, emb_size}, 0);
        } وگرنه {
            shape = {1, 1, -1, shape[3]};
        }
        بشکن;
    حالت 8:
        shape[0] = -1;
        بشکن;
    پیشفرض:
        FRONT_END_OP_CONVERSION_CHECK(نادرست, "Unsupported RESHAPE case: ", op_case);
    }
    اگر (!new_shape_گره) {
        new_shape_گره = ov::op::v0::Constant::create(ov::element::i64, {shape.size()}, shape);
    }
    auto res = std::make_shared<ov::op::v1::Reshape>(context.get_درون_داد(0), new_shape_گره, نادرست);
    بازگشت rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
