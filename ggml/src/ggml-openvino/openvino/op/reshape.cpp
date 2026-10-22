#ਅਨ੍ਤਰ੍ਭਾਵ "../node_context.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../op_table.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../utils.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <cstdint>
#ਅਨ੍ਤਰ੍ਭਾਵ <memory>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/core/node.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/core/node_output.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/frontend/exception.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/concat.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/constant.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/reshape.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <vector>

ਨਾਮਾਕਾਸ਼ ov {
ਨਾਮਾਕਾਸ਼ frontend {
ਨਾਮਾਕਾਸ਼ ggml {
ਨਾਮਾਕਾਸ਼ op {

Output_ਸਦਿਸ਼ translate_reshape(ਅਚਲ ਗੰਢ_Context & context) {
    ਅੰਕ_inputs_check(context, 1, 1);
    ਜੇ (context.get_ਨਿਵੇਸ਼(0).get_partial_shape().is_static() &&
        context.get_ਨਿਵੇਸ਼_shape(0) == context.get_output_shape()) {
        ਵਾਪਸ {context.get_ਨਿਵੇਸ਼(0)};
    }

    ਅਚਲ ਪੂਰਨ op_case = context.get_op_case();
    ਅਚਲ ਸਵੈਚਾਲਿਤ output_shape = context.get_output_shape().ਮੰਜ਼ਿਲ_shape();
    std::vector<int64_t> shape(output_shape.begin(), output_shape.end());
    std::shared_ptr<ov::Node> new_shape_ਗੰਢ;
    ਚੋਣ (op_case) {
    ਹਾਲਤ 0:
        ਰੋਕੋ;
    ਹਾਲਤ 1:
    ਹਾਲਤ 9:
        shape[1] = -1;
        ਜੇ (context.is_stateful() && op_case == 1) {
            shape.erase(shape.begin());
        }
        ਰੋਕੋ;
    ਹਾਲਤ 2:
    ਹਾਲਤ 3:
        shape[2] = -1;
        ਜੇ (op_case == 3) {
            shape[3] = 1;
        }
        ਰੋਕੋ;
    ਹਾਲਤ 4:
        ਵਾਪਸ {context.get_ਨਿਵੇਸ਼(0).get_ਗੰਢ_shared_ptr()->ਨਿਵੇਸ਼_ਮੁੱਲ(0)};
    ਹਾਲਤ 5:
    ਹਾਲਤ 7:
        shape = {1, 1, -1, shape[3]};
        ਜੇ (context.is_stateful() && op_case == 5) {
            shape.erase(shape.begin());
        }
        ਰੋਕੋ;
    ਹਾਲਤ 6:
        // Recurrent inputs keep the active sequence count separate from the token count.
        ਜੇ (context.has_ਨਿਵੇਸ਼("s_copy_active_slot_len")) {
            ਸਵੈਚਾਲਿਤ n_ਥਾਂ_ਸਰਗਰਮ_len = context.get_ਨਿਵੇਸ਼("s_copy_active_slot_len");
            ਸਵੈਚਾਲਿਤ emb_ਮਾਪ = ov::op::v0::Constant::create(ov::element::i64, {1}, {shape[3]});
            ਸਵੈਚਾਲਿਤ one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
            ਸਵੈਚਾਲਿਤ neg_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
            new_shape_ਗੰਢ =
                std::make_shared<ov::op::v0::Concat>(ov::Output_ਸਦਿਸ਼{one, n_ਥਾਂ_ਸਰਗਰਮ_len, neg_one, emb_ਮਾਪ}, 0);
        } ਨਹੀਂਤਾਂ {
            shape = {1, 1, -1, shape[3]};
        }
        ਰੋਕੋ;
    ਹਾਲਤ 8:
        shape[0] = -1;
        ਰੋਕੋ;
    ਮੂਲ:
        FRONT_ਸਿਰਾ_OP_CONVERSION_CHECK(ਝੂਠ, "Unsupported RESHAPE case: ", op_case);
    }
    ਜੇ (!new_shape_ਗੰਢ) {
        new_shape_ਗੰਢ = ov::op::v0::Constant::create(ov::element::i64, {shape.size()}, shape);
    }
    ਸਵੈਚਾਲਿਤ res = std::make_shared<ov::op::v1::Reshape>(context.get_ਨਿਵੇਸ਼(0), new_shape_ਗੰਢ, ਝੂਠ);
    ਵਾਪਸ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
