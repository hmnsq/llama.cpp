#ضمن "../node_context.h"
#ضمن "../op_table.h"
#ضمن "../utils.h"

#ضمن <cstdint>
#ضمن <memory>
#ضمن <openvino/core/node.hpp>
#ضمن <openvino/core/node_output.hpp>
#ضمن <openvino/frontend/exception.hpp>
#ضمن <openvino/op/concat.hpp>
#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/reshape.hpp>
#ضمن <vector>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق op {

Output_ویکټور translate_reshape(ثابت غوټه_Context & context) {
    عدد_inputs_check(context, 1, 1);
    إذا (context.get_ننوت(0).get_partial_shape().is_static() &&
        context.get_ننوت_shape(0) == context.get_output_shape()) {
        أرجع {context.get_ننوت(0)};
    }

    ثابت صحيح op_case = context.get_op_case();
    ثابت خپلکاری output_shape = context.get_output_shape().منزل_shape();
    std::vector<int64_t> shape(output_shape.begin(), output_shape.end());
    std::shared_ptr<ov::Node> new_shape_غوټه;
    بدل (op_case) {
    حالة 0:
        اكسر;
    حالة 1:
    حالة 9:
        shape[1] = -1;
        إذا (context.is_stateful() && op_case == 1) {
            shape.erase(shape.begin());
        }
        اكسر;
    حالة 2:
    حالة 3:
        shape[2] = -1;
        إذا (op_case == 3) {
            shape[3] = 1;
        }
        اكسر;
    حالة 4:
        أرجع {context.get_ننوت(0).get_غوټه_shared_ptr()->ننوت_ارزښت(0)};
    حالة 5:
    حالة 7:
        shape = {1, 1, -1, shape[3]};
        إذا (context.is_stateful() && op_case == 5) {
            shape.erase(shape.begin());
        }
        اكسر;
    حالة 6:
        // Recurrent inputs keep the active sequence count separate from the token count.
        إذا (context.has_ننوت("s_copy_active_slot_len")) {
            خپلکاری n_خونه_فعال_len = context.get_ننوت("s_copy_active_slot_len");
            خپلکاری emb_اندازه = ov::op::v0::Constant::create(ov::element::i64, {1}, {shape[3]});
            خپلکاری one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
            خپلکاری neg_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
            new_shape_غوټه =
                std::make_shared<ov::op::v0::Concat>(ov::Output_ویکټور{one, n_خونه_فعال_len, neg_one, emb_اندازه}, 0);
        } وإلا {
            shape = {1, 1, -1, shape[3]};
        }
        اكسر;
    حالة 8:
        shape[0] = -1;
        اكسر;
    افتراضي:
        FRONT_پای_ټکی_OP_CONVERSION_CHECK(خطأ, "Unsupported RESHAPE case: ", op_case);
    }
    إذا (!new_shape_غوټه) {
        new_shape_غوټه = ov::op::v0::Constant::create(ov::element::i64, {shape.size()}, shape);
    }
    خپلکاری res = std::make_shared<ov::op::v1::Reshape>(context.get_ننوت(0), new_shape_غوټه, خطأ);
    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
