#അന്തര്ഭാവ "../node_context.h"
#അന്തര്ഭാവ "../op_table.h"
#അന്തര്ഭാവ "../utils.h"

#അന്തര്ഭാവ <cstdint>
#അന്തര്ഭാവ <memory>
#അന്തര്ഭാവ <openvino/core/node.hpp>
#അന്തര്ഭാവ <openvino/core/node_output.hpp>
#അന്തര്ഭാവ <openvino/frontend/exception.hpp>
#അന്തര്ഭാവ <openvino/op/concat.hpp>
#അന്തര്ഭാവ <openvino/op/constant.hpp>
#അന്തര്ഭാവ <openvino/op/reshape.hpp>
#അന്തര്ഭാവ <vector>

നാമാകാശ ov {
നാമാകാശ frontend {
നാമാകാശ ggml {
നാമാകാശ op {

OutputVector translate_reshape(സ്ഥിരം കണ്ണി_Context & context) {
    അക്കം_inputs_check(context, 1, 1);
    എങ്കിൽ (context.get_നിവേശം(0).get_partial_shape().is_static() &&
        context.get_നിവേശം_shape(0) == context.get_output_shape()) {
        തിരികെ {context.get_നിവേശം(0)};
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ op_case = context.get_op_case();
    സ്ഥിരം auto output_shape = context.get_output_shape().to_shape();
    std::vector<int64_t> shape(output_shape.begin(), output_shape.end());
    std::shared_ptr<ov::Node> new_shape_കണ്ണി;
    തിരഞ്ഞെടുപ്പ് (op_case) {
    അവസ്ഥ 0:
        നിർത്തുക;
    അവസ്ഥ 1:
    അവസ്ഥ 9:
        shape[1] = -1;
        എങ്കിൽ (context.is_stateful() && op_case == 1) {
            shape.erase(shape.begin());
        }
        നിർത്തുക;
    അവസ്ഥ 2:
    അവസ്ഥ 3:
        shape[2] = -1;
        എങ്കിൽ (op_case == 3) {
            shape[3] = 1;
        }
        നിർത്തുക;
    അവസ്ഥ 4:
        തിരികെ {context.get_നിവേശം(0).get_കണ്ണി_shared_ptr()->നിവേശം_മൂല്യം(0)};
    അവസ്ഥ 5:
    അവസ്ഥ 7:
        shape = {1, 1, -1, shape[3]};
        എങ്കിൽ (context.is_stateful() && op_case == 5) {
            shape.erase(shape.begin());
        }
        നിർത്തുക;
    അവസ്ഥ 6:
        // Recurrent inputs keep the active sequence count separate from the token count.
        എങ്കിൽ (context.has_നിവേശം("s_copy_active_slot_len")) {
            auto n_slot_active_len = context.get_നിവേശം("s_copy_active_slot_len");
            auto emb_size = ov::op::v0::Constant::create(ov::element::i64, {1}, {shape[3]});
            auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
            auto neg_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
            new_shape_കണ്ണി =
                std::make_shared<ov::op::v0::Concat>(ov::OutputVector{one, n_slot_active_len, neg_one, emb_size}, 0);
        } അല്ലെങ്കിൽ {
            shape = {1, 1, -1, shape[3]};
        }
        നിർത്തുക;
    അവസ്ഥ 8:
        shape[0] = -1;
        നിർത്തുക;
    സ്ഥിരസ്ഥിതി:
        FRONT_END_OP_CONVERSION_CHECK(അസത്യം, "Unsupported RESHAPE case: ", op_case);
    }
    എങ്കിൽ (!new_shape_കണ്ണി) {
        new_shape_കണ്ണി = ov::op::v0::Constant::create(ov::element::i64, {shape.size()}, shape);
    }
    auto res = std::make_shared<ov::op::v1::Reshape>(context.get_നിവേശം(0), new_shape_കണ്ണി, അസത്യം);
    തിരികെ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
