#അന്തര്ഭാവ "../node_context.h"
#അന്തര്ഭാവ "../op_table.h"
#അന്തര്ഭാവ "../utils.h"

#അന്തര്ഭാവ <openvino/op/avg_pool.hpp>
#അന്തര്ഭാവ <openvino/op/max_pool.hpp>
#അന്തര്ഭാവ <openvino/op/convert.hpp>

നാമാകാശ ov {
നാമാകാശ frontend {
നാമാകാശ ggml {
നാമാകാശ op {

OutputVector translate_pool_2d(സ്ഥിരം കണ്ണി_Context & context) {
    അക്കം_inputs_check(context, 1, 1);
    സ്ഥിരം int32_t * params = context.get_output_op_params();

    സ്ഥിരം പൂർണ്ണസംഖ്യ k0 = params[1];
    സ്ഥിരം പൂർണ്ണസംഖ്യ k1 = params[2];
    സ്ഥിരം പൂർണ്ണസംഖ്യ s0 = params[3];
    സ്ഥിരം പൂർണ്ണസംഖ്യ s1 = params[4];
    സ്ഥിരം പൂർണ്ണസംഖ്യ p0 = params[5];
    സ്ഥിരം പൂർണ്ണസംഖ്യ p1 = params[6];

    സ്ഥിരം പൂർണ്ണസംഖ്യ op_case = context.get_op_case();
    ov::Output<Node> നിവേശം = context.get_നിവേശം(0);
    ov::Strides strides{static_cast<size_t>(s1), static_cast<size_t>(s0)};
    ov::Shape pads_begin{static_cast<size_t>(p1), static_cast<size_t>(p0)};
    ov::Shape pads_end{static_cast<size_t>(p1), static_cast<size_t>(p0)};
    ov::Shape kernel{static_cast<size_t>(k1), static_cast<size_t>(k0)};
    ov::Output<Node> res;

    തിരഞ്ഞെടുപ്പ് (op_case) {
    അവസ്ഥ 1:  // GGML_OP_POOL_MAX
    {
        res = std::make_shared<ov::op::v1::MaxPool>(നിവേശം, strides, pads_begin, pads_end, kernel);
        നിർത്തുക;
    }
    അവസ്ഥ 2:  // GGML_OP_POOL_AVG
    {
        res = std::make_shared<ov::op::v1::AvgPool>(നിവേശം, strides, pads_begin, pads_end, kernel, അസത്യം);
        നിർത്തുക;
    }
    സ്ഥിരസ്ഥിതി:
        നിർത്തുക;
    }
    തിരികെ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
