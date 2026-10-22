#അന്തര്ഭാവ "../node_context.h"
#അന്തര്ഭാവ "../op_table.h"
#അന്തര്ഭാവ "../utils.h"

#അന്തര്ഭാവ <openvino/op/transpose.hpp>

നാമാകാശ ov {
നാമാകാശ frontend {
നാമാകാശ ggml {
നാമാകാശ op {

OutputVector translate_transpose(സ്ഥിരം കണ്ണി_Context & context) {
    അക്കം_inputs_check(context, 1, 1);

    // Compute permute order from input/output shape and stride information
    // so it adapts to different input and output layouts.
    auto നിവേശം_stride = context.get_നിവേശം_stride(0);
    auto output_stride = context.get_output_stride();

    // Compute permute order by matching output and input stride rankings.
    // Build <stride, dim_index> pairs.
    std::vector<std::pair<size_t, പൂർണ്ണസംഖ്യ>> output_stride_dims;
    std::vector<std::pair<size_t, പൂർണ്ണസംഖ്യ>> നിവേശം_stride_dims;

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < 4; ++i) {
        output_stride_dims.push_back({output_stride[i], i});
        നിവേശം_stride_dims.push_back({നിവേശം_stride[i], i});
    }

    // Sort by stride in descending order.
    std::sort(output_stride_dims.rbegin(), output_stride_dims.rend());
    std::sort(നിവേശം_stride_dims.rbegin(), നിവേശം_stride_dims.rend());

    // Build permute order.
    std::vector<int64_t> permute_order(4);
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < 4; ++i) {
        പൂർണ്ണസംഖ്യ output_dim = output_stride_dims[i].second;
        പൂർണ്ണസംഖ്യ നിവേശം_dim = നിവേശം_stride_dims[i].second;
        permute_order[output_dim] = നിവേശം_dim;
    }

    auto നിവേശം = process_view_നിവേശം_new(context, 0);

    auto res = std::make_shared<ov::op::v1::Transpose>(
        നിവേശം, ov::op::v0::Constant::create(ov::element::i64, {4}, permute_order));
    തിരികെ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
