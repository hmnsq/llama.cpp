#అన్తర్భావ "../node_context.h"
#అన్తర్భావ "../op_table.h"
#అన్తర్భావ "../utils.h"

#అన్తర్భావ <openvino/op/transpose.hpp>

నామాకాశ ov {
నామాకాశ frontend {
నామాకాశ ggml {
నామాకాశ op {

Output_సదిశ translate_transpose(స్థిరరాశి కణుపు_Context & context) {
    అంకె_inputs_check(context, 1, 1);

    // Compute permute order from input/output shape and stride information
    // so it adapts to different input and output layouts.
    స్వయంచాలక ఆదానం_stride = context.get_ఆదానం_stride(0);
    స్వయంచాలక output_stride = context.get_output_stride();

    // Compute permute order by matching output and input stride rankings.
    // Build <stride, dim_index> pairs.
    std::vector<std::pair<size_t, పూర్ణాంకం>> output_stride_dims;
    std::vector<std::pair<size_t, పూర్ణాంకం>> ఆదానం_stride_dims;

    కొరకు (పూర్ణాంకం i = 0; i < 4; ++i) {
        output_stride_dims.push_back({output_stride[i], i});
        ఆదానం_stride_dims.push_back({ఆదానం_stride[i], i});
    }

    // Sort by stride in descending order.
    std::sort(output_stride_dims.rbegin(), output_stride_dims.rend());
    std::sort(ఆదానం_stride_dims.rbegin(), ఆదానం_stride_dims.rend());

    // Build permute order.
    std::vector<int64_t> క్రమమార్పు_క్రమం(4);
    కొరకు (పూర్ణాంకం i = 0; i < 4; ++i) {
        పూర్ణాంకం output_dim = output_stride_dims[i].second;
        పూర్ణాంకం ఆదానం_dim = ఆదానం_stride_dims[i].second;
        క్రమమార్పు_క్రమం[output_dim] = ఆదానం_dim;
    }

    స్వయంచాలక ఆదానం = process_view_ఆదానం_new(context, 0);

    స్వయంచాలక res = std::make_shared<ov::op::v1::Transpose>(
        ఆదానం, ov::op::v0::Constant::create(ov::element::i64, {4}, క్రమమార్పు_క్రమం));
    తిరిగి rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
