#અન્તર્ભાવ "../node_context.h"
#અન્તર્ભાવ "../op_table.h"
#અન્તર્ભાવ "../utils.h"

#અન્તર્ભાવ <openvino/op/avg_pool.hpp>
#અન્તર્ભાવ <openvino/op/max_pool.hpp>
#અન્તર્ભાવ <openvino/op/convert.hpp>

નામાકાશ ov {
નામાકાશ frontend {
નામાકાશ ggml {
નામાકાશ op {

OutputVector translate_pool_2d(અચલ ગાંઠ_Context & context) {
    અંક_inputs_check(context, 1, 1);
    અચલ int32_t * params = context.get_output_op_params();

    અચલ પૂર્ણાંક k0 = params[1];
    અચલ પૂર્ણાંક k1 = params[2];
    અચલ પૂર્ણાંક s0 = params[3];
    અચલ પૂર્ણાંક s1 = params[4];
    અચલ પૂર્ણાંક p0 = params[5];
    અચલ પૂર્ણાંક p1 = params[6];

    અચલ પૂર્ણાંક op_case = context.get_op_case();
    ov::Output<Node> આગત = context.get_આગત(0);
    ov::Strides strides{static_cast<size_t>(s1), static_cast<size_t>(s0)};
    ov::Shape pads_begin{static_cast<size_t>(p1), static_cast<size_t>(p0)};
    ov::Shape pads_end{static_cast<size_t>(p1), static_cast<size_t>(p0)};
    ov::Shape kernel{static_cast<size_t>(k1), static_cast<size_t>(k0)};
    ov::Output<Node> res;

    પસંદગી (op_case) {
    સ્થિતિ 1:  // GGML_OP_POOL_MAX
    {
        res = std::make_shared<ov::op::v1::MaxPool>(આગત, strides, pads_begin, pads_end, kernel);
        તોડો;
    }
    સ્થિતિ 2:  // GGML_OP_POOL_AVG
    {
        res = std::make_shared<ov::op::v1::AvgPool>(આગત, strides, pads_begin, pads_end, kernel, ખોટું);
        તોડો;
    }
    મૂળભૂત:
        તોડો;
    }
    પાછા rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
