#అన్తర్భావ "../node_context.h"
#అన్తర్భావ "../op_table.h"
#అన్తర్భావ "../utils.h"
#అన్తర్భావ "ggml.h"

#అన్తర్భావ <memory>
#అన్తర్భావ <openvino/core/type.hpp>
#అన్తర్భావ <openvino/op/constant.hpp>
#అన్తర్భావ <openvino/op/convert.hpp>
#అన్తర్భావ <openvino/op/divide.hpp>
#అన్తర్భావ <openvino/op/multiply.hpp>
#అన్తర్భావ <openvino/op/shape_of.hpp>
#అన్తర్భావ <openvino/op/sigmoid.hpp>
#అన్తర్భావ <openvino/op/swish.hpp>
#అన్తర్భావ <openvino/op/tile.hpp>
#అన్తర్భావ <openvino/op/util/precision_sensitive_attribute.hpp>
#అన్తర్భావ <vector>

నామాకాశ ov {
నామాకాశ frontend {
నామాకాశ ggml {
నామాకాశ op {

నామాకాశ {

తార్కిక is_silu_div_నమూనా(స్థిరరాశి ov::Output<ov::Node> & numerator,
                         స్థిరరాశి ov::Output<ov::Node> & denominator,
                         స్థిరరాశి కణుపు_Context & context) {
    ఒకవేళ (context.get_ఆదానం_కొలత() != 2) {
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి స్వయంచాలక * unary_op = reinterpret_cast<స్థిరరాశి ggml_unary_op *>(context.get_ఆదానం_op_params(0));
    ఒకవేళ (unary_op == శూన్యసూచిక || *unary_op != GGML_UNARY_OP_SILU) {
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి స్వయంచాలక denom_కణుపు = denominator.get_కణుపు_shared_ptr();

    ఒకవేళ (స్వయంచాలక swish = ov::as_type_ptr<ov::op::v4::Swish>(numerator.get_కణుపు_shared_ptr())) {
        తిరిగి swish->ఆదానం_విలువ(0).get_కణుపు_shared_ptr() == denom_కణుపు;
    }
    తిరిగి అబద్ధం;
}

ov::Output<ov::Node> repeat_ఆదానం_గమ్యం_సరిపోలిక(స్థిరరాశి కణుపు_Context & context,
                                           స్థిరరాశి ov::Output<ov::Node> & ఆదానం,
                                           స్థిరరాశి ov::Output<ov::Node> & లక్ష్యం,
                                           size_t ఆదానం_సూచీ) {
    స్థిరరాశి స్వయంచాలక ఆదానం_shape = context.get_ఆదానం_shape(ఆదానం_సూచీ);
    స్థిరరాశి స్వయంచాలక లక్ష్యం_shape = context.get_ఆదానం_shape(0);

    ఒకవేళ (ఆదానం_shape == లక్ష్యం_shape) {
        తిరిగి ఆదానం;
    }

    ఒకవేళ (ఆదానం_shape.rank().is_static() && లక్ష్యం_shape.rank().is_static()) {
        స్థిరరాశి స్వయంచాలక rank = static_cast<size_t>(ఆదానం_shape.rank().get_నిడివి());
        std::vector<int64_t> repeats(rank, 1);
        తార్కిక needs_repeat = అబద్ధం;

        కొరకు (size_t axis = 0; axis < rank; ++axis) {
            FRONT_చివర_OP_CONVERSION_CHECK(ఆదానం_shape[axis].is_static() && లక్ష్యం_shape[axis].is_static(),
                                          "DIV repeat requires static dimensions on both inputs");

            స్థిరరాశి int64_t ఆదానం_dim = ఆదానం_shape[axis].get_నిడివి();
            స్థిరరాశి int64_t లక్ష్యం_dim = లక్ష్యం_shape[axis].get_నిడివి();

            FRONT_చివర_OP_CONVERSION_CHECK(ఆదానం_dim > 0 && లక్ష్యం_dim > 0 && లక్ష్యం_dim % ఆదానం_dim == 0,
                                          "DIV input shape ", ఆదానం_shape, " cannot repeat to match ", లక్ష్యం_shape);

            repeats[axis] = లక్ష్యం_dim / ఆదానం_dim;
            needs_repeat = needs_repeat || repeats[axis] != 1;
        }

        ఒకవేళ (!needs_repeat) {
            తిరిగి ఆదానం;
        }

        స్వయంచాలక repeats_కణుపు = ov::op::v0::Constant::create(ov::element::i64, {repeats.size()}, repeats);
        తిరిగి std::make_shared<ov::op::v0::Tile>(ఆదానం, repeats_కణుపు);
    }

    స్వయంచాలక ఆదానం_shape_కణుపు = std::make_shared<ov::op::v3::Shape_యొక్క>(ఆదానం, ov::element::i64);
    స్వయంచాలక లక్ష్యం_shape_కణుపు = std::make_shared<ov::op::v3::Shape_యొక్క>(లక్ష్యం, ov::element::i64);
    స్వయంచాలక repeats_కణుపు = std::make_shared<ov::op::v1::Divide>(లక్ష్యం_shape_కణుపు, ఆదానం_shape_కణుపు);
    తిరిగి std::make_shared<ov::op::v0::Tile>(ఆదానం, repeats_కణుపు);
}

}  // namespace

Output_సదిశ translate_div(స్థిరరాశి కణుపు_Context & context) {
    అంకె_inputs_check(context, 2, 2);

    స్వయంచాలక ఆదానం_0 = process_view_ఆదానం_new(context, 0);
    స్వయంచాలక ఆదానం_1 = process_view_ఆదానం_new(context, 1);

    ఒకవేళ (is_silu_div_నమూనా(ఆదానం_0, ఆదానం_1, context)) {
        ov::Output<ov::Node> res = std::make_shared<ov::op::v0::Sigmoid>(ఆదానం_1);
        ఒకవేళ (res.get_element_type() != context.get_output_type()) {
            res = std::make_shared<ov::op::v0::Convert>(res, context.get_output_type());
        }
        తిరిగి rename_outputs_with_suffix({res}, context.get_name());
    }

    ఆదానం_1 = repeat_ఆదానం_గమ్యం_సరిపోలిక(context, ఆదానం_1, ఆదానం_0, 1);

    స్థిరరాశి స్వయంచాలక output_type = context.get_output_type();
    స్థిరరాశి తార్కిక use_f32_compute = ఆదానం_0.get_element_type() != ov::element::f32 ||
                                 ఆదానం_1.get_element_type() != ov::element::f32 || output_type != ov::element::f32;

    ఒకవేళ (use_f32_compute) {
        ఆదానం_0 = std::make_shared<ov::op::v0::Convert>(ఆదానం_0, ov::element::f32);
        ఆదానం_1 = std::make_shared<ov::op::v0::Convert>(ఆదానం_1, ov::element::f32);
    }

    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Divide>(ఆదానం_0, ఆదానం_1);
    ఒకవేళ (use_f32_compute) {
        // Keep the reciprocal/divide path in FP32. Without this hint, the GPU
        // plugin can still compress the subgraph back to FP16 and overflow on
        // small shexp gate values (e.g. silu(x) / x in qwen2moe).
        ov::mark_as_precision_sensitive(res.get_కణుపు_shared_ptr()->ఆదానం(0));
        ov::mark_as_precision_sensitive(res.get_కణుపు_shared_ptr()->ఆదానం(1));
    }
    ఒకవేళ (res.get_element_type() != output_type) {
        స్వయంచాలక output_convert = std::make_shared<ov::op::v0::Convert>(res, output_type);
        ఒకవేళ (use_f32_compute) {
            ov::mark_as_precision_sensitive(output_convert->ఆదానం(0));
        }
        res = output_convert;
    }
    తిరిగి rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
