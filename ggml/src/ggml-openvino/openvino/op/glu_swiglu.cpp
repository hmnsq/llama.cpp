#అన్తర్భావ "../node_context.h"
#అన్తర్భావ "../op_table.h"
#అన్తర్భావ "../utils.h"

#అన్తర్భావ <cstdint>
#అన్తర్భావ <limits>
#అన్తర్భావ <memory>
#అన్తర్భావ <openvino/core/node_output.hpp>
#అన్తర్భావ <openvino/op/add.hpp>
#అన్తర్భావ <openvino/op/clamp.hpp>
#అన్తర్భావ <openvino/op/constant.hpp>
#అన్తర్భావ <openvino/op/convert.hpp>
#అన్తర్భావ <openvino/op/multiply.hpp>
#అన్తర్భావ <openvino/op/slice.hpp>
#అన్తర్భావ <openvino/op/swish.hpp>

నామాకాశ ov {
నామాకాశ frontend {
నామాకాశ ggml {
నామాకాశ op {

స్థిర std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>> get_glu_inputs(స్థిరరాశి కణుపు_Context & context) {
    అంకె_inputs_check(context, 1, 2);

    ov::Output<ov::Node> src0;
    ov::Output<ov::Node> src1;
    ఒకవేళ (context.get_ఆదానం_కొలత() == 2) {
        // Inputs may be VIEW slices of a combined gate_up tensor (MoE experts):
        // resolve them so each half has its real sliced shape, not the base tensor.
        src0 = process_view_ఆదానం_new(context, 0);
        src1 = process_view_ఆదానం_new(context, 1);
    } లేకపోతే {
        // GGML splits along ne[0] (OV last axis) using floor division: nc = ne[0] / 2.
        // Both halves are nc elements; if the dimension is odd, the last element is dropped.
        // Use Slice instead of Split to handle odd dimensions correctly.
        // Resolve a VIEW input (e.g. non-contiguous slice) to its real shape first.
        స్వయంచాలక combined = process_view_ఆదానం_new(context, 0);
        స్వయంచాలక combined_shape = combined.get_partial_shape();
        int64_t చివరి_dim_val = combined_shape[combined_shape.rank().get_నిడివి() - 1].get_నిడివి();
        int64_t nc = చివరి_dim_val / 2;

        స్వయంచాలక axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
        స్వయంచాలక అడుగు = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        స్వయంచాలక start0 = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        స్వయంచాలక stop0 = ov::op::v0::Constant::create(ov::element::i64, {1}, {nc});
        స్వయంచాలక start1 = ov::op::v0::Constant::create(ov::element::i64, {1}, {nc});
        స్వయంచాలక stop1 = ov::op::v0::Constant::create(ov::element::i64, {1}, {2 * nc});

        src0 = std::make_shared<ov::op::v8::ముక్క>(combined, start0, stop0, అడుగు, axis);
        src1 = std::make_shared<ov::op::v8::ముక్క>(combined, start1, stop1, అడుగు, axis);
    }

    int32_t * params = context.get_output_op_params();
    స్థిరరాశి int32_t swapped = params[1];
    ఒకవేళ (swapped) {
        std::swap(src0, src1);
    }

    తిరిగి {src0, src1};
}

Output_సదిశ translate_glu_swiglu(స్థిరరాశి కణుపు_Context & context) {
    స్వయంచాలక [src0, src1] = get_glu_inputs(context);

    స్వయంచాలక silu = std::make_shared<ov::op::v4::Swish>(src0);
    స్వయంచాలక res = std::make_shared<ov::op::v1::Multiply>(silu, src1);

    తిరిగి rename_outputs_with_suffix({res}, context.get_name());
}

Output_సదిశ translate_glu_swiglu_oai(స్థిరరాశి కణుపు_Context & context) {
    స్వయంచాలక [src0, src1] = get_glu_inputs(context);

    స్థిరరాశి int32_t * params = context.get_output_op_params();
    స్థిరరాశి దశాంశం alpha = reinterpret_cast<స్థిరరాశి దశాంశం *>(params)[2];
    స్థిరరాశి దశాంశం హద్దు = reinterpret_cast<స్థిరరాశి దశాంశం *>(params)[3];

    // Compute in f32: alpha_const/one are f32, and Swish/Add need matching input types.
    స్వయంచాలక output_type = context.get_output_type();
    ఒకవేళ (src0.get_element_type() != ov::element::f32) {
        src0 = std::make_shared<ov::op::v0::Convert>(src0, ov::element::f32);
    }
    ఒకవేళ (src1.get_element_type() != ov::element::f32) {
        src1 = std::make_shared<ov::op::v0::Convert>(src1, ov::element::f32);
    }

    స్వయంచాలక gate = std::make_shared<ov::op::v0::Clamp>(src0, -std::numeric_limits<దశాంశం>::infinity(), హద్దు);
    స్వయంచాలక alpha_const = ov::op::v0::Constant::create(ov::element::f32, {}, {alpha});
    స్వయంచాలక బహిర్గతం_glu = std::make_shared<ov::op::v4::Swish>(gate, alpha_const);

    స్వయంచాలక పైకి = std::make_shared<ov::op::v0::Clamp>(src1, -హద్దు, హద్దు);
    స్వయంచాలక one = ov::op::v0::Constant::create(ov::element::f32, {}, {1.0f});
    స్వయంచాలక పైకి_plus_one = std::make_shared<ov::op::v1::కూడు>(పైకి, one);
    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Multiply>(బహిర్గతం_glu, పైకి_plus_one);
    ఒకవేళ (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    తిరిగి rename_outputs_with_suffix({res}, context.get_name());
}

Output_సదిశ translate_glu_swiglu_clamp(స్థిరరాశి కణుపు_Context & context) {
    స్వయంచాలక [src0, src1] = get_glu_inputs(context);

    స్థిరరాశి int32_t * params = context.get_output_op_params();
    స్థిరరాశి దశాంశం హద్దు = reinterpret_cast<స్థిరరాశి దశాంశం *>(params)[3];

    // Compute in f32: f16 Swish/Clamp rounding drifts past the 1e-7 test tolerance.
    స్వయంచాలక output_type = context.get_output_type();
    ఒకవేళ (src0.get_element_type() != ov::element::f32) {
        src0 = std::make_shared<ov::op::v0::Convert>(src0, ov::element::f32);
    }
    ఒకవేళ (src1.get_element_type() != ov::element::f32) {
        src1 = std::make_shared<ov::op::v0::Convert>(src1, ov::element::f32);
    }

    స్వయంచాలక gate = std::make_shared<ov::op::v0::Clamp>(src0, -std::numeric_limits<దశాంశం>::infinity(), హద్దు);
    స్వయంచాలక silu = std::make_shared<ov::op::v4::Swish>(gate);
    స్వయంచాలక పైకి = std::make_shared<ov::op::v0::Clamp>(src1, -హద్దు, హద్దు);
    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Multiply>(silu, పైకి);
    ఒకవేళ (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    తిరిగి rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
