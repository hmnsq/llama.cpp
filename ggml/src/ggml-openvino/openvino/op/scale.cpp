#అన్తర్భావ "../node_context.h"
#అన్తర్భావ "../op_table.h"
#అన్తర్భావ "../utils.h"

#అన్తర్భావ <openvino/core/except.hpp>
#అన్తర్భావ <openvino/op/add.hpp>
#అన్తర్భావ <openvino/op/concat.hpp>
#అన్తర్భావ <openvino/op/constant.hpp>
#అన్తర్భావ <openvino/op/convert.hpp>
#అన్తర్భావ <openvino/op/equal.hpp>
#అన్తర్భావ <openvino/op/gather.hpp>
#అన్తర్భావ <openvino/op/greater_eq.hpp>
#అన్తర్భావ <openvino/op/if.hpp>
#అన్తర్భావ <openvino/op/less.hpp>
#అన్తర్భావ <openvino/op/logical_or.hpp>
#అన్తర్భావ <openvino/op/multiply.hpp>
#అన్తర్భావ <openvino/op/range.hpp>
#అన్తర్భావ <openvino/op/reshape.hpp>
#అన్తర్భావ <openvino/op/shape_of.hpp>
#అన్తర్భావ <openvino/op/slice.hpp>
#అన్తర్భావ <openvino/op/squeeze.hpp>
#అన్తర్భావ <openvino/op/unsqueeze.hpp>
#అన్తర్భావ <vector>

నామాకాశ ov {
నామాకాశ frontend {
నామాకాశ ggml {
నామాకాశ op {

Output_సదిశ translate_scale(స్థిరరాశి కణుపు_Context & context) {
    అంకె_inputs_check(context, 1, 1);

    దశాంశం scale;
    దశాంశం bias;
    జ్ఞాపకం_నకలు(&scale, (దశాంశం *) context.get_output_op_params() + 0, పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&bias, (దశాంశం *) context.get_output_op_params() + 1, పరిమాణం(దశాంశం));

    స్వయంచాలక scale_కణుపు = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<దశాంశం>{scale});

    ఒకవేళ (context.get_op_case() == 2) {
        తిరిగి {context.get_ఆదానం(0)};
    }

    ఒకవేళ (context.get_op_case() == 1 && context.has_ఆదానం("cache_rs_reset_len")) {
        స్వయంచాలక cache_rs_reset_idx = context.get_ఆదానం("cache_rs_reset_idx");
        స్వయంచాలక cache_rs_reset_len = context.get_ఆదానం("cache_rs_reset_len");

        స్వయంచాలక cache_rs = context.get_ఆదానం(0);

        స్వయంచాలక cache_shape = std::make_shared<ov::op::v3::Shape_యొక్క>(cache_rs, ov::element::i64);
        స్వయంచాలక n_slots_1d = std::make_shared<ov::op::v8::Gather>(
            cache_shape, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2}),
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0}));
        స్వయంచాలక n_slots = std::make_shared<ov::op::v0::Squeeze>(n_slots_1d);

        స్వయంచాలక iota = std::make_shared<ov::op::v4::Range>(
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0}), n_slots,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1}), ov::element::i64);

        స్వయంచాలక idx_plus_len = std::make_shared<ov::op::v1::కూడు>(cache_rs_reset_idx, cache_rs_reset_len);
        స్వయంచాలక less_than_idx = std::make_shared<ov::op::v1::Less>(iota, cache_rs_reset_idx);
        స్వయంచాలక greater_equal_idx_plus_len = std::make_shared<ov::op::v1::GreaterEqual>(iota, idx_plus_len);
        స్వయంచాలక keep_ముసుగు = std::make_shared<ov::op::v1::LogicalOr>(less_than_idx, greater_equal_idx_plus_len);

        స్వయంచాలక keep_ముసుగు_f32 = std::make_shared<ov::op::v0::Convert>(keep_ముసుగు, ov::element::f32);
        స్వయంచాలక keep_ముసుగు_reshape = std::make_shared<ov::op::v0::Unsqueeze>(
            keep_ముసుగు_f32, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1}));

        స్వయంచాలక cleared_cache_rs = std::make_shared<ov::op::v1::Multiply>(cache_rs, keep_ముసుగు_reshape);
        తిరిగి rename_outputs_with_suffix({cleared_cache_rs}, context.get_name());
    }

    స్వయంచాలక scaled = std::make_shared<ov::op::v1::Multiply>(context.get_ఆదానం(0), scale_కణుపు);

    std::shared_ptr<ov::Node> res;
    ఒకవేళ (bias != 0.0f) {
        స్వయంచాలక bias_కణుపు =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<దశాంశం>{bias});
        res = std::make_shared<ov::op::v1::కూడు>(scaled, bias_కణుపు);
    } లేకపోతే {
        res = scaled;
    }

    తిరిగి rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
