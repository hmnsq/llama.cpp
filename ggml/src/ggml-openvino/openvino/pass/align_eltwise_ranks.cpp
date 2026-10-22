#అన్తర్భావ "align_eltwise_ranks.h"

#అన్తర్భావ <numeric>
#అన్తర్భావ <openvino/op/add.hpp>
#అన్తర్భావ <openvino/op/constant.hpp>
#అన్తర్భావ <openvino/op/divide.hpp>
#అన్తర్భావ <openvino/op/multiply.hpp>
#అన్తర్భావ <openvino/op/sqrt.hpp>
#అన్తర్భావ <openvino/op/subtract.hpp>
#అన్తర్భావ <openvino/op/unsqueeze.hpp>
#అన్తర్భావ <openvino/pass/pattern/op/wrap_type.hpp>

నామాకాశ ov {
నామాకాశ frontend {
నామాకాశ ggml {
నామాకాశ pass {

నామాకాశ {

// True for an RMS-norm output, x * (1 / sqrt(mean(x^2) + eps)) as translate_rms_norm builds it, optionally
// scaled by the norm weight.
తార్కిక is_rms_norm_output(స్థిరరాశి ov::Output<ov::Node> & value, పూర్ణాంకం లోతు = 1) {
    స్థిరరాశి స్వయంచాలక * కణుపు = value.get_కణుపు();
    ఒకవేళ (!ov::is_type<ov::op::v1::Multiply>(కణుపు)) {
        తిరిగి అబద్ధం;
    }
    కొరకు (స్థిరరాశి స్వయంచాలక & ఆదానం : కణుపు->ఆదానం_విలువలు()) {
        స్థిరరాశి స్వయంచాలక * src = ఆదానం.get_కణుపు();
        ఒకవేళ (ov::is_type<ov::op::v1::Divide>(src) && ov::is_type<ov::op::v0::Sqrt>(src->get_ఆదానం_కణుపు_ptr(1))) {
            తిరిగి నిజం;
        }
        ఒకవేళ (లోతు > 0 && is_rms_norm_output(ఆదానం, లోతు - 1)) {
            తిరిగి నిజం;
        }
    }
    తిరిగి అబద్ధం;
}

}  // namespace

Align_Eltwise_నిర్వాహ్యం_Ranks::Align_Eltwise_నిర్వాహ్యం_Ranks() {
    స్వయంచాలక eltwise_m = ov::pass::pattern::wrap_type<ov::op::v1::కూడు, ov::op::v1::Multiply, ov::op::v1::వ్యవకలనం>();

    స్థిరరాశి స్వయంచాలక callback = [ఇది](ov::pass::pattern::Matcher & m) {
        స్వయంచాలక కణుపు = m.get_సరిపోలిక_మూలం();
        ఒకవేళ (కణుపు->get_ఆదానం_కొలత() != 2) {
            తిరిగి అబద్ధం;
        }

        స్వయంచాలక lhs = కణుపు->ఆదానం_విలువ(0);
        స్వయంచాలక rhs = కణుపు->ఆదానం_విలువ(1);

        // A tensor-vs-Constant mismatch is the norm's own eps / 1-over-sqrt arithmetic. That
        // folds into the `rms` primitive itself instead of becoming a fused eltwise post-op,
        // so it is not affected and is left alone.
        ఒకవేళ (ov::is_type<ov::op::v0::Constant>(lhs.get_కణుపు()) ||
            ov::is_type<ov::op::v0::Constant>(rhs.get_కణుపు())) {
            తిరిగి అబద్ధం;
        }

        స్థిరరాశి స్వయంచాలక lhs_rank = lhs.get_partial_shape().rank();
        స్థిరరాశి స్వయంచాలక rhs_rank = rhs.get_partial_shape().rank();
        ఒకవేళ (lhs_rank.is_dynamic() || rhs_rank.is_dynamic() || lhs_rank == rhs_rank) {
            తిరిగి అబద్ధం;
        }

        స్థిరరాశి size_t shorter_idx = lhs_rank.get_నిడివి() < rhs_rank.get_నిడివి() ? 0 : 1;
        స్థిరరాశి స్వయంచాలక & shorter = shorter_idx == 0 ? lhs : rhs;

        // The defect is with the norm output as the higher-rank operand. When the norm output is the
        // lower-rank one (gemma-3 adds it to a rank-4 residual), unsqueezing it puts the Unsqueeze between
        // `rms` and its post-op, and the GPU plugin then computes the layer wrongly.
        ఒకవేళ (is_rms_norm_output(shorter)) {
            తిరిగి అబద్ధం;
        }
        స్థిరరాశి int64_t diff = std::నిరపేక్ష(lhs_rank.get_నిడివి() - rhs_rank.get_నిడివి());

        std::vector<int64_t> axes(static_cast<size_t>(diff));
        std::iota(axes.begin(), axes.end(), 0);
        స్వయంచాలక unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(
            shorter, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ axes.size() }, axes));

        కణుపు->ఆదానం(shorter_idx).replace_ఆధారం_output(unsqueeze->output(0));
        రిజిస్టరు_new_కణుపు(unsqueeze);
        తిరిగి నిజం;
    };

    రిజిస్టరు_matcher(
        std::make_shared<ov::pass::pattern::Matcher>(eltwise_m, "ov::frontend::ggml::pass::AlignEltwiseOperandRanks"),
        callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
