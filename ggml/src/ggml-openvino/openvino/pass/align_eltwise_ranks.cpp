#شامل "align_eltwise_ranks.h"

#شامل <numeric>
#شامل <openvino/op/add.hpp>
#شامل <openvino/op/constant.hpp>
#شامل <openvino/op/divide.hpp>
#شامل <openvino/op/multiply.hpp>
#شامل <openvino/op/sqrt.hpp>
#شامل <openvino/op/subtract.hpp>
#شامل <openvino/op/unsqueeze.hpp>
#شامل <openvino/pass/pattern/op/wrap_type.hpp>

فضاینام ov {
فضاینام frontend {
فضاینام ggml {
فضاینام pass {

فضاینام {

// True for an RMS-norm output, x * (1 / sqrt(mean(x^2) + eps)) as translate_rms_norm builds it, optionally
// scaled by the norm weight.
منطقی is_rms_norm_output(ثابت ov::Output<ov::Node> & value, صحیح depth = 1) {
    ثابت auto * گره = value.get_گره();
    اگر (!ov::is_type<ov::op::v1::Multiply>(گره)) {
        بازگشت نادرست;
    }
    برای (ثابت auto & درون_داد : گره->درون_داد_مقادیر()) {
        ثابت auto * src = درون_داد.get_گره();
        اگر (ov::is_type<ov::op::v1::Divide>(src) && ov::is_type<ov::op::v0::Sqrt>(src->get_درون_داد_گره_ptr(1))) {
            بازگشت درست;
        }
        اگر (depth > 0 && is_rms_norm_output(درون_داد, depth - 1)) {
            بازگشت درست;
        }
    }
    بازگشت نادرست;
}

}  // namespace

AlignEltwiseOperandRanks::AlignEltwiseOperandRanks() {
    auto eltwise_m = ov::pass::pattern::wrap_type<ov::op::v1::Add, ov::op::v1::Multiply, ov::op::v1::Subtract>();

    ثابت auto callback = [این](ov::pass::pattern::Matcher & m) {
        auto گره = m.get_match_ریشه();
        اگر (گره->get_درون_داد_size() != 2) {
            بازگشت نادرست;
        }

        auto lhs = گره->درون_داد_مقدار(0);
        auto rhs = گره->درون_داد_مقدار(1);

        // A tensor-vs-Constant mismatch is the norm's own eps / 1-over-sqrt arithmetic. That
        // folds into the `rms` primitive itself instead of becoming a fused eltwise post-op,
        // so it is not affected and is left alone.
        اگر (ov::is_type<ov::op::v0::Constant>(lhs.get_گره()) ||
            ov::is_type<ov::op::v0::Constant>(rhs.get_گره())) {
            بازگشت نادرست;
        }

        ثابت auto lhs_rank = lhs.get_partial_shape().rank();
        ثابت auto rhs_rank = rhs.get_partial_shape().rank();
        اگر (lhs_rank.is_dynamic() || rhs_rank.is_dynamic() || lhs_rank == rhs_rank) {
            بازگشت نادرست;
        }

        ثابت size_t shorter_idx = lhs_rank.get_length() < rhs_rank.get_length() ? 0 : 1;
        ثابت auto & shorter = shorter_idx == 0 ? lhs : rhs;

        // The defect is with the norm output as the higher-rank operand. When the norm output is the
        // lower-rank one (gemma-3 adds it to a rank-4 residual), unsqueezing it puts the Unsqueeze between
        // `rms` and its post-op, and the GPU plugin then computes the layer wrongly.
        اگر (is_rms_norm_output(shorter)) {
            بازگشت نادرست;
        }
        ثابت int64_t diff = std::abs(lhs_rank.get_length() - rhs_rank.get_length());

        std::vector<int64_t> axes(static_cast<size_t>(diff));
        std::iota(axes.begin(), axes.end(), 0);
        auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(
            shorter, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ axes.size() }, axes));

        گره->درون_داد(shorter_idx).replace_source_output(unsqueeze->output(0));
        register_new_گره(unsqueeze);
        بازگشت درست;
    };

    register_matcher(
        std::make_shared<ov::pass::pattern::Matcher>(eltwise_m, "ov::frontend::ggml::pass::AlignEltwiseOperandRanks"),
        callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
