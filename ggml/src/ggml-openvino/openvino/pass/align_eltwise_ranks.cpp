#ضمن "align_eltwise_ranks.h"

#ضمن <numeric>
#ضمن <openvino/op/add.hpp>
#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/divide.hpp>
#ضمن <openvino/op/multiply.hpp>
#ضمن <openvino/op/sqrt.hpp>
#ضمن <openvino/op/subtract.hpp>
#ضمن <openvino/op/unsqueeze.hpp>
#ضمن <openvino/pass/pattern/op/wrap_type.hpp>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق pass {

نطاق {

// True for an RMS-norm output, x * (1 / sqrt(mean(x^2) + eps)) as translate_rms_norm builds it, optionally
// scaled by the norm weight.
منطقي is_rms_norm_output(ثابت ov::Output<ov::Node> & value, صحيح عمق = 1) {
    ثابت خودکار * گره = value.get_گره();
    إذا (!ov::is_type<ov::op::v1::Multiply>(گره)) {
        أرجع خطأ;
    }
    لكل (ثابت خودکار & درون_داد : گره->درون_داد_مقادیر()) {
        ثابت خودکار * src = درون_داد.get_گره();
        إذا (ov::is_type<ov::op::v1::Divide>(src) && ov::is_type<ov::op::v0::Sqrt>(src->get_درون_داد_گره_ptr(1))) {
            أرجع درست;
        }
        إذا (عمق > 0 && is_rms_norm_output(درون_داد, عمق - 1)) {
            أرجع درست;
        }
    }
    أرجع خطأ;
}

}  // namespace

Align_Eltwise_عملوند_Ranks::Align_Eltwise_عملوند_Ranks() {
    خودکار eltwise_m = ov::pass::pattern::wrap_type<ov::op::v1::جمع_کن, ov::op::v1::Multiply, ov::op::v1::تفریق>();

    ثابت خودکار callback = [هذا](ov::pass::pattern::Matcher & m) {
        خودکار گره = m.get_تطابق_ریشه();
        إذا (گره->get_درون_داد_اندازه() != 2) {
            أرجع خطأ;
        }

        خودکار lhs = گره->درون_داد_مقدار(0);
        خودکار rhs = گره->درون_داد_مقدار(1);

        // A tensor-vs-Constant mismatch is the norm's own eps / 1-over-sqrt arithmetic. That
        // folds into the `rms` primitive itself instead of becoming a fused eltwise post-op,
        // so it is not affected and is left alone.
        إذا (ov::is_type<ov::op::v0::Constant>(lhs.get_گره()) ||
            ov::is_type<ov::op::v0::Constant>(rhs.get_گره())) {
            أرجع خطأ;
        }

        ثابت خودکار lhs_rank = lhs.get_partial_shape().rank();
        ثابت خودکار rhs_rank = rhs.get_partial_shape().rank();
        إذا (lhs_rank.is_dynamic() || rhs_rank.is_dynamic() || lhs_rank == rhs_rank) {
            أرجع خطأ;
        }

        ثابت size_t shorter_idx = lhs_rank.get_درازی() < rhs_rank.get_درازی() ? 0 : 1;
        ثابت خودکار & shorter = shorter_idx == 0 ? lhs : rhs;

        // The defect is with the norm output as the higher-rank operand. When the norm output is the
        // lower-rank one (gemma-3 adds it to a rank-4 residual), unsqueezing it puts the Unsqueeze between
        // `rms` and its post-op, and the GPU plugin then computes the layer wrongly.
        إذا (is_rms_norm_output(shorter)) {
            أرجع خطأ;
        }
        ثابت int64_t diff = std::قدر_مطلق(lhs_rank.get_درازی() - rhs_rank.get_درازی());

        std::vector<int64_t> axes(static_cast<size_t>(diff));
        std::iota(axes.begin(), axes.end(), 0);
        خودکار unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(
            shorter, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ axes.size() }, axes));

        گره->درون_داد(shorter_idx).replace_منبع_output(unsqueeze->output(0));
        ثبات_new_گره(unsqueeze);
        أرجع درست;
    };

    ثبات_matcher(
        std::make_shared<ov::pass::pattern::Matcher>(eltwise_m, "ov::frontend::ggml::pass::AlignEltwiseOperandRanks"),
        callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
