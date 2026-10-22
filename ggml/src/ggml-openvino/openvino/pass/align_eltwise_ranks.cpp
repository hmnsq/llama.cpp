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
منطقي is_rms_norm_output(ثابت ov::Output<ov::Node> & value, صحيح اونهائي = 1) {
    ثابت خودڪار * ڳنڍ = value.get_ڳنڍ();
    إذا (!ov::is_type<ov::op::v1::Multiply>(ڳنڍ)) {
        أرجع خطأ;
    }
    لكل (ثابت خودڪار & داخلا_2 : ڳنڍ->داخلا_قدرون()) {
        ثابت خودڪار * src = داخلا_2.get_ڳنڍ();
        إذا (ov::is_type<ov::op::v1::Divide>(src) && ov::is_type<ov::op::v0::Sqrt>(src->get_داخلا_ڳنڍ_ptr(1))) {
            أرجع سچ;
        }
        إذا (اونهائي > 0 && is_rms_norm_output(داخلا_2, اونهائي - 1)) {
            أرجع سچ;
        }
    }
    أرجع خطأ;
}

}  // namespace

Align_Eltwise_عمليو_Ranks::Align_Eltwise_عمليو_Ranks() {
    خودڪار eltwise_m = ov::pass::pattern::wrap_type<ov::op::v1::جوڙيو, ov::op::v1::Multiply, ov::op::v1::ڪٽيو>();

    ثابت خودڪار callback = [هذا](ov::pass::pattern::Matcher & m) {
        خودڪار ڳنڍ = m.get_ميلاپ_پاڙ();
        إذا (ڳنڍ->get_داخلا_سائيز() != 2) {
            أرجع خطأ;
        }

        خودڪار lhs = ڳنڍ->داخلا_قدر(0);
        خودڪار rhs = ڳنڍ->داخلا_قدر(1);

        // A tensor-vs-Constant mismatch is the norm's own eps / 1-over-sqrt arithmetic. That
        // folds into the `rms` primitive itself instead of becoming a fused eltwise post-op,
        // so it is not affected and is left alone.
        إذا (ov::is_type<ov::op::v0::Constant>(lhs.get_ڳنڍ()) ||
            ov::is_type<ov::op::v0::Constant>(rhs.get_ڳنڍ())) {
            أرجع خطأ;
        }

        ثابت خودڪار lhs_rank = lhs.get_partial_shape().rank();
        ثابت خودڪار rhs_rank = rhs.get_partial_shape().rank();
        إذا (lhs_rank.is_dynamic() || rhs_rank.is_dynamic() || lhs_rank == rhs_rank) {
            أرجع خطأ;
        }

        ثابت size_t shorter_idx = lhs_rank.get_ڊيگهائي() < rhs_rank.get_ڊيگهائي() ? 0 : 1;
        ثابت خودڪار & shorter = shorter_idx == 0 ? lhs : rhs;

        // The defect is with the norm output as the higher-rank operand. When the norm output is the
        // lower-rank one (gemma-3 adds it to a rank-4 residual), unsqueezing it puts the Unsqueeze between
        // `rms` and its post-op, and the GPU plugin then computes the layer wrongly.
        إذا (is_rms_norm_output(shorter)) {
            أرجع خطأ;
        }
        ثابت int64_t diff = std::مطلق_قدر(lhs_rank.get_ڊيگهائي() - rhs_rank.get_ڊيگهائي());

        std::vector<int64_t> axes(static_cast<size_t>(diff));
        std::iota(axes.begin(), axes.end(), 0);
        خودڪار unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(
            shorter, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ axes.size() }, axes));

        ڳنڍ->داخلا_2(shorter_idx).replace_ذريعو_output(unsqueeze->output(0));
        رجسٽر_new_ڳنڍ(unsqueeze);
        أرجع سچ;
    };

    رجسٽر_matcher(
        std::make_shared<ov::pass::pattern::Matcher>(eltwise_m, "ov::frontend::ggml::pass::AlignEltwiseOperandRanks"),
        callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
