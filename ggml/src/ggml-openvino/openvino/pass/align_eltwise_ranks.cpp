#ਅਨ੍ਤਰ੍ਭਾਵ "align_eltwise_ranks.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <numeric>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/add.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/constant.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/divide.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/multiply.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/sqrt.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/subtract.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/unsqueeze.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/pass/pattern/op/wrap_type.hpp>

ਨਾਮਾਕਾਸ਼ ov {
ਨਾਮਾਕਾਸ਼ frontend {
ਨਾਮਾਕਾਸ਼ ggml {
ਨਾਮਾਕਾਸ਼ pass {

ਨਾਮਾਕਾਸ਼ {

// True for an RMS-norm output, x * (1 / sqrt(mean(x^2) + eps)) as translate_rms_norm builds it, optionally
// scaled by the norm weight.
ਤਾਰਕਿਕ is_rms_norm_output(ਅਚਲ ov::Output<ov::Node> & value, ਪੂਰਨ ਡੂੰਘਾਈ = 1) {
    ਅਚਲ ਸਵੈਚਾਲਿਤ * ਗੰਢ = value.get_ਗੰਢ();
    ਜੇ (!ov::is_type<ov::op::v1::Multiply>(ਗੰਢ)) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & ਨਿਵੇਸ਼ : ਗੰਢ->ਨਿਵੇਸ਼_ਮੁੱਲਾਂ()) {
        ਅਚਲ ਸਵੈਚਾਲਿਤ * src = ਨਿਵੇਸ਼.get_ਗੰਢ();
        ਜੇ (ov::is_type<ov::op::v1::Divide>(src) && ov::is_type<ov::op::v0::Sqrt>(src->get_ਨਿਵੇਸ਼_ਗੰਢ_ptr(1))) {
            ਵਾਪਸ ਸੱਚ;
        }
        ਜੇ (ਡੂੰਘਾਈ > 0 && is_rms_norm_output(ਨਿਵੇਸ਼, ਡੂੰਘਾਈ - 1)) {
            ਵਾਪਸ ਸੱਚ;
        }
    }
    ਵਾਪਸ ਝੂਠ;
}

}  // namespace

Align_Eltwise_ਸੰਕਾਰਜ_Ranks::Align_Eltwise_ਸੰਕਾਰਜ_Ranks() {
    ਸਵੈਚਾਲਿਤ eltwise_m = ov::pass::pattern::wrap_type<ov::op::v1::ਜੋੜੋ, ov::op::v1::Multiply, ov::op::v1::ਘਟਾਓ>();

    ਅਚਲ ਸਵੈਚਾਲਿਤ callback = [ਇਹ](ov::pass::pattern::Matcher & m) {
        ਸਵੈਚਾਲਿਤ ਗੰਢ = m.get_ਮੇਲ_ਜੜ੍ਹ();
        ਜੇ (ਗੰਢ->get_ਨਿਵੇਸ਼_ਮਾਪ() != 2) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਸਵੈਚਾਲਿਤ lhs = ਗੰਢ->ਨਿਵੇਸ਼_ਮੁੱਲ(0);
        ਸਵੈਚਾਲਿਤ rhs = ਗੰਢ->ਨਿਵੇਸ਼_ਮੁੱਲ(1);

        // A tensor-vs-Constant mismatch is the norm's own eps / 1-over-sqrt arithmetic. That
        // folds into the `rms` primitive itself instead of becoming a fused eltwise post-op,
        // so it is not affected and is left alone.
        ਜੇ (ov::is_type<ov::op::v0::Constant>(lhs.get_ਗੰਢ()) ||
            ov::is_type<ov::op::v0::Constant>(rhs.get_ਗੰਢ())) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਅਚਲ ਸਵੈਚਾਲਿਤ lhs_rank = lhs.get_partial_shape().rank();
        ਅਚਲ ਸਵੈਚਾਲਿਤ rhs_rank = rhs.get_partial_shape().rank();
        ਜੇ (lhs_rank.is_dynamic() || rhs_rank.is_dynamic() || lhs_rank == rhs_rank) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਅਚਲ size_t shorter_idx = lhs_rank.get_ਲੰਮਾਈ() < rhs_rank.get_ਲੰਮਾਈ() ? 0 : 1;
        ਅਚਲ ਸਵੈਚਾਲਿਤ & shorter = shorter_idx == 0 ? lhs : rhs;

        // The defect is with the norm output as the higher-rank operand. When the norm output is the
        // lower-rank one (gemma-3 adds it to a rank-4 residual), unsqueezing it puts the Unsqueeze between
        // `rms` and its post-op, and the GPU plugin then computes the layer wrongly.
        ਜੇ (is_rms_norm_output(shorter)) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਅਚਲ int64_t diff = std::ਨਿਰਪੇਖ(lhs_rank.get_ਲੰਮਾਈ() - rhs_rank.get_ਲੰਮਾਈ());

        std::vector<int64_t> axes(static_cast<size_t>(diff));
        std::iota(axes.begin(), axes.end(), 0);
        ਸਵੈਚਾਲਿਤ unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(
            shorter, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ axes.size() }, axes));

        ਗੰਢ->ਨਿਵੇਸ਼(shorter_idx).replace_ਸਰੋਤ_output(unsqueeze->output(0));
        ਰਜਿਸਟਰ_new_ਗੰਢ(unsqueeze);
        ਵਾਪਸ ਸੱਚ;
    };

    ਰਜਿਸਟਰ_matcher(
        std::make_shared<ov::pass::pattern::Matcher>(eltwise_m, "ov::frontend::ggml::pass::AlignEltwiseOperandRanks"),
        callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
