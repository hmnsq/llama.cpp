#ଅନ୍ତର୍ଭାଵ "align_eltwise_ranks.h"

#ଅନ୍ତର୍ଭାଵ <numeric>
#ଅନ୍ତର୍ଭାଵ <openvino/op/add.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/constant.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/divide.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/multiply.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/sqrt.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/subtract.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/unsqueeze.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/pass/pattern/op/wrap_type.hpp>

ନାମାକାଶ ov {
ନାମାକାଶ frontend {
ନାମାକାଶ ggml {
ନାମାକାଶ pass {

ନାମାକାଶ {

// True for an RMS-norm output, x * (1 / sqrt(mean(x^2) + eps)) as translate_rms_norm builds it, optionally
// scaled by the norm weight.
ତାର୍କିକ is_rms_norm_output(ଅଚଳ ov::Output<ov::Node> & value, ପୂର୍ଣ୍ଣସଂଖ୍ୟା depth = 1) {
    ଅଚଳ auto * ଗଣ୍ଠି = value.get_ଗଣ୍ଠି();
    ଯଦି (!ov::is_type<ov::op::v1::Multiply>(ଗଣ୍ଠି)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ପାଇଁ (ଅଚଳ auto & ନିବେଶ : ଗଣ୍ଠି->ନିବେଶ_ମୂଲ୍ୟଗୁଡ଼ିକ()) {
        ଅଚଳ auto * src = ନିବେଶ.get_ଗଣ୍ଠି();
        ଯଦି (ov::is_type<ov::op::v1::Divide>(src) && ov::is_type<ov::op::v0::Sqrt>(src->get_ନିବେଶ_ଗଣ୍ଠି_ptr(1))) {
            ଫେରନ୍ତୁ ସତ୍ୟ;
        }
        ଯଦି (depth > 0 && is_rms_norm_output(ନିବେଶ, depth - 1)) {
            ଫେରନ୍ତୁ ସତ୍ୟ;
        }
    }
    ଫେରନ୍ତୁ ମିଥ୍ୟା;
}

}  // namespace

AlignEltwiseOperandRanks::AlignEltwiseOperandRanks() {
    auto eltwise_m = ov::pass::pattern::wrap_type<ov::op::v1::Add, ov::op::v1::Multiply, ov::op::v1::Subtract>();

    ଅଚଳ auto callback = [ଏହି](ov::pass::pattern::Matcher & m) {
        auto ଗଣ୍ଠି = m.get_match_ମୂଳ();
        ଯଦି (ଗଣ୍ଠି->get_ନିବେଶ_size() != 2) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        auto lhs = ଗଣ୍ଠି->ନିବେଶ_ମୂଲ୍ୟ(0);
        auto rhs = ଗଣ୍ଠି->ନିବେଶ_ମୂଲ୍ୟ(1);

        // A tensor-vs-Constant mismatch is the norm's own eps / 1-over-sqrt arithmetic. That
        // folds into the `rms` primitive itself instead of becoming a fused eltwise post-op,
        // so it is not affected and is left alone.
        ଯଦି (ov::is_type<ov::op::v0::Constant>(lhs.get_ଗଣ୍ଠି()) ||
            ov::is_type<ov::op::v0::Constant>(rhs.get_ଗଣ୍ଠି())) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        ଅଚଳ auto lhs_rank = lhs.get_partial_shape().rank();
        ଅଚଳ auto rhs_rank = rhs.get_partial_shape().rank();
        ଯଦି (lhs_rank.is_dynamic() || rhs_rank.is_dynamic() || lhs_rank == rhs_rank) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        ଅଚଳ size_t shorter_idx = lhs_rank.get_length() < rhs_rank.get_length() ? 0 : 1;
        ଅଚଳ auto & shorter = shorter_idx == 0 ? lhs : rhs;

        // The defect is with the norm output as the higher-rank operand. When the norm output is the
        // lower-rank one (gemma-3 adds it to a rank-4 residual), unsqueezing it puts the Unsqueeze between
        // `rms` and its post-op, and the GPU plugin then computes the layer wrongly.
        ଯଦି (is_rms_norm_output(shorter)) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        ଅଚଳ int64_t diff = std::abs(lhs_rank.get_length() - rhs_rank.get_length());

        std::vector<int64_t> axes(static_cast<size_t>(diff));
        std::iota(axes.begin(), axes.end(), 0);
        auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(
            shorter, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ axes.size() }, axes));

        ଗଣ୍ଠି->ନିବେଶ(shorter_idx).replace_source_output(unsqueeze->output(0));
        register_new_ଗଣ୍ଠି(unsqueeze);
        ଫେରନ୍ତୁ ସତ୍ୟ;
    };

    register_matcher(
        std::make_shared<ov::pass::pattern::Matcher>(eltwise_m, "ov::frontend::ggml::pass::AlignEltwiseOperandRanks"),
        callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
