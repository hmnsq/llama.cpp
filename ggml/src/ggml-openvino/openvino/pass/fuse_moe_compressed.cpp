#ಅನ್ತರ್ಭಾವ "fuse_moe_compressed.h"

#ಅನ್ತರ್ಭಾವ <cstring>
#ಅನ್ತರ್ಭಾವ <limits>
#ಅನ್ತರ್ಭಾವ <set>
#ಅನ್ತರ್ಭಾವ <memory>
#ಅನ್ತರ್ಭಾವ <openvino/core/graph_util.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/core/rt_info.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/constant.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/convert.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/gelu.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/multiply.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/reduce_sum.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/reshape.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/slice.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/squeeze.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/subtract.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/swish.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/transpose.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/unsqueeze.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/pass/constant_folding.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/pass/pattern/op/optional.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/pass/pattern/op/wrap_type.hpp>

#ಅನ್ತರ್ಭಾವ "../op/gather_matmul.hpp"
#ಅನ್ತರ್ಭಾವ "../op/moe_compressed.hpp"

ನಾಮಾಕಾಶ ov {
ನಾಮಾಕಾಶ frontend {
ನಾಮಾಕಾಶ ggml {
ನಾಮಾಕಾಶ pass {

ನಾಮಾಕಾಶ {

ರಚನೆ dequant_inputs {
    ov::Output<ov::Node> ತೂಕ;
    ov::Output<ov::Node> scale;
    ov::Output<ov::Node> zp;
    ತಾರ್ಕಿಕ has_zp = ಸುಳ್ಳು;
    ತಾರ್ಕಿಕ ok = ಸುಳ್ಳು;
};

// Peel the chain built by make_int4_weights/make_int8_weights back to its Constant inputs.
// Grouped weights keep the pre-Reshape rank-4 form [n_expert, n, k/group, group] with scale
// and zp at [n_expert, n, k/group, 1], which is the layout MOECompressed expects. Channel-wise
// weights stay rank-3 with a rank-3 scale and carry no zp.
dequant_inputs unwrap_dequant(ಸ್ಥಿರಾಂಕ ov::Output<ov::Node> & b) {
    dequant_inputs res;

    ಸ್ವಯಂಚಾಲಿತ ಗಂಟು = b.get_ಗಂಟು_shared_ptr();
    ಆಗ (ov::is_type<ov::op::v0::Convert>(ಗಂಟು) || ov::is_type<ov::op::v1::Reshape>(ಗಂಟು)) {
        ಗಂಟು = ಗಂಟು->get_ಒಳಹರಿವು_ಗಂಟು_shared_ptr(0);
    }

    ಸ್ವಯಂಚಾಲಿತ mul = ov::as_type_ptr<ov::op::v1::Multiply>(ಗಂಟು);
    ಆದರೆ (!mul) {
        ಹಿಂತಿರುಗಿ res;
    }
    res.scale = mul->ಒಳಹರಿವು_ಮೌಲ್ಯ(1);

    ಸ್ವಯಂಚಾಲಿತ lhs = mul->get_ಒಳಹರಿವು_ಗಂಟು_shared_ptr(0);
    ಆದರೆ (ಸ್ವಯಂಚಾಲಿತ sub = ov::as_type_ptr<ov::op::v1::ಕಳೆ>(lhs)) {
        // Take the zero point down to its Constant: an integer zp is wrapped in a Convert to f16,
        // and the op wants the integer form. A natively quantized expert instead carries an exact
        // f16 zp (-min/scale) with no integer behind it, which the MoE kernel does not accept.
        ಸ್ವಯಂಚಾಲಿತ zp_ಗಂಟು = sub->get_ಒಳಹರಿವು_ಗಂಟು_shared_ptr(1);
        ಆಗ (ov::is_type<ov::op::v0::Convert>(zp_ಗಂಟು)) {
            zp_ಗಂಟು = zp_ಗಂಟು->get_ಒಳಹರಿವು_ಗಂಟು_shared_ptr(0);
        }
        res.zp = zp_ಗಂಟು->output(0);
        res.has_zp = ನಿಜ;
        lhs = sub->get_ಒಳಹರಿವು_ಗಂಟು_shared_ptr(0);
    }
    ಆಗ (ov::is_type<ov::op::v0::Convert>(lhs)) {
        lhs = lhs->get_ಒಳಹರಿವು_ಗಂಟು_shared_ptr(0);
    }
    ಆದರೆ (!ov::is_type<ov::op::v0::Constant>(lhs)) {
        ಹಿಂತಿರುಗಿ res;
    }

    res.ತೂಕ = lhs->output(0);
    res.ok = res.scale.get_partial_shape().is_static() && res.ತೂಕ.get_partial_shape().is_static();
    ಹಿಂತಿರುಗಿ res;
}

size_t logical_k(ಸ್ಥಿರಾಂಕ ov::Shape & shape) {
    ಹಿಂತಿರುಗಿ shape.size() == 4 ? shape[2] * shape[3] : shape.back();
}

// Slice a [n_expert, m, ...] compressed weight/scale/zp Constant along axis 1 (m), [begin, end).
// Built by copying raw bytes directly instead of a graph Slice op: CommonOptimizations rewrites
// v8::Slice into v1::StridedSlice for constant folding regardless of disable_constant_folding
// (the rewrite doesn't carry the marking over), and that reference evaluator crashes on
// sub-byte (u4/i4) element types. Each axis-1 "row" (the trailing dims) is confirmed
// byte-aligned here -- weight/scale/zp trailing sizes are always whole groups -- so a
// byte-range memcpy per row is exact for both regular and sub-byte element types.
ov::Output<ov::Node> ತುಂಡು_experts_dim1(ಸ್ಥಿರಾಂಕ ov::Output<ov::Node> & tensor, int64_t begin, int64_t end) {
    ಸ್ವಯಂಚಾಲಿತ constant = ov::as_type_ptr<ov::op::v0::Constant>(tensor.get_ಗಂಟು_shared_ptr());
    OPENVINO_ASSERT(constant, "slice_experts_dim1 expects a Constant input");

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & shape = constant->get_shape();
    OPENVINO_ASSERT(shape.size() >= 2, "slice_experts_dim1 expects rank >= 2");
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & type = constant->get_element_type();

    size_t ಅಡ್ಡಸಾಲು_elems = 1;
    ಗಾಗಿ (size_t i = 2; i < shape.size(); ++i) {
        ಅಡ್ಡಸಾಲು_elems *= shape[i];
    }
    ಸ್ಥಿರಾಂಕ size_t ಅಡ್ಡಸಾಲು_ಬಿಟ್ಟುಗಳು = ಅಡ್ಡಸಾಲು_elems * type.bitwidth();
    OPENVINO_ASSERT(ಅಡ್ಡಸಾಲು_ಬಿಟ್ಟುಗಳು % 8 == 0, "slice_experts_dim1: row is not byte-aligned");
    ಸ್ಥಿರಾಂಕ size_t ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು = ಅಡ್ಡಸಾಲು_ಬಿಟ್ಟುಗಳು / 8;

    ov::Shape ಹೊರಹೋಗು_shape = shape;
    ಹೊರಹೋಗು_shape[1] = static_cast<size_t>(end - begin);

    std::vector<uint8_t> ಹೊರಹೋಗು_ದತ್ತಾಂಶ(shape[0] * static_cast<size_t>(end - begin) * ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು);
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * src = static_cast<ಸ್ಥಿರಾಂಕ uint8_t *>(constant->get_ದತ್ತಾಂಶ_ptr());
    ಸ್ಥಿರಾಂಕ size_t src_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು = shape[1] * ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು;
    ಗಾಗಿ (size_t e = 0; e < shape[0]; ++e) {
        std::ಸ್ಮರಣೆ_ನಕಲು(ಹೊರಹೋಗು_ದತ್ತಾಂಶ.ದತ್ತಾಂಶ() + e * static_cast<size_t>(end - begin) * ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು,
                   src + e * src_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು + static_cast<size_t>(begin) * ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು,
                   static_cast<size_t>(end - begin) * ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು);
    }

    ಹಿಂತಿರುಗಿ std::make_shared<ov::op::v0::Constant>(type, ಹೊರಹೋಗು_shape, ಹೊರಹೋಗು_ದತ್ತಾಂಶ.ದತ್ತಾಂಶ());
}

}  // namespace

FuseMoeCompressed::FuseMoeCompressed() {
    ಬಳಸಿ ನಾಮಾಕಾಶ ov::pass::pattern;

    // The gate and up projections each get their own Reshape/Transpose of the hidden state and
    // their own Reshape of the routing ids, so every branch needs its own sub-pattern. On GPU
    // mul_mat_id also converts the activations to f16 before the op and back to f32 after it,
    // so those Converts are matched as optional.
    ಸ್ವಯಂಚಾಲಿತ hidden_gate_m = any_ಒಳಹರಿವು();
    ಸ್ವಯಂಚಾಲಿತ a_gate_reshape_m = wrap_type<ov::op::v1::Reshape>({ hidden_gate_m, any_ಒಳಹರಿವು() });
    ಸ್ವಯಂಚಾಲಿತ a_gate_m =
        wrap_type<ov::op::v1::Transpose>({ optional<ov::op::v0::Convert>({ a_gate_reshape_m }), any_ಒಳಹರಿವು() });
    ಸ್ವಯಂಚಾಲಿತ hidden_ಮೇಲೆ_m = any_ಒಳಹರಿವು();
    ಸ್ವಯಂಚಾಲಿತ a_ಮೇಲೆ_m = wrap_type<ov::op::v1::Transpose>(
        { optional<ov::op::v0::Convert>({ wrap_type<ov::op::v1::Reshape>({ hidden_ಮೇಲೆ_m, any_ಒಳಹರಿವು() }) }),
          any_ಒಳಹರಿವು() });

    ಸ್ವಯಂಚಾಲಿತ gate_w_m = any_ಒಳಹರಿವು();
    ಸ್ವಯಂಚಾಲಿತ ಮೇಲೆ_w_m = any_ಒಳಹರಿವು();
    ಸ್ವಯಂಚಾಲಿತ ಕೆಳಕ್ಕೆ_w_m = any_ಒಳಹರಿವು();
    ಸ್ವಯಂಚಾಲಿತ ids_gate_m = any_ಒಳಹರಿವು();
    ಸ್ವಯಂಚಾಲಿತ ids_ಮೇಲೆ_m = any_ಒಳಹರಿವು();
    ಸ್ವಯಂಚಾಲಿತ ids_ಕೆಳಕ್ಕೆ_m = any_ಒಳಹರಿವು();

    ಸ್ವಯಂಚಾಲಿತ bgm_gate_m = wrap_type<ov::op::internal::GatherMatmul>({ a_gate_m, gate_w_m, ids_gate_m, any_ಒಳಹರಿವು() });
    ಸ್ವಯಂಚಾಲಿತ gate_u_m = optional<ov::op::v0::Convert>({ wrap_type<ov::op::v0::Unsqueeze>(
        { wrap_type<ov::op::v1::Transpose>({ bgm_gate_m, any_ಒಳಹರಿವು() }), any_ಒಳಹರಿವು() }) });

    ಸ್ವಯಂಚಾಲಿತ silu_m = wrap_type<ov::op::v4::Swish>({ gate_u_m });

    ಸ್ವಯಂಚಾಲಿತ bgm_ಮೇಲೆ_m = wrap_type<ov::op::internal::GatherMatmul>({ a_ಮೇಲೆ_m, ಮೇಲೆ_w_m, ids_ಮೇಲೆ_m, any_ಒಳಹರಿವು() });
    ಸ್ವಯಂಚಾಲಿತ ಮೇಲೆ_u_m = optional<ov::op::v0::Convert>({ wrap_type<ov::op::v0::Unsqueeze>(
        { wrap_type<ov::op::v1::Transpose>({ bgm_ಮೇಲೆ_m, any_ಒಳಹರಿವು() }), any_ಒಳಹರಿವು() }) });
    ಸ್ವಯಂಚಾಲಿತ swiglu_m = wrap_type<ov::op::v1::Multiply>({ silu_m, ಮೇಲೆ_u_m });

    ಸ್ವಯಂಚಾಲಿತ d_t_m = wrap_type<ov::op::v1::Transpose>(
        { optional<ov::op::v0::Convert>({ wrap_type<ov::op::v1::Reshape>({ swiglu_m, any_ಒಳಹರಿವು() }) }),
          any_ಒಳಹರಿವು() });
    ಸ್ವಯಂಚಾಲಿತ bgm_ಕೆಳಕ್ಕೆ_m = wrap_type<ov::op::internal::GatherMatmul>({ d_t_m, ಕೆಳಕ್ಕೆ_w_m, ids_ಕೆಳಕ್ಕೆ_m, any_ಒಳಹರಿವು() });
    ಸ್ವಯಂಚಾಲಿತ ಕೆಳಕ್ಕೆ_u_m = optional<ov::op::v0::Convert>({ wrap_type<ov::op::v0::Unsqueeze>(
        { wrap_type<ov::op::v1::Transpose>({ bgm_ಕೆಳಕ್ಕೆ_m, any_ಒಳಹರಿವು() }), any_ಒಳಹರಿವು() }) });

    ಸ್ವಯಂಚಾಲಿತ routing_m = any_ಒಳಹರಿವು();
    ಸ್ವಯಂಚಾಲಿತ weighted_m = wrap_type<ov::op::v1::Multiply>({ ಕೆಳಕ್ಕೆ_u_m, routing_m });
    ಸ್ವಯಂಚಾಲಿತ ಬೇರು_m = wrap_type<ov::op::v1::Reduce_ಮೊತ್ತ>({ weighted_m, any_ಒಳಹರಿವು() });

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ callback = [=](Matcher & m) {
        ಸ್ವಯಂಚಾಲಿತ & pm = m.get_ನಮೂನೆ_ಮೌಲ್ಯ_map();

        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ gate = unwrap_dequant(pm.at(gate_w_m));
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ ಮೇಲೆ = unwrap_dequant(pm.at(ಮೇಲೆ_w_m));
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ ಕೆಳಕ್ಕೆ = unwrap_dequant(pm.at(ಕೆಳಕ್ಕೆ_w_m));
        ಆದರೆ (!gate.ok || !ಮೇಲೆ.ok || !ಕೆಳಕ್ಕೆ.ok) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ gate_shape = gate.ತೂಕ.get_shape();
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ ಮೇಲೆ_shape = ಮೇಲೆ.ತೂಕ.get_shape();
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ ಕೆಳಕ್ಕೆ_shape = ಕೆಳಕ್ಕೆ.ತೂಕ.get_shape();
        ಆದರೆ (gate_shape != ಮೇಲೆ_shape || gate_shape.size() < 3 || ಕೆಳಕ್ಕೆ_shape.size() < 3) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        // MOECompressed carries one group_size and one has_zp for all three projections, so a
        // model whose down-proj is quantized differently from gate/up cannot be described. This
        // happens when ggml requantizes Q5_K/Q6_K experts to channel-wise int8.
        ಆದರೆ (gate.has_zp != ಕೆಳಕ್ಕೆ.has_zp || gate_shape.size() != ಕೆಳಕ್ಕೆ_shape.size()) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        // The kernel only takes an integer zero point (moe_3gemm_swiglu_opt validate_impl).
        ಆದರೆ (gate.has_zp) {
            ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ std::set<ov::element::Type> int_zp_types = { ov::element::u4, ov::element::i4,
                                                                     ov::element::u8, ov::element::i8 };
            ಆದರೆ (int_zp_types.count(gate.zp.get_element_type()) == 0 ||
                int_zp_types.count(ಕೆಳಕ್ಕೆ.zp.get_element_type()) == 0) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
        }

        // Config holds a single group_size for all three projections.
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ ಗುಂಪು_ದ = [](ಸ್ಥಿರಾಂಕ dequant_inputs & w) {
            ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ s = w.ತೂಕ.get_shape();
            ಹಿಂತಿರುಗಿ s.size() == 4 ? s[3] : logical_k(s);
        };
        ಆದರೆ (ಗುಂಪು_ದ(gate) != ಗುಂಪು_ದ(ಮೇಲೆ) || ಗುಂಪು_ದ(gate) != ಗುಂಪು_ದ(ಕೆಳಕ್ಕೆ)) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        // all three branches must route the same hidden state through the same experts
        ಆದರೆ (pm.at(hidden_gate_m) != pm.at(hidden_ಮೇಲೆ_m)) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಸ್ವಯಂಚಾಲಿತ ids = pm.at(ids_ಕೆಳಕ್ಕೆ_m);
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ ids_pshape = ids.get_partial_shape();
        ಆದರೆ (ids_pshape.rank().is_dynamic() || ids_pshape[ids_pshape.rank().get_ಉದ್ದಳತೆ() - 1].is_dynamic()) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಸ್ಥಿರಾಂಕ size_t ತುದಿ_k = ids_pshape[ids_pshape.rank().get_ಉದ್ದಳತೆ() - 1].get_ಉದ್ದಳತೆ();

        // routing weights arrive as [1, n_tokens, top_k, 1]; the op wants [..., top_k]
        ಸ್ವಯಂಚಾಲಿತ routing = pm.at(routing_m);
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ routing_pshape = routing.get_partial_shape();
        ಆದರೆ (routing_pshape.rank().is_dynamic() || routing_pshape.rank().get_ಉದ್ದಳತೆ() != 4 ||
            routing_pshape[3] != 1) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        // MOE requires routing weights and ids to have the same shape. Drop the trailing 1 of the
        // routing weights and give the ids the leading batch dim, so both become [1, n_tokens, top_k].
        routing = std::make_shared<ov::op::v0::Squeeze>(
            routing, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 3 }));
        ಆದರೆ (ids_pshape.rank().get_ಉದ್ದಳತೆ() == 2) {
            ids = std::make_shared<ov::op::v0::Unsqueeze>(
                ids, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 0 }));
        }
        ಆದರೆ (routing.get_partial_shape() != ids.get_partial_shape()) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಸ್ಥಿರಾಂಕ size_t ಕೆಳಕ್ಕೆ_k = logical_k(ಕೆಳಕ್ಕೆ_shape);
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ ಕೆಳಕ್ಕೆ_scale_shape = ಕೆಳಕ್ಕೆ.scale.get_shape();
        ಸ್ಥಿರಾಂಕ size_t ಕೆಳಕ್ಕೆ_groups = ಕೆಳಕ್ಕೆ_scale_shape.size() >= 3 ? ಕೆಳಕ್ಕೆ_scale_shape[2] : 1;

        ov::op::internal::MOECompressed::Config config;
        config.expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
        config.activation_type = ov::op::internal::MOE::Activation_type::SWIGLU;
        config.expert_alpha = 0.0f;
        config.expert_beta = 1.0f;
        config.gate_idx = 0;
        config.hidden_ಗಾತ್ರ = logical_k(gate_shape);
        config.inter_ಗಾತ್ರ = gate_shape[1];
        config.ಅಂಕೆ_expert = gate_shape[0];
        config.ಅಂಕೆ_shared_expert = 0;
        config.ತುದಿ_k = ತುದಿ_k;
        config.ಗುಂಪು_ಗಾತ್ರ = ಕೆಳಕ್ಕೆ_groups <= 1 ? std::numeric_limits<size_t>::max() : ಕೆಳಕ್ಕೆ_k / ಕೆಳಕ್ಕೆ_groups;
        config.has_batch_dim = ನಿಜ;
        config.has_zp = gate.has_zp;
        // dynamic makes the output follow the hidden state, so the plugin can lower this region
        // to f16 together with the rest of the graph
        config.ಹೊರಹೋಗು_type = ov::element::dynamic;

        ಸ್ವಯಂಚಾಲಿತ absent_zp = [] {
            ಸ್ವಯಂಚಾಲಿತ zp = std::make_shared<ov::op::v0::Constant>(ov::element::dynamic, ov::Shape{ 0 });
            ov::pass::disable_constant_folding(zp);
            ಹಿಂತಿರುಗಿ zp->output(0);
        };

        // MOE takes its output type from the hidden state. Transpose the activations before the
        // f16 Convert that mul_mat_id adds on GPU, so the op stays f32 like the block it replaces
        // and the plugin can lower the whole region uniformly.
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ a_transpose = pm.at(a_gate_m).get_ಗಂಟು_shared_ptr();
        ov::Output<ov::Node> hidden =
            std::make_shared<ov::op::v1::Transpose>(pm.at(a_gate_reshape_m), a_transpose->ಒಳಹರಿವು_ಮೌಲ್ಯ(1));

        ಸ್ಥಿರಾಂಕ ov::Output_ಸದಿಶ ವಾದಗಳು_2 = {
            hidden, routing, ids,
            gate.ತೂಕ,  gate.scale,   gate.has_zp ? gate.zp : absent_zp(),
            ಮೇಲೆ.ತೂಕ,    ಮೇಲೆ.scale,     ಮೇಲೆ.has_zp ? ಮೇಲೆ.zp : absent_zp(),
            ಕೆಳಕ್ಕೆ.ತೂಕ,  ಕೆಳಕ್ಕೆ.scale,   ಕೆಳಕ್ಕೆ.has_zp ? ಕೆಳಕ್ಕೆ.zp : absent_zp(),
        };

        ಸ್ವಯಂಚಾಲಿತ moe = std::make_shared<ov::op::internal::MOECompressed>(ವಾದಗಳು_2, config);

        // MOE takes its output type from the hidden state, which is f16 on GPU, while the rest of
        // the ggml graph works in f32.
        ov::Output<ov::Node> ಫಲಿತಾಂಶ = moe->output(0);
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ ಬೇರು_type = m.get_ಹೊಂದಿಕೆ_ಬೇರು()->get_output_element_type(0);
        ಆದರೆ (ಫಲಿತಾಂಶ.get_element_type() != ಬೇರು_type) {
            ಫಲಿತಾಂಶ = std::make_shared<ov::op::v0::Convert>(ಫಲಿತಾಂಶ, ಬೇರು_type);
        }

        ಫಲಿತಾಂಶ.get_ಗಂಟು_shared_ptr()->ಗಣ_friendly_name(m.get_ಹೊಂದಿಕೆ_ಬೇರು()->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), ಫಲಿತಾಂಶ.get_ಗಂಟು_shared_ptr());
        ov::replace_ಗಂಟು(m.get_ಹೊಂದಿಕೆ_ಬೇರು(), ಫಲಿತಾಂಶ.get_ಗಂಟು_shared_ptr());
        ರಿಜಿಸ್ಟರ್_new_ಗಂಟು(moe);
        ಹಿಂತಿರುಗಿ ನಿಜ;
    };

    ರಿಜಿಸ್ಟರ್_matcher(std::make_shared<Matcher>(ಬೇರು_m, "ov::frontend::ggml::pass::FuseMoeCompressed"), callback);
}

Fuse_Moe_Compressed_Fused_Gate_ಮೇಲೆ::Fuse_Moe_Compressed_Fused_Gate_ಮೇಲೆ() {
    ಬಳಸಿ ನಾಮಾಕಾಶ ov::pass::pattern;

    // A single GatherMatmul computes the fused gate+up projection; the gate/up split happens
    // AFTER the GEMM via two Slice ops on the last (feature) axis (see get_glu_inputs in
    // op/glu_geglu.cpp), unlike FuseMoeCompressed's two-separate-GatherMatmul models.
    ಸ್ವಯಂಚಾಲಿತ hidden_m = any_ಒಳಹರಿವು();
    ಸ್ವಯಂಚಾಲಿತ a_reshape_m = wrap_type<ov::op::v1::Reshape>({ hidden_m, any_ಒಳಹರಿವು() });
    ಸ್ವಯಂಚಾಲಿತ a_m = wrap_type<ov::op::v1::Transpose>({ optional<ov::op::v0::Convert>({ a_reshape_m }), any_ಒಳಹರಿವು() });

    ಸ್ವಯಂಚಾಲಿತ gate_ಮೇಲೆ_w_m = any_ಒಳಹರಿವು();
    ಸ್ವಯಂಚಾಲಿತ ids_gate_ಮೇಲೆ_m = any_ಒಳಹರಿವು();
    ಸ್ವಯಂಚಾಲಿತ bgm_fused_m = wrap_type<ov::op::internal::GatherMatmul>({ a_m, gate_ಮೇಲೆ_w_m, ids_gate_ಮೇಲೆ_m, any_ಒಳಹರಿವು() });
    // Two graph shapes reach here. The rank-4 (stateless) graph restores the batch dim with an
    // Unsqueeze after the Transpose and may Convert afterwards:
    //     GatherMatmul -> Transpose -> Unsqueeze -> [Convert] -> Slice
    // The rank-3 (stateful) graph never drops to a batch dim at all, so there is no Unsqueeze
    // here - it appears later, between the GEGLU and the down Reshape - and the Convert sits on
    // the other side of the Transpose:
    //     GatherMatmul -> Convert -> Transpose -> Slice
    // optional<T>({a, b}) matches T(a, b) or bare a, so one pattern covers both.
    ಸ್ವಯಂಚಾಲಿತ gu_t_m =
        wrap_type<ov::op::v1::Transpose>({ optional<ov::op::v0::Convert>({ bgm_fused_m }), any_ಒಳಹರಿವು() });
    ಸ್ವಯಂಚಾಲಿತ gu_u_m =
        optional<ov::op::v0::Convert>({ optional<ov::op::v0::Unsqueeze>({ gu_t_m, any_ಒಳಹರಿವು() }) });

    ಸ್ವಯಂಚಾಲಿತ gate_ತುಂಡು_m = wrap_type<ov::op::v8::ತುಂಡು>({ gu_u_m, any_ಒಳಹರಿವು(), any_ಒಳಹರಿವು(), any_ಒಳಹರಿವು(), any_ಒಳಹರಿವು() });
    ಸ್ವಯಂಚಾಲಿತ ಮೇಲೆ_ತುಂಡು_m = wrap_type<ov::op::v8::ತುಂಡು>({ gu_u_m, any_ಒಳಹರಿವು(), any_ಒಳಹರಿವು(), any_ಒಳಹರಿವು(), any_ಒಳಹರಿವು() });
    ಸ್ವಯಂಚಾಲಿತ gelu_m = wrap_type<ov::op::v7::Gelu>({ gate_ತುಂಡು_m });
    ಸ್ವಯಂಚಾಲಿತ geglu_m = wrap_type<ov::op::v1::Multiply>({ gelu_m, ಮೇಲೆ_ತುಂಡು_m });

    // The rank-3 graph inserts the Unsqueeze the gate/up branch lacked right here, before the
    // Reshape that feeds the down projection; the rank-4 graph goes straight from the GEGLU
    // into the Reshape.
    ಸ್ವಯಂಚಾಲಿತ d_t_m = wrap_type<ov::op::v1::Transpose>(
        { optional<ov::op::v0::Convert>({ wrap_type<ov::op::v1::Reshape>(
              { optional<ov::op::v0::Unsqueeze>({ geglu_m, any_ಒಳಹರಿವು() }), any_ಒಳಹರಿವು() }) }),
          any_ಒಳಹರಿವು() });
    ಸ್ವಯಂಚಾಲಿತ ಕೆಳಕ್ಕೆ_w_m = any_ಒಳಹರಿವು();
    ಸ್ವಯಂಚಾಲಿತ ids_ಕೆಳಕ್ಕೆ_m = any_ಒಳಹರಿವು();
    ಸ್ವಯಂಚಾಲಿತ bgm_ಕೆಳಕ್ಕೆ_m = wrap_type<ov::op::internal::GatherMatmul>({ d_t_m, ಕೆಳಕ್ಕೆ_w_m, ids_ಕೆಳಕ್ಕೆ_m, any_ಒಳಹರಿವು() });
    // Same two shapes as the gate/up branch above.
    ಸ್ವಯಂಚಾಲಿತ ಕೆಳಕ್ಕೆ_t_m =
        wrap_type<ov::op::v1::Transpose>({ optional<ov::op::v0::Convert>({ bgm_ಕೆಳಕ್ಕೆ_m }), any_ಒಳಹರಿವು() });
    ಸ್ವಯಂಚಾಲಿತ ಕೆಳಕ್ಕೆ_u_m =
        optional<ov::op::v0::Convert>({ optional<ov::op::v0::Unsqueeze>({ ಕೆಳಕ್ಕೆ_t_m, any_ಒಳಹರಿವು() }) });

    // gemma-4 applies an extra per-expert output scale to the down projection before the
    // router-weight multiply (llama-graph.cpp's ffn_down_exps.scale); FuseMoeCompressed's
    // Qwen-shaped models have no such scale, so this node is specific to this pattern.
    ಸ್ವಯಂಚಾಲಿತ ಕೆಳಕ್ಕೆ_scale_m = any_ಒಳಹರಿವು();
    ಸ್ವಯಂಚಾಲಿತ ಕೆಳಕ್ಕೆ_scaled_m = wrap_type<ov::op::v1::Multiply>({ ಕೆಳಕ್ಕೆ_u_m, ಕೆಳಕ್ಕೆ_scale_m });

    ಸ್ವಯಂಚಾಲಿತ routing_m = any_ಒಳಹರಿವು();
    ಸ್ವಯಂಚಾಲಿತ weighted_m = wrap_type<ov::op::v1::Multiply>({ ಕೆಳಕ್ಕೆ_scaled_m, routing_m });
    ಸ್ವಯಂಚಾಲಿತ ಬೇರು_m = wrap_type<ov::op::v1::Reduce_ಮೊತ್ತ>({ weighted_m, any_ಒಳಹರಿವು() });

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ callback = [=](Matcher & m) {
        ಸ್ವಯಂಚಾಲಿತ & pm = m.get_ನಮೂನೆ_ಮೌಲ್ಯ_map();

        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ gate_ಮೇಲೆ = unwrap_dequant(pm.at(gate_ಮೇಲೆ_w_m));
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ ಕೆಳಕ್ಕೆ = unwrap_dequant(pm.at(ಕೆಳಕ್ಕೆ_w_m));
        ಆದರೆ (!gate_ಮೇಲೆ.ok || !ಕೆಳಕ್ಕೆ.ok) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        // oneDNN's weight-decompression GEMM needs a real zero point, so a symmetric (no zp)
        // projection cannot be fused here -- e.g. the GPU-only Q4_0 fallback in
        // ggml_openvino_get_extracted_layout for grouped 8-bit expert weights.
        ಆದರೆ (!gate_ಮೇಲೆ.has_zp || !ಕೆಳಕ್ಕೆ.has_zp) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ gate_ಮೇಲೆ_shape = gate_ಮೇಲೆ.ತೂಕ.get_shape();
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ ಕೆಳಕ್ಕೆ_shape = ಕೆಳಕ್ಕೆ.ತೂಕ.get_shape();
        ಆದರೆ (gate_ಮೇಲೆ_shape.size() < 3 || ಕೆಳಕ್ಕೆ_shape.size() < 3 || gate_ಮೇಲೆ_shape.size() != ಕೆಳಕ್ಕೆ_shape.size()) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        // The op reads the zero point straight off a weight port, so it must already be an
        // integer Constant. Requantized experts (GGML_OPENVINO_REQUANT_KQUANT=q4_asym64_all)
        // are; a native Q4_K expert keeps an exact f16 zp and is left to the unfused path
        // rather than rounded here -- rounding would put a live node on that port and the
        // expert GEMM would read the wrong zero point.
        ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ std::set<ov::element::Type> int_types = { ov::element::u4, ov::element::i4,
                                                               ov::element::u8, ov::element::i8 };
        ಆದರೆ (int_types.count(gate_ಮೇಲೆ.ತೂಕ.get_element_type()) == 0 ||
            int_types.count(ಕೆಳಕ್ಕೆ.ತೂಕ.get_element_type()) == 0 ||
            int_types.count(gate_ಮೇಲೆ.zp.get_element_type()) == 0 ||
            int_types.count(ಕೆಳಕ್ಕೆ.zp.get_element_type()) == 0) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ ಗುಂಪು_ದ = [](ಸ್ಥಿರಾಂಕ dequant_inputs & w) {
            ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ s = w.ತೂಕ.get_shape();
            ಹಿಂತಿರುಗಿ s.size() == 4 ? s[3] : logical_k(s);
        };
        ಆದರೆ (ಗುಂಪು_ದ(gate_ಮೇಲೆ) != ಗುಂಪು_ದ(ಕೆಳಕ್ಕೆ)) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಸ್ವಯಂಚಾಲಿತ gelu_ಗಂಟು = ov::as_type_ptr<ov::op::v7::Gelu>(pm.at(gelu_m).get_ಗಂಟು_shared_ptr());
        ಆದರೆ (!gelu_ಗಂಟು || gelu_ಗಂಟು->get_approximation_ಬಹುಲಕ() != ov::op::Gelu_Approximation_ಬಹುಲಕ::ERF) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        // Read the actual gate/up split point from the matched Slice nodes instead of assuming
        // inter_size/2, so this stays correct if the model's ffn_dim convention ever changes.
        ಸ್ವಯಂಚಾಲಿತ gate_ತುಂಡು = ov::as_type_ptr<ov::op::v8::ತುಂಡು>(pm.at(gate_ತುಂಡು_m).get_ಗಂಟು_shared_ptr());
        ಸ್ವಯಂಚಾಲಿತ ಮೇಲೆ_ತುಂಡು = ov::as_type_ptr<ov::op::v8::ತುಂಡು>(pm.at(ಮೇಲೆ_ತುಂಡು_m).get_ಗಂಟು_shared_ptr());
        ಸ್ವಯಂಚಾಲಿತ gate_ಕೊನೆ_c = ov::as_type_ptr<ov::op::v0::Constant>(gate_ತುಂಡು->ಒಳಹರಿವು_ಮೌಲ್ಯ(2).get_ಗಂಟು_shared_ptr());
        ಸ್ವಯಂಚಾಲಿತ ಮೇಲೆ_begin_c = ov::as_type_ptr<ov::op::v0::Constant>(ಮೇಲೆ_ತುಂಡು->ಒಳಹರಿವು_ಮೌಲ್ಯ(1).get_ಗಂಟು_shared_ptr());
        ಆದರೆ (!gate_ಕೊನೆ_c || !ಮೇಲೆ_begin_c) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ gate_ಕೊನೆ = gate_ಕೊನೆ_c->cast_ಸದಿಶ<int64_t>().at(0);
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ ಮೇಲೆ_begin = ಮೇಲೆ_begin_c->cast_ಸದಿಶ<int64_t>().at(0);
        ಆದರೆ (gate_ಕೊನೆ != ಮೇಲೆ_begin || gate_ಕೊನೆ <= 0 || gate_ಕೊನೆ >= static_cast<int64_t>(gate_ಮೇಲೆ_shape[1])) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಸ್ಥಿರಾಂಕ int64_t mid = gate_ಕೊನೆ;
        ಸ್ಥಿರಾಂಕ int64_t ತುಂಬಿದ = static_cast<int64_t>(gate_ಮೇಲೆ_shape[1]);

        ಸ್ವಯಂಚಾಲಿತ ids = pm.at(ids_ಕೆಳಕ್ಕೆ_m);
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ ids_pshape = ids.get_partial_shape();
        ಆದರೆ (ids_pshape.rank().is_dynamic() || ids_pshape[ids_pshape.rank().get_ಉದ್ದಳತೆ() - 1].is_dynamic()) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಸ್ಥಿರಾಂಕ size_t ತುದಿ_k = ids_pshape[ids_pshape.rank().get_ಉದ್ದಳತೆ() - 1].get_ಉದ್ದಳತೆ();

        // Routing weights arrive as [1, n_tokens, top_k, 1] on the rank-4 (stateless) graph and
        // as [n_tokens, top_k, 1] on the rank-3 (stateful) one, which is the same thing without
        // the leading batch dim. Normalise the rank-3 form up to the rank-4 one so everything
        // below - and the op's own config - stays in the shape that is already validated on the
        // stateless path; the batch dim is taken back off the result at the end.
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ routing_ಒಳ = pm.at(routing_m);
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ routing_pshape = routing_ಒಳ.get_partial_shape();
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ routing_rank = routing_pshape.rank();
        ಆದರೆ (routing_rank.is_dynamic() || (routing_rank.get_ಉದ್ದಳತೆ() != 4 && routing_rank.get_ಉದ್ದಳತೆ() != 3) ||
            routing_pshape[routing_rank.get_ಉದ್ದಳತೆ() - 1] != 1) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ batchless = routing_rank.get_ಉದ್ದಳತೆ() == 3;

        // Fold gemma-4's per-expert output scale into the routing weights: the reduction is
        // sum_e(routing[e] * scale[e] * down_out[e]), and MOECompressed only takes one
        // per-expert weight, so pre-multiply it into routing here (same [.., top_k, 1] shape).
        ಸ್ವಯಂಚಾಲಿತ ಕೆಳಕ್ಕೆ_scale = pm.at(ಕೆಳಕ್ಕೆ_scale_m);
        ಆದರೆ (ಕೆಳಕ್ಕೆ_scale.get_partial_shape() != routing_pshape) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ov::Output<ov::Node> routing = std::make_shared<ov::op::v1::Multiply>(routing_ಒಳ, ಕೆಳಕ್ಕೆ_scale);
        ಆದರೆ (batchless) {
            routing = std::make_shared<ov::op::v0::Unsqueeze>(
                routing, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 0 }));
        }
        routing = std::make_shared<ov::op::v0::Squeeze>(
            routing, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 3 }));
        ಆದರೆ (ids_pshape.rank().get_ಉದ್ದಳತೆ() == 2) {
            ids = std::make_shared<ov::op::v0::Unsqueeze>(
                ids, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 0 }));
        }
        ಆದರೆ (routing.get_partial_shape() != ids.get_partial_shape()) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಸ್ಥಿರಾಂಕ size_t ಕೆಳಕ್ಕೆ_k = logical_k(ಕೆಳಕ್ಕೆ_shape);
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ ಕೆಳಕ್ಕೆ_scale_shape = ಕೆಳಕ್ಕೆ.scale.get_shape();
        ಸ್ಥಿರಾಂಕ size_t ಕೆಳಕ್ಕೆ_groups = ಕೆಳಕ್ಕೆ_scale_shape.size() >= 3 ? ಕೆಳಕ್ಕೆ_scale_shape[2] : 1;

        // Split the fused gate_up weight/scale/zp on the output axis. slice_experts_dim1 copies
        // raw bytes out of the Constant, so each half stays a Constant -- which is what the op
        // needs on its weight ports.
        ಸ್ಥಿರಾಂಕ ov::Output<ov::Node> gate_ತೂಕ = ತುಂಡು_experts_dim1(gate_ಮೇಲೆ.ತೂಕ, 0, mid);
        ಸ್ಥಿರಾಂಕ ov::Output<ov::Node> ಮೇಲೆ_ತೂಕ = ತುಂಡು_experts_dim1(gate_ಮೇಲೆ.ತೂಕ, mid, ತುಂಬಿದ);
        ಸ್ಥಿರಾಂಕ ov::Output<ov::Node> gate_scale = ತುಂಡು_experts_dim1(gate_ಮೇಲೆ.scale, 0, mid);
        ಸ್ಥಿರಾಂಕ ov::Output<ov::Node> ಮೇಲೆ_scale = ತುಂಡು_experts_dim1(gate_ಮೇಲೆ.scale, mid, ತುಂಬಿದ);
        ಸ್ಥಿರಾಂಕ ov::Output<ov::Node> gate_zp = ತುಂಡು_experts_dim1(gate_ಮೇಲೆ.zp, 0, mid);
        ಸ್ಥಿರಾಂಕ ov::Output<ov::Node> ಮೇಲೆ_zp = ತುಂಡು_experts_dim1(gate_ಮೇಲೆ.zp, mid, ತುಂಬಿದ);

        ov::op::internal::MOECompressed::Config config;
        config.expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
        config.activation_type = ov::op::internal::MOE::Activation_type::GEGLU_ERF;
        config.expert_alpha = 0.0f;
        config.expert_beta = 1.0f;
        config.gate_idx = 0;
        config.hidden_ಗಾತ್ರ = logical_k(gate_ಮೇಲೆ_shape);
        config.inter_ಗಾತ್ರ = static_cast<size_t>(mid);
        config.ಅಂಕೆ_expert = gate_ಮೇಲೆ_shape[0];
        config.ಅಂಕೆ_shared_expert = 0;
        config.ತುದಿ_k = ತುದಿ_k;
        config.ಗುಂಪು_ಗಾತ್ರ = ಕೆಳಕ್ಕೆ_groups <= 1 ? std::numeric_limits<size_t>::max() : ಕೆಳಕ್ಕೆ_k / ಕೆಳಕ್ಕೆ_groups;
        config.has_batch_dim = ನಿಜ;
        config.has_zp = ನಿಜ;
        config.ಹೊರಹೋಗು_type = ov::element::dynamic;

        // Rebuild the activations Transpose before mul_mat_id's f16 GPU Convert, so the op
        // stays f32 like the block it replaces (mirrors FuseMoeCompressed).
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ a_transpose = pm.at(a_m).get_ಗಂಟು_shared_ptr();
        ov::Output<ov::Node> hidden =
            std::make_shared<ov::op::v1::Transpose>(pm.at(a_reshape_m), a_transpose->ಒಳಹರಿವು_ಮೌಲ್ಯ(1));

        // w0 is the activated (gate) lane and w1 the multiplied (up) lane -- see
        // moe_3gemm_swiglu_opt.cpp: "scratch.up = up(x) * silu(gate(x))".
        ಸ್ಥಿರಾಂಕ ov::Output_ಸದಿಶ ವಾದಗಳು_2 = {
            hidden,      routing,     ids,
            gate_ತೂಕ, gate_scale,  gate_zp,
            ಮೇಲೆ_ತೂಕ,   ಮೇಲೆ_scale,    ಮೇಲೆ_zp,
            ಕೆಳಕ್ಕೆ.ತೂಕ, ಕೆಳಕ್ಕೆ.scale,  ಕೆಳಕ್ಕೆ.zp,
        };

        ಸ್ವಯಂಚಾಲಿತ moe = std::make_shared<ov::op::internal::MOECompressed>(ವಾದಗಳು_2, config);

        ov::Output<ov::Node> ಫಲಿತಾಂಶ = moe->output(0);
        // The op was fed the batched form, so it produces [1, n_tokens, hidden]. On the rank-3
        // graph the ReduceSum being replaced is [n_tokens, hidden], so drop the batch dim again.
        ಆದರೆ (batchless) {
            ಫಲಿತಾಂಶ = std::make_shared<ov::op::v0::Squeeze>(
                ಫಲಿತಾಂಶ, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 0 }));
        }
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ ಬೇರು_type = m.get_ಹೊಂದಿಕೆ_ಬೇರು()->get_output_element_type(0);
        ಆದರೆ (ಫಲಿತಾಂಶ.get_element_type() != ಬೇರು_type) {
            ಫಲಿತಾಂಶ = std::make_shared<ov::op::v0::Convert>(ಫಲಿತಾಂಶ, ಬೇರು_type);
        }
        ಆದರೆ (ಫಲಿತಾಂಶ.get_partial_shape() != m.get_ಹೊಂದಿಕೆ_ಬೇರು()->get_output_partial_shape(0)) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಫಲಿತಾಂಶ.get_ಗಂಟು_shared_ptr()->ಗಣ_friendly_name(m.get_ಹೊಂದಿಕೆ_ಬೇರು()->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), ಫಲಿತಾಂಶ.get_ಗಂಟು_shared_ptr());
        ov::replace_ಗಂಟು(m.get_ಹೊಂದಿಕೆ_ಬೇರು(), ಫಲಿತಾಂಶ.get_ಗಂಟು_shared_ptr());
        ರಿಜಿಸ್ಟರ್_new_ಗಂಟು(moe);
        ಹಿಂತಿರುಗಿ ನಿಜ;
    };

    ರಿಜಿಸ್ಟರ್_matcher(
        std::make_shared<Matcher>(ಬೇರು_m, "ov::frontend::ggml::pass::FuseMoeCompressedFusedGateUp"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
