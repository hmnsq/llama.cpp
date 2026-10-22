#شامل "fuse_moe_compressed.h"

#شامل <cstring>
#شامل <limits>
#شامل <set>
#شامل <memory>
#شامل <openvino/core/graph_util.hpp>
#شامل <openvino/core/rt_info.hpp>
#شامل <openvino/op/constant.hpp>
#شامل <openvino/op/convert.hpp>
#شامل <openvino/op/gelu.hpp>
#شامل <openvino/op/multiply.hpp>
#شامل <openvino/op/reduce_sum.hpp>
#شامل <openvino/op/reshape.hpp>
#شامل <openvino/op/slice.hpp>
#شامل <openvino/op/squeeze.hpp>
#شامل <openvino/op/subtract.hpp>
#شامل <openvino/op/swish.hpp>
#شامل <openvino/op/transpose.hpp>
#شامل <openvino/op/unsqueeze.hpp>
#شامل <openvino/pass/constant_folding.hpp>
#شامل <openvino/pass/pattern/op/optional.hpp>
#شامل <openvino/pass/pattern/op/wrap_type.hpp>

#شامل "../op/gather_matmul.hpp"
#شامل "../op/moe_compressed.hpp"

نامخلا ov {
نامخلا frontend {
نامخلا ggml {
نامخلا pass {

نامخلا {

ساخت dequant_inputs {
    ov::Output<ov::Node> وزن;
    ov::Output<ov::Node> scale;
    ov::Output<ov::Node> zp;
    منطقی has_zp = جھوٹ;
    منطقی ok = جھوٹ;
};

// Peel the chain built by make_int4_weights/make_int8_weights back to its Constant inputs.
// Grouped weights keep the pre-Reshape rank-4 form [n_expert, n, k/group, group] with scale
// and zp at [n_expert, n, k/group, 1], which is the layout MOECompressed expects. Channel-wise
// weights stay rank-3 with a rank-3 scale and carry no zp.
dequant_inputs unwrap_dequant(ثابت ov::Output<ov::Node> & b) {
    dequant_inputs res;

    خودکار گرہ = b.get_گرہ_shared_ptr();
    جب (ov::is_type<ov::op::v0::Convert>(گرہ) || ov::is_type<ov::op::v1::Reshape>(گرہ)) {
        گرہ = گرہ->get_داخلہ_گرہ_shared_ptr(0);
    }

    خودکار mul = ov::as_type_ptr<ov::op::v1::Multiply>(گرہ);
    اگر (!mul) {
        واپس res;
    }
    res.scale = mul->داخلہ_قدر(1);

    خودکار lhs = mul->get_داخلہ_گرہ_shared_ptr(0);
    اگر (خودکار sub = ov::as_type_ptr<ov::op::v1::تفریق>(lhs)) {
        // Take the zero point down to its Constant: an integer zp is wrapped in a Convert to f16,
        // and the op wants the integer form. A natively quantized expert instead carries an exact
        // f16 zp (-min/scale) with no integer behind it, which the MoE kernel does not accept.
        خودکار zp_گرہ = sub->get_داخلہ_گرہ_shared_ptr(1);
        جب (ov::is_type<ov::op::v0::Convert>(zp_گرہ)) {
            zp_گرہ = zp_گرہ->get_داخلہ_گرہ_shared_ptr(0);
        }
        res.zp = zp_گرہ->output(0);
        res.has_zp = سچ;
        lhs = sub->get_داخلہ_گرہ_shared_ptr(0);
    }
    جب (ov::is_type<ov::op::v0::Convert>(lhs)) {
        lhs = lhs->get_داخلہ_گرہ_shared_ptr(0);
    }
    اگر (!ov::is_type<ov::op::v0::Constant>(lhs)) {
        واپس res;
    }

    res.وزن = lhs->output(0);
    res.ok = res.scale.get_partial_shape().is_static() && res.وزن.get_partial_shape().is_static();
    واپس res;
}

size_t logical_k(ثابت ov::Shape & shape) {
    واپس shape.size() == 4 ? shape[2] * shape[3] : shape.back();
}

// Slice a [n_expert, m, ...] compressed weight/scale/zp Constant along axis 1 (m), [begin, end).
// Built by copying raw bytes directly instead of a graph Slice op: CommonOptimizations rewrites
// v8::Slice into v1::StridedSlice for constant folding regardless of disable_constant_folding
// (the rewrite doesn't carry the marking over), and that reference evaluator crashes on
// sub-byte (u4/i4) element types. Each axis-1 "row" (the trailing dims) is confirmed
// byte-aligned here -- weight/scale/zp trailing sizes are always whole groups -- so a
// byte-range memcpy per row is exact for both regular and sub-byte element types.
ov::Output<ov::Node> قاش_experts_dim1(ثابت ov::Output<ov::Node> & tensor, int64_t begin, int64_t end) {
    خودکار constant = ov::as_type_ptr<ov::op::v0::Constant>(tensor.get_گرہ_shared_ptr());
    OPENVINO_ASSERT(constant, "slice_experts_dim1 expects a Constant input");

    ثابت خودکار & shape = constant->get_shape();
    OPENVINO_ASSERT(shape.size() >= 2, "slice_experts_dim1 expects rank >= 2");
    ثابت خودکار & type = constant->get_element_type();

    size_t صف_elems = 1;
    برائے (size_t i = 2; i < shape.size(); ++i) {
        صف_elems *= shape[i];
    }
    ثابت size_t صف_بٹ = صف_elems * type.bitwidth();
    OPENVINO_ASSERT(صف_بٹ % 8 == 0, "slice_experts_dim1: row is not byte-aligned");
    ثابت size_t صف_بائٹیں = صف_بٹ / 8;

    ov::Shape برآمد_shape = shape;
    برآمد_shape[1] = static_cast<size_t>(end - begin);

    std::vector<uint8_t> برآمد_ڈیٹا(shape[0] * static_cast<size_t>(end - begin) * صف_بائٹیں);
    ثابت خودکار * src = static_cast<ثابت uint8_t *>(constant->get_ڈیٹا_ptr());
    ثابت size_t src_صف_بائٹیں = shape[1] * صف_بائٹیں;
    برائے (size_t e = 0; e < shape[0]; ++e) {
        std::حافظہ_نقل(برآمد_ڈیٹا.ڈیٹا() + e * static_cast<size_t>(end - begin) * صف_بائٹیں,
                   src + e * src_صف_بائٹیں + static_cast<size_t>(begin) * صف_بائٹیں,
                   static_cast<size_t>(end - begin) * صف_بائٹیں);
    }

    واپس std::make_shared<ov::op::v0::Constant>(type, برآمد_shape, برآمد_ڈیٹا.ڈیٹا());
}

}  // namespace

FuseMoeCompressed::FuseMoeCompressed() {
    استعمال نامخلا ov::pass::pattern;

    // The gate and up projections each get their own Reshape/Transpose of the hidden state and
    // their own Reshape of the routing ids, so every branch needs its own sub-pattern. On GPU
    // mul_mat_id also converts the activations to f16 before the op and back to f32 after it,
    // so those Converts are matched as optional.
    خودکار hidden_gate_m = any_داخلہ();
    خودکار a_gate_reshape_m = wrap_type<ov::op::v1::Reshape>({ hidden_gate_m, any_داخلہ() });
    خودکار a_gate_m =
        wrap_type<ov::op::v1::Transpose>({ optional<ov::op::v0::Convert>({ a_gate_reshape_m }), any_داخلہ() });
    خودکار hidden_اوپر_m = any_داخلہ();
    خودکار a_اوپر_m = wrap_type<ov::op::v1::Transpose>(
        { optional<ov::op::v0::Convert>({ wrap_type<ov::op::v1::Reshape>({ hidden_اوپر_m, any_داخلہ() }) }),
          any_داخلہ() });

    خودکار gate_w_m = any_داخلہ();
    خودکار اوپر_w_m = any_داخلہ();
    خودکار نیچے_کو_w_m = any_داخلہ();
    خودکار ids_gate_m = any_داخلہ();
    خودکار ids_اوپر_m = any_داخلہ();
    خودکار ids_نیچے_کو_m = any_داخلہ();

    خودکار bgm_gate_m = wrap_type<ov::op::internal::GatherMatmul>({ a_gate_m, gate_w_m, ids_gate_m, any_داخلہ() });
    خودکار gate_u_m = optional<ov::op::v0::Convert>({ wrap_type<ov::op::v0::Unsqueeze>(
        { wrap_type<ov::op::v1::Transpose>({ bgm_gate_m, any_داخلہ() }), any_داخلہ() }) });

    خودکار silu_m = wrap_type<ov::op::v4::Swish>({ gate_u_m });

    خودکار bgm_اوپر_m = wrap_type<ov::op::internal::GatherMatmul>({ a_اوپر_m, اوپر_w_m, ids_اوپر_m, any_داخلہ() });
    خودکار اوپر_u_m = optional<ov::op::v0::Convert>({ wrap_type<ov::op::v0::Unsqueeze>(
        { wrap_type<ov::op::v1::Transpose>({ bgm_اوپر_m, any_داخلہ() }), any_داخلہ() }) });
    خودکار swiglu_m = wrap_type<ov::op::v1::Multiply>({ silu_m, اوپر_u_m });

    خودکار d_t_m = wrap_type<ov::op::v1::Transpose>(
        { optional<ov::op::v0::Convert>({ wrap_type<ov::op::v1::Reshape>({ swiglu_m, any_داخلہ() }) }),
          any_داخلہ() });
    خودکار bgm_نیچے_کو_m = wrap_type<ov::op::internal::GatherMatmul>({ d_t_m, نیچے_کو_w_m, ids_نیچے_کو_m, any_داخلہ() });
    خودکار نیچے_کو_u_m = optional<ov::op::v0::Convert>({ wrap_type<ov::op::v0::Unsqueeze>(
        { wrap_type<ov::op::v1::Transpose>({ bgm_نیچے_کو_m, any_داخلہ() }), any_داخلہ() }) });

    خودکار routing_m = any_داخلہ();
    خودکار weighted_m = wrap_type<ov::op::v1::Multiply>({ نیچے_کو_u_m, routing_m });
    خودکار جڑ_m = wrap_type<ov::op::v1::Reduce_مجموع>({ weighted_m, any_داخلہ() });

    ثابت خودکار callback = [=](Matcher & m) {
        خودکار & pm = m.get_نمونہ_قدر_map();

        ثابت خودکار gate = unwrap_dequant(pm.at(gate_w_m));
        ثابت خودکار اوپر = unwrap_dequant(pm.at(اوپر_w_m));
        ثابت خودکار نیچے_کو = unwrap_dequant(pm.at(نیچے_کو_w_m));
        اگر (!gate.ok || !اوپر.ok || !نیچے_کو.ok) {
            واپس جھوٹ;
        }

        ثابت خودکار gate_shape = gate.وزن.get_shape();
        ثابت خودکار اوپر_shape = اوپر.وزن.get_shape();
        ثابت خودکار نیچے_کو_shape = نیچے_کو.وزن.get_shape();
        اگر (gate_shape != اوپر_shape || gate_shape.size() < 3 || نیچے_کو_shape.size() < 3) {
            واپس جھوٹ;
        }

        // MOECompressed carries one group_size and one has_zp for all three projections, so a
        // model whose down-proj is quantized differently from gate/up cannot be described. This
        // happens when ggml requantizes Q5_K/Q6_K experts to channel-wise int8.
        اگر (gate.has_zp != نیچے_کو.has_zp || gate_shape.size() != نیچے_کو_shape.size()) {
            واپس جھوٹ;
        }

        // The kernel only takes an integer zero point (moe_3gemm_swiglu_opt validate_impl).
        اگر (gate.has_zp) {
            ساکن ثابت std::set<ov::element::Type> int_zp_types = { ov::element::u4, ov::element::i4,
                                                                     ov::element::u8, ov::element::i8 };
            اگر (int_zp_types.count(gate.zp.get_element_type()) == 0 ||
                int_zp_types.count(نیچے_کو.zp.get_element_type()) == 0) {
                واپس جھوٹ;
            }
        }

        // Config holds a single group_size for all three projections.
        ثابت خودکار گروہ_کا = [](ثابت dequant_inputs & w) {
            ثابت خودکار s = w.وزن.get_shape();
            واپس s.size() == 4 ? s[3] : logical_k(s);
        };
        اگر (گروہ_کا(gate) != گروہ_کا(اوپر) || گروہ_کا(gate) != گروہ_کا(نیچے_کو)) {
            واپس جھوٹ;
        }

        // all three branches must route the same hidden state through the same experts
        اگر (pm.at(hidden_gate_m) != pm.at(hidden_اوپر_m)) {
            واپس جھوٹ;
        }

        خودکار ids = pm.at(ids_نیچے_کو_m);
        ثابت خودکار ids_pshape = ids.get_partial_shape();
        اگر (ids_pshape.rank().is_dynamic() || ids_pshape[ids_pshape.rank().get_لمبائی() - 1].is_dynamic()) {
            واپس جھوٹ;
        }
        ثابت size_t چوٹی_k = ids_pshape[ids_pshape.rank().get_لمبائی() - 1].get_لمبائی();

        // routing weights arrive as [1, n_tokens, top_k, 1]; the op wants [..., top_k]
        خودکار routing = pm.at(routing_m);
        ثابت خودکار routing_pshape = routing.get_partial_shape();
        اگر (routing_pshape.rank().is_dynamic() || routing_pshape.rank().get_لمبائی() != 4 ||
            routing_pshape[3] != 1) {
            واپس جھوٹ;
        }
        // MOE requires routing weights and ids to have the same shape. Drop the trailing 1 of the
        // routing weights and give the ids the leading batch dim, so both become [1, n_tokens, top_k].
        routing = std::make_shared<ov::op::v0::Squeeze>(
            routing, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 3 }));
        اگر (ids_pshape.rank().get_لمبائی() == 2) {
            ids = std::make_shared<ov::op::v0::Unsqueeze>(
                ids, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 0 }));
        }
        اگر (routing.get_partial_shape() != ids.get_partial_shape()) {
            واپس جھوٹ;
        }

        ثابت size_t نیچے_کو_k = logical_k(نیچے_کو_shape);
        ثابت خودکار نیچے_کو_scale_shape = نیچے_کو.scale.get_shape();
        ثابت size_t نیچے_کو_groups = نیچے_کو_scale_shape.size() >= 3 ? نیچے_کو_scale_shape[2] : 1;

        ov::op::internal::MOECompressed::Config config;
        config.expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
        config.activation_type = ov::op::internal::MOE::Activation_type::SWIGLU;
        config.expert_alpha = 0.0f;
        config.expert_beta = 1.0f;
        config.gate_idx = 0;
        config.hidden_سائز = logical_k(gate_shape);
        config.inter_سائز = gate_shape[1];
        config.عدد_expert = gate_shape[0];
        config.عدد_shared_expert = 0;
        config.چوٹی_k = چوٹی_k;
        config.گروہ_سائز = نیچے_کو_groups <= 1 ? std::numeric_limits<size_t>::max() : نیچے_کو_k / نیچے_کو_groups;
        config.has_batch_dim = سچ;
        config.has_zp = gate.has_zp;
        // dynamic makes the output follow the hidden state, so the plugin can lower this region
        // to f16 together with the rest of the graph
        config.برآمد_type = ov::element::dynamic;

        خودکار absent_zp = [] {
            خودکار zp = std::make_shared<ov::op::v0::Constant>(ov::element::dynamic, ov::Shape{ 0 });
            ov::pass::disable_constant_folding(zp);
            واپس zp->output(0);
        };

        // MOE takes its output type from the hidden state. Transpose the activations before the
        // f16 Convert that mul_mat_id adds on GPU, so the op stays f32 like the block it replaces
        // and the plugin can lower the whole region uniformly.
        ثابت خودکار a_transpose = pm.at(a_gate_m).get_گرہ_shared_ptr();
        ov::Output<ov::Node> hidden =
            std::make_shared<ov::op::v1::Transpose>(pm.at(a_gate_reshape_m), a_transpose->داخلہ_قدر(1));

        ثابت ov::Output_سمتیہ دلائل_2 = {
            hidden, routing, ids,
            gate.وزن,  gate.scale,   gate.has_zp ? gate.zp : absent_zp(),
            اوپر.وزن,    اوپر.scale,     اوپر.has_zp ? اوپر.zp : absent_zp(),
            نیچے_کو.وزن,  نیچے_کو.scale,   نیچے_کو.has_zp ? نیچے_کو.zp : absent_zp(),
        };

        خودکار moe = std::make_shared<ov::op::internal::MOECompressed>(دلائل_2, config);

        // MOE takes its output type from the hidden state, which is f16 on GPU, while the rest of
        // the ggml graph works in f32.
        ov::Output<ov::Node> نتیجہ = moe->output(0);
        ثابت خودکار جڑ_type = m.get_مطابقت_جڑ()->get_output_element_type(0);
        اگر (نتیجہ.get_element_type() != جڑ_type) {
            نتیجہ = std::make_shared<ov::op::v0::Convert>(نتیجہ, جڑ_type);
        }

        نتیجہ.get_گرہ_shared_ptr()->سیٹ_friendly_name(m.get_مطابقت_جڑ()->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), نتیجہ.get_گرہ_shared_ptr());
        ov::replace_گرہ(m.get_مطابقت_جڑ(), نتیجہ.get_گرہ_shared_ptr());
        رجسٹر_new_گرہ(moe);
        واپس سچ;
    };

    رجسٹر_matcher(std::make_shared<Matcher>(جڑ_m, "ov::frontend::ggml::pass::FuseMoeCompressed"), callback);
}

Fuse_Moe_Compressed_Fused_Gate_اوپر::Fuse_Moe_Compressed_Fused_Gate_اوپر() {
    استعمال نامخلا ov::pass::pattern;

    // A single GatherMatmul computes the fused gate+up projection; the gate/up split happens
    // AFTER the GEMM via two Slice ops on the last (feature) axis (see get_glu_inputs in
    // op/glu_geglu.cpp), unlike FuseMoeCompressed's two-separate-GatherMatmul models.
    خودکار hidden_m = any_داخلہ();
    خودکار a_reshape_m = wrap_type<ov::op::v1::Reshape>({ hidden_m, any_داخلہ() });
    خودکار a_m = wrap_type<ov::op::v1::Transpose>({ optional<ov::op::v0::Convert>({ a_reshape_m }), any_داخلہ() });

    خودکار gate_اوپر_w_m = any_داخلہ();
    خودکار ids_gate_اوپر_m = any_داخلہ();
    خودکار bgm_fused_m = wrap_type<ov::op::internal::GatherMatmul>({ a_m, gate_اوپر_w_m, ids_gate_اوپر_m, any_داخلہ() });
    // Two graph shapes reach here. The rank-4 (stateless) graph restores the batch dim with an
    // Unsqueeze after the Transpose and may Convert afterwards:
    //     GatherMatmul -> Transpose -> Unsqueeze -> [Convert] -> Slice
    // The rank-3 (stateful) graph never drops to a batch dim at all, so there is no Unsqueeze
    // here - it appears later, between the GEGLU and the down Reshape - and the Convert sits on
    // the other side of the Transpose:
    //     GatherMatmul -> Convert -> Transpose -> Slice
    // optional<T>({a, b}) matches T(a, b) or bare a, so one pattern covers both.
    خودکار gu_t_m =
        wrap_type<ov::op::v1::Transpose>({ optional<ov::op::v0::Convert>({ bgm_fused_m }), any_داخلہ() });
    خودکار gu_u_m =
        optional<ov::op::v0::Convert>({ optional<ov::op::v0::Unsqueeze>({ gu_t_m, any_داخلہ() }) });

    خودکار gate_قاش_m = wrap_type<ov::op::v8::قاش>({ gu_u_m, any_داخلہ(), any_داخلہ(), any_داخلہ(), any_داخلہ() });
    خودکار اوپر_قاش_m = wrap_type<ov::op::v8::قاش>({ gu_u_m, any_داخلہ(), any_داخلہ(), any_داخلہ(), any_داخلہ() });
    خودکار gelu_m = wrap_type<ov::op::v7::Gelu>({ gate_قاش_m });
    خودکار geglu_m = wrap_type<ov::op::v1::Multiply>({ gelu_m, اوپر_قاش_m });

    // The rank-3 graph inserts the Unsqueeze the gate/up branch lacked right here, before the
    // Reshape that feeds the down projection; the rank-4 graph goes straight from the GEGLU
    // into the Reshape.
    خودکار d_t_m = wrap_type<ov::op::v1::Transpose>(
        { optional<ov::op::v0::Convert>({ wrap_type<ov::op::v1::Reshape>(
              { optional<ov::op::v0::Unsqueeze>({ geglu_m, any_داخلہ() }), any_داخلہ() }) }),
          any_داخلہ() });
    خودکار نیچے_کو_w_m = any_داخلہ();
    خودکار ids_نیچے_کو_m = any_داخلہ();
    خودکار bgm_نیچے_کو_m = wrap_type<ov::op::internal::GatherMatmul>({ d_t_m, نیچے_کو_w_m, ids_نیچے_کو_m, any_داخلہ() });
    // Same two shapes as the gate/up branch above.
    خودکار نیچے_کو_t_m =
        wrap_type<ov::op::v1::Transpose>({ optional<ov::op::v0::Convert>({ bgm_نیچے_کو_m }), any_داخلہ() });
    خودکار نیچے_کو_u_m =
        optional<ov::op::v0::Convert>({ optional<ov::op::v0::Unsqueeze>({ نیچے_کو_t_m, any_داخلہ() }) });

    // gemma-4 applies an extra per-expert output scale to the down projection before the
    // router-weight multiply (llama-graph.cpp's ffn_down_exps.scale); FuseMoeCompressed's
    // Qwen-shaped models have no such scale, so this node is specific to this pattern.
    خودکار نیچے_کو_scale_m = any_داخلہ();
    خودکار نیچے_کو_scaled_m = wrap_type<ov::op::v1::Multiply>({ نیچے_کو_u_m, نیچے_کو_scale_m });

    خودکار routing_m = any_داخلہ();
    خودکار weighted_m = wrap_type<ov::op::v1::Multiply>({ نیچے_کو_scaled_m, routing_m });
    خودکار جڑ_m = wrap_type<ov::op::v1::Reduce_مجموع>({ weighted_m, any_داخلہ() });

    ثابت خودکار callback = [=](Matcher & m) {
        خودکار & pm = m.get_نمونہ_قدر_map();

        ثابت خودکار gate_اوپر = unwrap_dequant(pm.at(gate_اوپر_w_m));
        ثابت خودکار نیچے_کو = unwrap_dequant(pm.at(نیچے_کو_w_m));
        اگر (!gate_اوپر.ok || !نیچے_کو.ok) {
            واپس جھوٹ;
        }

        // oneDNN's weight-decompression GEMM needs a real zero point, so a symmetric (no zp)
        // projection cannot be fused here -- e.g. the GPU-only Q4_0 fallback in
        // ggml_openvino_get_extracted_layout for grouped 8-bit expert weights.
        اگر (!gate_اوپر.has_zp || !نیچے_کو.has_zp) {
            واپس جھوٹ;
        }

        ثابت خودکار gate_اوپر_shape = gate_اوپر.وزن.get_shape();
        ثابت خودکار نیچے_کو_shape = نیچے_کو.وزن.get_shape();
        اگر (gate_اوپر_shape.size() < 3 || نیچے_کو_shape.size() < 3 || gate_اوپر_shape.size() != نیچے_کو_shape.size()) {
            واپس جھوٹ;
        }

        // The op reads the zero point straight off a weight port, so it must already be an
        // integer Constant. Requantized experts (GGML_OPENVINO_REQUANT_KQUANT=q4_asym64_all)
        // are; a native Q4_K expert keeps an exact f16 zp and is left to the unfused path
        // rather than rounded here -- rounding would put a live node on that port and the
        // expert GEMM would read the wrong zero point.
        ساکن ثابت std::set<ov::element::Type> int_types = { ov::element::u4, ov::element::i4,
                                                               ov::element::u8, ov::element::i8 };
        اگر (int_types.count(gate_اوپر.وزن.get_element_type()) == 0 ||
            int_types.count(نیچے_کو.وزن.get_element_type()) == 0 ||
            int_types.count(gate_اوپر.zp.get_element_type()) == 0 ||
            int_types.count(نیچے_کو.zp.get_element_type()) == 0) {
            واپس جھوٹ;
        }

        ثابت خودکار گروہ_کا = [](ثابت dequant_inputs & w) {
            ثابت خودکار s = w.وزن.get_shape();
            واپس s.size() == 4 ? s[3] : logical_k(s);
        };
        اگر (گروہ_کا(gate_اوپر) != گروہ_کا(نیچے_کو)) {
            واپس جھوٹ;
        }

        خودکار gelu_گرہ = ov::as_type_ptr<ov::op::v7::Gelu>(pm.at(gelu_m).get_گرہ_shared_ptr());
        اگر (!gelu_گرہ || gelu_گرہ->get_approximation_منوال() != ov::op::Gelu_Approximation_منوال::ERF) {
            واپس جھوٹ;
        }

        // Read the actual gate/up split point from the matched Slice nodes instead of assuming
        // inter_size/2, so this stays correct if the model's ffn_dim convention ever changes.
        خودکار gate_قاش = ov::as_type_ptr<ov::op::v8::قاش>(pm.at(gate_قاش_m).get_گرہ_shared_ptr());
        خودکار اوپر_قاش = ov::as_type_ptr<ov::op::v8::قاش>(pm.at(اوپر_قاش_m).get_گرہ_shared_ptr());
        خودکار gate_انتہا_c = ov::as_type_ptr<ov::op::v0::Constant>(gate_قاش->داخلہ_قدر(2).get_گرہ_shared_ptr());
        خودکار اوپر_begin_c = ov::as_type_ptr<ov::op::v0::Constant>(اوپر_قاش->داخلہ_قدر(1).get_گرہ_shared_ptr());
        اگر (!gate_انتہا_c || !اوپر_begin_c) {
            واپس جھوٹ;
        }
        ثابت خودکار gate_انتہا = gate_انتہا_c->cast_سمتیہ<int64_t>().at(0);
        ثابت خودکار اوپر_begin = اوپر_begin_c->cast_سمتیہ<int64_t>().at(0);
        اگر (gate_انتہا != اوپر_begin || gate_انتہا <= 0 || gate_انتہا >= static_cast<int64_t>(gate_اوپر_shape[1])) {
            واپس جھوٹ;
        }
        ثابت int64_t mid = gate_انتہا;
        ثابت int64_t بھرا = static_cast<int64_t>(gate_اوپر_shape[1]);

        خودکار ids = pm.at(ids_نیچے_کو_m);
        ثابت خودکار ids_pshape = ids.get_partial_shape();
        اگر (ids_pshape.rank().is_dynamic() || ids_pshape[ids_pshape.rank().get_لمبائی() - 1].is_dynamic()) {
            واپس جھوٹ;
        }
        ثابت size_t چوٹی_k = ids_pshape[ids_pshape.rank().get_لمبائی() - 1].get_لمبائی();

        // Routing weights arrive as [1, n_tokens, top_k, 1] on the rank-4 (stateless) graph and
        // as [n_tokens, top_k, 1] on the rank-3 (stateful) one, which is the same thing without
        // the leading batch dim. Normalise the rank-3 form up to the rank-4 one so everything
        // below - and the op's own config - stays in the shape that is already validated on the
        // stateless path; the batch dim is taken back off the result at the end.
        ثابت خودکار routing_اندر = pm.at(routing_m);
        ثابت خودکار routing_pshape = routing_اندر.get_partial_shape();
        ثابت خودکار routing_rank = routing_pshape.rank();
        اگر (routing_rank.is_dynamic() || (routing_rank.get_لمبائی() != 4 && routing_rank.get_لمبائی() != 3) ||
            routing_pshape[routing_rank.get_لمبائی() - 1] != 1) {
            واپس جھوٹ;
        }
        ثابت منطقی batchless = routing_rank.get_لمبائی() == 3;

        // Fold gemma-4's per-expert output scale into the routing weights: the reduction is
        // sum_e(routing[e] * scale[e] * down_out[e]), and MOECompressed only takes one
        // per-expert weight, so pre-multiply it into routing here (same [.., top_k, 1] shape).
        خودکار نیچے_کو_scale = pm.at(نیچے_کو_scale_m);
        اگر (نیچے_کو_scale.get_partial_shape() != routing_pshape) {
            واپس جھوٹ;
        }
        ov::Output<ov::Node> routing = std::make_shared<ov::op::v1::Multiply>(routing_اندر, نیچے_کو_scale);
        اگر (batchless) {
            routing = std::make_shared<ov::op::v0::Unsqueeze>(
                routing, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 0 }));
        }
        routing = std::make_shared<ov::op::v0::Squeeze>(
            routing, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 3 }));
        اگر (ids_pshape.rank().get_لمبائی() == 2) {
            ids = std::make_shared<ov::op::v0::Unsqueeze>(
                ids, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 0 }));
        }
        اگر (routing.get_partial_shape() != ids.get_partial_shape()) {
            واپس جھوٹ;
        }

        ثابت size_t نیچے_کو_k = logical_k(نیچے_کو_shape);
        ثابت خودکار نیچے_کو_scale_shape = نیچے_کو.scale.get_shape();
        ثابت size_t نیچے_کو_groups = نیچے_کو_scale_shape.size() >= 3 ? نیچے_کو_scale_shape[2] : 1;

        // Split the fused gate_up weight/scale/zp on the output axis. slice_experts_dim1 copies
        // raw bytes out of the Constant, so each half stays a Constant -- which is what the op
        // needs on its weight ports.
        ثابت ov::Output<ov::Node> gate_وزن = قاش_experts_dim1(gate_اوپر.وزن, 0, mid);
        ثابت ov::Output<ov::Node> اوپر_وزن = قاش_experts_dim1(gate_اوپر.وزن, mid, بھرا);
        ثابت ov::Output<ov::Node> gate_scale = قاش_experts_dim1(gate_اوپر.scale, 0, mid);
        ثابت ov::Output<ov::Node> اوپر_scale = قاش_experts_dim1(gate_اوپر.scale, mid, بھرا);
        ثابت ov::Output<ov::Node> gate_zp = قاش_experts_dim1(gate_اوپر.zp, 0, mid);
        ثابت ov::Output<ov::Node> اوپر_zp = قاش_experts_dim1(gate_اوپر.zp, mid, بھرا);

        ov::op::internal::MOECompressed::Config config;
        config.expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
        config.activation_type = ov::op::internal::MOE::Activation_type::GEGLU_ERF;
        config.expert_alpha = 0.0f;
        config.expert_beta = 1.0f;
        config.gate_idx = 0;
        config.hidden_سائز = logical_k(gate_اوپر_shape);
        config.inter_سائز = static_cast<size_t>(mid);
        config.عدد_expert = gate_اوپر_shape[0];
        config.عدد_shared_expert = 0;
        config.چوٹی_k = چوٹی_k;
        config.گروہ_سائز = نیچے_کو_groups <= 1 ? std::numeric_limits<size_t>::max() : نیچے_کو_k / نیچے_کو_groups;
        config.has_batch_dim = سچ;
        config.has_zp = سچ;
        config.برآمد_type = ov::element::dynamic;

        // Rebuild the activations Transpose before mul_mat_id's f16 GPU Convert, so the op
        // stays f32 like the block it replaces (mirrors FuseMoeCompressed).
        ثابت خودکار a_transpose = pm.at(a_m).get_گرہ_shared_ptr();
        ov::Output<ov::Node> hidden =
            std::make_shared<ov::op::v1::Transpose>(pm.at(a_reshape_m), a_transpose->داخلہ_قدر(1));

        // w0 is the activated (gate) lane and w1 the multiplied (up) lane -- see
        // moe_3gemm_swiglu_opt.cpp: "scratch.up = up(x) * silu(gate(x))".
        ثابت ov::Output_سمتیہ دلائل_2 = {
            hidden,      routing,     ids,
            gate_وزن, gate_scale,  gate_zp,
            اوپر_وزن,   اوپر_scale,    اوپر_zp,
            نیچے_کو.وزن, نیچے_کو.scale,  نیچے_کو.zp,
        };

        خودکار moe = std::make_shared<ov::op::internal::MOECompressed>(دلائل_2, config);

        ov::Output<ov::Node> نتیجہ = moe->output(0);
        // The op was fed the batched form, so it produces [1, n_tokens, hidden]. On the rank-3
        // graph the ReduceSum being replaced is [n_tokens, hidden], so drop the batch dim again.
        اگر (batchless) {
            نتیجہ = std::make_shared<ov::op::v0::Squeeze>(
                نتیجہ, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 0 }));
        }
        ثابت خودکار جڑ_type = m.get_مطابقت_جڑ()->get_output_element_type(0);
        اگر (نتیجہ.get_element_type() != جڑ_type) {
            نتیجہ = std::make_shared<ov::op::v0::Convert>(نتیجہ, جڑ_type);
        }
        اگر (نتیجہ.get_partial_shape() != m.get_مطابقت_جڑ()->get_output_partial_shape(0)) {
            واپس جھوٹ;
        }

        نتیجہ.get_گرہ_shared_ptr()->سیٹ_friendly_name(m.get_مطابقت_جڑ()->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), نتیجہ.get_گرہ_shared_ptr());
        ov::replace_گرہ(m.get_مطابقت_جڑ(), نتیجہ.get_گرہ_shared_ptr());
        رجسٹر_new_گرہ(moe);
        واپس سچ;
    };

    رجسٹر_matcher(
        std::make_shared<Matcher>(جڑ_m, "ov::frontend::ggml::pass::FuseMoeCompressedFusedGateUp"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
