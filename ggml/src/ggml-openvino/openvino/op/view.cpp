#ਅਨ੍ਤਰ੍ਭਾਵ "../op_table.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../utils.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/concat.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/constant.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/gather.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/reshape.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/shape_of.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/slice.hpp>

ਨਾਮਾਕਾਸ਼ ov {
ਨਾਮਾਕਾਸ਼ frontend {
ਨਾਮਾਕਾਸ਼ ggml {
ਨਾਮਾਕਾਸ਼ op {

Output_ਸਦਿਸ਼ translate_view(ਅਚਲ ਗੰਢ_Context & context) {
    ਅੰਕ_inputs_check(context, 1, 1);

    ਜੇ (context.get_op_case() == 1) {
        // Static-mode identity pass-through for VIEWs over a GATED_DELTA_NET combined output or
        // the conv_input CONCAT; the consuming op (CPY/RMS_NORM) does its own runtime-correct
        // slicing on the full tensor (see ggml-decoder.cpp compute_op_case, GGML_OP_VIEW).
        ਵਾਪਸ {context.get_ਨਿਵੇਸ਼(0)};
    }

    ਜੇ (!context.is_static()) {
        // On the stateless/non-static path VIEW is normally a no-op (consumers re-slice).
        // EXCEPTION: the MoE expert aggregation slices each expert plane out of
        // ffn_moe_weighted [n_embd, n_expert_used, n_tokens] with ggml_view_2d and then
        // sums the planes with a chain of ADDs (llama-graph.cpp). Those ADDs read this
        // VIEW node directly from the tensor map and do NOT re-slice, so a no-op here
        // makes every plane the full tensor and the expert sum collapses. Materialize the
        // single-expert slice here. Gated by name (ffn_moe_weighted...view) so it can't
        // affect any other view.
        ਅਚਲ std::string & vname = context.get_name();
        ਜੇ (vname.find("ffn_moe_weighted") != std::string::npos) {
            ਸਵੈਚਾਲਿਤ src_ps = context.get_ਨਿਵੇਸ਼_shape(0);
            ਸਵੈਚਾਲਿਤ dst_ps = context.get_output_shape();
            ਜੇ (src_ps.rank().is_static() && dst_ps.rank().is_static() && src_ps.rank() == dst_ps.rank() &&
                src_ps.is_static() && dst_ps.is_static()) {
                ਸਵੈਚਾਲਿਤ sst = context.get_ਨਿਵੇਸ਼_stride(0);
                ਸਵੈਚਾਲਿਤ dst = context.get_output_stride();
                size_t voff = context.get_output_op_ਵਿੱਥ();
                ਸਵੈਚਾਲਿਤ ss = src_ps.ਮੰਜ਼ਿਲ_shape();
                ਸਵੈਚਾਲਿਤ dd = dst_ps.ਮੰਜ਼ਿਲ_shape();
                ਅਚਲ size_t nd = ss.size();
                // Stateful models drop the leading size-1 batch dim, so the real OV tensor can
                // be one rank lower than the ggml shape metadata above. Axis indices derived
                // from that metadata must be shifted down by the difference before they are
                // used as OV axes. Bail out if the dims we would drop are not all 1.
                ਅਚਲ ਸਵੈਚਾਲਿਤ ਵਿੱਚ_rank = context.get_ਨਿਵੇਸ਼(0).get_partial_shape().rank();
                ਅਚਲ ਪੂਰਨ axis_ਖਿਸਕਾਓ =
                    ਵਿੱਚ_rank.is_static() ? (ਪੂਰਨ) nd - (ਪੂਰਨ) ਵਿੱਚ_rank.get_ਲੰਮਾਈ() : 0;
                ਤਾਰਕਿਕ ਖਿਸਕਾਓ_ok = axis_ਖਿਸਕਾਓ >= 0 && (size_t) axis_ਖਿਸਕਾਓ < nd;
                ਲਈ (ਪੂਰਨ a = 0; a < axis_ਖਿਸਕਾਓ && ਖਿਸਕਾਓ_ok; ++a) {
                    ਖਿਸਕਾਓ_ok = (ss[a] == 1 && dd[a] == 1);
                }
                ਜੇ (ਖਿਸਕਾਓ_ok && sst.size() == nd && dst.size() == nd) {
                    // Map each dst axis of size>1 to a src axis with equal (size,stride);
                    // the unmatched src axis of size>1 is the indexed expert axis.
                    // dst_to_src[d] records which src axis each dst axis came from, so we can
                    // later pull the dynamic (token) dim from the right source axis at runtime.
                    std::vector<ਤਾਰਕਿਕ> ਵਰਤਿਆ(nd, ਝੂਠ);
                    std::vector<ਪੂਰਨ> dst_ਮੰਜ਼ਿਲ_src(nd, -1);
                    ਤਾਰਕਿਕ ok = ਸੱਚ;
                    ਲਈ (size_t d = 0; d < nd; ++d) {
                        ਜੇ (dd[d] == 1) {
                            ਜਾਰੀ;
                        }
                        ਪੂਰਨ ਮਿਲਿਆ = -1;
                        ਲਈ (size_t s = 0; s < nd; ++s) {
                            ਜੇ (!ਵਰਤਿਆ[s] && ss[s] == dd[d] && sst[s] == dst[d]) {
                                ਮਿਲਿਆ = (ਪੂਰਨ) s;
                                ਰੋਕੋ;
                            }
                        }
                        ਜੇ (ਮਿਲਿਆ < 0) {
                            ok = ਝੂਠ;
                            ਰੋਕੋ;
                        }
                        ਵਰਤਿਆ[ਮਿਲਿਆ] = ਸੱਚ;
                        dst_ਮੰਜ਼ਿਲ_src[d] = ਮਿਲਿਆ;
                    }
                    ਪੂਰਨ dropped = -1;
                    ਜੇ (ok) {
                        ਲਈ (size_t s = 0; s < nd; ++s) {
                            ਜੇ (!ਵਰਤਿਆ[s] && ss[s] > 1) {
                                ਜੇ (dropped >= 0) {
                                    ok = ਝੂਠ;
                                    ਰੋਕੋ;
                                }
                                dropped = (ਪੂਰਨ) s;
                            }
                        }
                    }
                    ਜੇ (ok && dropped >= 0) {
                        ਅਚਲ size_t dstr = sst[dropped];
                        ਅਚਲ int64_t dsz = (int64_t) ss[dropped];
                        ਜੇ (dstr > 0 && voff % dstr == 0) {
                            ਅਚਲ int64_t sel = (int64_t) (voff / dstr);
                            ਜੇ (sel >= 0 && sel < dsz) {
                                ov::Output<ov::Node> sl = std::make_shared<ov::op::v8::ਟੁਕੜਾ>(
                                    context.get_ਨਿਵੇਸ਼(0),
                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {sel}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {sel + 1}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1},
                                                                 {dropped - axis_ਖਿਸਕਾਓ}));
                                // Build the reshape target from the (concrete) dst shape, but
                                // keep the dynamic token axis dynamic instead of freezing it
                                // to the captured n_tokens. Without this the constant dst
                                // shape bakes in the prefill token count and the static value
                                // flows downstream, turning every later decoder layer static
                                // (the GPU in-place-concat KV-cache bug). The token axis is
                                // PERMUTED between the sliced input and the dst (e.g. input
                                // [1,tok,expert,emb] -> dst [1,1,tok,emb]), so special_zero
                                // (which copies the same-position dim) is not enough: pull the
                                // dynamic dim from the correct SOURCE axis via ShapeOf+Gather
                                // and place it at the dst token position.
                                ਅਚਲ int32_t dyn = context.get_op_dynamic_dim();  // output ggml axis, -1 if none
                                // still in ggml metadata axis space; get_shape() reverses ggml order
                                ਪੂਰਨ dst_ov_axis = (dyn != -1) ? ((ਪੂਰਨ) nd - 1 - (ਪੂਰਨ) dyn) : -1;
                                ਪੂਰਨ src_ov_axis = (dst_ov_axis >= axis_ਖਿਸਕਾਓ && dst_ov_axis < (ਪੂਰਨ) nd)
                                                      ? dst_ਮੰਜ਼ਿਲ_src[dst_ov_axis]
                                                      : -1;
                                ਜੇ (dst_ov_axis >= 0 && src_ov_axis >= axis_ਖਿਸਕਾਓ) {
                                    // target = concat of per-axis scalars; the token axis is a
                                    // runtime Gather of the slice's shape, the rest are constants.
                                    ਸਵੈਚਾਲਿਤ sl_shape = std::make_shared<ov::op::v3::Shape_ਦਾ>(sl, ov::element::i64);
                                    ਸਵੈਚਾਲਿਤ tok_dim = std::make_shared<ov::op::v8::Gather>(
                                        sl_shape,
                                        ov::op::v0::Constant::create(ov::element::i64, {1},
                                                                     {src_ov_axis - axis_ਖਿਸਕਾਓ}),
                                        ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
                                    ov::Output_ਸਦਿਸ਼ parts;
                                    ਲਈ (ਪੂਰਨ a = axis_ਖਿਸਕਾਓ; a < (ਪੂਰਨ) nd; ++a) {
                                        ਜੇ (a == dst_ov_axis) {
                                            parts.push_back(tok_dim);
                                        } ਨਹੀਂਤਾਂ {
                                            parts.push_back(ov::op::v0::Constant::create(
                                                ov::element::i64, {1}, {(int64_t) dd[a]}));
                                        }
                                    }
                                    ਸਵੈਚਾਲਿਤ dc = std::make_shared<ov::op::v0::Concat>(parts, 0);
                                    ਸਵੈਚਾਲਿਤ rs = std::make_shared<ov::op::v1::Reshape>(sl, dc, ਝੂਠ);
                                    ਵਾਪਸ rename_outputs_with_suffix({rs}, context.get_name());
                                }
                                std::vector<int64_t> dd_ov(dd.begin() + axis_ਖਿਸਕਾਓ, dd.end());
                                ਸਵੈਚਾਲਿਤ dc = ov::op::v0::Constant::create(ov::element::i64, {dd_ov.size()}, dd_ov);
                                ਸਵੈਚਾਲਿਤ rs = std::make_shared<ov::op::v1::Reshape>(sl, dc, ਝੂਠ);
                                ਵਾਪਸ rename_outputs_with_suffix({rs}, context.get_name());
                            }
                        }
                    }
                }
            }
        }
        ਵਾਪਸ {context.get_ਨਿਵੇਸ਼(0)};
    }

    ਸਵੈਚਾਲਿਤ ਨਿਵੇਸ਼ = context.get_ਨਿਵੇਸ਼(0);
    ਸਵੈਚਾਲਿਤ src_shape = context.get_ਨਿਵੇਸ਼_shape(0);
    ਸਵੈਚਾਲਿਤ dst_shape = context.get_output_shape();

    ਜੇ (src_shape.rank().is_dynamic() || dst_shape.rank().is_dynamic()) {
        ਵਾਪਸ {ਨਿਵੇਸ਼};
    }

    int64_t src_elems = 1;
    int64_t dst_elems = 1;
    ਲਈ (int64_t i = 0; i < src_shape.rank().get_ਲੰਮਾਈ(); ++i) {
        ਜੇ (src_shape[i].is_dynamic()) {
            ਵਾਪਸ {ਨਿਵੇਸ਼};
        }
        src_elems *= src_shape[i].get_ਲੰਮਾਈ();
    }
    ਲਈ (int64_t i = 0; i < dst_shape.rank().get_ਲੰਮਾਈ(); ++i) {
        ਜੇ (dst_shape[i].is_dynamic()) {
            ਵਾਪਸ {ਨਿਵੇਸ਼};
        }
        dst_elems *= dst_shape[i].get_ਲੰਮਾਈ();
    }

    ਜੇ (dst_elems >= src_elems) {
        ਵਾਪਸ {ਨਿਵੇਸ਼};
    }

    ਸਵੈਚਾਲਿਤ src_stride = context.get_ਨਿਵੇਸ਼_stride(0);
    ਸਵੈਚਾਲਿਤ dst_stride = context.get_output_stride();
    size_t view_ਵਿੱਥ = context.get_output_op_ਵਿੱਥ();

    ਤਾਰਕਿਕ same_stride = (src_stride.size() == dst_stride.size());
    ਜੇ (same_stride) {
        ਲਈ (size_t i = 0; i < src_stride.size(); ++i) {
            ਜੇ (src_stride[i] != dst_stride[i]) {
                same_stride = ਝੂਠ;
                ਰੋਕੋ;
            }
        }
    }

    ਜੇ (!same_stride) {
        ਵਾਪਸ {ਨਿਵੇਸ਼};
    }

    ਸਵੈਚਾਲਿਤ src_ov_shape = src_shape.ਮੰਜ਼ਿਲ_shape();
    ਸਵੈਚਾਲਿਤ dst_ov_shape = dst_shape.ਮੰਜ਼ਿਲ_shape();
    size_t ndims = src_ov_shape.size();
    ਜੇ (dst_ov_shape.size() != ndims) {
        ਵਾਪਸ {ਨਿਵੇਸ਼};
    }

    std::vector<ਪੂਰਨ> diff_dims;
    ਲਈ (size_t i = 0; i < ndims; ++i) {
        ਜੇ (src_ov_shape[i] != dst_ov_shape[i]) {
            diff_dims.push_back(static_cast<ਪੂਰਨ>(i));
        }
    }

    ਜੇ (diff_dims.size() != 1) {
        ਵਾਪਸ {ਨਿਵੇਸ਼};
    }

    ਪੂਰਨ ਟੁਕੜਾ_dim = diff_dims[0];
    int64_t dim_ਮਾਪ = static_cast<int64_t>(src_ov_shape[ਟੁਕੜਾ_dim]);

    size_t ov_stride_for_dim = 1;
    ਲਈ (size_t i = ਟੁਕੜਾ_dim + 1; i < ndims; ++i) {
        ov_stride_for_dim *= src_ov_shape[i];
    }
    size_t elem_ਮਾਪ = src_stride.back();
    ਜੇ (elem_ਮਾਪ == 0) {
        elem_ਮਾਪ = 1;
    }

    int64_t begin_val = 0;
    ਜੇ (ov_stride_for_dim > 0 && elem_ਮਾਪ > 0) {
        begin_val = static_cast<int64_t>((view_ਵਿੱਥ / elem_ਮਾਪ) / ov_stride_for_dim);
    }
    int64_t ਸਿਰਾ_val = begin_val + static_cast<int64_t>(dst_ov_shape[ਟੁਕੜਾ_dim]);

    ਜੇ (begin_val < 0 || ਸਿਰਾ_val > dim_ਮਾਪ) {
        ਵਾਪਸ {ਨਿਵੇਸ਼};
    }

    ਸਵੈਚਾਲਿਤ sliced =
        std::make_shared<ov::op::v8::ਟੁਕੜਾ>(ਨਿਵੇਸ਼, ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val}),
                                            ov::op::v0::Constant::create(ov::element::i64, {1}, {ਸਿਰਾ_val}),
                                            ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                            ov::op::v0::Constant::create(ov::element::i64, {1}, {ਟੁਕੜਾ_dim}));

    sliced->ਸੈੱਟ_friendly_name(context.get_output_name());
    ਵਾਪਸ {sliced->output(0)};
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
