#ಅನ್ತರ್ಭಾವ "../op_table.h"
#ಅನ್ತರ್ಭಾವ "../utils.h"

#ಅನ್ತರ್ಭಾವ <openvino/op/concat.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/constant.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/gather.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/reshape.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/shape_of.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/slice.hpp>

ನಾಮಾಕಾಶ ov {
ನಾಮಾಕಾಶ frontend {
ನಾಮಾಕಾಶ ggml {
ನಾಮಾಕಾಶ op {

Output_ಸದಿಶ translate_view(ಸ್ಥಿರಾಂಕ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 1, 1);

    ಆದರೆ (context.get_op_case() == 1) {
        // Static-mode identity pass-through for VIEWs over a GATED_DELTA_NET combined output or
        // the conv_input CONCAT; the consuming op (CPY/RMS_NORM) does its own runtime-correct
        // slicing on the full tensor (see ggml-decoder.cpp compute_op_case, GGML_OP_VIEW).
        ಹಿಂತಿರುಗಿ {context.get_ಒಳಹರಿವು(0)};
    }

    ಆದರೆ (!context.is_static()) {
        // On the stateless/non-static path VIEW is normally a no-op (consumers re-slice).
        // EXCEPTION: the MoE expert aggregation slices each expert plane out of
        // ffn_moe_weighted [n_embd, n_expert_used, n_tokens] with ggml_view_2d and then
        // sums the planes with a chain of ADDs (llama-graph.cpp). Those ADDs read this
        // VIEW node directly from the tensor map and do NOT re-slice, so a no-op here
        // makes every plane the full tensor and the expert sum collapses. Materialize the
        // single-expert slice here. Gated by name (ffn_moe_weighted...view) so it can't
        // affect any other view.
        ಸ್ಥಿರಾಂಕ std::string & vname = context.get_name();
        ಆದರೆ (vname.find("ffn_moe_weighted") != std::string::npos) {
            ಸ್ವಯಂಚಾಲಿತ src_ps = context.get_ಒಳಹರಿವು_shape(0);
            ಸ್ವಯಂಚಾಲಿತ dst_ps = context.get_output_shape();
            ಆದರೆ (src_ps.rank().is_static() && dst_ps.rank().is_static() && src_ps.rank() == dst_ps.rank() &&
                src_ps.is_static() && dst_ps.is_static()) {
                ಸ್ವಯಂಚಾಲಿತ sst = context.get_ಒಳಹರಿವು_stride(0);
                ಸ್ವಯಂಚಾಲಿತ dst = context.get_output_stride();
                size_t voff = context.get_output_op_ಸ್ಥಾನಾಂತರ();
                ಸ್ವಯಂಚಾಲಿತ ss = src_ps.ಗಮ್ಯ_shape();
                ಸ್ವಯಂಚಾಲಿತ dd = dst_ps.ಗಮ್ಯ_shape();
                ಸ್ಥಿರಾಂಕ size_t nd = ss.size();
                // Stateful models drop the leading size-1 batch dim, so the real OV tensor can
                // be one rank lower than the ggml shape metadata above. Axis indices derived
                // from that metadata must be shifted down by the difference before they are
                // used as OV axes. Bail out if the dims we would drop are not all 1.
                ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ ಒಳ_rank = context.get_ಒಳಹರಿವು(0).get_partial_shape().rank();
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ axis_ಸರಿಸು =
                    ಒಳ_rank.is_static() ? (ಪೂರ್ಣಾಂಕ) nd - (ಪೂರ್ಣಾಂಕ) ಒಳ_rank.get_ಉದ್ದಳತೆ() : 0;
                ತಾರ್ಕಿಕ ಸರಿಸು_ok = axis_ಸರಿಸು >= 0 && (size_t) axis_ಸರಿಸು < nd;
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ a = 0; a < axis_ಸರಿಸು && ಸರಿಸು_ok; ++a) {
                    ಸರಿಸು_ok = (ss[a] == 1 && dd[a] == 1);
                }
                ಆದರೆ (ಸರಿಸು_ok && sst.size() == nd && dst.size() == nd) {
                    // Map each dst axis of size>1 to a src axis with equal (size,stride);
                    // the unmatched src axis of size>1 is the indexed expert axis.
                    // dst_to_src[d] records which src axis each dst axis came from, so we can
                    // later pull the dynamic (token) dim from the right source axis at runtime.
                    std::vector<ತಾರ್ಕಿಕ> ಬಳಸಿದ(nd, ಸುಳ್ಳು);
                    std::vector<ಪೂರ್ಣಾಂಕ> dst_ಗಮ್ಯ_src(nd, -1);
                    ತಾರ್ಕಿಕ ok = ನಿಜ;
                    ಗಾಗಿ (size_t d = 0; d < nd; ++d) {
                        ಆದರೆ (dd[d] == 1) {
                            ಮುಂದುವರಿ;
                        }
                        ಪೂರ್ಣಾಂಕ ಸಿಕ್ಕಿತು = -1;
                        ಗಾಗಿ (size_t s = 0; s < nd; ++s) {
                            ಆದರೆ (!ಬಳಸಿದ[s] && ss[s] == dd[d] && sst[s] == dst[d]) {
                                ಸಿಕ್ಕಿತು = (ಪೂರ್ಣಾಂಕ) s;
                                ನಿಲ್ಲಿಸು;
                            }
                        }
                        ಆದರೆ (ಸಿಕ್ಕಿತು < 0) {
                            ok = ಸುಳ್ಳು;
                            ನಿಲ್ಲಿಸು;
                        }
                        ಬಳಸಿದ[ಸಿಕ್ಕಿತು] = ನಿಜ;
                        dst_ಗಮ್ಯ_src[d] = ಸಿಕ್ಕಿತು;
                    }
                    ಪೂರ್ಣಾಂಕ dropped = -1;
                    ಆದರೆ (ok) {
                        ಗಾಗಿ (size_t s = 0; s < nd; ++s) {
                            ಆದರೆ (!ಬಳಸಿದ[s] && ss[s] > 1) {
                                ಆದರೆ (dropped >= 0) {
                                    ok = ಸುಳ್ಳು;
                                    ನಿಲ್ಲಿಸು;
                                }
                                dropped = (ಪೂರ್ಣಾಂಕ) s;
                            }
                        }
                    }
                    ಆದರೆ (ok && dropped >= 0) {
                        ಸ್ಥಿರಾಂಕ size_t dstr = sst[dropped];
                        ಸ್ಥಿರಾಂಕ int64_t dsz = (int64_t) ss[dropped];
                        ಆದರೆ (dstr > 0 && voff % dstr == 0) {
                            ಸ್ಥಿರಾಂಕ int64_t sel = (int64_t) (voff / dstr);
                            ಆದರೆ (sel >= 0 && sel < dsz) {
                                ov::Output<ov::Node> sl = std::make_shared<ov::op::v8::ತುಂಡು>(
                                    context.get_ಒಳಹರಿವು(0),
                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {sel}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {sel + 1}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1},
                                                                 {dropped - axis_ಸರಿಸು}));
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
                                ಸ್ಥಿರಾಂಕ int32_t dyn = context.get_op_dynamic_dim();  // output ggml axis, -1 if none
                                // still in ggml metadata axis space; get_shape() reverses ggml order
                                ಪೂರ್ಣಾಂಕ dst_ov_axis = (dyn != -1) ? ((ಪೂರ್ಣಾಂಕ) nd - 1 - (ಪೂರ್ಣಾಂಕ) dyn) : -1;
                                ಪೂರ್ಣಾಂಕ src_ov_axis = (dst_ov_axis >= axis_ಸರಿಸು && dst_ov_axis < (ಪೂರ್ಣಾಂಕ) nd)
                                                      ? dst_ಗಮ್ಯ_src[dst_ov_axis]
                                                      : -1;
                                ಆದರೆ (dst_ov_axis >= 0 && src_ov_axis >= axis_ಸರಿಸು) {
                                    // target = concat of per-axis scalars; the token axis is a
                                    // runtime Gather of the slice's shape, the rest are constants.
                                    ಸ್ವಯಂಚಾಲಿತ sl_shape = std::make_shared<ov::op::v3::Shape_ದ>(sl, ov::element::i64);
                                    ಸ್ವಯಂಚಾಲಿತ tok_dim = std::make_shared<ov::op::v8::Gather>(
                                        sl_shape,
                                        ov::op::v0::Constant::create(ov::element::i64, {1},
                                                                     {src_ov_axis - axis_ಸರಿಸು}),
                                        ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
                                    ov::Output_ಸದಿಶ parts;
                                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ a = axis_ಸರಿಸು; a < (ಪೂರ್ಣಾಂಕ) nd; ++a) {
                                        ಆದರೆ (a == dst_ov_axis) {
                                            parts.push_back(tok_dim);
                                        } ಇಲ್ಲದಿದ್ದರೆ {
                                            parts.push_back(ov::op::v0::Constant::create(
                                                ov::element::i64, {1}, {(int64_t) dd[a]}));
                                        }
                                    }
                                    ಸ್ವಯಂಚಾಲಿತ dc = std::make_shared<ov::op::v0::Concat>(parts, 0);
                                    ಸ್ವಯಂಚಾಲಿತ rs = std::make_shared<ov::op::v1::Reshape>(sl, dc, ಸುಳ್ಳು);
                                    ಹಿಂತಿರುಗಿ rename_outputs_with_suffix({rs}, context.get_name());
                                }
                                std::vector<int64_t> dd_ov(dd.begin() + axis_ಸರಿಸು, dd.end());
                                ಸ್ವಯಂಚಾಲಿತ dc = ov::op::v0::Constant::create(ov::element::i64, {dd_ov.size()}, dd_ov);
                                ಸ್ವಯಂಚಾಲಿತ rs = std::make_shared<ov::op::v1::Reshape>(sl, dc, ಸುಳ್ಳು);
                                ಹಿಂತಿರುಗಿ rename_outputs_with_suffix({rs}, context.get_name());
                            }
                        }
                    }
                }
            }
        }
        ಹಿಂತಿರುಗಿ {context.get_ಒಳಹರಿವು(0)};
    }

    ಸ್ವಯಂಚಾಲಿತ ಒಳಹರಿವು = context.get_ಒಳಹರಿವು(0);
    ಸ್ವಯಂಚಾಲಿತ src_shape = context.get_ಒಳಹರಿವು_shape(0);
    ಸ್ವಯಂಚಾಲಿತ dst_shape = context.get_output_shape();

    ಆದರೆ (src_shape.rank().is_dynamic() || dst_shape.rank().is_dynamic()) {
        ಹಿಂತಿರುಗಿ {ಒಳಹರಿವು};
    }

    int64_t src_elems = 1;
    int64_t dst_elems = 1;
    ಗಾಗಿ (int64_t i = 0; i < src_shape.rank().get_ಉದ್ದಳತೆ(); ++i) {
        ಆದರೆ (src_shape[i].is_dynamic()) {
            ಹಿಂತಿರುಗಿ {ಒಳಹರಿವು};
        }
        src_elems *= src_shape[i].get_ಉದ್ದಳತೆ();
    }
    ಗಾಗಿ (int64_t i = 0; i < dst_shape.rank().get_ಉದ್ದಳತೆ(); ++i) {
        ಆದರೆ (dst_shape[i].is_dynamic()) {
            ಹಿಂತಿರುಗಿ {ಒಳಹರಿವು};
        }
        dst_elems *= dst_shape[i].get_ಉದ್ದಳತೆ();
    }

    ಆದರೆ (dst_elems >= src_elems) {
        ಹಿಂತಿರುಗಿ {ಒಳಹರಿವು};
    }

    ಸ್ವಯಂಚಾಲಿತ src_stride = context.get_ಒಳಹರಿವು_stride(0);
    ಸ್ವಯಂಚಾಲಿತ dst_stride = context.get_output_stride();
    size_t view_ಸ್ಥಾನಾಂತರ = context.get_output_op_ಸ್ಥಾನಾಂತರ();

    ತಾರ್ಕಿಕ same_stride = (src_stride.size() == dst_stride.size());
    ಆದರೆ (same_stride) {
        ಗಾಗಿ (size_t i = 0; i < src_stride.size(); ++i) {
            ಆದರೆ (src_stride[i] != dst_stride[i]) {
                same_stride = ಸುಳ್ಳು;
                ನಿಲ್ಲಿಸು;
            }
        }
    }

    ಆದರೆ (!same_stride) {
        ಹಿಂತಿರುಗಿ {ಒಳಹರಿವು};
    }

    ಸ್ವಯಂಚಾಲಿತ src_ov_shape = src_shape.ಗಮ್ಯ_shape();
    ಸ್ವಯಂಚಾಲಿತ dst_ov_shape = dst_shape.ಗಮ್ಯ_shape();
    size_t ndims = src_ov_shape.size();
    ಆದರೆ (dst_ov_shape.size() != ndims) {
        ಹಿಂತಿರುಗಿ {ಒಳಹರಿವು};
    }

    std::vector<ಪೂರ್ಣಾಂಕ> diff_dims;
    ಗಾಗಿ (size_t i = 0; i < ndims; ++i) {
        ಆದರೆ (src_ov_shape[i] != dst_ov_shape[i]) {
            diff_dims.push_back(static_cast<ಪೂರ್ಣಾಂಕ>(i));
        }
    }

    ಆದರೆ (diff_dims.size() != 1) {
        ಹಿಂತಿರುಗಿ {ಒಳಹರಿವು};
    }

    ಪೂರ್ಣಾಂಕ ತುಂಡು_dim = diff_dims[0];
    int64_t dim_ಗಾತ್ರ = static_cast<int64_t>(src_ov_shape[ತುಂಡು_dim]);

    size_t ov_stride_for_dim = 1;
    ಗಾಗಿ (size_t i = ತುಂಡು_dim + 1; i < ndims; ++i) {
        ov_stride_for_dim *= src_ov_shape[i];
    }
    size_t elem_ಗಾತ್ರ = src_stride.back();
    ಆದರೆ (elem_ಗಾತ್ರ == 0) {
        elem_ಗಾತ್ರ = 1;
    }

    int64_t begin_val = 0;
    ಆದರೆ (ov_stride_for_dim > 0 && elem_ಗಾತ್ರ > 0) {
        begin_val = static_cast<int64_t>((view_ಸ್ಥಾನಾಂತರ / elem_ಗಾತ್ರ) / ov_stride_for_dim);
    }
    int64_t ಕೊನೆ_val = begin_val + static_cast<int64_t>(dst_ov_shape[ತುಂಡು_dim]);

    ಆದರೆ (begin_val < 0 || ಕೊನೆ_val > dim_ಗಾತ್ರ) {
        ಹಿಂತಿರುಗಿ {ಒಳಹರಿವು};
    }

    ಸ್ವಯಂಚಾಲಿತ sliced =
        std::make_shared<ov::op::v8::ತುಂಡು>(ಒಳಹರಿವು, ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val}),
                                            ov::op::v0::Constant::create(ov::element::i64, {1}, {ಕೊನೆ_val}),
                                            ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                            ov::op::v0::Constant::create(ov::element::i64, {1}, {ತುಂಡು_dim}));

    sliced->ಗಣ_friendly_name(context.get_output_name());
    ಹಿಂತಿರುಗಿ {sliced->output(0)};
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
