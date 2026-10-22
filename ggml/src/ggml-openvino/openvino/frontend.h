// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ہدایت once

#شامل <openvino/frontend/frontend.hpp>

namespace ov {
namespace frontend {
namespace ggml {

class Front_انتہا {
عوامی:
    Front_انتہا();

    ساکن std::shared_ptr<Model> convert(ثابت داخلہ_Model::Ptr & model, bool naive = false);
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
