// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#അന്തര്ഭാവ <openvino/frontend/frontend.hpp>

namespace ov {
namespace frontend {
namespace ggml {

class FrontEnd {
പൊതു:
    FrontEnd();

    സ്ഥിതം std::shared_ptr<Model> convert(സ്ഥിരം നിവേശം_Model::Ptr & model, bool naive = false);
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
