#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-vulkan-common.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <cstdint>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdlib>
#ਅਨ੍ਤਰ੍ਭਾਵ <mutex>
#ਅਨ੍ਤਰ੍ਭਾਵ <shared_mutex>

ਸਥਿਰ ਖਾਲੀ * portable_aligned_malloc(size_t align, size_t size) {
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ _WIN32
    ਵਾਪਸ _aligned_malloc(size, align);
#ਨਹੀਂਤਾਂ
    ਖਾਲੀ * ptr = ਖਾਲੀ_ਸੂਚਕ;
    ਜੇ (posix_memalign(&ptr, align, size) != 0) {
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }
    ਵਾਪਸ ptr;
#ਜੇ_ਅੰਤ
}

ਸਥਿਰ ਖਾਲੀ portable_aligned_free(ਖਾਲੀ * ptr) {
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ _WIN32
    _aligned_free(ptr);
#ਨਹੀਂਤਾਂ
    std::ਮੁਕਤ(ptr);
#ਜੇ_ਅੰਤ
}

ggml_backend_ਬਫਰ_type_i ggml_backend_vk_ਬਫਰ_type_interface = {
    /* .get_name            = */ ggml_backend_vk_ਬਫਰ_type_name,
    /* .alloc_buffer        = */ ggml_backend_vk_ਬਫਰ_type_alloc_ਬਫਰ,
    /* .alloc_buffer_n      = */ ਨਿਰਾ,
    /* .get_alignment       = */ ggml_backend_vk_ਬਫਰ_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_vk_ਬਫਰ_type_get_max_ਮਾਪ,
    /* .get_alloc_size      = */ ggml_backend_vk_ਬਫਰ_type_get_alloc_ਮਾਪ,
    /* .get_alloc_size_n    = */ ਨਿਰਾ,
    /* .is_host             = */ ਨਿਰਾ,
};

ਸਥਿਰ std::vector<uint32_t> ggml_vk_ਲੱਭੋ_ਸਮ੍ਰਿਤੀ_properties(ਅਚਲ vk::Physical_Device_ਸਮ੍ਰਿਤੀ_Properties* mem_props, vk::ਸਮ੍ਰਿਤੀ_Requirements* mem_req, vk::ਸਮ੍ਰਿਤੀ_Property_Flags flags) {
    std::vector<uint32_t> indices;

    ਲਈ (uint32_t i = 0; i < mem_props->ਸਮ੍ਰਿਤੀ_Type_ਗਿਣਤੀ; ++i) {
        vk::ਸਮ੍ਰਿਤੀ_Type ਸਮ੍ਰਿਤੀ_type = mem_props->ਸਮ੍ਰਿਤੀ_Types[i];
        ਜੇ ((mem_req->ਸਮ੍ਰਿਤੀ_Type_ਬਿੱਟ & ((uint64_t)1 << i)) &&
            (flags & ਸਮ੍ਰਿਤੀ_type.propertyFlags) == flags &&
            mem_props->ਸਮ੍ਰਿਤੀ_Heaps[ਸਮ੍ਰਿਤੀ_type.ਢੇਰ_ਸੂਚਕਾਂਕ].size >= mem_req->size) {
            indices.push_back(i);
        }
    }
    ਵਾਪਸ indices;
}

ਸਥਿਰ vk_ਬਫਰ ggml_vk_create_ਬਫਰ(vk_device& device, size_t size, ਅਚਲ std::initializer_list<vk::ਸਮ੍ਰਿਤੀ_Property_Flags> & req_flags_list,
                                       ਖਾਲੀ *import_ptr = ਖਾਲੀ_ਸੂਚਕ) {
    VK_LOG_DEBUG("ggml_vk_create_buffer(" << device->name << ", " << size << ", " << to_string(req_flags_list.begin()[0]) << ", " << to_string(req_flags_list.begin()[req_flags_list.size()-1]) << ")");
    ਜੇ (size > device->max_ਬਫਰ_ਮਾਪ) {
        ਸੁੱਟੋ vk::ਬਾਹਰਲਾ_ਦਾ_Device_ਸਮ੍ਰਿਤੀ_Error("Requested buffer size exceeds device buffer size limit");
    }

    vk_ਬਫਰ buf = std::make_shared<vk_ਬਫਰ_struct>();

    ਜੇ (size == 0) {
        buf->size = 0;
        ਵਾਪਸ buf;
    }

    vk::ਬਫਰ_Usage_Flags usage_flags = vk::ਬਫਰ_Usage_Flag_ਬਿੱਟ::e_Storage_ਬਫਰ | vk::ਬਫਰ_Usage_Flag_ਬਿੱਟ::eTransferSrc | vk::ਬਫਰ_Usage_Flag_ਬਿੱਟ::eTransferDst;
    vk::ਸਮ੍ਰਿਤੀ_Allocate_Flags mem_flags {};
    ਜੇ (device->ਬਫਰ_device_ਪਤਾ) {
        usage_flags |= vk::ਬਫਰ_Usage_Flag_ਬਿੱਟ::e_Shader_Device_ਪਤਾ;
        mem_flags |= vk::ਸਮ੍ਰਿਤੀ_Allocate_Flag_ਬਿੱਟ::e_Device_ਪਤਾ;
    }

    vk::ਬਫਰ_Create_Info ਬਫਰ_create_info{
        vk::ਬਫਰ_Create_Flags(),
        size,
        usage_flags,
        vk::Sharing_ਬਹੁਲਕ::eExclusive,
        0,
        ਖਾਲੀ_ਸੂਚਕ,
    };

    vk::External_ਸਮ੍ਰਿਤੀ_ਬਫਰ_Create_Info external_ਸਮ੍ਰਿਤੀ_bci;
    ਜੇ (import_ptr) {
        external_ਸਮ੍ਰਿਤੀ_bci.handleTypes = vk::External_ਸਮ੍ਰਿਤੀ_Handle_Type_Flag_ਬਿੱਟ::e_Host_ਵੰਡ_EXT;
        ਬਫਰ_create_info.ਸੈੱਟ_PNext(&external_ਸਮ੍ਰਿਤੀ_bci);
    }

    buf->ਬਫਰ = device->device.create_ਬਫਰ(ਬਫਰ_create_info);

    vk::ਸਮ੍ਰਿਤੀ_Requirements mem_req = device->device.get_ਬਫਰ_ਸਮ੍ਰਿਤੀ_Requirements(buf->ਬਫਰ);

    vk::Physical_Device_ਸਮ੍ਰਿਤੀ_Properties mem_props = device->physical_device.get_ਸਮ੍ਰਿਤੀ_Properties();

    ਅਚਲ vk::ਸਮ੍ਰਿਤੀ_ਤਰਜੀਹ_Allocate_Info_EXT mem_ਤਰਜੀਹ_info { 1.0f };

    vk::ਸਮ੍ਰਿਤੀ_Allocate_Flags_Info mem_flags_info { mem_flags };

    ਜੇ (device->ਸਮ੍ਰਿਤੀ_ਤਰਜੀਹ) {
        mem_flags_info.ਸੈੱਟ_PNext(&mem_ਤਰਜੀਹ_info);
    }

    ਜੇ (import_ptr) {
        vk::ਸਮ੍ਰਿਤੀ_Host_ਸੂਚਕ_Properties_EXT host_ਸੂਚਕ_props;
        ਕੋਸ਼ਿਸ਼ {
            host_ਸੂਚਕ_props = device->device.get_ਸਮ੍ਰਿਤੀ_Host_ਸੂਚਕ_Properties_EXT(vk::External_ਸਮ੍ਰਿਤੀ_Handle_Type_Flag_ਬਿੱਟ::e_Host_ਵੰਡ_EXT, import_ptr);
        } ਫੜੋ (vk::SystemError& e) {
            GGML_LOG_WARN("ggml_vulkan: Failed getMemoryHostPointerPropertiesEXT (%s)\n", e.what());
            device->device.destroy_ਬਫਰ(buf->ਬਫਰ);
            ਵਾਪਸ {};
        }
        vk::Physical_Device_ਸਮ੍ਰਿਤੀ_Properties mem_props = device->physical_device.get_ਸਮ੍ਰਿਤੀ_Properties();

        uint32_t ਸਮ੍ਰਿਤੀ_type_idx;
        vk::ਸਮ੍ਰਿਤੀ_Property_Flags property_flags = *req_flags_list.begin();
        ਲਈ (ਸਮ੍ਰਿਤੀ_type_idx = 0; ਸਮ੍ਰਿਤੀ_type_idx < 32; ++ਸਮ੍ਰਿਤੀ_type_idx) {
            ਜੇ (!(host_ਸੂਚਕ_props.ਸਮ੍ਰਿਤੀ_Type_ਬਿੱਟ & (1u << ਸਮ੍ਰਿਤੀ_type_idx))) {
                ਜਾਰੀ;
            }
            ਜੇ (!(mem_req.ਸਮ੍ਰਿਤੀ_Type_ਬਿੱਟ & (1u << ਸਮ੍ਰਿਤੀ_type_idx))) {
                ਜਾਰੀ;
            }

            vk::ਸਮ੍ਰਿਤੀ_Type ਸਮ੍ਰਿਤੀ_type = mem_props.ਸਮ੍ਰਿਤੀ_Types[ਸਮ੍ਰਿਤੀ_type_idx];
            // check for visible+coherent+cached. Other flags (e.g. devicelocal) are allowed
            ਜੇ ((ਸਮ੍ਰਿਤੀ_type.propertyFlags & property_flags) == property_flags) {
                property_flags = ਸਮ੍ਰਿਤੀ_type.propertyFlags;
                ਰੋਕੋ;
            }
        }
        ਜੇ (ਸਮ੍ਰਿਤੀ_type_idx == 32) {
            GGML_LOG_WARN("ggml_vulkan: Memory type for host allocation not found\n");
            device->device.destroy_ਬਫਰ(buf->ਬਫਰ);
            ਵਾਪਸ {};
        }

        buf->ਸਮ੍ਰਿਤੀ_property_flags = mem_props.ਸਮ੍ਰਿਤੀ_Types[ਸਮ੍ਰਿਤੀ_type_idx].propertyFlags;
        ਕੋਸ਼ਿਸ਼ {
            vk::Import_ਸਮ੍ਰਿਤੀ_Host_ਸੂਚਕ_Info_EXT import_info;
            import_info.handleType = vk::External_ਸਮ੍ਰਿਤੀ_Handle_Type_Flag_ਬਿੱਟ::e_Host_ਵੰਡ_EXT;
            import_info.p_Host_ਸੂਚਕ = import_ptr;
            import_info.ਸੈੱਟ_PNext(&mem_flags_info);
            buf->device_ਸਮ੍ਰਿਤੀ = device->device.allocate_ਸਮ੍ਰਿਤੀ({ size, ਸਮ੍ਰਿਤੀ_type_idx, &import_info });
        } ਫੜੋ (ਅਚਲ vk::SystemError& e) {
        }
    } ਨਹੀਂਤਾਂ {
        ਲਈ (ਸਵੈਚਾਲਿਤ it = req_flags_list.begin(); it != req_flags_list.end(); it++) {
            ਅਚਲ ਸਵੈਚਾਲਿਤ & req_flags = *it;

            ਅਚਲ std::vector<uint32_t> ਸਮ੍ਰਿਤੀ_type_indices = ggml_vk_ਲੱਭੋ_ਸਮ੍ਰਿਤੀ_properties(&mem_props, &mem_req, req_flags);

            ਜੇ (ਸਮ੍ਰਿਤੀ_type_indices.ਖਾਲੀ_ਥਾਵਾਂ()) {
                ਜਾਰੀ;
            }

            ਤਾਰਕਿਕ ਪੂਰਾ = ਝੂਠ;

            ਲਈ (ਸਵੈਚਾਲਿਤ mtype_it = ਸਮ੍ਰਿਤੀ_type_indices.begin(); mtype_it != ਸਮ੍ਰਿਤੀ_type_indices.end(); mtype_it++) {
                ਕੋਸ਼ਿਸ਼ {
                    buf->device_ਸਮ੍ਰਿਤੀ = device->device.allocate_ਸਮ੍ਰਿਤੀ({ mem_req.size, *mtype_it, &mem_flags_info });
                    buf->ਸਮ੍ਰਿਤੀ_property_flags = mem_props.ਸਮ੍ਰਿਤੀ_Types[*mtype_it].propertyFlags;
                    ਪੂਰਾ = ਸੱਚ;
                    ਰੋਕੋ;
                } ਫੜੋ (ਅਚਲ vk::SystemError& e) {
                    // loop and retry
                    // during last attempt throw the exception
                    ਜੇ (it + 1 == req_flags_list.end() && mtype_it + 1 == ਸਮ੍ਰਿਤੀ_type_indices.end()) {
                        device->device.destroy_ਬਫਰ(buf->ਬਫਰ);
                        ਸੁੱਟੋ e;
                    }
                }
            }

            ਜੇ (ਪੂਰਾ) {
                ਰੋਕੋ;
            }
        }
    }

    ਜੇ (!buf->device_ਸਮ੍ਰਿਤੀ) {
        device->device.destroy_ਬਫਰ(buf->ਬਫਰ);
        ਸੁੱਟੋ vk::ਬਾਹਰਲਾ_ਦਾ_Device_ਸਮ੍ਰਿਤੀ_Error("No suitable memory type found");
    }

    buf->ptr = ਖਾਲੀ_ਸੂਚਕ;

    ਜੇ (import_ptr) {
        buf->ptr = import_ptr;
    } ਨਹੀਂਤਾਂ {
        ਜੇ (buf->ਸਮ੍ਰਿਤੀ_property_flags & vk::ਸਮ੍ਰਿਤੀ_Property_Flag_ਬਿੱਟ::eHostVisible) {
            buf->ptr = device->device.map_ਸਮ੍ਰਿਤੀ(buf->device_ਸਮ੍ਰਿਤੀ, 0, VK_ਸਾਰਾ_ਮਾਪ);
        }
    }

    device->device.bind_ਬਫਰ_ਸਮ੍ਰਿਤੀ(buf->ਬਫਰ, buf->device_ਸਮ੍ਰਿਤੀ, 0);

    buf->device = device;
    buf->size = size;

    ਜੇ (device->ਬਫਰ_device_ਪਤਾ) {
        ਅਚਲ vk::ਬਫਰ_Device_ਪਤਾ_Info ਪਤਾ_Info(buf->ਬਫਰ);
        buf->bda_addr = device->device.get_ਬਫਰ_ਪਤਾ(ਪਤਾ_Info);
    }

    device->ਸਮ੍ਰਿਤੀ_logger->log_ਵੰਡ(buf, size);

    ਵਾਪਸ buf;
}

vk_ਬਫਰ ggml_vk_create_ਬਫਰ_check(vk_device& device, size_t size, vk::ਸਮ੍ਰਿਤੀ_Property_Flags req_flags, vk::ਸਮ੍ਰਿਤੀ_Property_Flags fallback_flags) {
    ਕੋਸ਼ਿਸ਼ {
        ਵਾਪਸ ggml_vk_create_ਬਫਰ(device, size, {req_flags, fallback_flags});
    } ਫੜੋ (ਅਚਲ vk::SystemError& e) {
        std::cerr << "ggml_vulkan: Memory allocation of size " << size << " failed." << std::ਸਤਰ_ਅੰਤ;
        std::cerr << "ggml_vulkan: " << e.what() << std::ਸਤਰ_ਅੰਤ;
        ਸੁੱਟੋ e;
    }
}

vk_ਬਫਰ ggml_vk_create_ਬਫਰ_device(vk_device& device, size_t size) {
    vk_ਬਫਰ buf;
    ਕੋਸ਼ਿਸ਼ {
        ਜੇ (device->prefer_host_ਸਮ੍ਰਿਤੀ) {
            buf = ggml_vk_create_ਬਫਰ(device, size, {vk::ਸਮ੍ਰਿਤੀ_Property_Flag_ਬਿੱਟ::eHostVisible | vk::ਸਮ੍ਰਿਤੀ_Property_Flag_ਬਿੱਟ::eHostCoherent,
                                                       vk::ਸਮ੍ਰਿਤੀ_Property_Flag_ਬਿੱਟ::eDeviceLocal});
        } ਨਹੀਂਤਾਂ ਜੇ (device->uma) {
            // On UMA, prefer host-visible memory so direct tensor borrowing works.
            // If unavailable, fall back to device-local memory.
            buf = ggml_vk_create_ਬਫਰ(device, size, {vk::ਸਮ੍ਰਿਤੀ_Property_Flag_ਬਿੱਟ::eDeviceLocal | vk::ਸਮ੍ਰਿਤੀ_Property_Flag_ਬਿੱਟ::eHostVisible | vk::ਸਮ੍ਰਿਤੀ_Property_Flag_ਬਿੱਟ::eHostCoherent,
                                                       vk::ਸਮ੍ਰਿਤੀ_Property_Flag_ਬਿੱਟ::eDeviceLocal,
                                                       vk::ਸਮ੍ਰਿਤੀ_Property_Flag_ਬਿੱਟ::eHostVisible | vk::ਸਮ੍ਰਿਤੀ_Property_Flag_ਬਿੱਟ::eHostCoherent});
        } ਨਹੀਂਤਾਂ ਜੇ (device->disable_host_visible_vidmem) {
            ਜੇ (device->allow_sysmem_fallback) {
                buf = ggml_vk_create_ਬਫਰ(device, size, {vk::ਸਮ੍ਰਿਤੀ_Property_Flag_ਬਿੱਟ::eDeviceLocal,
                                                           vk::ਸਮ੍ਰਿਤੀ_Property_Flag_ਬਿੱਟ::eHostVisible | vk::ਸਮ੍ਰਿਤੀ_Property_Flag_ਬਿੱਟ::eHostCoherent});
            } ਨਹੀਂਤਾਂ {
                buf = ggml_vk_create_ਬਫਰ(device, size, {vk::ਸਮ੍ਰਿਤੀ_Property_Flag_ਬਿੱਟ::eDeviceLocal});
            }
        } ਨਹੀਂਤਾਂ {
            // use rebar if available, otherwise fallback to device only visible memory
            ਜੇ (device->allow_sysmem_fallback) {
                buf = ggml_vk_create_ਬਫਰ(device, size, {vk::ਸਮ੍ਰਿਤੀ_Property_Flag_ਬਿੱਟ::eDeviceLocal | vk::ਸਮ੍ਰਿਤੀ_Property_Flag_ਬਿੱਟ::eHostVisible | vk::ਸਮ੍ਰਿਤੀ_Property_Flag_ਬਿੱਟ::eHostCoherent,
                                                           vk::ਸਮ੍ਰਿਤੀ_Property_Flag_ਬਿੱਟ::eDeviceLocal,
                                                           vk::ਸਮ੍ਰਿਤੀ_Property_Flag_ਬਿੱਟ::eHostVisible | vk::ਸਮ੍ਰਿਤੀ_Property_Flag_ਬਿੱਟ::eHostCoherent});
            } ਨਹੀਂਤਾਂ {
                buf = ggml_vk_create_ਬਫਰ(device, size, {vk::ਸਮ੍ਰਿਤੀ_Property_Flag_ਬਿੱਟ::eDeviceLocal | vk::ਸਮ੍ਰਿਤੀ_Property_Flag_ਬਿੱਟ::eHostVisible | vk::ਸਮ੍ਰਿਤੀ_Property_Flag_ਬਿੱਟ::eHostCoherent,
                                                           vk::ਸਮ੍ਰਿਤੀ_Property_Flag_ਬਿੱਟ::eDeviceLocal});
            }
        }
    } ਫੜੋ (ਅਚਲ vk::SystemError& e) {
        std::cerr << "ggml_vulkan: Device memory allocation of size " << size << " failed." << std::ਸਤਰ_ਅੰਤ;
        std::cerr << "ggml_vulkan: " << e.what() << std::ਸਤਰ_ਅੰਤ;
        ਸੁੱਟੋ e;
    }

    ਵਾਪਸ buf;
}

ਖਾਲੀ ggml_vk_destroy_ਬਫਰ(vk_ਬਫਰ& buf) {
    ਜੇ (buf == ਖਾਲੀ_ਸੂਚਕ) {
        ਵਾਪਸ;
    }

    ਜੇ (buf->device != ਖਾਲੀ_ਸੂਚਕ) {
        buf->device->ਸਮ੍ਰਿਤੀ_logger->log_deallocation(buf);
    }

    buf.reset();
}

// caller must hold host_allocations_mutex
ਸਥਿਰ size_t ggml_vk_host_ਵੰਡ_ਲੱਭੋ(ਅਚਲ ਖਾਲੀ * ptr) {
    ਲਈ (size_t i = 0; i < vk_instance.host_allocations.size(); i++) {
        ਅਚਲ uint8_t * ਅਧਾਰ = (ਅਚਲ uint8_t *) vk_instance.host_allocations[i].ਅਧਾਰ;
        ਜੇ (ptr >= ਅਧਾਰ && ptr < ਅਧਾਰ + vk_instance.host_allocations[i].size) {
            ਵਾਪਸ i;
        }
    }
    ਵਾਪਸ SIZE_MAX;
}

ਖਾਲੀ * ggml_vk_host_malloc(size_t size) {
    VK_LOG_ਸਮ੍ਰਿਤੀ("ggml_vk_host_malloc(" << size << ")");

    ggml_vk_instance_init();

    ਜੇ (vk_instance.multi_device) {
        ਅਚਲ size_t align      = std::max<size_t>(vk_instance.host_import_alignment, TENSOR_ALIGNMENT);
        ਅਚਲ size_t alloc_ਮਾਪ = (size + align - 1) / align * align;

        ਖਾਲੀ * ptr = portable_aligned_malloc(align, alloc_ਮਾਪ);
        ਜੇ (ptr == ਖਾਲੀ_ਸੂਚਕ) {
            ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
        }
        vk_host_ਵੰਡ ਇੰਦਰਾਜ;
        ਇੰਦਰਾਜ.ਅਧਾਰ = ptr;
        ਇੰਦਰਾਜ.size = alloc_ਮਾਪ;

        std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);
        vk_instance.host_allocations.push_back(std::move(ਇੰਦਰਾਜ));
        ਵਾਪਸ ptr;
    }

    vk_device dev0 = ggml_vk_get_device(0);
    vk_ਬਫਰ buf;
    ਕੋਸ਼ਿਸ਼ {
        buf = ggml_vk_create_ਬਫਰ(dev0, size,
            {vk::ਸਮ੍ਰਿਤੀ_Property_Flag_ਬਿੱਟ::eHostVisible | vk::ਸਮ੍ਰਿਤੀ_Property_Flag_ਬਿੱਟ::eHostCoherent | vk::ਸਮ੍ਰਿਤੀ_Property_Flag_ਬਿੱਟ::eHostCached,
             vk::ਸਮ੍ਰਿਤੀ_Property_Flag_ਬਿੱਟ::eHostVisible | vk::ਸਮ੍ਰਿਤੀ_Property_Flag_ਬਿੱਟ::eHostCoherent});
    } ਫੜੋ (vk::SystemError&) {
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }

    ਜੇ (!(buf->ਸਮ੍ਰਿਤੀ_property_flags & vk::ਸਮ੍ਰਿਤੀ_Property_Flag_ਬਿੱਟ::eHostVisible)) {
        fprintf(stderr, "WARNING: failed to allocate %.2f MB of pinned memory\n", size / 1024.0 / 1024.0);
        ggml_vk_destroy_ਬਫਰ(buf);
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }

    vk_host_ਵੰਡ ਇੰਦਰਾਜ;
    ਇੰਦਰਾਜ.ਅਧਾਰ             = buf->ptr;
    ਇੰਦਰਾਜ.size             = size;
    ਇੰਦਰਾਜ.views[dev0->idx] = buf;

    std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);
    vk_instance.host_allocations.push_back(std::move(ਇੰਦਰਾਜ));
    ਵਾਪਸ buf->ptr;
}

ਖਾਲੀ ggml_vk_host_free(ਖਾਲੀ * ptr) {
    ਜੇ (ptr == ਖਾਲੀ_ਸੂਚਕ) {
        ਵਾਪਸ;
    }
    VK_LOG_ਸਮ੍ਰਿਤੀ("ggml_vk_host_free(" << ptr << ")");

    vk_host_ਵੰਡ ਇੰਦਰਾਜ;
    {
        std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);
        ਅਚਲ size_t ਸੂਚਕਾਂਕ = ggml_vk_host_ਵੰਡ_ਲੱਭੋ(ptr);
        ਜੇ (ਸੂਚਕਾਂਕ == SIZE_MAX) {
            fprintf(stderr, "WARNING: failed to free pinned memory: memory not in map\n");
            ਵਾਪਸ;
        }
        ਇੰਦਰਾਜ = std::move(vk_instance.host_allocations[ਸੂਚਕਾਂਕ]);
        vk_instance.host_allocations.erase(vk_instance.host_allocations.begin() + ਸੂਚਕਾਂਕ);
    }

    ਲਈ (ਸਵੈਚਾਲਿਤ & view : ਇੰਦਰਾਜ.views) {
        ggml_vk_destroy_ਬਫਰ(view);
    }
    ਜੇ (vk_instance.multi_device) {
        portable_aligned_free(ਇੰਦਰਾਜ.ਅਧਾਰ);
    }
}

ਖਾਲੀ ggml_vk_host_get(ਅਚਲ vk_device& device, ਅਚਲ ਖਾਲੀ * ptr, vk_ਬਫਰ& buf, size_t& buf_ਵਿੱਥ) {
    buf = ਖਾਲੀ_ਸੂਚਕ;
    buf_ਵਿੱਥ = 0;

    std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);

    ਅਚਲ size_t ਸੂਚਕਾਂਕ = ggml_vk_host_ਵੰਡ_ਲੱਭੋ(ptr);
    ਜੇ (ਸੂਚਕਾਂਕ == SIZE_MAX) {
        ਵਾਪਸ;
    }
    vk_host_ਵੰਡ & ਇੰਦਰਾਜ = vk_instance.host_allocations[ਸੂਚਕਾਂਕ];

    ਜੇ (ਇੰਦਰਾਜ.views[device->idx] == ਖਾਲੀ_ਸੂਚਕ) {
        ਅਚਲ uint32_t ਦੋਆਧਾਰੀ_ਅੰਕ = 1u << device->idx;
        ਜੇ (!vk_instance.multi_device || !vk_instance.device_host_ਬਫਰ_import[device->idx] || (vk_instance.host_import_failed & ਦੋਆਧਾਰੀ_ਅੰਕ)) {
            ਵਾਪਸ;
        }
        GGML_ASSERT(device->external_ਸਮ੍ਰਿਤੀ_host);
        // const only propagates from the const context; the pointee is mutable
        ਇੰਦਰਾਜ.views[device->idx] = ggml_vk_ਬਫਰ_from_host_ptr(const_cast<vk_device&>(device), ਇੰਦਰਾਜ.ਅਧਾਰ, ਇੰਦਰਾਜ.size);
        ਜੇ (ਇੰਦਰਾਜ.views[device->idx] == ਖਾਲੀ_ਸੂਚਕ) {
            vk_instance.host_import_failed |= ਦੋਆਧਾਰੀ_ਅੰਕ;
            GGML_LOG_WARN("ggml_vulkan: %s: host memory import failed, using staging\n", device->name.c_str());
            ਵਾਪਸ;
        }
    }

    buf = ਇੰਦਰਾਜ.views[device->idx];
    buf_ਵਿੱਥ = (ਅਚਲ uint8_t *) ptr - (ਅਚਲ uint8_t *) ਇੰਦਰਾਜ.ਅਧਾਰ;
}

ਖਾਲੀ ggml_vk_ensure_sync_staging_ਬਫਰ(vk_device& device, size_t size) {
    ਜੇ (device->sync_staging == ਖਾਲੀ_ਸੂਚਕ || device->sync_staging->size < size) {
        VK_LOG_ਸਮ੍ਰਿਤੀ("ggml_vk_ensure_sync_staging_buffer(" << size << ")");
        ggml_vk_destroy_ਬਫਰ(device->sync_staging);
        device->sync_staging = ggml_vk_create_ਬਫਰ_check(device, size,
            vk::ਸਮ੍ਰਿਤੀ_Property_Flag_ਬਿੱਟ::eHostVisible | vk::ਸਮ੍ਰਿਤੀ_Property_Flag_ਬਿੱਟ::eHostCoherent | vk::ਸਮ੍ਰਿਤੀ_Property_Flag_ਬਿੱਟ::eHostCached,
            vk::ਸਮ੍ਰਿਤੀ_Property_Flag_ਬਿੱਟ::eHostVisible | vk::ਸਮ੍ਰਿਤੀ_Property_Flag_ਬਿੱਟ::eHostCoherent);
    }
}

ਖਾਲੀ ggml_vk_ensure_sync_staging_ਬਫਰ(ggml_backend_vk_context * ctx, size_t size) {
    ਜੇ (ctx->sync_staging == ਖਾਲੀ_ਸੂਚਕ || ctx->sync_staging->size < size) {
        VK_LOG_ਸਮ੍ਰਿਤੀ("ggml_vk_ensure_sync_staging_buffer(" << size << ")");
        ggml_vk_destroy_ਬਫਰ(ctx->sync_staging);
        ctx->sync_staging = ggml_vk_create_ਬਫਰ_check(ctx->device, size,
            vk::ਸਮ੍ਰਿਤੀ_Property_Flag_ਬਿੱਟ::eHostVisible | vk::ਸਮ੍ਰਿਤੀ_Property_Flag_ਬਿੱਟ::eHostCoherent | vk::ਸਮ੍ਰਿਤੀ_Property_Flag_ਬਿੱਟ::eHostCached,
            vk::ਸਮ੍ਰਿਤੀ_Property_Flag_ਬਿੱਟ::eHostVisible | vk::ਸਮ੍ਰਿਤੀ_Property_Flag_ਬਿੱਟ::eHostCoherent);
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_vk_ਬਫਰ_write_nc_async(ggml_backend_vk_context * ctx, vk_context& subctx, vk_ਬਫਰ& dst, size_t ਵਿੱਥ, ਅਚਲ ggml_tensor * tensor, ਤਾਰਕਿਕ sync_staging = ਝੂਠ) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_nc_async(" << tensor << ")");
    GGML_ASSERT(!ggml_is_contiguous(tensor));
    // Buffer is already mapped
    ਜੇ(dst->ਸਮ੍ਰਿਤੀ_property_flags & vk::ਸਮ੍ਰਿਤੀ_Property_Flag_ਬਿੱਟ::eHostVisible) {
        std::cerr << "ggml_vulkan: buffer_write_nc_async dst buffer is host_visible. Use synchronous write." << std::ਸਤਰ_ਅੰਤ;
        GGML_ABORT("fatal error");
    }
    // Check if src is pinned memory
    vk_ਬਫਰ buf = ਖਾਲੀ_ਸੂਚਕ;
    size_t buf_ਵਿੱਥ = 0;
    ggml_vk_host_get(ctx->device, tensor->ਅੰਕੜੇ, buf, buf_ਵਿੱਥ);

    ਅਚਲ uint64_t ne0 = tensor->ne[0];
    ਅਚਲ uint64_t ne1 = tensor->ne[1];
    ਅਚਲ uint64_t ne2 = tensor->ne[2];
    ਅਚਲ uint64_t ne3 = tensor->ne[3];
    ਅਚਲ uint64_t nb0 = tensor->nb[0];
    ਅਚਲ uint64_t nb1 = tensor->nb[1];
    ਅਚਲ uint64_t nb2 = tensor->nb[2];
    ਅਚਲ uint64_t nb3 = tensor->nb[3];
    ਅਚਲ ggml_type type = tensor->type;
    ਅਚਲ uint64_t ts = ggml_type_ਮਾਪ(type);
    ਅਚਲ uint64_t bs = ggml_blck_ਮਾਪ(type);

    ਅਚਲ uint64_t dstnb0 = ts;
    ਅਚਲ uint64_t dstnb1 = dstnb0*(ne0/bs);
    ਅਚਲ uint64_t dstnb2 = dstnb1*ne1;
    ਅਚਲ uint64_t dstnb3 = dstnb2*ne2;

    ਅਚਲ uint64_t ne = ggml_nelements(tensor);

    ਜੇ (buf != ਖਾਲੀ_ਸੂਚਕ) {
        // Memory is pinned, use as staging buffer
        std::vector<vk::ਬਫਰ_Copy> slices;

        ਲਈ (uint64_t i3 = 0; i3 < ne3; i3++) {
            ਲਈ (uint64_t i2 = 0; i2 < ne2; i2++) {
                // Find longest contiguous slice
                ਜੇ (ne1*nb1 == dstnb2) {
                    slices.push_back({ buf_ਵਿੱਥ + i3*nb3 + i2*nb2, ਵਿੱਥ + i3*dstnb3 + i2*dstnb2, dstnb2 });
                } ਨਹੀਂਤਾਂ {
                    ਲਈ (uint64_t i1 = 0; i1 < ne1; i1++) {
                        ਜੇ (ne0*nb0/bs == dstnb1) {
                            slices.push_back({ buf_ਵਿੱਥ + i3*nb3 + i2*nb2 + i1*nb1, ਵਿੱਥ + i3*dstnb3 + i2*dstnb2 + i1*dstnb1, dstnb1 });
                        } ਨਹੀਂਤਾਂ {
                            ਅਚਲ uint64_t s_off = buf_ਵਿੱਥ + i3*nb3 + i2*nb2 + i1*nb1;
                            ਅਚਲ uint64_t d_off = ਵਿੱਥ + i3*dstnb3 + i2*dstnb2 + i1*dstnb1;
                            ਲਈ (uint64_t i0 = 0; i0 < ne0; i0++) {
                                slices.push_back({ s_off + i0*nb0, d_off + i0*dstnb0, dstnb0 });
                            }
                        }
                    }
                }
            }
        }

        ggml_vk_sync_buffers(ctx, subctx);
        subctx->s->ਬਫਰ->buf.copy_ਬਫਰ(buf->ਬਫਰ, dst->ਬਫਰ, slices);
        ਵਾਪਸ;
    }

    ਜੇ (!sync_staging) {
        GGML_ABORT("Asynchronous write to non-pinned memory not supported");
    }

    // Staging buffer required
    vk_ਬਫਰ& staging = ctx->device->sync_staging;
    ਅਚਲ uint64_t copy_ਮਾਪ = ts*ne/bs;
    ggml_vk_ensure_sync_staging_ਬਫਰ(ctx->device, copy_ਮਾਪ);
    Vk_ਬਫਰ_Copy buf_copy{ 0, ਵਿੱਥ, copy_ਮਾਪ };

    ggml_vk_sync_buffers(ctx, subctx);
    vk_Cmd_Copy_ਬਫਰ(subctx->s->ਬਫਰ->buf, (Vk_ਬਫਰ)staging->ਬਫਰ, (Vk_ਬਫਰ)dst->ਬਫਰ, 1, &buf_copy);

    ਲਈ (uint64_t i3 = 0; i3 < ne3; i3++) {
        ਲਈ (uint64_t i2 = 0; i2 < ne2; i2++) {
            // Find longest contiguous slice
            ਜੇ (ne1*nb1 == dstnb2) {
                deferred_memcpy((uint8_t *)staging->ptr + i3*dstnb3 + i2*dstnb2, (ਅਚਲ uint8_t *) tensor->ਅੰਕੜੇ + buf_ਵਿੱਥ + i3*nb3 + i2*nb2, dstnb2, &subctx->ਵਿੱਚ_memcpys);
            } ਨਹੀਂਤਾਂ {
                ਲਈ (uint64_t i1 = 0; i1 < ne1; i1++) {
                    ਜੇ (ne0*nb0/bs == dstnb1) {
                        deferred_memcpy((uint8_t *)staging->ptr + i3*dstnb3 + i2*dstnb2 + i1*dstnb1, (ਅਚਲ uint8_t *) tensor->ਅੰਕੜੇ + buf_ਵਿੱਥ + i3*nb3 + i2*nb2 + i1*nb1, dstnb1, &subctx->ਵਿੱਚ_memcpys);
                    } ਨਹੀਂਤਾਂ {
                        ਅਚਲ uint64_t s_off = buf_ਵਿੱਥ + i3*nb3 + i2*nb2 + i1*nb1;
                        ਅਚਲ uint64_t d_off = i3*dstnb3 + i2*dstnb2 + i1*dstnb1;
                        ਲਈ (uint64_t i0 = 0; i0 < ne0; i0++) {
                            deferred_memcpy((uint8_t *)staging->ptr + d_off + i0*dstnb0, (ਅਚਲ uint8_t *) tensor->ਅੰਕੜੇ + s_off + i0*nb0, dstnb0, &subctx->ਵਿੱਚ_memcpys);
                        }
                    }
                }
            }
        }
    }
}

ਤਾਰਕਿਕ ggml_vk_ਬਫਰ_write_2d_async(vk_context subctx, vk_ਬਫਰ& dst, size_t ਵਿੱਥ, ਅਚਲ ਖਾਲੀ * src, size_t spitch, size_t dpitch, size_t ਚੌੜਾਈ, size_t ਉਚਾਈ, ਤਾਰਕਿਕ sync_staging) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_2d_async(" << ਚੌੜਾਈ << ", " << ਉਚਾਈ << ")");
    // Check if src is pinned memory
    vk_ਬਫਰ buf = ਖਾਲੀ_ਸੂਚਕ;
    size_t buf_ਵਿੱਥ = 0;
    ggml_vk_host_get(dst->device, src, buf, buf_ਵਿੱਥ);

    ਜੇ (buf != ਖਾਲੀ_ਸੂਚਕ) {
        // Memory is pinned, use as staging buffer
        std::vector<vk::ਬਫਰ_Copy> slices(1);
        ਜੇ (ਚੌੜਾਈ == spitch && ਚੌੜਾਈ == dpitch) {
            // Only do single write if stride is equal
            slices[0].src_ਵਿੱਥ_2 = buf_ਵਿੱਥ;
            slices[0].dst_ਵਿੱਥ_2 = ਵਿੱਥ;
            slices[0].size = ਚੌੜਾਈ * ਉਚਾਈ;
        } ਨਹੀਂਤਾਂ {
            slices.resize(ਉਚਾਈ);
            ਲਈ (size_t i = 0; i < ਉਚਾਈ; i++) {
                slices[i].src_ਵਿੱਥ_2 = buf_ਵਿੱਥ + i * spitch;
                slices[i].dst_ਵਿੱਥ_2 = ਵਿੱਥ + i * dpitch;
                slices[i].size = ਚੌੜਾਈ;
            }
        }

        ggml_vk_sync_buffers(ਖਾਲੀ_ਸੂਚਕ, subctx);
        subctx->s->ਬਫਰ->buf.copy_ਬਫਰ(buf->ਬਫਰ, dst->ਬਫਰ, slices);
        ਵਾਪਸ ਸੱਚ;
    }
    VK_LOG_DEBUG("STAGING");

    ਜੇ (!sync_staging) {
        // copy was not handled caller needs to fall back
        ਵਾਪਸ ਝੂਠ;
    }

    // Staging buffer required
    ਅਚਲ size_t staging_ਮਾਪ = ਚੌੜਾਈ * ਉਚਾਈ;
    ggml_vk_ensure_sync_staging_ਬਫਰ(dst->device, staging_ਮਾਪ);

    vk_ਬਫਰ& staging_ਬਫਰ = dst->device->sync_staging;

    std::vector<vk::ਬਫਰ_Copy> slices(1);
    ਜੇ (ਚੌੜਾਈ == dpitch) {
        slices[0].src_ਵਿੱਥ_2 = 0;
        slices[0].dst_ਵਿੱਥ_2 = ਵਿੱਥ;
        slices[0].size = staging_ਮਾਪ;
    } ਨਹੀਂਤਾਂ {
        slices.resize(ਉਚਾਈ);
        ਲਈ (size_t i = 0; i < ਉਚਾਈ; i++) {
            slices[i].src_ਵਿੱਥ_2 = i * ਚੌੜਾਈ;
            slices[i].dst_ਵਿੱਥ_2 = ਵਿੱਥ + i * dpitch;
            slices[i].size = ਚੌੜਾਈ;
        }
    }

    ggml_vk_sync_buffers(ਖਾਲੀ_ਸੂਚਕ, subctx);
    subctx->s->ਬਫਰ->buf.copy_ਬਫਰ(staging_ਬਫਰ->ਬਫਰ, dst->ਬਫਰ, slices);

    ਜੇ (ਚੌੜਾਈ == spitch) {
        deferred_memcpy((uint8_t *)staging_ਬਫਰ->ptr, src, staging_ਮਾਪ, &subctx->ਵਿੱਚ_memcpys);
    } ਨਹੀਂਤਾਂ {
        ਲਈ (size_t i = 0; i < ਉਚਾਈ; i++) {
            deferred_memcpy((uint8_t *)staging_ਬਫਰ->ptr + i * ਚੌੜਾਈ, (ਅਚਲ uint8_t *) src + i * spitch, ਚੌੜਾਈ, &subctx->ਵਿੱਚ_memcpys);
        }
    }
    ਵਾਪਸ ਸੱਚ;
}

ਤਾਰਕਿਕ ggml_vk_ਬਫਰ_write_async(vk_context subctx, vk_ਬਫਰ& dst, size_t ਵਿੱਥ, ਅਚਲ ਖਾਲੀ * src, size_t size, ਤਾਰਕਿਕ sync_staging) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_async(" << size << ")");
    ਵਾਪਸ ggml_vk_ਬਫਰ_write_2d_async(subctx, dst, ਵਿੱਥ, src, size, size, size, 1, sync_staging);
}

ਖਾਲੀ ggml_vk_ਬਫਰ_write_2d(vk_ਬਫਰ& dst, size_t ਵਿੱਥ, ਅਚਲ ਖਾਲੀ * src, size_t spitch, size_t dpitch, size_t ਚੌੜਾਈ, size_t ਉਚਾਈ) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_2d(" << ਚੌੜਾਈ << ", " << ਉਚਾਈ << ")");
    // Buffer is already mapped
    ਜੇ(dst->ਸਮ੍ਰਿਤੀ_property_flags & vk::ਸਮ੍ਰਿਤੀ_Property_Flag_ਬਿੱਟ::eHostVisible) {
        GGML_ASSERT(dst->ਸਮ੍ਰਿਤੀ_property_flags & vk::ਸਮ੍ਰਿਤੀ_Property_Flag_ਬਿੱਟ::eHostCoherent);

        ਜੇ (ਚੌੜਾਈ == spitch && ਚੌੜਾਈ == dpitch) {
            ਸਮ੍ਰਿਤੀ_ਨਕਲ((uint8_t *)dst->ptr + ਵਿੱਥ, src, ਚੌੜਾਈ * ਉਚਾਈ);
        } ਨਹੀਂਤਾਂ {
            ਲਈ (size_t i = 0; i < ਉਚਾਈ; i++) {
                ਸਮ੍ਰਿਤੀ_ਨਕਲ((uint8_t *)dst->ptr + ਵਿੱਥ + i * dpitch, (ਅਚਲ uint8_t *) src + i * spitch, ਚੌੜਾਈ);
            }
        }
    } ਨਹੀਂਤਾਂ {
        std::lock_guard<std::recursive_mutex> guard(dst->device->mutex);

        vk_context subctx = ggml_vk_create_temporary_context(dst->device->transfer_ਕਤਾਰ->cmd_pool);
        ggml_vk_ctx_begin(dst->device, subctx);
        ਤਾਰਕਿਕ ret = ggml_vk_ਬਫਰ_write_2d_async(subctx, dst, ਵਿੱਥ, src, spitch, dpitch, ਚੌੜਾਈ, ਉਚਾਈ, ਸੱਚ);
        GGML_ASSERT(ret);
        ggml_vk_ctx_ਸਿਰਾ(subctx);

        ਲਈ (ਸਵੈਚਾਲਿਤ& cpy : subctx->ਵਿੱਚ_memcpys) {
            ਸਮ੍ਰਿਤੀ_ਨਕਲ(cpy.dst, cpy.src, cpy.n);
        }

        ਲਈ (ਸਵੈਚਾਲਿਤ& mset : subctx->memsets) {
            memset(mset.dst, mset.val, mset.n);
        }

        ggml_vk_submit(subctx, dst->device->fence);
        VK_CHECK(dst->device->device.waitForFences({ dst->device->fence }, ਸੱਚ, UINT64_MAX), "vk_buffer_write_2d waitForFences", dst->device);
        dst->device->device.resetFences({ dst->device->fence });
        ggml_vk_ਕਤਾਰ_command_pools_cleanup(dst->device);
    }
}

ਖਾਲੀ ggml_vk_ਬਫਰ_write(vk_ਬਫਰ& dst, size_t ਵਿੱਥ, ਅਚਲ ਖਾਲੀ * src, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_write(" << size << ")");
    ggml_vk_ਬਫਰ_write_2d(dst, ਵਿੱਥ, src, size, size, size, 1);
}

ਤਾਰਕਿਕ ggml_vk_ਬਫਰ_read_2d_async(vk_context subctx, vk_ਬਫਰ& src, size_t ਵਿੱਥ, ਖਾਲੀ * dst, size_t spitch, size_t dpitch, size_t ਚੌੜਾਈ, size_t ਉਚਾਈ, ਤਾਰਕਿਕ sync_staging) {
    VK_LOG_DEBUG("ggml_vk_buffer_read_2d_async(offset=" << ਵਿੱਥ << ", width=" << ਚੌੜਾਈ << ", height=" << ਉਚਾਈ << ")");
    GGML_ASSERT(ਚੌੜਾਈ > 0);
    GGML_ASSERT(ਉਚਾਈ > 0);
    GGML_ASSERT(src != ਖਾਲੀ_ਸੂਚਕ);

    // TODO: staging_offset is not used

    // Check if dst is pinned memory
    vk_ਬਫਰ buf = ਖਾਲੀ_ਸੂਚਕ;
    size_t buf_ਵਿੱਥ = 0;
    ggml_vk_host_get(src->device, dst, buf, buf_ਵਿੱਥ);

    std::vector<vk::ਬਫਰ_Copy> slices(1);
    ਜੇ (ਚੌੜਾਈ == spitch && ਚੌੜਾਈ == dpitch) {
        // Only do single write if stride is equal
        slices[0].src_ਵਿੱਥ_2 = ਵਿੱਥ;
        slices[0].dst_ਵਿੱਥ_2 = buf_ਵਿੱਥ;
        slices[0].size = ਚੌੜਾਈ * ਉਚਾਈ;
    } ਨਹੀਂਤਾਂ {
        slices.resize(ਉਚਾਈ);
        ਲਈ (size_t i = 0; i < ਉਚਾਈ; i++) {
            slices[i].src_ਵਿੱਥ_2 = ਵਿੱਥ + i * spitch;
            slices[i].dst_ਵਿੱਥ_2 = buf_ਵਿੱਥ + i * dpitch;
            slices[i].size = ਚੌੜਾਈ;
        }
    }

    ਜੇ (buf != ਖਾਲੀ_ਸੂਚਕ) {
        // Memory is pinned, use as staging buffer
        ggml_vk_sync_buffers(ਖਾਲੀ_ਸੂਚਕ, subctx);
        subctx->s->ਬਫਰ->buf.copy_ਬਫਰ(src->ਬਫਰ, buf->ਬਫਰ, slices);

        ਵਾਪਸ ਸੱਚ;
    }
    VK_LOG_DEBUG("STAGING");

    ਜੇ (!sync_staging) {
        // copy was not handled caller needs to fall back
        ਵਾਪਸ ਝੂਠ;
    }

    // Fall back to staging buffer
    ਅਚਲ size_t staging_ਮਾਪ = ਚੌੜਾਈ * ਉਚਾਈ;
    ggml_vk_ensure_sync_staging_ਬਫਰ(src->device, staging_ਮਾਪ);

    vk_ਬਫਰ& staging_ਬਫਰ = src->device->sync_staging;

    std::vector<vk::ਬਫਰ_Copy> staging_slices(1);
    ਜੇ (ਚੌੜਾਈ == spitch) {
        staging_slices[0].src_ਵਿੱਥ_2 = ਵਿੱਥ;
        staging_slices[0].dst_ਵਿੱਥ_2 = 0;
        staging_slices[0].size = staging_ਮਾਪ;
    } ਨਹੀਂਤਾਂ {
        staging_slices.resize(ਉਚਾਈ);
        ਲਈ (size_t i = 0; i < ਉਚਾਈ; i++) {
            staging_slices[i].src_ਵਿੱਥ_2 = ਵਿੱਥ + i * spitch;
            staging_slices[i].dst_ਵਿੱਥ_2 = i * ਚੌੜਾਈ;
            staging_slices[i].size = ਚੌੜਾਈ;
        }
    }

    ggml_vk_sync_buffers(ਖਾਲੀ_ਸੂਚਕ, subctx);
    subctx->s->ਬਫਰ->buf.copy_ਬਫਰ(src->ਬਫਰ, staging_ਬਫਰ->ਬਫਰ, staging_slices);

    ਜੇ (ਚੌੜਾਈ == dpitch) {
        deferred_memcpy(dst, staging_ਬਫਰ->ptr, staging_ਮਾਪ, &subctx->ਬਾਹਰਲਾ_memcpys);
    } ਨਹੀਂਤਾਂ {
        ਲਈ (size_t i = 0; i < ਉਚਾਈ; i++) {
            deferred_memcpy((uint8_t *) dst + i * dpitch, (ਅਚਲ uint8_t *) staging_ਬਫਰ->ptr + i * ਚੌੜਾਈ, ਚੌੜਾਈ, &subctx->ਬਾਹਰਲਾ_memcpys);
        }
    }
    ਵਾਪਸ ਸੱਚ;
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_vk_ਬਫਰ_read_async(vk_context subctx, vk_ਬਫਰ& src, size_t ਵਿੱਥ, ਖਾਲੀ * dst, size_t size, ਤਾਰਕਿਕ sync_staging = ਝੂਠ) {
    ਵਾਪਸ ggml_vk_ਬਫਰ_read_2d_async(subctx, src, ਵਿੱਥ, dst, size, size, size, 1, sync_staging);
}

ਖਾਲੀ ggml_vk_ਬਫਰ_read_2d(vk_ਬਫਰ& src, size_t ਵਿੱਥ, ਖਾਲੀ * dst, size_t spitch, size_t dpitch, size_t ਚੌੜਾਈ, size_t ਉਚਾਈ) {
    VK_LOG_DEBUG("ggml_vk_buffer_read_2d(" << src->ਬਫਰ << ", " << ਵਿੱਥ << ", " << ਚੌੜਾਈ << ", " << ਉਚਾਈ << ")");

    // If the device is not an UMA device the memory is host-accessible through rebar. While writing
    // through PCIe is sufficient fast reading back data from PCIe is slower than going through
    // the HW device to host copy path.
    // AMD UMA: uncached host-visible memory is write-combined, CPU reads are slow
    ਅਚਲ ਤਾਰਕਿਕ slow_host_read = src->device->vendor_id == VK_VENDOR_ID_AMD &&
                                !(src->ਸਮ੍ਰਿਤੀ_property_flags & vk::ਸਮ੍ਰਿਤੀ_Property_Flag_ਬਿੱਟ::eHostCached);
    ਜੇ(src->ਸਮ੍ਰਿਤੀ_property_flags & vk::ਸਮ੍ਰਿਤੀ_Property_Flag_ਬਿੱਟ::eHostVisible && src->device->uma && !slow_host_read) {
        GGML_ASSERT(src->ਸਮ੍ਰਿਤੀ_property_flags & vk::ਸਮ੍ਰਿਤੀ_Property_Flag_ਬਿੱਟ::eHostCoherent);

        std::lock_guard<std::recursive_mutex> guard(src->device->mutex);
        vk_context subctx = ggml_vk_create_temporary_context(src->device->compute_ਕਤਾਰ->cmd_pool);
        ggml_vk_ctx_begin(src->device, subctx);
        subctx->s->ਬਫਰ->buf.pipelineBarrier(
            vk::Pipeline_Stage_Flag_ਬਿੱਟ::eComputeShader | vk::Pipeline_Stage_Flag_ਬਿੱਟ::eTransfer,
            vk::Pipeline_Stage_Flag_ਬਿੱਟ::eHost,
            {},
            { { vk::Access_Flag_ਬਿੱਟ::eShaderWrite | vk::Access_Flag_ਬਿੱਟ::eTransferWrite,
                vk::Access_Flag_ਬਿੱਟ::eHostRead } },
            {}, {});
        ggml_vk_ctx_ਸਿਰਾ(subctx);
        ggml_vk_submit(subctx, src->device->fence);
        VK_CHECK(src->device->device.waitForFences({ src->device->fence }, ਸੱਚ, UINT64_MAX),
                 "vk_buffer_read_2d uma waitForFences", src->device);
        src->device->device.resetFences({ src->device->fence });
        ggml_vk_ਕਤਾਰ_command_pools_cleanup(src->device);

        ਜੇ (ਚੌੜਾਈ == spitch && ਚੌੜਾਈ == dpitch) {
            ਸਮ੍ਰਿਤੀ_ਨਕਲ(dst, (ਅਚਲ uint8_t *) src->ptr + ਵਿੱਥ, ਚੌੜਾਈ * ਉਚਾਈ);
        } ਨਹੀਂਤਾਂ {
            ਲਈ (size_t i = 0; i < ਉਚਾਈ; i++) {
                ਸਮ੍ਰਿਤੀ_ਨਕਲ((uint8_t *) dst + i * dpitch, (ਅਚਲ uint8_t *) src->ptr + ਵਿੱਥ + i * spitch, ਚੌੜਾਈ);
            }
        }
    } ਨਹੀਂਤਾਂ {
        std::lock_guard<std::recursive_mutex> guard(src->device->mutex);

        vk_context subctx = ggml_vk_create_temporary_context(src->device->transfer_ਕਤਾਰ->cmd_pool);
        ggml_vk_ctx_begin(src->device, subctx);
        ਤਾਰਕਿਕ ret = ggml_vk_ਬਫਰ_read_2d_async(subctx, src, ਵਿੱਥ, dst, spitch, dpitch, ਚੌੜਾਈ, ਉਚਾਈ, ਸੱਚ);
        GGML_ASSERT(ret);
        ggml_vk_ctx_ਸਿਰਾ(subctx);

        ggml_vk_submit(subctx, src->device->fence);
        VK_CHECK(src->device->device.waitForFences({ src->device->fence }, ਸੱਚ, UINT64_MAX), "vk_buffer_read_2d waitForFences", src->device);
        src->device->device.resetFences({ src->device->fence });
        ggml_vk_ਕਤਾਰ_command_pools_cleanup(src->device);

        ਲਈ (ਸਵੈਚਾਲਿਤ& cpy : subctx->ਬਾਹਰਲਾ_memcpys) {
            ਸਮ੍ਰਿਤੀ_ਨਕਲ(cpy.dst, cpy.src, cpy.n);
        }
    }
}

ਖਾਲੀ ggml_vk_ਬਫਰ_read(vk_ਬਫਰ& src, size_t ਵਿੱਥ, ਖਾਲੀ * dst, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_read(" << src->ਬਫਰ << ", " << ਵਿੱਥ << ", " << size << ")");
    ggml_vk_ਬਫਰ_read_2d(src, ਵਿੱਥ, dst, size, size, size, 1);
}

ਖਾਲੀ ggml_vk_ਬਫਰ_copy_async(vk_context& ctx, vk_ਬਫਰ& dst, size_t dst_ਵਿੱਥ, vk_ਬਫਰ& src, size_t src_ਵਿੱਥ, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_copy_async(" << size << ")");
    // Make sure both buffers are on same device
    GGML_ASSERT(src->device == dst->device);

    Vk_ਬਫਰ_Copy bc{ src_ਵਿੱਥ, dst_ਵਿੱਥ, size };

    vk_Cmd_Copy_ਬਫਰ(ctx->s->ਬਫਰ->buf, (Vk_ਬਫਰ)src->ਬਫਰ, (Vk_ਬਫਰ)dst->ਬਫਰ, 1, &bc);
}

ਖਾਲੀ ggml_vk_ਬਫਰ_copy(vk_ਬਫਰ& dst, size_t dst_ਵਿੱਥ, vk_ਬਫਰ& src, size_t src_ਵਿੱਥ, size_t size) {
    ਜੇ (src->device == dst->device) {
        std::lock_guard<std::recursive_mutex> guard(src->device->mutex);
        VK_LOG_DEBUG("ggml_vk_buffer_copy(SINGLE_DEVICE, " << size << ")");
        // Copy within the device
        vk_context subctx = ggml_vk_create_temporary_context(src->device->transfer_ਕਤਾਰ->cmd_pool);
        ggml_vk_ctx_begin(src->device, subctx);
        ggml_vk_ਬਫਰ_copy_async(subctx, dst, dst_ਵਿੱਥ, src, src_ਵਿੱਥ, size);
        ggml_vk_ctx_ਸਿਰਾ(subctx);
        ggml_vk_submit(subctx, src->device->fence);
        VK_CHECK(src->device->device.waitForFences({ src->device->fence }, ਸੱਚ, UINT64_MAX), "vk_buffer_copy waitForFences", src->device);
        src->device->device.resetFences({ src->device->fence });
        ggml_vk_ਕਤਾਰ_command_pools_cleanup(src->device);
    } ਨਹੀਂਤਾਂ {
        VK_LOG_DEBUG("ggml_vk_buffer_copy(MULTI_DEVICE, " << size << ")");
        // Copy device to device
        ggml_vk_ensure_sync_staging_ਬਫਰ(src->device, size);

        // Copy to src staging buffer
        ggml_vk_ਬਫਰ_copy(src->device->sync_staging, 0, src, src_ਵਿੱਥ, size);
        // Copy to dst buffer
        ggml_vk_ਬਫਰ_write(dst, dst_ਵਿੱਥ, src->device->sync_staging->ptr, size);
    }
}

ਖਾਲੀ ggml_vk_ਬਫਰ_memset_async(vk_context& ctx, vk_ਬਫਰ& dst, size_t ਵਿੱਥ, uint32_t c, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_memset_async(" << ਵਿੱਥ << ", " << c << ", " << size << ")");

    ਜੇ (dst->ਸਮ੍ਰਿਤੀ_property_flags & vk::ਸਮ੍ਰਿਤੀ_Property_Flag_ਬਿੱਟ::eHostVisible &&
        dst->device->uma) {
        deferred_memset((uint8_t*)dst->ptr + ਵਿੱਥ, c, size, &ctx->memsets);
        ਵਾਪਸ;
    }

    // Fall back to GPU fillBuffer for non-UMA or non-host-visible buffers
    ctx->s->ਬਫਰ->buf.fill_ਬਫਰ(dst->ਬਫਰ, ਵਿੱਥ, size, c);
}

ਖਾਲੀ ggml_vk_ਬਫਰ_memset(vk_ਬਫਰ& dst, size_t ਵਿੱਥ, uint32_t c, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_memset(" << ਵਿੱਥ << ", " << c << ", " << size << ")");

    ਜੇ (dst->ਸਮ੍ਰਿਤੀ_property_flags & vk::ਸਮ੍ਰਿਤੀ_Property_Flag_ਬਿੱਟ::eHostVisible &&
        dst->device->uma) {
        memset((uint8_t*)dst->ptr + ਵਿੱਥ, c, size);
        ਵਾਪਸ;
    }

    std::lock_guard<std::recursive_mutex> guard(dst->device->mutex);
    vk_context subctx = ggml_vk_create_temporary_context(dst->device->transfer_ਕਤਾਰ->cmd_pool);
    ggml_vk_ctx_begin(dst->device, subctx);
    subctx->s->ਬਫਰ->buf.fill_ਬਫਰ(dst->ਬਫਰ, ਵਿੱਥ, size, c);
    ggml_vk_ctx_ਸਿਰਾ(subctx);

    ggml_vk_submit(subctx, dst->device->fence);
    VK_CHECK(dst->device->device.waitForFences({ dst->device->fence }, ਸੱਚ, UINT64_MAX), "vk_memset waitForFences", dst->device);
    dst->device->device.resetFences({ dst->device->fence });
    ggml_vk_ਕਤਾਰ_command_pools_cleanup(dst->device);
}

ggml_backend_ਬਫਰ_i ggml_backend_vk_ਬਫਰ_interface = {
    /* .free_buffer     = */ ggml_backend_vk_ਬਫਰ_free_ਬਫਰ,
    /* .get_base        = */ ggml_backend_vk_ਬਫਰ_get_ਅਧਾਰ,
    /* .init_tensor     = */ ggml_backend_vk_ਬਫਰ_init_tensor,
    /* .memset_tensor   = */ ggml_backend_vk_ਬਫਰ_memset_tensor,
    /* .set_tensor      = */ ggml_backend_vk_ਬਫਰ_ਸੈੱਟ_tensor,
    /* .get_tensor      = */ ggml_backend_vk_ਬਫਰ_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_vk_ਬਫਰ_ਸੈੱਟ_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_vk_ਬਫਰ_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_vk_ਬਫਰ_cpy_tensor,
    /* .clear           = */ ggml_backend_vk_ਬਫਰ_clear,
    /* .reset           = */ ਨਿਰਾ,
};

vk_ਬਫਰ ggml_vk_ਬਫਰ_from_host_ptr(vk_device & device, ਖਾਲੀ * ptr, size_t size) {
    ਜੇ (!device->external_ਸਮ੍ਰਿਤੀ_host) {
        ਵਾਪਸ {};
    }

    uintptr_t uptr = reinterpret_cast<uintptr_t>(ptr);
    ਜੇ (uptr & (device->min_imported_host_ਸੂਚਕ_alignment - 1)) {
        ਵਾਪਸ {};
    }
    ਜੇ (size & (device->min_imported_host_ਸੂਚਕ_alignment - 1)) {
        ਵਾਪਸ {};
    }

    ਅਚਲ vk::ਸਮ੍ਰਿਤੀ_Property_Flags property_flags = vk::ਸਮ੍ਰਿਤੀ_Property_Flag_ਬਿੱਟ::eHostVisible | vk::ਸਮ੍ਰਿਤੀ_Property_Flag_ਬਿੱਟ::eHostCoherent | vk::ਸਮ੍ਰਿਤੀ_Property_Flag_ਬਿੱਟ::eHostCached;

    vk_ਬਫਰ buf {};
    ਕੋਸ਼ਿਸ਼ {
        buf = ggml_vk_create_ਬਫਰ(device, size, { property_flags }, ptr);
    } ਫੜੋ (vk::SystemError& e) {
        GGML_LOG_WARN("ggml_vulkan: Failed ggml_vk_create_buffer (%s)\n", e.what());
    }

    ਵਾਪਸ buf;
}

