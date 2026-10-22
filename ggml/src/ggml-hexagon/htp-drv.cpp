#अन्तर्भाव <filesystem>
#अन्तर्भाव <set>
#अन्तर्भाव <sstream>
#अन्तर्भाव <string>

#ifdef _WIN32
#   define WIN32_LEAN_AND_MEAN
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#   include <winevt.h>
#अन्यथा
#   include <dlfcn.h>
#   include <unistd.h>
#endif

#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#pragma clang diagnostic ignored "-Wmissing-prototypes"
#pragma clang diagnostic ignored "-Wsign-compare"
#pragma clang diagnostic ignored "-Wlanguage-extension-token"
#pragma clang diagnostic ignored "-Wmicrosoft-enum-value"
#pragma clang diagnostic ignored "-Wnested-anon-types"

#अन्तर्भाव "ggml-impl.h"
#अन्तर्भाव "htp-drv.h"
#अन्तर्भाव "libdl.h"

#अन्तर्भाव <domain.h>

//
// Driver API types
//

प्रकारव्याख्या शून्य * (*rpcmem_alloc_pfn_t)(पूर्णांक heapid, uint32_t flags, पूर्णांक size);
प्रकारव्याख्या शून्य * (*rpcmem_alloc2_pfn_t)(पूर्णांक heapid, uint32_t flags, माप_प्रकार size);
प्रकारव्याख्या शून्य   (*rpcmem_free_pfn_t)(शून्य * po);
प्रकारव्याख्या पूर्णांक    (*rpcmem_to_fd_pfn_t)(शून्य * po);

प्रकारव्याख्या AEEResult (*dspqueue_create_pfn_t)(पूर्णांक                 domain,
                                           uint32_t            flags,
                                           uint32_t            req_queue_size,
                                           uint32_t            resp_queue_size,
                                           dspqueue_callback_t packet_callback,
                                           dspqueue_callback_t error_callback,
                                           शून्य *              callback_context,
                                           dspqueue_t *        queue);
प्रकारव्याख्या AEEResult (*dspqueue_close_pfn_t)(dspqueue_t queue);
प्रकारव्याख्या AEEResult (*dspqueue_export_pfn_t)(dspqueue_t queue, uint64_t *queue_id);
प्रकारव्याख्या AEEResult (*dspqueue_write_pfn_t)(dspqueue_t queue, uint32_t flags,
                                          uint32_t अंक_buffers,
                                          संरचना dspqueue_buffer *buffers,
                                          uint32_t message_length,
                                          स्थिरांक uint8_t *message,
                                          uint32_t timeout_us);
प्रकारव्याख्या AEEResult (*dspqueue_read_pfn_t)(dspqueue_t queue, uint32_t *flags,
                                         uint32_t max_buffers, uint32_t *अंक_buffers,
                                         संरचना dspqueue_buffer *buffers,
                                         uint32_t max_message_length,
                                         uint32_t *message_length, uint8_t *message,
                                         uint32_t timeout_us);
प्रकारव्याख्या AEEResult (*dspqueue_read_noblock_pfn_t)(dspqueue_t queue, uint32_t *flags,
                                                 uint32_t max_buffers, uint32_t *अंक_buffers,
                                                 संरचना dspqueue_buffer *buffers,
                                                 uint32_t max_message_length,
                                                 uint32_t *message_length, uint8_t *message);
प्रकारव्याख्या पूर्णांक (*fastrpc_mmap_pfn_t)(पूर्णांक domain, पूर्णांक fd, शून्य *addr, पूर्णांक offset, माप_प्रकार length, गणना fastrpc_map_flags flags);
प्रकारव्याख्या पूर्णांक (*fastrpc_munmap_pfn_t)(पूर्णांक domain, पूर्णांक fd, शून्य *addr, माप_प्रकार length);

प्रकारव्याख्या पूर्णांक (*remote_handle64_open_pfn_t)(स्थिरांक अक्षर* name, remote_handle64 *ph);
प्रकारव्याख्या पूर्णांक (*remote_handle64_invoke_pfn_t)(remote_handle64 h, uint32_t dwScalars, remote_arg *pra);
प्रकारव्याख्या पूर्णांक (*remote_handle64_close_pfn_t)(remote_handle h);
प्रकारव्याख्या पूर्णांक (*remote_handle_control_pfn_t)(uint32_t req, शून्य* data, uint32_t datalen);
प्रकारव्याख्या पूर्णांक (*remote_handle64_control_pfn_t)(remote_handle64 h, uint32_t req, शून्य* data, uint32_t datalen);
प्रकारव्याख्या पूर्णांक (*remote_session_control_pfn_t)(uint32_t req, शून्य *data, uint32_t datalen);
प्रकारव्याख्या पूर्णांक (*remote_system_request_pfn_t)(system_req_payload * req);

//
// Driver API pfns
//

rpcmem_alloc_pfn_t  rpcmem_alloc_pfn  = nullptr;
rpcmem_alloc2_pfn_t rpcmem_alloc2_pfn = nullptr;
rpcmem_free_pfn_t   rpcmem_free_pfn   = nullptr;
rpcmem_to_fd_pfn_t  rpcmem_to_fd_pfn  = nullptr;

fastrpc_mmap_pfn_t   fastrpc_mmap_pfn   = nullptr;
fastrpc_munmap_pfn_t fastrpc_munmap_pfn = nullptr;

dspqueue_create_pfn_t       dspqueue_create_pfn       = nullptr;
dspqueue_close_pfn_t        dspqueue_close_pfn        = nullptr;
dspqueue_export_pfn_t       dspqueue_export_pfn       = nullptr;
dspqueue_write_pfn_t        dspqueue_write_pfn        = nullptr;
dspqueue_read_pfn_t         dspqueue_read_pfn         = nullptr;
dspqueue_read_noblock_pfn_t dspqueue_read_noblock_pfn = nullptr;

remote_handle64_open_pfn_t    remote_handle64_open_pfn    = nullptr;
remote_handle64_invoke_pfn_t  remote_handle64_invoke_pfn  = nullptr;
remote_handle64_close_pfn_t   remote_handle64_close_pfn   = nullptr;
remote_handle_control_pfn_t   remote_handle_control_pfn   = nullptr;
remote_handle64_control_pfn_t remote_handle64_control_pfn = nullptr;
remote_session_control_pfn_t  remote_session_control_pfn  = nullptr;
remote_system_request_pfn_t   remote_system_request_pfn   = nullptr;

//
// Driver API
//

शून्य * rpcmem_alloc(पूर्णांक heapid, uint32_t flags, पूर्णांक size) {
    परत rpcmem_alloc_pfn(heapid, flags, size);
}

शून्य * rpcmem_alloc2(पूर्णांक heapid, uint32_t flags, माप_प्रकार size) {
    जर (rpcmem_alloc2_pfn) {
        परत rpcmem_alloc2_pfn(heapid, flags, size);
    } अन्यथा {
        GGML_LOG_INFO("ggml-hex: rpcmem_alloc2 not found, falling back to rpcmem_alloc\n");
        परत rpcmem_alloc_pfn(heapid, flags, size);
    }
}

शून्य rpcmem_free(शून्य * po) {
    परत rpcmem_free_pfn(po);
}

पूर्णांक rpcmem_to_fd(शून्य * po) {
    परत rpcmem_to_fd_pfn(po);
}

HTPDRV_API पूर्णांक fastrpc_mmap(पूर्णांक domain, पूर्णांक fd, शून्य * addr, पूर्णांक offset, माप_प्रकार length, गणना fastrpc_map_flags flags) {
    परत fastrpc_mmap_pfn(domain, fd, addr, offset, length, flags);
}

HTPDRV_API पूर्णांक fastrpc_munmap(पूर्णांक domain, पूर्णांक fd, शून्य * addr, माप_प्रकार length) {
    परत fastrpc_munmap_pfn(domain, fd, addr, length);
}

AEEResult dspqueue_create(पूर्णांक                 domain,
                          uint32_t            flags,
                          uint32_t            req_queue_size,
                          uint32_t            resp_queue_size,
                          dspqueue_callback_t packet_callback,
                          dspqueue_callback_t error_callback,
                          शून्य *              callback_context,
                          dspqueue_t *        queue) {
    परत dspqueue_create_pfn(domain, flags, req_queue_size, resp_queue_size, packet_callback, error_callback,
                               callback_context, queue);
}

AEEResult dspqueue_close(dspqueue_t queue) {
    परत dspqueue_close_pfn(queue);
}

AEEResult dspqueue_export(dspqueue_t queue, uint64_t * queue_id) {
    परत dspqueue_export_pfn(queue, queue_id);
}

AEEResult dspqueue_write(dspqueue_t               queue,
                         uint32_t                 flags,
                         uint32_t                 अंक_buffers,
                         संरचना dspqueue_buffer * buffers,
                         uint32_t                 message_length,
                         स्थिरांक uint8_t *          message,
                         uint32_t                 timeout_us) {
    परत dspqueue_write_pfn(queue, flags, अंक_buffers, buffers, message_length, message, timeout_us);
}

AEEResult dspqueue_read(dspqueue_t               queue,
                        uint32_t *               flags,
                        uint32_t                 max_buffers,
                        uint32_t *               अंक_buffers,
                        संरचना dspqueue_buffer * buffers,
                        uint32_t                 max_message_length,
                        uint32_t *               message_length,
                        uint8_t *                message,
                        uint32_t                 timeout_us) {
#ifdef _WIN32
    जर (timeout_us == 0) {
        परत dspqueue_read_noblock_pfn(queue, flags, max_buffers, अंक_buffers, buffers, max_message_length,
                                     message_length, message);
    }
#endif
    परत dspqueue_read_pfn(queue, flags, max_buffers, अंक_buffers, buffers, max_message_length, message_length,
                             message, timeout_us);
}

HTPDRV_API पूर्णांक remote_handle64_open(स्थिरांक अक्षर * name, remote_handle64 * ph) {
    परत remote_handle64_open_pfn(name, ph);
}

HTPDRV_API पूर्णांक remote_handle64_invoke(remote_handle64 h, uint32_t dwScalars, remote_arg * pra) {
    परत remote_handle64_invoke_pfn(h, dwScalars, pra);
}

HTPDRV_API पूर्णांक remote_handle64_close(remote_handle64 h) {
    परत remote_handle64_close_pfn(h);
}

HTPDRV_API पूर्णांक remote_handle_control(uint32_t req, शून्य * data, uint32_t datalen) {
    परत remote_handle_control_pfn(req, data, datalen);
}

HTPDRV_API पूर्णांक remote_handle64_control(remote_handle64 h, uint32_t req, शून्य * data, uint32_t datalen) {
    परत remote_handle64_control_pfn(h, req, data, datalen);
}

HTPDRV_API पूर्णांक remote_session_control(uint32_t req, शून्य * data, uint32_t datalen) {
    परत remote_session_control_pfn(req, data, datalen);
}

HTPDRV_API पूर्णांक remote_system_request(system_req_payload * req) {
    जर (!remote_system_request_pfn) {
        परत AEE_EUNSUPPORTEDAPI;
    }
    परत remote_system_request_pfn(req);
}

#ifdef _WIN32

स्थिर std::string wstr_to_str(std::wstring_view wstr) {
    std::string result;
    जर (wstr.empty()) {
        परत result;
    }
    स्वतः bytes_needed = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS,
                                            wstr.data(), (पूर्णांक) wstr.size(),
                                            nullptr, 0, nullptr, nullptr);
    जर (bytes_needed == 0) {
        GGML_LOG_ERROR("ggml-hex: WideCharToMultiByte failed. Error %lu\n", GetLastError());
        फेका std::runtime_error("Invalid wstring input");
    }

    result.resize(bytes_needed, '\0');
    पूर्णांक bytes_written = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS,
                                            wstr.data(), (पूर्णांक) wstr.size(),
                                            result.data(), bytes_needed,
                                            nullptr, nullptr);
    जर (bytes_written == 0) {
        GGML_LOG_ERROR("ggml-hex: WideCharToMultiByte failed. Error %lu\n", GetLastError());
        फेका std::runtime_error("Wstring conversion failed");
    }
    परत result;
}

स्थिर std::string get_driver_path() {
    std::wstring serviceName = L"qcnspmcdm";
    std::string result;

    // Get a handle to the SCM database.
    SC_HANDLE schSCManager = OpenSCManagerW(रिक्त, रिक्त, STANDARD_RIGHTS_READ);
    जर (nullptr == schSCManager) {
        GGML_LOG_ERROR("ggml-hex: Failed to open SCManager. Error: %lu\n", GetLastError());
        परत result;
    }

    // Get a handle to the service.
    SC_HANDLE schService = OpenServiceW(schSCManager,           // SCM database
                                        serviceName.c_str(),    // name of service
                                        SERVICE_QUERY_CONFIG);  // need query config access

    जर (nullptr == schService) {
        GGML_LOG_ERROR("ggml-hex: Failed to open qcnspmcdm service. Error: %lu\n", GetLastError());
        CloseServiceHandle(schSCManager);
        परत result;
    }

    // Store the size of buffer used as an output.
    DWORD bufferSize;
    जर (!QueryServiceConfigW(schService, रिक्त, 0, &bufferSize) &&
        (GetLastError() != ERROR_INSUFFICIENT_BUFFER)) {
        GGML_LOG_ERROR("ggml-hex: Failed to query service config. Error: %lu\n", GetLastError());
        CloseServiceHandle(schService);
        CloseServiceHandle(schSCManager);
        परत result;
    }
    // Get the configuration of the service.
    LPQUERY_SERVICE_CONFIGW serviceConfig =
        static_cast<LPQUERY_SERVICE_CONFIGW>(LocalAlloc(LMEM_FIXED, bufferSize));
    जर (!QueryServiceConfigW(schService, serviceConfig, bufferSize, &bufferSize)) {
        ख_लिखो(मानक_त्रुटि, "ggml-hex: Failed to query service config. Error: %lu\n", GetLastError());
        LocalFree(serviceConfig);
        CloseServiceHandle(schService);
        CloseServiceHandle(schSCManager);
        परत result;
    }

    // Read the driver file path get its parent directory
    std::wstring driverPath = std::wstring(serviceConfig->lp_द्विमान_Path_Name);
    driverPath = driverPath.substr(0, driverPath.find_last_of(L"\\"));

    // Clean up resources
    LocalFree(serviceConfig);
    CloseServiceHandle(schService);
    CloseServiceHandle(schSCManager);

    // Driver path would contain invalid path string, like:
    // \SystemRoot\System32\DriverStore\FileRepository\qcadsprpc8280.inf_arm64_c2b9460c9a072f37
    // "\SystemRoot" should be replace with a correct one (e.g. C:\Windows)
    स्थिरांक std::wstring system_मूळ_Placeholder = L"\\SystemRoot";
    जर (0 != driverPath.compare(0, system_मूळ_Placeholder.length(), system_मूळ_Placeholder)) {
        GGML_LOG_ERROR("ggml-hex: String pattern not found in driver path.\n");
        परत result;
    }

    // Replace \SystemRoot with an absolute path from system ENV windir
    स्थिरांक std::wstring system_मूळ_Env = L"windir";

    // Query the number of wide characters this variable requires
    DWORD अंक_शब्द = GetEnvironmentVariableW(system_मूळ_Env.c_str(), रिक्त, 0);
    जर (अंक_शब्द == 0) {
        GGML_LOG_ERROR("ggml-hex: Failed get systemRoot environment variable\n");
        परत result;
    }

    // Query the actual system root name from environment variable
    std::vector<ब_अक्षर_प्रकार> system_मूळ(अंक_शब्द + 1);
    अंक_शब्द = GetEnvironmentVariableW(system_मूळ_Env.c_str(), system_मूळ.data(), अंक_शब्द + 1);
    जर (अंक_शब्द == 0) {
        GGML_LOG_ERROR("ggml-hex: Failed to read windir environment variable\n");
        परत result;
    }
    driverPath.replace(0, system_मूळ_Placeholder.length(), std::wstring(system_मूळ.data()));

    परत wstr_to_str(driverPath);
}

#endif

using dl_handle_ptr = std::unique_ptr<dl_handle, dl_handle_deleter>;

पूर्णांक htpdrv_init() {
    स्थिर dl_handle_ptr lib_cdsp_rpc_handle = nullptr;
    स्थिर तार्किक initialized = असत्य;
#ifdef _WIN32
    std::string drv_path = get_driver_path() + "\\" + "libcdsprpc.dll";
#अन्यथा
    std::string drv_path = "libcdsprpc.so";
#endif
    जर (initialized) {
        GGML_LOG_INFO("ggml-hex: Driver already loaded\n");
        परत AEE_SUCCESS;
    }
    GGML_LOG_INFO("ggml-hex: Loading driver %s\n", drv_path.c_str());

    fs::path path{ drv_path.c_str() };
    dl_handle_ptr handle { dl_load_library(path) };
    जर (!handle) {
        GGML_LOG_ERROR("ggml-hex: failed to load %s: %s\n", path.u8string().c_str(), dl_error());
        परत AEE_EUNABLETOLOAD;
    }

#define dlsym(drv, type, pfn, चिन्ह, ignore)                               \
    करा {                                                                    \
        pfn = (type) dl_get_sym(drv, #symbol);                              \
        जर (!ignore && nullptr == pfn) {                                    \
            GGML_LOG_ERROR("ggml-hex: failed to dlsym %s\n", #symbol);      \
            परत AEE_EUNABLETOLOAD;                                       \
        }                                                                   \
    } पर्यंत (0)

    dlsym(handle.get(), rpcmem_alloc_pfn_t, rpcmem_alloc_pfn, rpcmem_alloc, असत्य);
    dlsym(handle.get(), rpcmem_alloc2_pfn_t, rpcmem_alloc2_pfn, rpcmem_alloc2, सत्य);
    dlsym(handle.get(), rpcmem_free_pfn_t, rpcmem_free_pfn, rpcmem_free, असत्य);
    dlsym(handle.get(), rpcmem_to_fd_pfn_t, rpcmem_to_fd_pfn, rpcmem_to_fd, असत्य);
    dlsym(handle.get(), fastrpc_mmap_pfn_t, fastrpc_mmap_pfn, fastrpc_mmap, असत्य);
    dlsym(handle.get(), fastrpc_munmap_pfn_t, fastrpc_munmap_pfn, fastrpc_munmap, असत्य);
    dlsym(handle.get(), dspqueue_create_pfn_t, dspqueue_create_pfn, dspqueue_create, असत्य);
    dlsym(handle.get(), dspqueue_close_pfn_t, dspqueue_close_pfn, dspqueue_close, असत्य);
    dlsym(handle.get(), dspqueue_export_pfn_t, dspqueue_export_pfn, dspqueue_export, असत्य);
    dlsym(handle.get(), dspqueue_write_pfn_t, dspqueue_write_pfn, dspqueue_write, असत्य);
    dlsym(handle.get(), dspqueue_read_pfn_t, dspqueue_read_pfn, dspqueue_read, असत्य);
    dlsym(handle.get(), dspqueue_read_noblock_pfn_t, dspqueue_read_noblock_pfn, dspqueue_read_noblock, असत्य);
    dlsym(handle.get(), remote_handle64_open_pfn_t, remote_handle64_open_pfn, remote_handle64_open, असत्य);
    dlsym(handle.get(), remote_handle64_invoke_pfn_t, remote_handle64_invoke_pfn, remote_handle64_invoke, असत्य);
    dlsym(handle.get(), remote_handle_control_pfn_t, remote_handle_control_pfn, remote_handle_control, असत्य);
    dlsym(handle.get(), remote_handle64_control_pfn_t, remote_handle64_control_pfn, remote_handle64_control, असत्य);
    dlsym(handle.get(), remote_session_control_pfn_t, remote_session_control_pfn, remote_session_control, असत्य);
    dlsym(handle.get(), remote_handle64_close_pfn_t, remote_handle64_close_pfn, remote_handle64_close, असत्य);
    dlsym(handle.get(), remote_system_request_pfn_t, remote_system_request_pfn, remote_system_request, सत्य);

    lib_cdsp_rpc_handle = std::move(handle);
    initialized         = सत्य;

    परत AEE_SUCCESS;
}

domain * htpdrv_get_domain(पूर्णांक domain_id) {
    पूर्णांक i    = 0;
    पूर्णांक size = आकार(supported_domains) / आकार(domain);

    साठी (i = 0; i < size; i++) {
        जर (supported_domains[i].id == domain_id) {
            परत &supported_domains[i];
        }
    }

    परत रिक्त;
}

पूर्णांक htpdrv_get_arch(पूर्णांक domain, पूर्णांक * arch) {
    जर (!remote_handle_control_pfn) {
        GGML_LOG_ERROR("ggml-hex: remote_handle_control is not supported on this device\n");
        परत AEE_EUNSUPPORTEDAPI;
    }

    संरचना remote_dsp_capability arch_ver;
    arch_ver.domain       = (uint32_t) domain;
    arch_ver.attribute_ID = ARCH_VER;
    arch_ver.capability   = (uint32_t) 0;

    पूर्णांक err = remote_handle_control(DSPRPC_GET_DSP_INFO, &arch_ver, आकार(arch_ver));
    जर ((err & 0xff) == (AEE_EUNSUPPORTEDAPI & 0xff)) {
        GGML_LOG_ERROR("ggml-hex: FastRPC capability API is not supported on this device\n");
        परत AEE_EUNSUPPORTEDAPI;
    }

    जर (err != AEE_SUCCESS) {
        GGML_LOG_ERROR("ggml-hex: FastRPC capability query failed (err %d)\n", err);
        परत err;
    }

    uint32_t val = arch_ver.capability & 0xff;
    *arch = (पूर्णांक) ((val >> 4) * 10 + (val & 0x0f));
    परत 0;
}
