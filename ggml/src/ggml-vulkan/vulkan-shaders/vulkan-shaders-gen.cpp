#ضمن <iostream>
#ضمن <fstream>
#ضمن <sstream>
#ضمن <string>
#ضمن <stdexcept>
#ضمن <array>
#ضمن <vector>
#ضمن <map>
#ضمن <thread>
#ضمن <mutex>
#ضمن <future>
#ضمن <queue>
#ضمن <condition_variable>
#ضمن <atomic>
#ضمن <cstdio>
#ضمن <cstring>
#ضمن <cstdlib>
#ضمن <cassert>
#ضمن <algorithm>
#ضمن <sys/stat.h>
#ضمن <sys/types.h>
#ضمن <filesystem>

#که_تعریف_شوی _WIN32
    #تعریف_کړه NOMINMAX
    #ضمن <windows.h>
    #ضمن <direct.h> // For _mkdir on Windows
#وإلا
    #ضمن <unistd.h>
    #ضمن <sys/wait.h>
    #ضمن <fcntl.h>
#د_که_پای

#تعریف_کړه ASYNCIO_CONCURRENCY 64

std::mutex قلف;
std::vector<std::pair<std::string, std::string>> shader_fnames;
// Set when any shader subprocess fails (non-zero exit / stderr / launch failure) so the
// build is stopped instead of silently producing a broken libggml-vulkan. (issue #24393)
ساكن std::atomic<منطقي> compile_failed{خطأ};
std::locale c_locale("C");

std::string GLSLC = "glslc";
std::string ننوت_filepath = "";
std::string output_dir = "/tmp";
std::string موخه_hpp = "";
std::string موخه_cpp = "";

ثابت std::vector<std::string> type_نومونه = {
    "f32",
    "f16",
    "q1_0",
    "q2_0",
    "q4_0",
    "q4_1",
    "q5_0",
    "q5_1",
    "q8_0",
    "q2_k",
    "q3_k",
    "q4_k",
    "q5_k",
    "q6_k",
    "iq1_s",
    "iq1_m",
    "iq2_xxs",
    "iq2_xs",
    "iq2_s",
    "iq3_xxs",
    "iq3_s",
    "iq4_xs",
    "iq4_nl",
    "mxfp4",
    "nvfp4",
    "tq1_0",
    "tq2_0",
    "bf16",
};

تعداد MatMulIdType {
    NONE,
    DEFAULT,
    SUBGROUP,
};

نطاق {

صحيح execute_command(std::vector<std::string>& command, std::string& stdout_str, std::string& stderr_str) {
#که_تعریف_شوی _WIN32
    HANDLE stdout_read, stdout_write;
    HANDLE stderr_read, stderr_write;
    SECURITY_ځانګړتیاوې sa = { حجم(SECURITY_ځانګړتیاوې), عدم, TRUE };

    إذا (!CreatePipe(&stdout_read, &stdout_write, &sa, 0) ||
        !سېټ_Handle_Information(stdout_read, HANDLE_FLAG_INHERIT, 0)) {
        ارم std::runtime_error("Failed to create stdout pipe");
    }

    إذا (!CreatePipe(&stderr_read, &stderr_write, &sa, 0) ||
        !سېټ_Handle_Information(stderr_read, HANDLE_FLAG_INHERIT, 0)) {
        ارم std::runtime_error("Failed to create stderr pipe");
    }

    PROCESS_INFORMATION pi;
    STARTUPINFOA si = {};
    si.cb = حجم(STARTUPINFOA);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = stdout_write;
    si.hStdError = stderr_write;

    std::string cmd;
    لكل (ثابت خپلکاری& part : command) {
        cmd += part + " ";
    }

    إذا (!CreateProcessA(عدم, cmd.معلومات(), عدم, عدم, TRUE, 0, عدم, عدم, &si, &pi)) {
        ارم std::runtime_error("Failed to create process");
    }

    CloseHandle(stdout_write);
    CloseHandle(stderr_write);

    std::array<حرف, 128> بفر;
    DWORD بایټونه_read;

    طالما (ReadFile(stdout_read, بفر.معلومات(), (DWORD)بفر.size(), &بایټونه_read, عدم) && بایټونه_read > 0) {
        stdout_str.append(بفر.معلومات(), بایټونه_read);
    }

    طالما (ReadFile(stderr_read, بفر.معلومات(), (DWORD)بفر.size(), &بایټونه_read, عدم) && بایټونه_read > 0) {
        stderr_str.append(بفر.معلومات(), بایټونه_read);
    }

    CloseHandle(stdout_read);
    CloseHandle(stderr_read);
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD exit_کوډ = 1;
    Get_Exit_کوډ_Process(pi.hProcess, &exit_کوډ);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    أرجع (صحيح)exit_کوډ;
#وإلا
    صحيح stdout_pipe[2];
    صحيح stderr_pipe[2];

    إذا (أنبوب(stdout_pipe) != 0 || أنبوب(stderr_pipe) != 0) {
        ارم std::runtime_error("Failed to create pipes");
    }

    pid_t pid = fork();
    إذا (pid < 0) {
        std::cerr << strerror(errno) << "\n";
        ارم std::runtime_error("Failed to fork process");
    }

    std::vector<حرف*> argv;
    لكل (std::string& part : command) {
        argv.push_back(part.معلومات());
    }
    argv.push_back(تش_اشاره);

    إذا (pid == 0) {
        close(stdout_pipe[0]);
        close(stderr_pipe[0]);
        dup2(stdout_pipe[1], STDOUT_FILENO);
        dup2(stderr_pipe[1], STDERR_FILENO);
        close(stdout_pipe[1]);
        close(stderr_pipe[1]);
        execvp(argv[0], argv.معلومات());
        _exit(EXIT_FAILURE);
    } وإلا {
        close(stdout_pipe[1]);
        close(stderr_pipe[1]);

        std::array<حرف, 128> بفر;
        ssize_t بایټونه_read;

        طالما ((بایټونه_read = اقرأ(stdout_pipe[0], بفر.معلومات(), بفر.size())) > 0) {
            stdout_str.append(بفر.معلومات(), بایټونه_read);
        }

        طالما ((بایټونه_read = اقرأ(stderr_pipe[0], بفر.معلومات(), بفر.size())) > 0) {
            stderr_str.append(بفر.معلومات(), بایټونه_read);
        }

        close(stdout_pipe[0]);
        close(stderr_pipe[0]);
        صحيح حالت = 0;
        waitpid(pid, &حالت, 0);
        أرجع WIFEXITED(حالت) ? WEXITSTATUS(حالت) : -1;
    }
#د_که_پای
}

منطقي directory_exists(ثابت std::string& path) {
    بنية stat info;
    إذا (stat(path.c_str(), &info) != 0) {
        أرجع خطأ; // Path doesn't exist or can't be accessed
    }
    أرجع (info.st_mode & S_IFDIR) != 0; // Check if it is a directory
}

منطقي create_directory(ثابت std::string& path) {
#که_تعریف_شوی _WIN32
    أرجع _mkdir(path.c_str()) == 0 || errno == EEXIST; // EEXIST means the directory already exists
#وإلا
    أرجع mkdir(path.c_str(), 0755) == 0 || errno == EEXIST; // 0755 is the directory permissions
#د_که_پای
}

std::string منزل_uppercase(ثابت std::string& ننوت) {
    std::string پایله = ننوت;
    لكل (حرف& c : پایله) {
        c = std::لوی_توری(c);
    }
    أرجع پایله;
}

منطقي string_starts_with(ثابت std::string& str, ثابت std::string& prefix) {
    إذا (prefix.size() > str.size()) {
        أرجع خطأ;
    }
    أرجع std::equal(prefix.begin(), prefix.end(), str.begin());
}

منطقي string_پایونه_with(ثابت std::string& str, ثابت std::string& suffix) {
    إذا (suffix.size() > str.size()) {
        أرجع خطأ;
    }
    أرجع std::equal(suffix.rbegin(), suffix.rend(), str.rbegin());
}

منطقي is_quantized_type(ثابت std::string& type_name) {
    أرجع type_name != "f32" && type_name != "f16" && type_name != "bf16";
}

منطقي is_legacy_quant(ثابت std::string& type_name) {
    أرجع type_name == "q2_0" || type_name == "q4_0" || type_name == "q4_1" || type_name == "q5_0" || type_name == "q5_1" || type_name == "q8_0";
}

منطقي is_k_quant(ثابت std::string& type_name) {
    أرجع string_پایونه_with(type_name, "_k");
}

منطقي is_iq_quant(ثابت std::string& type_name) {
    أرجع string_starts_with(type_name, "iq");
}

منطقي is_lut_quant(ثابت std::string& type_name) {
    أرجع is_iq_quant(type_name) || type_name == "mxfp4" || type_name == "nvfp4";
}

std::string lut_بارول_vec_a(ثابت std::string& type_name) {
    إذا (type_name == "iq1_s" || type_name == "iq1_m" || type_name == "iq2_xxs" || type_name == "iq2_xs" || type_name == "iq2_s" || type_name == "iq4_xs") {
        أرجع "8";
    }
    أرجع "4";
}

ساكن ثابت حرف لار_separator = '/';

std::string join_paths(ثابت std::string& path1, ثابت std::string& path2) {
    أرجع path1 + لار_separator + path2;
}

std::string basename(ثابت std::string &path) {
    أرجع path.substr(path.find_last_of("/\\") + 1);
}

std::stringstream make_generic_stringstream() {
    std::stringstream ss;
    ss.imbue(c_locale);
    أرجع ss;
}

std::string read_دوګونی_file(ثابت std::string& path, منطقي may_not_exist = خطأ) {
    FILE* f = fopen(path.c_str(), "rb");
    إذا (!f) {
        إذا (!may_not_exist) {
            std::cerr << "Error opening file: " << path << " (" << strerror(errno) << ")\n";
        }
        أرجع {};
    }

    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    fseek(f, 0, SEEK_SET);

    std::string معلومات(size, '\0');
    size_t read_اندازه = fread(معلومات.معلومات(), 1, size, f);
    fclose(f);
    إذا (read_اندازه != size) {
        std::cerr << "Error reading file: " << path << " (" << strerror(errno) << ")\n";
        أرجع {};
    }

    أرجع معلومات;
}

فراغ write_دوګونی_file(ثابت std::string& path, ثابت std::string& content) {
    FILE* f = fopen(path.c_str(), "wb");
    إذا (!f) {
        std::cerr << "Error opening file for writing: " << path << " (" << strerror(errno) << ")\n";
        أرجع;
    }

    size_t write_اندازه = fwrite(content.معلومات(), 1, content.size(), f);
    fclose(f);
    إذا (write_اندازه != content.size()) {
        std::cerr << "Error writing file: " << path << " (" << strerror(errno) << ")\n";
        أرجع;
    }
}

فراغ write_file_if_بدل_شوی(ثابت std::string& path, ثابت std::string& content) {
    std::string existing = read_دوګونی_file(path, ریښتیا);
    إذا (existing != content) {
        write_دوګونی_file(path, content);
    }
}


// variables to track number of compiles in progress
ساكن uint32_t compile_شمېر = 0;
ساكن std::mutex compile_شمېر_mutex;
ساكن std::condition_variable compile_شمېر_cond;
ساكن منطقي generate_dep_file = ریښتیا;

فراغ decrement_compile_شمېر(uint32_t * count) {
    إذا (count) {
        std::lock_guard<std::mutex> guard(compile_شمېر_mutex);
        أكد(compile_شمېر > 0);
        compile_شمېر--;
        compile_شمېر_cond.notify_all();
    }
}

کارول compile_شمېر_guard = std::unique_ptr<uint32_t, decltype(&decrement_compile_شمېر)>;

compile_شمېر_guard acquire_compile_خونه() {
    // wait until fewer than N compiles are in progress.
    // 16 is an arbitrary limit, the goal is to avoid "failed to create pipe" errors.
    uint32_t N = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
    std::unique_lock<std::mutex> guard(compile_شمېر_mutex);
    compile_شمېر_cond.wait(guard, [N] { أرجع compile_شمېر < N; });
    compile_شمېر++;
    أرجع compile_شمېر_guard(&compile_شمېر, &decrement_compile_شمېر);
}

فراغ string_منزل_spv_func(std::string name, std::string دننه_لار, std::string بهر_لار, std::map<std::string, std::string> تعریفوي, منطقي coopmat, منطقي dep_file, compile_شمېر_guard خونه) {
    std::string موخه_env = (name.find("_cm2") != std::string::npos) ? "--target-env=vulkan1.3" : "--target-env=vulkan1.2";

    #که_تعریف_شوی _WIN32
        std::vector<std::string> cmd = {GLSLC, "-fshader-stage=compute", موخه_env, "\"" + دننه_لار + "\"", "-o", "\"" + بهر_لار + "\""};
    #وإلا
        std::vector<std::string> cmd = {GLSLC, "-fshader-stage=compute", موخه_env, دننه_لار, "-o", بهر_لار};
    #د_که_پای

    // disable spirv-opt for coopmat shaders for https://github.com/ggml-org/llama.cpp/issues/10734
    // disable spirv-opt for bf16 shaders for https://github.com/ggml-org/llama.cpp/issues/15344
    // disable spirv-opt for rope shaders for https://github.com/ggml-org/llama.cpp/issues/16860
    // disable spirv-opt for dot2 shaders (spirv-opt doesn't recognize SPV_VALVE_mixed_float_dot_product capability)
    إذا (!coopmat && name.find("bf16") == std::string::npos && name.find("rope") == std::string::npos && name.find("_dot2") == std::string::npos) {
        cmd.push_back("-O");
    }

    إذا (dep_file) {
        cmd.push_back("-MD");
        cmd.push_back("-MF");
#که_تعریف_شوی _WIN32
        cmd.push_back("\"" + موخه_cpp + ".d\"");
#وإلا
        cmd.push_back(موخه_cpp + ".d");
#د_که_پای
    }

    #که_تعریف_شوی GGML_VULKAN_SHADER_DEBUG_INFO
        cmd.push_back("-g");
    #د_که_پای

    لكل (ثابت خپلکاری& define : تعریفوي) {
        cmd.push_back("-D" + define.لومړی + "=" + define.second);
    }

    std::string command;
    لكل (ثابت خپلکاری& part : cmd) {
        command += part + " ";
    }

    std::string stdout_str, stderr_str;
    حاول {
        // std::cout << "Executing command: ";
        // for (const auto& part : cmd) {
        //     std::cout << part << " ";
        // }
        // std::cout << std::endl;

        صحيح exit_کوډ = execute_command(cmd, stdout_str, stderr_str);
        إذا (exit_کوډ != 0 || !stderr_str.تشه()) {
            std::cerr << "cannot compile " << name << " (exit code " << exit_کوډ << ")\n\n";
            لكل (ثابت خپلکاری& part : cmd) {
                std::cerr << part << " ";
            }
            std::cerr << "\n\n" << stderr_str << std::کرښه_پای;
            compile_failed = ریښتیا;
            أرجع;
        }

        إذا (dep_file) {
            // replace .spv output path with the embed .cpp path which is used as output in CMakeLists.txt
            std::string dep = read_دوګونی_file(موخه_cpp + ".d", ریښتیا);
            إذا (!dep.تشه()) {
                size_t pos = dep.find(بهر_لار);
                إذا (pos != std::string::npos) {
                    dep.replace(pos, بهر_لار.اوږدتیا(), موخه_cpp);
                }
                write_دوګونی_file(موخه_cpp + ".d", dep);
            }
        }

        std::lock_guard<std::mutex> guard(قلف);
        shader_fnames.push_back(std::make_pair(name, بهر_لار));
    } امسك (ثابت std::exception& e) {
        std::cerr << "Error executing command for " << name << ": " << e.what() << std::کرښه_پای;
        compile_failed = ریښتیا;
    }
}

std::map<std::string, std::string> یوځای_maps(ثابت std::map<std::string, std::string>& a, ثابت std::map<std::string, std::string>& b) {
    std::map<std::string, std::string> پایله = a;
    پایله.insert(b.begin(), b.end());
    أرجع پایله;
}

ساكن std::deque<std::future<فراغ>> compiles;
فراغ string_منزل_spv(std::string name, ثابت std::string& سرچینه, ثابت std::map<std::string, std::string>& تعریفوي, منطقي fp16 = ریښتیا, منطقي coopmat = خطأ, منطقي coopmat2 = خطأ, منطقي f16acc = خطأ, ثابت std::string& suffix = "") {
    name = name + (f16acc ? "_f16acc" : "") + (coopmat ? "_cm1" : "") + (coopmat2 ? "_cm2" : (fp16 ? "" : "_fp32")) + suffix;
    std::string بهر_لار = join_paths(output_dir, name + ".spv");

    إذا (ننوت_filepath == "") {
        // No input source to compile, only generate header for all shaders
        shader_fnames.push_back(std::pair(name, بهر_لار));
        أرجع;
    } وإلا إذا (basename(ننوت_filepath) != سرچینه) {
        // Only compile shader variants matching the input filename
        أرجع;
    }

    compile_شمېر_guard خونه = acquire_compile_خونه();
    compiles.push_back(std::async(
        string_منزل_spv_func, name, ننوت_filepath, بهر_لار, تعریفوي, coopmat, generate_dep_file, std::move(خونه)));
    // Don't write the same dep file from multiple processes
    generate_dep_file = خطأ;

    // Clean up completed futures - don't accumulate virtual memory for completed threads' stacks.
    طالما (!compiles.تشه() && compiles.front().wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        compiles.pop_front();
    }
}

فراغ matmul_shaders(منطقي fp16, MatMulIdType matmul_id_type, منطقي coopmat, منطقي coopmat2, منطقي f16acc, منطقي dot2 = خطأ) {
    std::string بارول_vec = coopmat2 ? "1" : fp16 ? "8" : "4";
    std::string aligned_b_type_f32 = coopmat2 ? "float" : fp16 ? "mat2x4" : "vec4";
    std::string aligned_b_type_f16 = coopmat2 ? "float16_t" : fp16 ? "f16mat2x4" : "f16vec4";
    std::string dot2_sfx = dot2 ? "_dot2" : "";

    std::map<std::string, std::string> بنسټ_dict;
    std::string shader_name = "matmul";

    إذا (matmul_id_type == MatMulIdType::DEFAULT) {
        بنسټ_dict["MUL_MAT_ID"] = "1";
        shader_name = "matmul_id";
    } وإلا إذا (matmul_id_type == MatMulIdType::SUBGROUP) {
        بنسټ_dict["MUL_MAT_ID"] = "1";
        بنسټ_dict["MUL_MAT_ID_USE_SUBGROUPS"] = "1";
        shader_name = "matmul_id_subgroup";
    }

    إذا (fp16) {
        بنسټ_dict["FLOAT16"] = "1";
    }

    بنسټ_dict["ACC_TYPE"     ] = f16acc ? "float16_t" : "float";
    بنسټ_dict["ACC_TYPEV2"   ] = f16acc ? "f16vec2"   : "vec2";
    بنسټ_dict["ACC_TYPE_VEC4"] = f16acc ? "f16vec4"   : "vec4";
    إذا (f16acc) {
        بنسټ_dict["ACC_TYPE_MAX"] = "float16_t(65504.0)";
        بنسټ_dict["ACC_F16"] = "1";
    }

    إذا (coopmat) {
        بنسټ_dict["COOPMAT"] = "1";
    }
#إذا defined(GGML_VULKAN_COOPMAT2_DECODE_ویکټور_GLSLC_SUPPORT)
    إذا (coopmat2) {
        بنسټ_dict["GGML_VULKAN_COOPMAT2_DECODE_VECTOR"] = "1";
    }
#د_که_پای

    إذا (dot2) {
        بنسټ_dict["DOT2_F16"] = "1";
    }

    ثابت std::string سرچینه_name = coopmat2 ? "mul_mm_cm2.comp" : "mul_mm.comp";

    خپلکاری ثابت &FLOAT_TYPE = [&](صحيح vec, ثابت std::string &t) -> std::string {
        بدل (vec) {
        حالة 1:
            إذا (t == "bf16") {
                // scalar path promotes to float
                إذا (!coopmat && !coopmat2) {
                    أرجع "float";
                }
                أرجع "bfloat16_t";
            }
            إذا (coopmat2 || fp16) {
                أرجع "float16_t";
            }
            أرجع "float";
        حالة 2:
            إذا (t == "bf16") {
                // scalar path promotes to float
                إذا (!coopmat && !coopmat2) {
                    أرجع "vec2";
                }
                أرجع "bf16vec2";
            }
            إذا (coopmat2 || fp16) {
                أرجع "f16vec2";
            }
            أرجع "vec2";
        حالة 4:
            إذا (t == "bf16") {
                // scalar path promotes to float
                إذا (!coopmat && !coopmat2) {
                    أرجع "vec4";
                }
                أرجع "bf16vec4";
            }
            إذا (coopmat2 || fp16) {
                أرجع "f16vec4";
            }
            أرجع "vec4";
        حالة 8:
            إذا (t == "bf16") {
                // scalar path promotes to float
                إذا (!coopmat && !coopmat2) {
                    أرجع "mat2x4";
                }
                ارم std::runtime_error("bf16 vec8 not supported");
            }
            إذا (coopmat2 || fp16) {
                أرجع "f16mat2x4";
            }
            أرجع "mat2x4";
        افتراضي:
            ارم std::runtime_error("invalid vector size");
        }
    };

    ثابت std::map<std::string, std::string> float_type_dict_f16 = {
        {"FLOAT_TYPE",   FLOAT_TYPE(1, "f16")},
        {"FLOAT_TYPEV2", FLOAT_TYPE(2, "f16")},
        {"FLOAT_TYPEV4", FLOAT_TYPE(4, "f16")},
        {"FLOAT_TYPEV8", FLOAT_TYPE(8, "f16")},
    };

    // Shaders with f16 B_TYPE
    string_منزل_spv(shader_name + "_f32_f16" + dot2_sfx, سرچینه_name, یوځای_maps(یوځای_maps(بنسټ_dict, float_type_dict_f16), {{"DATA_A_F32", "1"}, {"LOAD_VEC_A", بارول_vec}, {"LOAD_VEC_B", بارول_vec}, {"B_TYPE", aligned_b_type_f16}, {"B_TYPE_SCALAR", "float16_t"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);

    string_منزل_spv(shader_name + "_f16" + dot2_sfx, سرچینه_name, یوځای_maps(یوځای_maps(بنسټ_dict, float_type_dict_f16), {{"DATA_A_F16", "1"}, {"LOAD_VEC_A", بارول_vec}, {"LOAD_VEC_B", بارول_vec}, {"B_TYPE", aligned_b_type_f16}, {"B_TYPE_SCALAR", "float16_t"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);

    // bf16
    {
        // For aligned matmul loads
        std::string بارول_vec_a = coopmat2 ? "1" : "4";

        // scalar path promotes to float
        std::string منزل_float_type = (coopmat || coopmat2) ? "uintBitsToBFloat16EXT" : "bf16_to_fp32";

        ثابت std::map<std::string, std::string> float_type_dict_bf16 = {
            {"FLOAT_TYPE",   FLOAT_TYPE(1, "bf16")},
            {"FLOAT_TYPEV2", FLOAT_TYPE(2, "bf16")},
            {"FLOAT_TYPEV4", FLOAT_TYPE(4, "bf16")},
        };

        // If bfloat16 is not supported, then only compile the scalar (promote to fp32) shader
#إذا !defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        إذا (!(coopmat || coopmat2))
#د_که_پای
        {
            إذا (!dot2) {
                string_منزل_spv(shader_name + "_bf16", سرچینه_name, یوځای_maps(یوځای_maps(بنسټ_dict, float_type_dict_bf16), {{"TO_FLOAT_TYPE", منزل_float_type}, {"DATA_A_BF16", "1"}, {"LOAD_VEC_A", بارول_vec_a}, {"LOAD_VEC_B", "4"}, {"B_TYPE", coopmat2 ? "bfloat16_t" : "u16vec4"}, {"B_TYPE_SCALAR", coopmat2 ? "bfloat16_t" : "uint16_t"}, {"B_TYPEV4", "bf16vec4"}, {"D_TYPE", "float"}, {"B_IS_FLOAT", "1"}, {"DATA_B_BF16", "1"}}), fp16, coopmat, coopmat2, f16acc);
            }
        }
    }

    لكل (ثابت خپلکاری& tname : type_نومونه) {
        إذا (tname == "bf16") {
            استمر;
        }

        // Float types keep per-type compilation (different accumulation loop structure)
        إذا (tname == "f32" || tname == "f16") {
            std::string معلومات_a_کلی = "DATA_A_" + منزل_uppercase(tname);

            ثابت std::map<std::string, std::string> float_type_dict = {
                {"FLOAT_TYPE",   FLOAT_TYPE(1, tname)},
                {"FLOAT_TYPEV2", FLOAT_TYPE(2, tname)},
                {"FLOAT_TYPEV4", FLOAT_TYPE(4, tname)},
                {"FLOAT_TYPEV8", FLOAT_TYPE(8, tname)},
            };

            إذا (!coopmat2) {
                string_منزل_spv(shader_name + "_" + tname + "_f32" + dot2_sfx, سرچینه_name, یوځای_maps(یوځای_maps(بنسټ_dict, float_type_dict), {{معلومات_a_کلی, "1"}, {"LOAD_VEC_A", بارول_vec}, {"LOAD_VEC_B", بارول_vec}, {"B_TYPE", aligned_b_type_f32}, {"B_TYPE_SCALAR", "float"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);
            }
            استمر;
        }

        std::string معلومات_a_کلی = "DATA_A_" + منزل_uppercase(tname);
        ثابت std::map<std::string, std::string> float_type_dict = {
            {"FLOAT_TYPE",   FLOAT_TYPE(1, tname)},
            {"FLOAT_TYPEV2", FLOAT_TYPE(2, tname)},
            {"FLOAT_TYPEV4", FLOAT_TYPE(4, tname)},
            {"FLOAT_TYPEV8", FLOAT_TYPE(8, tname)},
        };

#إذا defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
        إذا (!f16acc && !coopmat && !coopmat2 && !dot2 && (is_legacy_quant(tname) || is_k_quant(tname) || tname == "mxfp4" || tname == "iq3_s" || tname == "iq4_xs")) {
            string_منزل_spv(shader_name + "_" + tname + "_q8_1", "mul_mmq.comp", یوځای_maps(یوځای_maps(بنسټ_dict, float_type_dict), {{معلومات_a_کلی, "1"}, {"D_TYPE", "float"},}), fp16, coopmat, coopmat2, f16acc);
        }
#د_که_پای

        إذا (!f16acc && coopmat && (tname == "q4_0" || tname == "q4_1" || tname == "q5_0" || tname == "q5_1" || tname == "q8_0" || tname == "iq4_nl" || tname == "iq4_xs" || tname == "mxfp4"
                     || tname == "q3_k" || tname == "q4_k" || tname == "q5_k" || tname == "q6_k" || tname == "nvfp4")) {
            string_منزل_spv(shader_name + "_" + tname + "_q8_1", "mul_mmq_cm1.comp", یوځای_maps(یوځای_maps(بنسټ_dict, float_type_dict), {{معلومات_a_کلی, "1"}, {"D_TYPE", "float"}, {"D_TYPE_VEC4", "vec4"}}), fp16, coopmat, coopmat2, f16acc);
        }

        إذا (is_lut_quant(tname)) {
            std::string lva = lut_بارول_vec_a(tname);

            string_منزل_spv(shader_name + "_" + tname + "_f16" + dot2_sfx, سرچینه_name, یوځای_maps(یوځای_maps(بنسټ_dict, float_type_dict), {{معلومات_a_کلی, "1"}, {"LOAD_VEC_A", lva}, {"LOAD_VEC_B", بارول_vec}, {"B_TYPE", aligned_b_type_f16}, {"B_TYPE_SCALAR", "float16_t"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);

            إذا (!coopmat2) {
                string_منزل_spv(shader_name + "_" + tname + "_f32" + dot2_sfx, سرچینه_name, یوځای_maps(یوځای_maps(بنسټ_dict, float_type_dict), {{معلومات_a_کلی, "1"}, {"LOAD_VEC_A", lva}, {"LOAD_VEC_B", بارول_vec}, {"B_TYPE", aligned_b_type_f32}, {"B_TYPE_SCALAR", "float"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);
            }

#إذا defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
            إذا ((tname == "mxfp4" || tname == "nvfp4") && (coopmat || coopmat2)) {
                string_منزل_spv(shader_name + "_" + tname + "_f16_ocp" + dot2_sfx, سرچینه_name, یوځای_maps(یوځای_maps(بنسټ_dict, float_type_dict), {{معلومات_a_کلی, "1"}, {"USE_OCP_FP4", "1"}, {"LOAD_VEC_A", lva}, {"LOAD_VEC_B", بارول_vec}, {"B_TYPE", aligned_b_type_f16}, {"B_TYPE_SCALAR", "float16_t"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);
                إذا (!coopmat2) {
                    string_منزل_spv(shader_name + "_" + tname + "_f32_ocp" + dot2_sfx, سرچینه_name, یوځای_maps(یوځای_maps(بنسټ_dict, float_type_dict), {{معلومات_a_کلی, "1"}, {"USE_OCP_FP4", "1"}, {"LOAD_VEC_A", lva}, {"LOAD_VEC_B", بارول_vec}, {"B_TYPE", aligned_b_type_f32}, {"B_TYPE_SCALAR", "float"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);
                }
            }
#د_که_پای
            استمر;
        }

        // dedicated shader needed due to regression on Ampere
        إذا (coopmat2 && (tname == "q4_k" || tname == "q5_k")) {
            string_منزل_spv(shader_name + "_" + tname + "_f16" + dot2_sfx, سرچینه_name, یوځای_maps(یوځای_maps(بنسټ_dict, float_type_dict), {{معلومات_a_کلی, "1"}, {"LOAD_VEC_A", بارول_vec}, {"LOAD_VEC_B", بارول_vec}, {"B_TYPE", aligned_b_type_f16}, {"B_TYPE_SCALAR", "float16_t"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);
        }
    }

    // Quant shader: one SPIR-V for all quant types, selected via MmTypeA spec constant
    {
        ثابت std::map<std::string, std::string> quant_float_type_dict = {
            {"FLOAT_TYPE",   FLOAT_TYPE(1, "q4_0")},
            {"FLOAT_TYPEV2", FLOAT_TYPE(2, "q4_0")},
            {"FLOAT_TYPEV4", FLOAT_TYPE(4, "q4_0")},
            {"FLOAT_TYPEV8", FLOAT_TYPE(8, "q4_0")},
        };

        string_منزل_spv(shader_name + "_quant_f16" + dot2_sfx, سرچینه_name, یوځای_maps(یوځای_maps(بنسټ_dict, quant_float_type_dict), {{"MULMAT_QUANT", "1"}, {"LOAD_VEC_B", بارول_vec}, {"B_TYPE", aligned_b_type_f16}, {"B_TYPE_SCALAR", "float16_t"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);

        إذا (!coopmat2) {
            string_منزل_spv(shader_name + "_quant_f32" + dot2_sfx, سرچینه_name, یوځای_maps(یوځای_maps(بنسټ_dict, quant_float_type_dict), {{"MULMAT_QUANT", "1"}, {"LOAD_VEC_B", بارول_vec}, {"B_TYPE", aligned_b_type_f32}, {"B_TYPE_SCALAR", "float"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);
        }
    }
}

فراغ process_shaders() {
    // matmul
    لكل (ثابت MatMulIdType& matmul_id_type : {MatMulIdType::NONE, MatMulIdType::DEFAULT, MatMulIdType::SUBGROUP}) {
        // No coopmats
        // fp32
        matmul_shaders(خطأ, matmul_id_type, خطأ, خطأ, خطأ);

        // fp16, fp32acc and fp16acc
        matmul_shaders(ریښتیا, matmul_id_type, خطأ, خطأ, خطأ);
        matmul_shaders(ریښتیا, matmul_id_type, خطأ, خطأ, ریښتیا);

        // dot2 variants (scalar fp16 only)
        matmul_shaders(ریښتیا, matmul_id_type, خطأ, خطأ, خطأ, ریښتیا);
        matmul_shaders(ریښتیا, matmul_id_type, خطأ, خطأ, ریښتیا,  ریښتیا);

        إذا (matmul_id_type != MatMulIdType::DEFAULT) {
#إذا defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
            // Coopmat, fp32acc and fp16acc
            matmul_shaders(ریښتیا, matmul_id_type, ریښتیا, خطأ, خطأ);
            matmul_shaders(ریښتیا, matmul_id_type, ریښتیا, خطأ, ریښتیا);
#د_که_پای

#إذا defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
            // Coopmat2, fp32acc and fp16acc
            matmul_shaders(ریښتیا, matmul_id_type, خطأ, ریښتیا, خطأ);
            matmul_shaders(ریښتیا, matmul_id_type, خطأ, ریښتیا, ریښتیا);
#د_که_پای
        }
    }

    لكل (ثابت منطقي& fp16 : {خطأ, ریښتیا}) {
        std::map<std::string, std::string> بنسټ_dict;
        إذا (fp16) {
            بنسټ_dict = {{"FLOAT_TYPE", "float16_t"}, {"FLOAT_TYPEV2", "f16vec2"}, {"FLOAT_TYPEV4", "f16vec4"}, {"FLOAT16", "1"}, {"FLOAT_TYPE_MAX", "float16_t(65504.0)"}};
        } وإلا {
            بنسټ_dict = {{"FLOAT_TYPE", "float"}, {"FLOAT_TYPEV2", "vec2"}, {"FLOAT_TYPEV4", "vec4"}};
        }

        // flash attention
        لكل (ثابت منطقي& f16acc : {خطأ, ریښتیا}) {
            std::map<std::string, std::string> fa_بنسټ_dict = بنسټ_dict;
            fa_بنسټ_dict["ACC_TYPE"] = fp16 && f16acc ? "float16_t" : "float";
            fa_بنسټ_dict["ACC_TYPEV2"] = fp16 && f16acc ? "f16vec2" : "vec2";
            fa_بنسټ_dict["ACC_TYPEV4"] = fp16 && f16acc ? "f16vec4" : "vec4";
            // Compile IQ4_NL support into all FA variants so its shared LUT is available when K or V uses it.
            fa_بنسټ_dict["DATA_A_IQ4_NL"] = "1";
            إذا (fp16 && f16acc) {
                fa_بنسټ_dict["ACC_TYPE_MAX"] = "float16_t(65504.0)";
            }

            إذا (fp16) {
#إذا defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
                string_منزل_spv("flash_attn_f32_f16", "flash_attn_cm2.comp",
                    یوځای_maps(fa_بنسټ_dict, {{"Q_TYPE", "float"}, {"D_TYPE", "float"}, {"D_TYPEV4", "vec4"}}), fp16, خطأ, ریښتیا, f16acc);
#د_که_پای

#إذا defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
                string_منزل_spv("flash_attn_f32_f16", "flash_attn_cm1.comp",
                    یوځای_maps(fa_بنسټ_dict, {{"Q_TYPE", "float"}, {"D_TYPE", "float"}, {"D_TYPEV4", "vec4"}, {"COOPMAT", "1"}}), fp16, ریښتیا, خطأ, f16acc);
#د_که_پای
            }

            string_منزل_spv("flash_attn_f32_f16", "flash_attn.comp",
                یوځای_maps(fa_بنسټ_dict, {{"Q_TYPE", "float"}, {"D_TYPE", "float"}, {"D_TYPEV4", "vec4"}}), fp16, خطأ, خطأ, f16acc);

            إذا (fp16) {
                string_منزل_spv("flash_attn_f32_f16_dot2", "flash_attn.comp",
                    یوځای_maps(fa_بنسټ_dict, {{"Q_TYPE", "float"}, {"D_TYPE", "float"}, {"D_TYPEV4", "vec4"}, {"DOT2_F16", "1"}}), fp16, خطأ, خطأ, f16acc);
            }

#إذا defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
            string_منزل_spv("flash_attn_f32_f16", "flash_attn.comp",
                یوځای_maps(fa_بنسټ_dict, {{"Q_TYPE", "float"}, {"D_TYPE", "float"}, {"D_TYPEV4", "vec4"}, {"MMQ", "1"}, {"FA_MMQ_MIXED", "1"}}), fp16, خطأ, خطأ, f16acc, "_int8");
#د_که_پای
        }
    }

    ثابت std::map<std::string, std::string> fa_bf16_dict = {
        {"FLOAT_TYPE",   "bfloat16_t"},
        {"FLOAT_TYPEV2", "bf16vec2"},
        {"FLOAT_TYPEV4", "bf16vec4"},
        {"ACC_TYPE",     "float"},
        {"ACC_TYPEV2",   "vec2"},
        {"ACC_TYPEV4",   "vec4"},
        {"BFLOAT16",     "1"},
    };

#إذا defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    string_منزل_spv("flash_attn_f32_f16_bf16", "flash_attn_cm1.comp",
        یوځای_maps(fa_bf16_dict, {{"Q_TYPE", "float"}, {"D_TYPE", "float"}, {"D_TYPEV4", "vec4"}, {"COOPMAT", "1"}}),
        ریښتیا, ریښتیا, خطأ, خطأ);
#د_که_پای

#إذا defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT) && defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
    string_منزل_spv("flash_attn_f32_f16_bf16", "flash_attn_cm2.comp",
        یوځای_maps(fa_bf16_dict, {{"Q_TYPE", "float"}, {"D_TYPE", "float"}, {"D_TYPEV4", "vec4"}}),
        ریښتیا, خطأ, ریښتیا, خطأ);
#د_که_پای

    std::map<std::string, std::string> بنسټ_dict = {{"FLOAT_TYPE", "float"}, {"FLOAT_TYPEV2", "vec2"}};

    لكل (ثابت خپلکاری& tname : type_نومونه) {
        // mul mat vec
        std::string معلومات_a_کلی = "DATA_A_" + منزل_uppercase(tname);
        std::string shader = (string_پایونه_with(tname, "_k") || string_starts_with(tname, "iq1_") || string_starts_with(tname, "iq2_") || string_starts_with(tname, "iq3_") || tname == "iq4_xs" || tname == "tq2_0" || tname == "tq1_0") ? "mul_mat_vec_" + tname + ".comp" : "mul_mat_vec.comp";

        string_منزل_spv("mul_mat_vec_" + tname + "_f32_f32", shader, یوځای_maps(بنسټ_dict, {{معلومات_a_کلی, "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}}));
        string_منزل_spv("mul_mat_vec_" + tname + "_f16_f32", shader, یوځای_maps(بنسټ_dict, {{معلومات_a_کلی, "1"}, {"B_TYPE", "float16_t"}, {"B_TYPEV2", "f16vec2"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}}));

        string_منزل_spv("mul_mat_vec_" + tname + "_f32_f32_subgroup", shader, یوځای_maps(بنسټ_dict, {{معلومات_a_کلی, "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}}));
        string_منزل_spv("mul_mat_vec_" + tname + "_f16_f32_subgroup", shader, یوځای_maps(بنسټ_dict, {{معلومات_a_کلی, "1"}, {"B_TYPE", "float16_t"}, {"B_TYPEV2", "f16vec2"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}}));

        string_منزل_spv("mul_mat_vec_" + tname + "_f32_f32_subgroup_no_shmem", shader, یوځای_maps(بنسټ_dict, {{معلومات_a_کلی, "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));
        string_منزل_spv("mul_mat_vec_" + tname + "_f16_f32_subgroup_no_shmem", shader, یوځای_maps(بنسټ_dict, {{معلومات_a_کلی, "1"}, {"B_TYPE", "float16_t"}, {"B_TYPEV2", "f16vec2"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));

#إذا defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
        إذا (tname == "mxfp4" || tname == "nvfp4") {
            string_منزل_spv("mul_mat_vec_" + tname + "_f32_f32_ocp", shader, یوځای_maps(بنسټ_dict, {{معلومات_a_کلی, "1"}, {"USE_OCP_FP4", "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}}));
            string_منزل_spv("mul_mat_vec_" + tname + "_f16_f32_ocp", shader, یوځای_maps(بنسټ_dict, {{معلومات_a_کلی, "1"}, {"USE_OCP_FP4", "1"}, {"B_TYPE", "float16_t"}, {"B_TYPEV2", "f16vec2"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}}));
            string_منزل_spv("mul_mat_vec_" + tname + "_f32_f32_ocp_subgroup", shader, یوځای_maps(بنسټ_dict, {{معلومات_a_کلی, "1"}, {"USE_OCP_FP4", "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}}));
            string_منزل_spv("mul_mat_vec_" + tname + "_f16_f32_ocp_subgroup", shader, یوځای_maps(بنسټ_dict, {{معلومات_a_کلی, "1"}, {"USE_OCP_FP4", "1"}, {"B_TYPE", "float16_t"}, {"B_TYPEV2", "f16vec2"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}}));
            string_منزل_spv("mul_mat_vec_" + tname + "_f32_f32_ocp_subgroup_no_shmem", shader, یوځای_maps(بنسټ_dict, {{معلومات_a_کلی, "1"}, {"USE_OCP_FP4", "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));
            string_منزل_spv("mul_mat_vec_" + tname + "_f16_f32_ocp_subgroup_no_shmem", shader, یوځای_maps(بنسټ_dict, {{معلومات_a_کلی, "1"}, {"USE_OCP_FP4", "1"}, {"B_TYPE", "float16_t"}, {"B_TYPEV2", "f16vec2"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));
            string_منزل_spv("mul_mat_vec_id_" + tname + "_f32_f32_ocp", shader, یوځای_maps(بنسټ_dict, {{"MUL_MAT_ID", "1"}, {معلومات_a_کلی, "1"}, {"USE_OCP_FP4", "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}}));
            string_منزل_spv("mul_mat_vec_id_" + tname + "_f32_f32_ocp_subgroup", shader, یوځای_maps(بنسټ_dict, {{"MUL_MAT_ID", "1"}, {معلومات_a_کلی, "1"}, {"USE_OCP_FP4", "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}}));
            string_منزل_spv("mul_mat_vec_id_" + tname + "_f32_f32_ocp_subgroup_no_shmem", shader, یوځای_maps(بنسټ_dict, {{"MUL_MAT_ID", "1"}, {معلومات_a_کلی, "1"}, {"USE_OCP_FP4", "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));
        }
#د_که_پای

        string_منزل_spv("mul_mat_vec_id_" + tname + "_f32_f32", shader, یوځای_maps(بنسټ_dict, {{"MUL_MAT_ID", "1"}, {معلومات_a_کلی, "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}}));
        string_منزل_spv("mul_mat_vec_id_" + tname + "_f32_f32_subgroup", shader, یوځای_maps(بنسټ_dict, {{"MUL_MAT_ID", "1"}, {معلومات_a_کلی, "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}}));
        string_منزل_spv("mul_mat_vec_id_" + tname + "_f32_f32_subgroup_no_shmem", shader, یوځای_maps(بنسټ_dict, {{"MUL_MAT_ID", "1"}, {معلومات_a_کلی, "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));

        // mul mat vec with integer dot product
#إذا defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
        إذا (is_legacy_quant(tname) || tname == "mxfp4" || is_k_quant(tname) || tname == "iq1_s" || tname == "iq1_m" || tname == "iq4_xs") {
            string_منزل_spv("mul_mat_vec_" + tname + "_q8_1_f32", "mul_mat_vecq.comp", یوځای_maps(بنسټ_dict, {{معلومات_a_کلی, "1"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"FLOAT_TYPEV2", "vec2"}, {"ACC_TYPE", "float"}}));
            string_منزل_spv("mul_mat_vec_" + tname + "_q8_1_f32_subgroup", "mul_mat_vecq.comp", یوځای_maps(بنسټ_dict, {{معلومات_a_کلی, "1"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"FLOAT_TYPEV2", "vec2"}, {"ACC_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}}));
            string_منزل_spv("mul_mat_vec_" + tname + "_q8_1_f32_subgroup_no_shmem", "mul_mat_vecq.comp", یوځای_maps(بنسټ_dict, {{معلومات_a_کلی, "1"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"FLOAT_TYPEV2", "vec2"}, {"ACC_TYPE", "float"}, {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));

            string_منزل_spv("mul_mat_vec_id_" + tname + "_q8_1_f32", "mul_mat_vecq.comp", یوځای_maps(بنسټ_dict, {{"MUL_MAT_ID", "1"}, {معلومات_a_کلی, "1"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"FLOAT_TYPEV2", "vec2"}, {"ACC_TYPE", "float"}}));
            string_منزل_spv("mul_mat_vec_id_" + tname + "_q8_1_f32_subgroup", "mul_mat_vecq.comp", یوځای_maps(بنسټ_dict, {{"MUL_MAT_ID", "1"}, {معلومات_a_کلی, "1"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"FLOAT_TYPEV2", "vec2"}, {"ACC_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}}));
            string_منزل_spv("mul_mat_vec_id_" + tname + "_q8_1_f32_subgroup_no_shmem", "mul_mat_vecq.comp", یوځای_maps(بنسټ_dict, {{"MUL_MAT_ID", "1"}, {معلومات_a_کلی, "1"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"FLOAT_TYPEV2", "vec2"}, {"ACC_TYPE", "float"}, {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));
        }
#د_که_پای

        // Dequant shaders
        إذا (tname != "f16" && tname != "bf16") {
            string_منزل_spv("dequant_" + tname, "dequant_" + tname + ".comp", یوځای_maps(بنسټ_dict, {{معلومات_a_کلی, "1"}, {"D_TYPE", "float16_t"}}));
        }
        // Fused dequant+transpose variant for FA quant-KV (per-head-contiguous f16 scratch).
        إذا (tname == "q8_0") {
            string_منزل_spv("dequant_" + tname + "_transpose", "dequant_" + tname + ".comp", یوځای_maps(بنسټ_dict, {{معلومات_a_کلی, "1"}, {"D_TYPE", "float16_t"}, {"DEQUANT_TRANSPOSE", "1"}}));
        }

        shader = (tname == "f32" || tname == "f16" || tname == "bf16") ? "get_rows.comp" : "get_rows_quant.comp";

        إذا (tname == "f16") {
            string_منزل_spv("get_rows_" + tname, shader, یوځای_maps(بنسټ_dict, {{"TEMP_TYPE", "FLOAT_TYPE"}, {معلومات_a_کلی, "1"}, {"B_TYPE", "int"}, {"D_TYPE", "float16_t"}, {"OPTIMIZATION_ERROR_WORKAROUND", "1"}}));
        } وإلا {
            string_منزل_spv("get_rows_" + tname, shader, یوځای_maps(بنسټ_dict, {{"TEMP_TYPE", "FLOAT_TYPE"}, {معلومات_a_کلی, "1"}, {"B_TYPE", "int"}, {"D_TYPE", "float16_t"}}));
        }
        string_منزل_spv("get_rows_" + tname + "_f32", shader, یوځای_maps(بنسټ_dict, {{"TEMP_TYPE", "FLOAT_TYPE"}, {معلومات_a_کلی, "1"}, {"B_TYPE", "int"}, {"D_TYPE", "float"}}));
    }

    string_منزل_spv("get_rows_i32", "get_rows.comp", {{"TEMP_TYPE", "uint"}, {"A_TYPE", "uint"}, {"B_TYPE", "int"}, {"D_TYPE", "uint"}});

    string_منزل_spv("mul_mat_vec_p021_f16_f32_subgroup_add", "mul_mat_vec_p021.comp", {{"A_TYPE", "float16_t"}, {"A_TYPEV4", "f16vec4"}, {"B_TYPE", "float"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}});
    string_منزل_spv("mul_mat_vec_p021_f16_f32",              "mul_mat_vec_p021.comp", {{"A_TYPE", "float16_t"}, {"A_TYPEV4", "f16vec4"}, {"B_TYPE", "float"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}});
    string_منزل_spv("mul_mat_vec_nc_f16_f32", "mul_mat_vec_nc.comp", {{"A_TYPE", "float16_t"}, {"A_TYPEV4", "f16vec4"}, {"B_TYPE", "float"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}});

    // Norms
    string_منزل_spv("norm_f32", "norm.comp", یوځای_maps(بنسټ_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_منزل_spv("group_norm_f32", "group_norm.comp", یوځای_maps(بنسټ_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_منزل_spv("rms_norm_f32", "rms_norm.comp", یوځای_maps(بنسټ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_منزل_spv("rms_norm_mul_add_f32", "rms_norm.comp", یوځای_maps(بنسټ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"RMS_NORM_ADD_FUSION", "1"}}));
    string_منزل_spv("rms_norm_mul_add_partials_f32", "rms_norm_partials.comp", یوځای_maps(بنسټ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"RMS_NORM_ADD_FUSION", "1"}}));
    string_منزل_spv("rms_norm_set_rows_f32_f32", "rms_norm.comp", یوځای_maps(بنسټ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"RMS_NORM_SET_ROWS_FUSION", "1"}}));
    string_منزل_spv("rms_norm_set_rows_f32_f16", "rms_norm.comp", یوځای_maps(بنسټ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float16_t"}, {"RMS_NORM_SET_ROWS_FUSION", "1"}}));
    string_منزل_spv("rms_norm_partials_f32", "rms_norm_partials.comp", یوځای_maps(بنسټ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_منزل_spv("rms_norm_mul_rope_f32_f32", "rms_norm.comp", یوځای_maps(بنسټ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"ROPE_D_TYPE", "float"}, {"RMS_NORM_ROPE_FUSION", "1"}}));
    string_منزل_spv("rms_norm_mul_rope_f32_f16", "rms_norm.comp", یوځای_maps(بنسټ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"ROPE_D_TYPE", "float16_t"}, {"RMS_NORM_ROPE_FUSION", "1"}}));
    string_منزل_spv("rms_norm_back_f32", "rms_norm_back.comp", یوځای_maps(بنسټ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_منزل_spv("l2_norm_f32", "l2_norm.comp", یوځای_maps(بنسټ_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));

    string_منزل_spv("cpy_32", "copy.comp", {{"A_TYPE", "uint"}, {"D_TYPE", "uint"}});
    string_منزل_spv("cpy_f32_f16", "copy.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float16_t"}});
    string_منزل_spv("cpy_16", "copy.comp", {{"A_TYPE", "uint16_t"}, {"D_TYPE", "uint16_t"}});
    string_منزل_spv("cpy_f16_f32", "copy.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float"}, {"OPTIMIZATION_ERROR_WORKAROUND", "1"}});
    string_منزل_spv("cpy_f32_bf16","copy.comp", {{"A_TYPE", "float"}, {"D_TYPE", "uint16_t"}, {"DATA_D_BF16", "1"}});
    string_منزل_spv("cpy_bf16_f32","copy.comp", {{"A_TYPE", "uint16_t"}, {"D_TYPE", "float"}, {"DATA_A_BF16", "1"}});
    string_منزل_spv("cpy_bf16_f16","copy.comp", {{"A_TYPE", "uint16_t"}, {"D_TYPE", "float16_t"}, {"DATA_A_BF16", "1"}});
    string_منزل_spv("contig_cpy_32", "contig_copy.comp", {{"A_TYPE", "uint"}, {"D_TYPE", "uint"}});
    string_منزل_spv("contig_cpy_f32_i32", "contig_copy.comp", {{"A_TYPE", "float"}, {"D_TYPE", "int"}});
    string_منزل_spv("contig_cpy_i32_f32", "contig_copy.comp", {{"A_TYPE", "int"}, {"D_TYPE", "float"}});
    string_منزل_spv("contig_cpy_f32_f16", "contig_copy.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float16_t"}});
    string_منزل_spv("contig_cpy_16", "contig_copy.comp", {{"A_TYPE", "uint16_t"}, {"D_TYPE", "uint16_t"}});
    string_منزل_spv("contig_cpy_f16_f32", "contig_copy.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float"}, {"OPTIMIZATION_ERROR_WORKAROUND", "1"}});
    string_منزل_spv("contig_cpy_f32_bf16","contig_copy.comp",{{"A_TYPE", "float"}, {"D_TYPE", "uint16_t"}, {"DATA_D_BF16", "1"}});
    string_منزل_spv("contig_cpy_bf16_f32","contig_copy.comp",{{"A_TYPE", "uint16_t"}, {"D_TYPE", "float"}, {"DATA_A_BF16", "1"}});
    string_منزل_spv("contig_cpy_bf16_f16","contig_copy.comp",{{"A_TYPE", "uint16_t"}, {"D_TYPE", "float16_t"}, {"DATA_A_BF16", "1"}});
    string_منزل_spv("cpy_f32_i32", "copy.comp", {{"A_TYPE", "float"}, {"D_TYPE", "int"}});
    string_منزل_spv("cpy_i32_f32", "copy.comp", {{"A_TYPE", "int"}, {"D_TYPE", "float"}});

    string_منزل_spv("cpy_transpose_16", "copy_transpose.comp", {{"A_TYPE", "uint16_t"}, {"D_TYPE", "uint16_t"}});
    string_منزل_spv("cpy_transpose_32", "copy_transpose.comp", {{"A_TYPE", "uint"}, {"D_TYPE", "uint"}});
    string_منزل_spv("cpy_transpose_02_16", "copy_transpose_02.comp", {{"A_TYPE", "uint16_t"}, {"D_TYPE", "uint16_t"}});
    string_منزل_spv("cpy_transpose_02_32", "copy_transpose_02.comp", {{"A_TYPE", "uint"}, {"D_TYPE", "uint"}});

    لكل (std::string t : {"q1_0", "q2_0", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "iq4_nl"}) {
        string_منزل_spv("cpy_f32_" + t, "copy_to_quant.comp", {{"DATA_A_" + منزل_uppercase(t), "1"}, {"S_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});
        string_منزل_spv("cpy_" + t + "_f32", "copy_from_quant.comp", {{"DATA_A_" + منزل_uppercase(t), "1"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});
    }

    لكل (خپلکاری src : {std::pair{"f32", "float"}, std::pair{"f16", "float16_t"}}) {
        لكل (std::string dst : {"f32", "f16", "bf16", "q1_0", "q2_0", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "iq4_nl"}) {
            string_منزل_spv("set_rows_" + std::string(src.لومړی) + "_" + dst + "_i32", "copy_to_quant.comp", {{"SET_ROWS", "1"}, {"DATA_A_" + منزل_uppercase(dst), "1"}, {"B_TYPE", "uint"}, {"B_SIZE", "32"}, {"S_TYPE", src.second}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});
            string_منزل_spv("set_rows_" + std::string(src.لومړی) + "_" + dst + "_i64", "copy_to_quant.comp", {{"SET_ROWS", "1"}, {"DATA_A_" + منزل_uppercase(dst), "1"}, {"B_TYPE", "uvec2"}, {"B_SIZE", "64"}, {"S_TYPE", src.second}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});
        }
    }

    خپلکاری get_type_str = [](منطقي f16) {
        أرجع f16 ? "float16_t" : "float";
    };
    خپلکاری get_suffix = [](منطقي src0_f16, منطقي src1_f16, منطقي dst_f16) {
        std::string s;
        s += std::string(src0_f16 ? "_f16" : "_f32");
        s += std::string(src1_f16 ? "_f16" : "_f32");
        s += std::string(dst_f16 ? "_f16" : "_f32");
        أرجع s;
    };
    لكل (std::string op : {"add", "sub", "mul", "div", "add_rms", }) {
    لكل (خپلکاری src0_f16 : {خطأ, ریښتیا}) {
    لكل (خپلکاری src1_f16 : {خطأ, ریښتیا}) {
    لكل (خپلکاری dst_f16  : {خطأ, ریښتیا}) {
        خپلکاری سرچینه = op == "add_rms" ? std::string("add") : op;
        خپلکاری name = op + get_suffix(src0_f16, src1_f16, dst_f16);
        خپلکاری جمع_کړه_rms = op == "add_rms" ? "1" : "0";
        string_منزل_spv(name.c_str(), سرچینه + ".comp", {{"A_TYPE", get_type_str(src0_f16)}, {"B_TYPE", get_type_str(src1_f16)}, {"D_TYPE", get_type_str(dst_f16)}, {"FLOAT_TYPE", "float"}, {"ADD_RMS" , جمع_کړه_rms}});
    }
    }
    }
    }

    string_منزل_spv("sub_f32", "sub.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});

    string_منزل_spv("acc_f32", "acc.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});

    string_منزل_spv("split_k_reduce", "mul_mat_split_k_reduce.comp", {});
    string_منزل_spv("fa_split_k_reduce", "flash_attn_split_k_reduce.comp", {});

    string_منزل_spv("fa_mask_opt", "flash_attn_mask_opt.comp", {});

#إذا defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    string_منزل_spv("fa_decode_ph1", "flash_attn_decode_phase_1.comp", {}, ریښتیا, ریښتیا, خطأ, خطأ);
    string_منزل_spv("fa_decode_ph2", "flash_attn_decode_phase_2.comp", {}, ریښتیا, ریښتیا, خطأ, خطأ);
#د_که_پای

    string_منزل_spv("fa_sparse_compact", "flash_attn_sparse_compact.comp", {});
    string_منزل_spv("fa_sparse_compact_subgroup", "flash_attn_sparse_compact.comp", {{"USE_SUBGROUPS", "1"}});

    string_منزل_spv("quantize_q8_1", "quantize_q8_1.comp", {});
    string_منزل_spv("quantize_q8_1_subgroup", "quantize_q8_1.comp", {{"USE_SUBGROUPS", "1"}});

    string_منزل_spv("quantize_q8_1_x4", "quantize_q8_1.comp", {{"QBLOCK_X4", "1"}});
    string_منزل_spv("quantize_q8_1_x4_subgroup", "quantize_q8_1.comp", {{"QBLOCK_X4", "1"}, {"USE_SUBGROUPS", "1"}});

    string_منزل_spv("mul_f32", "mul.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});

    string_منزل_spv("div_f32", "div.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});

    string_منزل_spv("repeat_i32", "repeat.comp", {{"A_TYPE", "int32_t"}, {"D_TYPE", "int32_t"}});
    string_منزل_spv("repeat_back_f32", "repeat_back.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}});
    string_منزل_spv("get_rows_back_f32", "get_rows_back.comp", {{"A_TYPE", "float"}, {"B_TYPE", "int"}, {"D_TYPE", "float"}});

    string_منزل_spv("repeat_i16", "repeat.comp", {{"A_TYPE", "int16_t"}, {"D_TYPE", "int16_t"}});

    string_منزل_spv("scale_f32", "scale.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});

    string_منزل_spv("pad_f32", "pad.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}});
    string_منزل_spv("pad_reflect_1d_f32", "pad_reflect_1d.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}});

    string_منزل_spv("concat_i8", "concat.comp", {{"A_TYPE", "uint8_t"}, {"B_TYPE", "uint8_t"}, {"D_TYPE", "uint8_t"}});
    string_منزل_spv("concat_i16", "concat.comp", {{"A_TYPE", "uint16_t"}, {"B_TYPE", "uint16_t"}, {"D_TYPE", "uint16_t"}});
    string_منزل_spv("concat_i32", "concat.comp", {{"A_TYPE", "uint"}, {"B_TYPE", "uint"}, {"D_TYPE", "uint"}});
    string_منزل_spv("concat_i64", "concat.comp", {{"A_TYPE", "uvec2"}, {"B_TYPE", "uvec2"}, {"D_TYPE", "uvec2"}});

    string_منزل_spv("upscale_f32", "upscale.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}});

    string_منزل_spv("exp_f16",        "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_exp"}});
    string_منزل_spv("exp_f32",        "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_exp"}});
    string_منزل_spv("expm1_f16",      "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_expm1"}});
    string_منزل_spv("expm1_f32",      "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_expm1"}});

    string_منزل_spv("log_f16",        "log.comp",         {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_منزل_spv("log_f32",        "log.comp",         {{"A_TYPE", "float"},       {"D_TYPE", "float"}});
    string_منزل_spv("gelu_f16",       "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_gelu"}});
    string_منزل_spv("gelu_f32",       "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_gelu"}});
    string_منزل_spv("gelu_erf_f16",   "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_gelu_erf"}});
    string_منزل_spv("gelu_erf_f32",   "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_gelu_erf"}});
    string_منزل_spv("gelu_quick_f16", "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_gelu_quick"}});
    string_منزل_spv("gelu_quick_f32", "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_gelu_quick"}});
    string_منزل_spv("silu_f16",       "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_silu"}});
    string_منزل_spv("silu_f32",       "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_silu"}});
    string_منزل_spv("relu_f16",       "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_relu"}});
    string_منزل_spv("relu_f32",       "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_relu"}});
    string_منزل_spv("sqr_f16",        "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_sqr"}});
    string_منزل_spv("sqr_f32",        "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_sqr"}});
    string_منزل_spv("sqrt_f16",       "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_sqrt"}});
    string_منزل_spv("sqrt_f32",       "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_sqrt"}});
    string_منزل_spv("sin_f16",        "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_sin"}});
    string_منزل_spv("sin_f32",        "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_sin"}});
    string_منزل_spv("cos_f16",        "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_cos"}});
    string_منزل_spv("cos_f32",        "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_cos"}});
    string_منزل_spv("clamp_f16",      "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_clamp"}});
    string_منزل_spv("clamp_f32",      "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_clamp"}});
    string_منزل_spv("leaky_relu_f16", "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_leaky_relu"}});
    string_منزل_spv("leaky_relu_f32", "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_leaky_relu"}});
    string_منزل_spv("neg_f16",        "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_neg"}});
    string_منزل_spv("neg_f32",        "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_neg"}});
    string_منزل_spv("tanh_f16",       "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_tanh"}});
    string_منزل_spv("tanh_f32",       "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_tanh"}});
    string_منزل_spv("sigmoid_f16",    "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_sigmoid"}});
    string_منزل_spv("sigmoid_f32",    "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_sigmoid"}});
    string_منزل_spv("hardsigmoid_f16","unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_hardsigmoid"}});
    string_منزل_spv("hardsigmoid_f32","unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_hardsigmoid"}});
    string_منزل_spv("hardswish_f16",  "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_hardswish"}});
    string_منزل_spv("hardswish_f32",  "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_hardswish"}});
    string_منزل_spv("abs_f16",        "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_abs"}});
    string_منزل_spv("abs_f32",        "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_abs"}});
    string_منزل_spv("elu_f16",        "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_elu"}});
    string_منزل_spv("elu_f32",        "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_elu"}});
    string_منزل_spv("xielu_f16",      "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_xielu"}});
    string_منزل_spv("xielu_f32",      "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_xielu"}});
    string_منزل_spv("sgn_f16",        "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_sgn"}});
    string_منزل_spv("sgn_f32",        "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_sgn"}});

    string_منزل_spv("tri_f16",        "tri.comp",         {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_منزل_spv("tri_f32",        "tri.comp",         {{"A_TYPE", "float"},       {"D_TYPE", "float"}});
    string_منزل_spv("diag_f16",       "diag.comp",        {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_منزل_spv("diag_f32",       "diag.comp",        {{"A_TYPE", "float"},       {"D_TYPE", "float"}});

    string_منزل_spv("softplus_f16",   "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_softplus"}});
    string_منزل_spv("softplus_f32",   "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_softplus"}});

    string_منزل_spv("gelu_mul_f32",    "unary.comp",      {{"A_TYPE", "float"},       {"B_TYPE", "float"},     {"D_TYPE", "float"},     {"FLOAT_TYPE", "float"}, {"OP", "op_gelu"},     {"UNARY_MUL_FUSION", "1"}});
    string_منزل_spv("gelu_mul_f16",    "unary.comp",      {{"A_TYPE", "float16_t"},   {"B_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"FLOAT_TYPE", "float"}, {"OP", "op_gelu"},     {"UNARY_MUL_FUSION", "1"}});
    string_منزل_spv("sigmoid_mul_f32", "unary.comp",      {{"A_TYPE", "float"},       {"B_TYPE", "float"},     {"D_TYPE", "float"},     {"FLOAT_TYPE", "float"}, {"OP", "op_sigmoid"},  {"UNARY_MUL_FUSION", "1"}});
    string_منزل_spv("sigmoid_mul_f16", "unary.comp",      {{"A_TYPE", "float16_t"},   {"B_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"FLOAT_TYPE", "float"}, {"OP", "op_sigmoid"},  {"UNARY_MUL_FUSION", "1"}});
    string_منزل_spv("silu_mul_f32",    "unary.comp",      {{"A_TYPE", "float"},       {"B_TYPE", "float"},     {"D_TYPE", "float"},     {"FLOAT_TYPE", "float"}, {"OP", "op_silu"},     {"UNARY_MUL_FUSION", "1"}});
    string_منزل_spv("silu_mul_f16",    "unary.comp",      {{"A_TYPE", "float16_t"},   {"B_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"FLOAT_TYPE", "float"}, {"OP", "op_silu"},     {"UNARY_MUL_FUSION", "1"}});
    string_منزل_spv("softplus_mul_f32","unary.comp",      {{"A_TYPE", "float"},       {"B_TYPE", "float"},     {"D_TYPE", "float"},     {"FLOAT_TYPE", "float"}, {"OP", "op_softplus"}, {"UNARY_MUL_FUSION", "1"}});
    string_منزل_spv("softplus_mul_f16","unary.comp",      {{"A_TYPE", "float16_t"},   {"B_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"FLOAT_TYPE", "float"}, {"OP", "op_softplus"}, {"UNARY_MUL_FUSION", "1"}});

    string_منزل_spv("add1_f16_f16",   "add1.comp",        {{"A_TYPE", "float16_t"},   {"B_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"FLOAT_TYPE", "float"}});
    string_منزل_spv("add1_f16_f32",   "add1.comp",        {{"A_TYPE", "float16_t"},   {"B_TYPE", "float"}, {"D_TYPE", "float16_t"}, {"FLOAT_TYPE", "float"}});
    string_منزل_spv("add1_f32_f32",   "add1.comp",        {{"A_TYPE", "float"},       {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});
    string_منزل_spv("arange_f32",     "arange.comp",      {{"A_TYPE", "float"},       {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});
    string_منزل_spv("fill_f32",       "fill.comp",        {{"D_TYPE", "float"},       {"FLOAT_TYPE", "float"}});
    string_منزل_spv("fill_f16",       "fill.comp",        {{"D_TYPE", "float16_t"},   {"FLOAT_TYPE", "float"}});
    string_منزل_spv("step_f16",       "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_step"}});
    string_منزل_spv("step_f32",       "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_step"}});
    string_منزل_spv("round_f16",      "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_round"}});
    string_منزل_spv("round_f32",      "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_round"}});
    string_منزل_spv("ceil_f16",       "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_ceil"}});
    string_منزل_spv("ceil_f32",       "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_ceil"}});
    string_منزل_spv("floor_f16",      "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_floor"}});
    string_منزل_spv("floor_f32",      "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_floor"}});
    string_منزل_spv("trunc_f16",      "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_trunc"}});
    string_منزل_spv("trunc_f32",      "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_trunc"}});

    string_منزل_spv("geglu_f16",      "geglu.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_منزل_spv("geglu_f32",      "geglu.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"}});
    string_منزل_spv("reglu_f16",      "reglu.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_منزل_spv("reglu_f32",      "reglu.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"}});
    string_منزل_spv("swiglu_f16",     "swiglu.comp",      {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_منزل_spv("swiglu_f32",     "swiglu.comp",      {{"A_TYPE", "float"},       {"D_TYPE", "float"}});
    string_منزل_spv("swiglu_oai_f16", "swiglu_oai.comp",  {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_منزل_spv("swiglu_oai_f32", "swiglu_oai.comp",  {{"A_TYPE", "float"},       {"D_TYPE", "float"}});
    string_منزل_spv("swiglu_clamp_f16", "swiglu_clamp.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}});
    string_منزل_spv("swiglu_clamp_f32", "swiglu_clamp.comp", {{"A_TYPE", "float"},     {"D_TYPE", "float"}});
    string_منزل_spv("geglu_erf_f16",  "geglu_erf.comp",   {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_منزل_spv("geglu_erf_f32",  "geglu_erf.comp",   {{"A_TYPE", "float"},       {"D_TYPE", "float"}});
    string_منزل_spv("geglu_quick_f16","geglu_quick.comp", {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_منزل_spv("geglu_quick_f32","geglu_quick.comp", {{"A_TYPE", "float"},       {"D_TYPE", "float"}});

    string_منزل_spv("silu_back_f32",  "silu_back.comp",   {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}});

    string_منزل_spv("diag_mask_inf_f32", "diag_mask_inf.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}});

    string_منزل_spv("soft_max_f32", "soft_max.comp", یوځای_maps(بنسټ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_منزل_spv("soft_max_f32_f16", "soft_max.comp", یوځای_maps(بنسټ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float16_t"}, {"D_TYPE", "float"}}));
    string_منزل_spv("soft_max_back_f32", "soft_max_back.comp", یوځای_maps(بنسټ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));

    string_منزل_spv("soft_max_large1_f32", "soft_max_large1.comp", یوځای_maps(بنسټ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_منزل_spv("soft_max_large2_f32", "soft_max_large2.comp", یوځای_maps(بنسټ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_منزل_spv("soft_max_large3_f32", "soft_max_large3.comp", یوځای_maps(بنسټ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_منزل_spv("soft_max_large1_f32_f16", "soft_max_large1.comp", یوځای_maps(بنسټ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float16_t"}, {"D_TYPE", "float"}}));
    string_منزل_spv("soft_max_large2_f32_f16", "soft_max_large2.comp", یوځای_maps(بنسټ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float16_t"}, {"D_TYPE", "float"}}));
    string_منزل_spv("soft_max_large3_f32_f16", "soft_max_large3.comp", یوځای_maps(بنسټ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float16_t"}, {"D_TYPE", "float"}}));

    string_منزل_spv("rope_norm_f32", "rope_norm.comp", {{"A_TYPE", "float"}, {"ROPE_D_TYPE", "float"}});
    string_منزل_spv("rope_norm_f16", "rope_norm.comp", {{"A_TYPE", "float16_t"}, {"ROPE_D_TYPE", "float16_t"}});
    string_منزل_spv("rope_norm_f32_f16", "rope_norm.comp", {{"A_TYPE", "float"}, {"ROPE_D_TYPE", "float16_t"}});

    string_منزل_spv("rope_neox_f32", "rope_neox.comp", {{"A_TYPE", "float"}, {"ROPE_D_TYPE", "float"}});
    string_منزل_spv("rope_neox_f16", "rope_neox.comp", {{"A_TYPE", "float16_t"}, {"ROPE_D_TYPE", "float16_t"}});
    string_منزل_spv("rope_neox_f32_f16", "rope_neox.comp", {{"A_TYPE", "float"}, {"ROPE_D_TYPE", "float16_t"}});

    string_منزل_spv("rope_multi_f32", "rope_multi.comp", {{"A_TYPE", "float"}, {"ROPE_D_TYPE", "float"}});
    string_منزل_spv("rope_multi_f16", "rope_multi.comp", {{"A_TYPE", "float16_t"}, {"ROPE_D_TYPE", "float16_t"}});
    string_منزل_spv("rope_multi_f32_f16", "rope_multi.comp", {{"A_TYPE", "float"}, {"ROPE_D_TYPE", "float16_t"}});

    string_منزل_spv("rope_vision_f32", "rope_vision.comp", {{"A_TYPE", "float"}, {"ROPE_D_TYPE", "float"}});
    string_منزل_spv("rope_vision_f16", "rope_vision.comp", {{"A_TYPE", "float16_t"}, {"ROPE_D_TYPE", "float16_t"}});

    string_منزل_spv("argsort_f32", "argsort.comp", {{"A_TYPE", "float"}});
    string_منزل_spv("argsort_large_f32", "argsort_large.comp", {{"A_TYPE", "float"}});

    string_منزل_spv("topk_argsort_f32", "topk_argsort.comp", {{"A_TYPE", "float"}});
    string_منزل_spv("topk_nary_search_f32", "topk_nary_search.comp", {{"A_TYPE", "float"}});
    string_منزل_spv("topk_radix_select_f32", "topk_radix_select.comp", {{"A_TYPE", "float"}});

    string_منزل_spv("argmax_f32", "argmax.comp", یوځای_maps(بنسټ_dict, {{"A_TYPE", "float"}, {"D_TYPE", "int"}}));
    string_منزل_spv("sum_rows_f32", "sum_rows.comp", یوځای_maps(بنسټ_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_منزل_spv("cross_entropy_loss_f32", "cross_entropy_loss.comp", یوځای_maps(بنسټ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_منزل_spv("cross_entropy_loss_back_f32", "cross_entropy_loss_back.comp", یوځای_maps(بنسټ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_منزل_spv("fwht_f32", "fwht.comp", {});
    string_منزل_spv("fwht_shmem_f32", "fwht.comp", {{"FWHT_SHMEM", "1"}});
    string_منزل_spv("count_equal_i32", "count_equal.comp", یوځای_maps(بنسټ_dict, {{"A_TYPE", "int"}, {"B_TYPE", "int"}, {"D_TYPE", "int"}}));
    string_منزل_spv("dsv4_hc_comb_f32", "dsv4_hc_comb.comp", {});
    string_منزل_spv("dsv4_hc_pre_f32",  "dsv4_hc_pre.comp",  {});
    string_منزل_spv("dsv4_hc_post_f32", "dsv4_hc_post.comp", {});
    string_منزل_spv("cumsum_f32", "cumsum.comp", یوځای_maps(بنسټ_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_منزل_spv("cumsum_multipass1_f32", "cumsum_multipass1.comp", یوځای_maps(بنسټ_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_منزل_spv("cumsum_multipass2_f32", "cumsum_multipass2.comp", یوځای_maps(بنسټ_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));

    string_منزل_spv("count_experts", "count_experts.comp", یوځای_maps(بنسټ_dict, {{"A_TYPE", "uint"}, {"D_TYPE", "uint"}}));
    string_منزل_spv("count_experts_subgroup", "count_experts.comp", یوځای_maps(بنسټ_dict, {{"A_TYPE", "uint"}, {"D_TYPE", "uint"}, {"USE_SUBGROUPS", "1"}}));

    لكل (std::string dim_str : {"", "_3d"}) {
        لكل (منطقي bda : {خطأ, ریښتیا}) {
            std::string bda_str = bda ? "_bda" : "";
            std::string bda_def = bda ? "1" : "0";
            string_منزل_spv("im2col" + dim_str + "_f32" + bda_str, "im2col" + dim_str + ".comp", یوځای_maps(بنسټ_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}, {"D_SIZE", "4"}, {"BDA", bda_def}}));
            string_منزل_spv("im2col" + dim_str + "_f32_f16" + bda_str, "im2col" + dim_str + ".comp", یوځای_maps(بنسټ_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float16_t"}, {"D_SIZE", "2"}, {"BDA", bda_def}}));
        }
    }

    string_منزل_spv("out_prod_f32", "out_prod.comp", {});

    string_منزل_spv("timestep_embedding_f32", "timestep_embedding.comp", یوځای_maps(بنسټ_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));

    string_منزل_spv("conv_transpose_1d_f32", "conv_transpose_1d.comp", {{"A_TYPE", "float"},  {"B_TYPE", "float"}, {"D_TYPE", "float"}});
    string_منزل_spv("col2im_1d_f32",  "col2im_1d.comp", {{"DATA_A_F32", "1"},  {"A_TYPE", "float"},     {"D_TYPE", "float"}});
    string_منزل_spv("col2im_1d_f16",  "col2im_1d.comp", {{"DATA_A_F16", "1"},  {"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}});
    string_منزل_spv("col2im_1d_bf16", "col2im_1d.comp", {{"DATA_A_BF16", "1"}, {"A_TYPE", "uint16_t"},  {"D_TYPE", "uint16_t"}});

    string_منزل_spv("snake_f32",  "snake.comp", {{"DATA_A_F32", "1"},  {"A_TYPE", "float"},     {"D_TYPE", "float"}});
    string_منزل_spv("snake_f16",  "snake.comp", {{"DATA_A_F16", "1"},  {"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}});
    string_منزل_spv("snake_bf16", "snake.comp", {{"DATA_A_BF16", "1"}, {"DATA_D_BF16", "1"}, {"A_TYPE", "uint16_t"},  {"D_TYPE", "uint16_t"}});

    string_منزل_spv("pool1d_f32", "pool1d.comp", یوځای_maps(بنسټ_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_منزل_spv("pool2d_f32", "pool2d.comp", یوځای_maps(بنسټ_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_منزل_spv("pool2d_subgroup_f32", "pool2d.comp", یوځای_maps(بنسټ_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}, {"USE_SUBGROUP", "1"}}));

    string_منزل_spv("rwkv_wkv6_f32", "wkv6.comp", یوځای_maps(بنسټ_dict, {{"A_TYPE", "float"}}));

    string_منزل_spv("gated_linear_attn_f32", "gla.comp", یوځای_maps(بنسټ_dict, {{"A_TYPE", "float"}}));

    // Compile IQ4_NL support in so its shared LUT is available when K uses it.
    // K quant type is selected at runtime via the FaTypeK spec constant.
    std::map<std::string, std::string> li_dict = {{"FLOAT_TYPE", "float"}, {"FLOAT_TYPEV4", "vec4"}, {"DATA_A_IQ4_NL", "1"}};
    string_منزل_spv("lightning_indexer_f32", "lightning_indexer.comp", li_dict);

    string_منزل_spv("rwkv_wkv7_f32", "wkv7.comp", یوځای_maps(بنسټ_dict, {{"A_TYPE", "float"}}));

    string_منزل_spv("gated_delta_net_f32", "gated_delta_net.comp", یوځای_maps(بنسټ_dict, {{"FLOAT_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}, {"USE_SUBGROUP_CLUSTERED", "1"}}));
    string_منزل_spv("gated_delta_net_f32_nocluster", "gated_delta_net.comp", یوځای_maps(بنسټ_dict, {{"FLOAT_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}, {"USE_SUBGROUP_CLUSTERED", "0"}}));
    string_منزل_spv("gated_delta_net_f32_shmem", "gated_delta_net.comp", یوځای_maps(بنسټ_dict, {{"FLOAT_TYPE", "float"}, {"USE_SUBGROUP_ADD", "0"}, {"USE_SUBGROUP_CLUSTERED", "0"}}));

    string_منزل_spv("opt_step_adamw_f32", "opt_step_adamw.comp", یوځای_maps(بنسټ_dict, {{"A_TYPE", "float"}}));
    string_منزل_spv("opt_step_sgd_f32", "opt_step_sgd.comp", یوځای_maps(بنسټ_dict, {{"A_TYPE", "float"}}));

    string_منزل_spv("solve_tri_f32", "solve_tri.comp", یوځای_maps(بنسټ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));

    لكل (خپلکاری transpose : {خطأ, ریښتیا}) {
        لكل (خپلکاری unroll : {خطأ, ریښتیا}) {
            لكل (خپلکاری a_f16 : {خطأ, ریښتیا}) {
                std::map<std::string, std::string> تعریفوي = {
                    {"A_TYPE", a_f16 ? "float16_t" : "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"},
                    {"USE_COLLECTIVES", "1"}, {"UNROLL", unroll ? "[[unroll]]" : ""},
                };
                إذا (transpose) تعریفوي["TRANSPOSE"] = "1";
                std::string name = std::string(transpose ? "conv_transpose_2d": "conv2d")
                    + (a_f16 ? "_f16" : "") + "_f32";
                string_منزل_spv(name + (unroll ? "_unroll" : ""), "conv2d_mm.comp", تعریفوي);
#إذا defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
                إذا (unroll) {
                    خپلکاری cm2_تعریفوي = تعریفوي;
                    cm2_تعریفوي["COOPMAT2"] = "1";
                    string_منزل_spv(name, "conv2d_mm.comp", cm2_تعریفوي, ریښتیا, خطأ, ریښتیا);
                }
#د_که_پای
#إذا defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
                إذا (unroll) {
                    خپلکاری cm1_تعریفوي = تعریفوي;
                    cm1_تعریفوي["COOPMAT"] = "1";
                    string_منزل_spv(name, "conv2d_mm.comp", cm1_تعریفوي, ریښتیا, ریښتیا, خطأ);
                }
#د_که_پای
            }
        }
    }

    لكل (خپلکاری unroll : {خطأ, ریښتیا}) {
        لكل (خپلکاری a_f16 : {خطأ, ریښتیا}) {
            std::map<std::string, std::string> تعریفوي = {
                {"A_TYPE", a_f16 ? "float16_t" : "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"},
                {"UNROLL", unroll ? "[[unroll]]" : ""},
            };
            std::string name = std::string("conv3d") + (a_f16 ? "_f16" : "") + "_f32";
            string_منزل_spv(name + (unroll ? "_unroll" : ""), "conv3d_mm.comp", تعریفوي);
#إذا defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
            إذا (unroll) {
                خپلکاری cm2_تعریفوي = تعریفوي;
                cm2_تعریفوي["COOPMAT2"] = "1";
                string_منزل_spv(name, "conv3d_mm.comp", cm2_تعریفوي, ریښتیا, خطأ, ریښتیا);
            }
#د_که_پای
#إذا defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
            إذا (unroll) {
                خپلکاری cm1_تعریفوي = تعریفوي;
                cm1_تعریفوي["COOPMAT"] = "1";
                string_منزل_spv(name, "conv3d_mm.comp", cm1_تعریفوي, ریښتیا, ریښتیا, خطأ);
            }
#د_که_پای
        }
    }

    string_منزل_spv("conv2d_dw_whcn_f32", "conv2d_dw.comp", یوځای_maps(بنسټ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"WHCN", "1"}}));
    string_منزل_spv("conv2d_dw_cwhn_f32", "conv2d_dw.comp", یوځای_maps(بنسټ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"CWHN", "1"}}));
    string_منزل_spv("conv2d_dw_whcn_f16_f32", "conv2d_dw.comp", یوځای_maps(بنسټ_dict, {{"A_TYPE", "float16_t"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"WHCN", "1"}}));
    string_منزل_spv("conv2d_dw_cwhn_f16_f32", "conv2d_dw.comp", یوځای_maps(بنسټ_dict, {{"A_TYPE", "float16_t"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"CWHN", "1"}}));

    string_منزل_spv("roll_f32", "roll.comp", یوځای_maps(بنسټ_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));

    string_منزل_spv("add_id_f32", "add_id.comp", یوځای_maps(بنسټ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));

    string_منزل_spv("multi_add_f32", "multi_add.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"ADD_RMS" , "0"}});
    string_منزل_spv("multi_add_rms_f32", "multi_add.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"ADD_RMS" , "1"}});

    string_منزل_spv("ssm_scan_f32",          "ssm_scan.comp", {{"A_TYPE", "float"}});
    string_منزل_spv("ssm_scan_subgroup_f32", "ssm_scan.comp", {{"A_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}});

    string_منزل_spv("ssm_conv_f32", "ssm_conv.comp", {{"A_TYPE", "float"}});

    string_منزل_spv("topk_moe_f32", "topk_moe.comp", {});

    لكل (خپلکاری &c : compiles) {
        c.wait();
    }
}

فراغ write_output_files() {
    std::stringstream hdr = make_generic_stringstream();
    std::stringstream src = make_generic_stringstream();

    hdr << "#include <cstdint>\n\n";
    src << "#include \"" << basename(موخه_hpp) << "\"\n\n";

    std::sort(shader_fnames.begin(), shader_fnames.end());
    لكل (ثابت خپلکاری& pair : shader_fnames) {
        ثابت std::string& name = pair.لومړی;
        #که_تعریف_شوی _WIN32
            std::string path = pair.second;
            std::replace(path.begin(), path.end(), '/', '\\' );
        #وإلا
            ثابت std::string& path = pair.second;
        #د_که_پای

        hdr << "extern const uint64_t " << name << "_len;\n";
        hdr << "extern const unsigned char " << name << "_data[];\n\n";

        إذا (ننوت_filepath != "") {
            std::string معلومات = read_دوګونی_file(path);
            إذا (معلومات.تشه()) {
                استمر;
            }

            src << "const uint64_t " << name << "_len = " << معلومات.size() << ";\n";
            src << "const unsigned char " << name << "_data[" << معلومات.size() << "] = {\n" << std::hex;
            خپلکاری بایټونه = reinterpret_cast<ثابت uint8_t*>(معلومات.معلومات());
            لكل (size_t i = 0; i < معلومات.size(); ++i) {
                src << "0x" << static_cast<صحيح>(بایټونه[i]) << ",";
                إذا ((i + 1) % 12 == 0) src << "\n";
            }
            src << std::dec << "\n};\n\n";
        }
    }

    std::string suffixes[2] = {"_f32", "_f16"};
    لكل (std::string op : {"add", "sub", "mul", "div", "add_rms"}) {
        hdr << "extern const void * " << op << "_data[2][2][2];\n";
        hdr << "extern const uint64_t " << op << "_len[2][2][2];\n";

        std::string op_file = op == "add_rms" ? "add.comp" : std::string(op) + ".comp";
        إذا (basename(ننوت_filepath) != op_file) {
            استمر;
        }
        std::stringstream معلومات = make_generic_stringstream();
        std::stringstream len  = make_generic_stringstream();
        معلومات << "const void * " << op << "_data[2][2][2] = ";
        len  << "const uint64_t " << op << "_len[2][2][2] = ";
        لكل (uint32_t t0 = 0; t0 < 2; ++t0) {
            إذا (t0 == 0) {
                معلومات << "{";
                len  << "{";
            }
            لكل (uint32_t t1 = 0; t1 < 2; ++t1) {
                إذا (t1 == 0) {
                    معلومات << "{";
                    len  << "{";
                }
                لكل (uint32_t t2 = 0; t2 < 2; ++t2) {
                    إذا (t2 == 0) {
                        معلومات << "{";
                        len  << "{";
                    }
                    معلومات << op << suffixes[t0] << suffixes[t1] << suffixes[t2];
                    len  << op << suffixes[t0] << suffixes[t1] << suffixes[t2];
                    معلومات << "_data,";
                    len  << "_len,";
                    إذا (t2 == 1) {
                        معلومات << "}, ";
                        len  << "}, ";
                    }
                }
                إذا (t1 == 1) {
                    معلومات << "}, ";
                    len  << "}, ";
                }
            }
            إذا (t0 == 1) {
                معلومات << "};\n";
                len  << "};\n";
            }
        }
        src << معلومات.str();
        src << len.str();
    }

    std::vector<std::string> btypes = {"f16", "f32"};

#إذا defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
    btypes.push_back("q8_1");
#د_که_پای

    لكل (ثابت std::string& btype : btypes) {
    لكل (ثابت خپلکاری& tname : type_نومونه) {
        إذا (btype == "q8_1" && !is_legacy_quant(tname) && tname != "mxfp4" && !is_k_quant(tname) && tname != "iq1_s" && tname != "iq1_m" && tname != "iq4_xs") {
            استمر;
        }
        hdr << "extern const void * arr_dmmv_"   << tname << "_" << btype << "_f32_data[3];\n";
        hdr << "extern const uint64_t arr_dmmv_" << tname << "_" << btype << "_f32_len[3];\n";
        إذا (basename(ننوت_filepath) == "mul_mat_vec.comp") {
            src << "const void * arr_dmmv_"   << tname << "_" << btype << "_f32_data[3] = {mul_mat_vec_" << tname << "_" << btype << "_f32_data, mul_mat_vec_" << tname << "_" << btype << "_f32_subgroup_data, mul_mat_vec_" << tname << "_" << btype << "_f32_subgroup_no_shmem_data};\n";
            src << "const uint64_t arr_dmmv_" << tname << "_" << btype << "_f32_len[3] =  {mul_mat_vec_" << tname << "_" << btype << "_f32_len,  mul_mat_vec_" << tname << "_" << btype << "_f32_subgroup_len, mul_mat_vec_"  << tname << "_" << btype << "_f32_subgroup_no_shmem_len};\n";
        }

        إذا (btype == "f16") {
            استمر;
        }
        hdr << "extern const void * arr_dmmv_id_"   << tname << "_" << btype << "_f32_data[3];\n";
        hdr << "extern const uint64_t arr_dmmv_id_" << tname << "_" << btype << "_f32_len[3];\n";
        إذا (basename(ننوت_filepath) == "mul_mat_vec.comp") {
            src << "const void * arr_dmmv_id_"   << tname << "_" << btype << "_f32_data[3] = {mul_mat_vec_id_" << tname << "_" << btype << "_f32_data, mul_mat_vec_id_" << tname << "_" << btype << "_f32_subgroup_data, mul_mat_vec_id_" << tname << "_" << btype << "_f32_subgroup_no_shmem_data};\n";
            src << "const uint64_t arr_dmmv_id_" << tname << "_" << btype << "_f32_len[3] =  {mul_mat_vec_id_" << tname << "_" << btype << "_f32_len,  mul_mat_vec_id_" << tname << "_" << btype << "_f32_subgroup_len, mul_mat_vec_id_"  << tname << "_" << btype << "_f32_subgroup_no_shmem_len};\n";
        }
    }
    }

#إذا defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
    لكل (ثابت std::string& btype : {"f16", "f32"}) {
    لكل (ثابت std::string& tname : {"mxfp4", "nvfp4"}) {
        hdr << "extern const void * arr_dmmv_"   << tname << "_" << btype << "_f32_ocp_data[3];\n";
        hdr << "extern const uint64_t arr_dmmv_" << tname << "_" << btype << "_f32_ocp_len[3];\n";
        إذا (basename(ننوت_filepath) == "mul_mat_vec.comp") {
            src << "const void * arr_dmmv_"   << tname << "_" << btype << "_f32_ocp_data[3] = {mul_mat_vec_" << tname << "_" << btype << "_f32_ocp_data, mul_mat_vec_" << tname << "_" << btype << "_f32_ocp_subgroup_data, mul_mat_vec_" << tname << "_" << btype << "_f32_ocp_subgroup_no_shmem_data};\n";
            src << "const uint64_t arr_dmmv_" << tname << "_" << btype << "_f32_ocp_len[3] =  {mul_mat_vec_" << tname << "_" << btype << "_f32_ocp_len,  mul_mat_vec_" << tname << "_" << btype << "_f32_ocp_subgroup_len, mul_mat_vec_"  << tname << "_" << btype << "_f32_ocp_subgroup_no_shmem_len};\n";
        }
    }
    }
    لكل (ثابت std::string& tname : {"mxfp4", "nvfp4"}) {
        hdr << "extern const void * arr_dmmv_id_"   << tname << "_f32_f32_ocp_data[3];\n";
        hdr << "extern const uint64_t arr_dmmv_id_" << tname << "_f32_f32_ocp_len[3];\n";
        إذا (basename(ننوت_filepath) == "mul_mat_vec.comp") {
            src << "const void * arr_dmmv_id_"   << tname << "_f32_f32_ocp_data[3] = {mul_mat_vec_id_" << tname << "_f32_f32_ocp_data, mul_mat_vec_id_" << tname << "_f32_f32_ocp_subgroup_data, mul_mat_vec_id_" << tname << "_f32_f32_ocp_subgroup_no_shmem_data};\n";
            src << "const uint64_t arr_dmmv_id_" << tname << "_f32_f32_ocp_len[3] =  {mul_mat_vec_id_" << tname << "_f32_f32_ocp_len,  mul_mat_vec_id_" << tname << "_f32_f32_ocp_subgroup_len, mul_mat_vec_id_"  << tname << "_f32_f32_ocp_subgroup_no_shmem_len};\n";
        }
    }
#د_که_پای

    إذا (ننوت_filepath == "") {
        write_file_if_بدل_شوی(موخه_hpp, hdr.str());
    }
    إذا (موخه_cpp != "") {
        write_دوګونی_file(موخه_cpp, src.str());
    }
}

} // namespace

صحيح اصلي(صحيح argc, حرف** argv) {
    std::map<std::string, std::string> دلیلونه_2;
    لكل (صحيح i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        إذا (arg.rfind("--", 0) == 0) {
            إذا (i + 1 < argc && argv[i + 1][0] != '-') {
                دلیلونه_2[arg] = argv[i + 1];
                ++i;
            } وإلا {
                دلیلونه_2[arg] = "";
            }
        }
    }

    إذا (دلیلونه_2.find("--glslc") != دلیلونه_2.end()) {
        GLSLC = دلیلونه_2["--glslc"]; // Path to glslc
    }
    إذا (دلیلونه_2.find("--source") != دلیلونه_2.end()) {
        ننوت_filepath = دلیلونه_2["--source"]; // The shader source file to compile
    }
    إذا (دلیلونه_2.find("--output-dir") != دلیلونه_2.end()) {
        output_dir = دلیلونه_2["--output-dir"]; // Directory for containing SPIR-V output
    }
    إذا (دلیلونه_2.find("--target-hpp") != دلیلونه_2.end()) {
        موخه_hpp = دلیلونه_2["--target-hpp"]; // Path to generated header file
    }
    إذا (دلیلونه_2.find("--target-cpp") != دلیلونه_2.end()) {
        موخه_cpp = دلیلونه_2["--target-cpp"]; // Path to generated cpp file
    }

    إذا (!directory_exists(output_dir)) {
        إذا (!create_directory(output_dir)) {
            std::cerr << "Error creating output directory: " << output_dir << "\n";
            أرجع EXIT_FAILURE;
        }
    }

    process_shaders();

    إذا (compile_failed) {
        std::cerr << "vulkan-shaders-gen: one or more shaders failed to compile" << std::کرښه_پای;
        أرجع EXIT_FAILURE;
    }

    write_output_files();

    أرجع EXIT_SUCCESS;
}
