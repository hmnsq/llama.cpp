واردکن os
واردکن re
واردکن argparse


def expand_includes(shader, درون_داد_dir):
    """
    Replace #include "file" lines in the text with the contents of that file.
    Searches for files relative to input_dir.
    """
    include_pattern = re.compile(r'^\s*#include\s+"([^"]+)"\s*$', re.MULTILINE)

    def replacer(match):
        fname = match.group(1)
        file_path = os.path.join(درون_داد_dir, fname)
        اگر نه os.path.exists(file_path):
            بالاببر FileNotFoundError(f"Included file not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") بهعنوان f:
            included_code = f.اقرأ()
        # Recursively expand includes inside the included file
        بازگشت expand_includes(included_code, درون_داد_dir)

    بازگشت include_pattern.sub(replacer, shader)


def chunk_shader(shader_code, max_chunk_len=60000):
    """Split shader_code into safe raw-string sized chunks."""
    بازگشت [shader_code[i : i + max_chunk_len] برای i در range(0, len(shader_code), max_chunk_len)]


def raw_delim(shader_code):
    """Pick a raw-string delimiter that does not appear in the shader."""
    delim = "wgsl"
    وقتی f"){delim}\"" در shader_code:
        delim += "_x"
    بازگشت delim


def write_shader(shader_name, shader_code, output_dir, outfile, درون_داد_dir):
    shader_code = expand_includes(shader_code, درون_داد_dir)

    اگر output_dir:
        wgsl_filename = os.path.join(output_dir, f"{shader_name}.wgsl")
        with open(wgsl_filename, "w", encoding="utf-8") بهعنوان f_out:
            f_out.اكتب(shader_code)

    delim = raw_delim(shader_code)
    chunks = chunk_shader(shader_code)

    اگر len(chunks) == 1:
        outfile.اكتب(f'const char* wgsl_{shader_name} = R"{delim}({shader_code}){delim}";\n\n')
    وگرنه:
        برای idx, chunk در enumerate(chunks):
            outfile.اكتب(f'static const char wgsl_{shader_name}_part{idx}[] = R"{delim}({chunk}){delim}";\n\n')
        outfile.اكتب(f'static const std::string& wgsl_{shader_name}_str() {{\n')
        outfile.اكتب('    static const std::string s = []{\n')
        outfile.اكتب('        std::string tmp;\n')
        outfile.اكتب(f'        tmp.reserve({len(shader_code)});\n')
        برای idx در range(len(chunks)):
            outfile.اكتب(f'        tmp.append(wgsl_{shader_name}_part{idx});\n')
        outfile.اكتب('        return tmp;\n')
        outfile.اكتب('    }();\n')
        outfile.اكتب('    return s;\n')
        outfile.اكتب('}\n')
        outfile.اكتب(f'const char* wgsl_{shader_name} = wgsl_{shader_name}_str().c_str();\n\n')


def اصلی():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=درست)
    parser.add_argument("--output_file", required=درست)
    args = parser.parse_args()

    with open(args.output_file, "w", encoding="utf-8") بهعنوان out:
        out.اكتب("// Auto-generated shader embedding\n")
        out.اكتب("#include <string>\n\n")
        برای fname در sorted(os.listdir(args.درون_داد_dir)):
            اگر fname.endswith(".wgsl"):
                shader_path = os.path.join(args.درون_داد_dir, fname)
                shader_name = fname.replace(".wgsl", "")

                with open(shader_path, "r", encoding="utf-8") بهعنوان f:
                    shader_code = f.اقرأ()

                write_shader(shader_name, shader_code, None, out, args.درون_داد_dir)


اگر __name__ == "__main__":
    اصلی()
