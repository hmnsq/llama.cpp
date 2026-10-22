استورد os
استورد re
استورد argparse


def expand_includes(shader, مدخل_dir):
    """
    Replace #include "file" lines in the text with the contents of that file.
    Searches for files relative to input_dir.
    """
    include_pattern = re.compile(r'^\s*#include\s+"([^"]+)"\s*$', re.MULTILINE)

    def replacer(match):
        fname = match.group(1)
        file_path = os.path.join(مدخل_dir, fname)
        إذا ليس os.path.exists(file_path):
            ارفع FileNotFoundError(f"Included file not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") كـ f:
            included_code = f.اقرأ()
        # Recursively expand includes inside the included file
        أرجع expand_includes(included_code, مدخل_dir)

    أرجع include_pattern.sub(replacer, shader)


def chunk_shader(shader_code, max_chunk_len=60000):
    """Split shader_code into safe raw-string sized chunks."""
    أرجع [shader_code[i : i + max_chunk_len] لكل i في range(0, len(shader_code), max_chunk_len)]


def raw_delim(shader_code):
    """Pick a raw-string delimiter that does not appear in the shader."""
    delim = "wgsl"
    طالما f"){delim}\"" في shader_code:
        delim += "_x"
    أرجع delim


def write_shader(shader_name, shader_code, output_dir, outfile, مدخل_dir):
    shader_code = expand_includes(shader_code, مدخل_dir)

    إذا output_dir:
        wgsl_filename = os.path.join(output_dir, f"{shader_name}.wgsl")
        with open(wgsl_filename, "w", encoding="utf-8") كـ f_out:
            f_out.اكتب(shader_code)

    delim = raw_delim(shader_code)
    chunks = chunk_shader(shader_code)

    إذا len(chunks) == 1:
        outfile.اكتب(f'const char* wgsl_{shader_name} = R"{delim}({shader_code}){delim}";\n\n')
    وإلا:
        لكل idx, chunk في enumerate(chunks):
            outfile.اكتب(f'static const char wgsl_{shader_name}_part{idx}[] = R"{delim}({chunk}){delim}";\n\n')
        outfile.اكتب(f'static const std::string& wgsl_{shader_name}_str() {{\n')
        outfile.اكتب('    static const std::string s = []{\n')
        outfile.اكتب('        std::string tmp;\n')
        outfile.اكتب(f'        tmp.reserve({len(shader_code)});\n')
        لكل idx في range(len(chunks)):
            outfile.اكتب(f'        tmp.append(wgsl_{shader_name}_part{idx});\n')
        outfile.اكتب('        return tmp;\n')
        outfile.اكتب('    }();\n')
        outfile.اكتب('    return s;\n')
        outfile.اكتب('}\n')
        outfile.اكتب(f'const char* wgsl_{shader_name} = wgsl_{shader_name}_str().c_str();\n\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=صحيح)
    parser.add_argument("--output_file", required=صحيح)
    وسائط_2 = parser.parse_وسائط()

    with open(وسائط_2.output_file, "w", encoding="utf-8") كـ out:
        out.اكتب("// Auto-generated shader embedding\n")
        out.اكتب("#include <string>\n\n")
        لكل fname في sorted(os.listdir(وسائط_2.مدخل_dir)):
            إذا fname.endswith(".wgsl"):
                shader_path = os.path.join(وسائط_2.مدخل_dir, fname)
                shader_name = fname.replace(".wgsl", "")

                with open(shader_path, "r", encoding="utf-8") كـ f:
                    shader_code = f.اقرأ()

                write_shader(shader_name, shader_code, None, out, وسائط_2.مدخل_dir)


إذا __name__ == "__main__":
    main()
