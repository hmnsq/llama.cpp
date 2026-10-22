আনযন os
আনযন re
আনযন argparse


def expand_includes(shader, প্ৰবেশ_dir):
    """
    Replace #include "file" lines in the text with the contents of that file.
    Searches for files relative to input_dir.
    """
    include_pattern = re.compile(r'^\s*#include\s+"([^"]+)"\s*$', re.MULTILINE)

    def replacer(match):
        fname = match.group(1)
        file_path = os.path.join(প্ৰবেশ_dir, fname)
        যদি নহয় os.path.exists(file_path):
            উত্থাপয FileNotFoundError(f"Included file not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") ইৱ f:
            included_code = f.পঠ()
        # Recursively expand includes inside the included file
        ঘূৰাই expand_includes(included_code, প্ৰবেশ_dir)

    ঘূৰাই include_pattern.sub(replacer, shader)


def chunk_shader(shader_code, max_chunk_len=60000):
    """Split shader_code into safe raw-string sized chunks."""
    ঘূৰাই [shader_code[i : i + max_chunk_len] বাবে i অন্তঃ range(0, len(shader_code), max_chunk_len)]


def raw_delim(shader_code):
    """Pick a raw-string delimiter that does not appear in the shader."""
    delim = "wgsl"
    যেতিয়ালৈ f"){delim}\"" অন্তঃ shader_code:
        delim += "_x"
    ঘূৰাই delim


def write_shader(shader_name, shader_code, output_dir, outfile, প্ৰবেশ_dir):
    shader_code = expand_includes(shader_code, প্ৰবেশ_dir)

    যদি output_dir:
        wgsl_filename = os.path.join(output_dir, f"{shader_name}.wgsl")
        with open(wgsl_filename, "w", encoding="utf-8") ইৱ f_out:
            f_out.লিখ(shader_code)

    delim = raw_delim(shader_code)
    chunks = chunk_shader(shader_code)

    যদি len(chunks) == 1:
        outfile.লিখ(f'const char* wgsl_{shader_name} = R"{delim}({shader_code}){delim}";\n\n')
    অন্যথা:
        বাবে idx, chunk অন্তঃ enumerate(chunks):
            outfile.লিখ(f'static const char wgsl_{shader_name}_part{idx}[] = R"{delim}({chunk}){delim}";\n\n')
        outfile.লিখ(f'static const std::string& wgsl_{shader_name}_str() {{\n')
        outfile.লিখ('    static const std::string s = []{\n')
        outfile.লিখ('        std::string tmp;\n')
        outfile.লিখ(f'        tmp.reserve({len(shader_code)});\n')
        বাবে idx অন্তঃ range(len(chunks)):
            outfile.লিখ(f'        tmp.append(wgsl_{shader_name}_part{idx});\n')
        outfile.লিখ('        return tmp;\n')
        outfile.লিখ('    }();\n')
        outfile.লিখ('    return s;\n')
        outfile.লিখ('}\n')
        outfile.লিখ(f'const char* wgsl_{shader_name} = wgsl_{shader_name}_str().c_str();\n\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=সঁচা)
    parser.add_argument("--output_file", required=সঁচা)
    args = parser.parse_args()

    with open(args.output_file, "w", encoding="utf-8") ইৱ out:
        out.লিখ("// Auto-generated shader embedding\n")
        out.লিখ("#include <string>\n\n")
        বাবে fname অন্তঃ sorted(os.listdir(args.প্ৰবেশ_dir)):
            যদি fname.endswith(".wgsl"):
                shader_path = os.path.join(args.প্ৰবেশ_dir, fname)
                shader_name = fname.replace(".wgsl", "")

                with open(shader_path, "r", encoding="utf-8") ইৱ f:
                    shader_code = f.পঠ()

                write_shader(shader_name, shader_code, None, out, args.প্ৰবেশ_dir)


যদি __name__ == "__main__":
    main()
