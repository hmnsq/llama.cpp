आनयन os
आनयन re
आनयन argparse


def expand_includes(shader, आगत_dir):
    """
    Replace #include "file" lines in the text with the contents of that file.
    Searches for files relative to input_dir.
    """
    include_pattern = re.compile(r'^\s*#include\s+"([^"]+)"\s*$', re.MULTILINE)

    def replacer(match):
        fname = match.group(1)
        file_path = os.path.join(आगत_dir, fname)
        यदि होइन os.path.exists(file_path):
            उत्थापय FileNotFoundError(f"Included file not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") इव f:
            included_code = f.पठ()
        # Recursively expand includes inside the included file
        फर्काऊ expand_includes(included_code, आगत_dir)

    फर्काऊ include_pattern.sub(replacer, shader)


def chunk_shader(shader_code, max_chunk_len=60000):
    """Split shader_code into safe raw-string sized chunks."""
    फर्काऊ [shader_code[i : i + max_chunk_len] लागि i अन्तः range(0, len(shader_code), max_chunk_len)]


def raw_delim(shader_code):
    """Pick a raw-string delimiter that does not appear in the shader."""
    delim = "wgsl"
    जबसम्म f"){delim}\"" अन्तः shader_code:
        delim += "_x"
    फर्काऊ delim


def write_shader(shader_name, shader_code, output_dir, outfile, आगत_dir):
    shader_code = expand_includes(shader_code, आगत_dir)

    यदि output_dir:
        wgsl_filename = os.path.join(output_dir, f"{shader_name}.wgsl")
        with open(wgsl_filename, "w", encoding="utf-8") इव f_out:
            f_out.लिख(shader_code)

    delim = raw_delim(shader_code)
    chunks = chunk_shader(shader_code)

    यदि len(chunks) == 1:
        outfile.लिख(f'const char* wgsl_{shader_name} = R"{delim}({shader_code}){delim}";\n\n')
    अन्यथा:
        लागि idx, chunk अन्तः enumerate(chunks):
            outfile.लिख(f'static const char wgsl_{shader_name}_part{idx}[] = R"{delim}({chunk}){delim}";\n\n')
        outfile.लिख(f'static const std::string& wgsl_{shader_name}_str() {{\n')
        outfile.लिख('    static const std::string s = []{\n')
        outfile.लिख('        std::string tmp;\n')
        outfile.लिख(f'        tmp.reserve({len(shader_code)});\n')
        लागि idx अन्तः range(len(chunks)):
            outfile.लिख(f'        tmp.append(wgsl_{shader_name}_part{idx});\n')
        outfile.लिख('        return tmp;\n')
        outfile.लिख('    }();\n')
        outfile.लिख('    return s;\n')
        outfile.लिख('}\n')
        outfile.लिख(f'const char* wgsl_{shader_name} = wgsl_{shader_name}_str().c_str();\n\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=सत्य)
    parser.add_argument("--output_file", required=सत्य)
    args = parser.parse_args()

    with open(args.output_file, "w", encoding="utf-8") इव out:
        out.लिख("// Auto-generated shader embedding\n")
        out.लिख("#include <string>\n\n")
        लागि fname अन्तः sorted(os.listdir(args.आगत_dir)):
            यदि fname.endswith(".wgsl"):
                shader_path = os.path.join(args.आगत_dir, fname)
                shader_name = fname.replace(".wgsl", "")

                with open(shader_path, "r", encoding="utf-8") इव f:
                    shader_code = f.पठ()

                write_shader(shader_name, shader_code, None, out, args.आगत_dir)


यदि __name__ == "__main__":
    main()
