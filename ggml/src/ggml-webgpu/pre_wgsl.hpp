#ifndef PRE_WGSL_HPP
#define PRE_WGSL_HPP

#অন্তর্ভাব <cctype>
#অন্তর্ভাব <fstream>
#অন্তর্ভাব <sstream>
#অন্তর্ভাব <stdexcept>
#অন্তর্ভাব <string>
#অন্তর্ভাব <string_view>
#অন্তর্ভাব <unordered_map>
#অন্তর্ভাব <unordered_set>
#অন্তর্ভাব <vector>

নামাকাশ pre_wgsl {

//==============================================================
// Options
//==============================================================
কাঠামো Options {
    std::string              include_path = ".";
    std::vector<std::string> macros;
};

//==============================================================
// Utility: trim
//==============================================================
স্থির std::string trim(ধ্রুবক std::string & s) {
    size_t a = 0;
    যখন (a < s.size() && std::isspace((unsigned অক্ষর) s[a])) {
        a++;
    }
    size_t b = s.size();
    যখন (b > a && std::isspace((unsigned অক্ষর) s[b - 1])) {
        b--;
    }
    ফেরত s.substr(a, b - a);
}

স্থির std::string trim_মান(std::istream & is) {
    std::ostringstream ss;
    ss << is.rdbuf();
    ফেরত trim(ss.str());
}

স্থির যৌক্তিক isIdentChar(অক্ষর c) {
    ফেরত std::isalnum(static_cast<unsigned অক্ষর>(c)) || c == '_';
}

স্থির যৌক্তিক endsWithContinuation(ধ্রুবক std::string & সারি) {
    size_t i = সারি.size();
    যখন (i > 0 && std::isspace((unsigned অক্ষর) সারি[i - 1])) {
        i--;
    }
    ফেরত i > 0 && সারি[i - 1] == '\\';
}

স্থির শূন্য stripContinuation(std::string & সারি) {
    size_t i = সারি.size();
    যখন (i > 0 && std::isspace((unsigned অক্ষর) সারি[i - 1])) {
        i--;
    }
    যদি (i > 0 && সারি[i - 1] == '\\') {
        সারি.erase(i - 1);
    }
}

স্থির std::string expandMacrosRecursiveInternal(ধ্রুবক std::string &                                  সারি,
                                                 ধ্রুবক std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting);

স্থির std::string expand_Macro_মান(ধ্রুবক std::string &                                  name,
                                    ধ্রুবক std::unordered_map<std::string, std::string> & macros,
                                    std::unordered_set<std::string> &                    visiting) {
    যদি (visiting.count(name)) {
        ছোড়ো std::runtime_error("Recursive macro: " + name);
    }
    visiting.insert(name);

    auto it = macros.find(name);
    যদি (it == macros.end()) {
        visiting.erase(name);
        ফেরত name;
    }

    ধ্রুবক std::string & value = it->second;
    যদি (value.empty()) {
        visiting.erase(name);
        ফেরত "";
    }

    std::string expanded = expandMacrosRecursiveInternal(value, macros, visiting);
    visiting.erase(name);
    ফেরত expanded;
}

স্থির std::string expandMacrosRecursiveInternal(ধ্রুবক std::string &                                  সারি,
                                                 ধ্রুবক std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting) {
    std::string ফলাফল;
    ফলাফল.reserve(সারি.size());

    size_t i = 0;
    যখন (i < সারি.size()) {
        যদি (isIdentChar(সারি[i])) {
            size_t start = i;
            যখন (i < সারি.size() && isIdentChar(সারি[i])) {
                i++;
            }
            std::string token = সারি.substr(start, i - start);

            auto it = macros.find(token);
            যদি (it != macros.end()) {
                ফলাফল += expand_Macro_মান(token, macros, visiting);
            } অন্যথা {
                ফলাফল += token;
            }
        } অন্যথা {
            ফলাফল += সারি[i];
            i++;
        }
    }

    ফেরত ফলাফল;
}

স্থির std::string expandMacrosRecursive(ধ্রুবক std::string &                                  সারি,
                                         ধ্রুবক std::unordered_map<std::string, std::string> & macros) {
    std::unordered_set<std::string> visiting;
    ফেরত expandMacrosRecursiveInternal(সারি, macros, visiting);
}

//==============================================================
// Tokenizer for expressions in #if/#elif
//==============================================================
শ্রেণী রাশিমালা_Lexer {
  সর্বজনীন:
    গণনা Kind { END, IDENT, সংখ্যা_2, OP, LPAREN, RPAREN };

    কাঠামো Tok {
        Kind        kind;
        std::string text;
    };

    explicit রাশিমালা_Lexer(std::string_view sv) : src(sv), pos(0) {}

    Tok পরবর্তী() {
        skipWS();
        যদি (pos >= src.size()) {
            ফেরত { END, "" };
        }

        অক্ষর c = src[pos];

        //সংখ্যা
        যদি (std::isdigit((unsigned অক্ষর) c)) {
            size_t start = pos;
            যখন (pos < src.size() && std::isdigit((unsigned অক্ষর) src[pos])) {
                pos++;
            }
            ফেরত { সংখ্যা_2, std::string(src.substr(start, pos - start)) };
        }

        // identifier
        যদি (std::isalpha((unsigned অক্ষর) c) || c == '_') {
            size_t start = pos;
            যখন (pos < src.size() && (std::isalnum((unsigned অক্ষর) src[pos]) || src[pos] == '_')) {
                pos++;
            }
            ফেরত { IDENT, std::string(src.substr(start, pos - start)) };
        }

        যদি (c == '(') {
            pos++;
            ফেরত { LPAREN, "(" };
        }
        যদি (c == ')') {
            pos++;
            ফেরত { RPAREN, ")" };
        }

        // multi-char operators
        স্থির ধ্রুবক অক্ষর * two_ops[] = { "==", "!=", "<=", ">=", "&&", "||", "<<", ">>" };
        জন্য (auto op : two_ops) {
            যদি (src.substr(pos, 2) == op) {
                pos += 2;
                ফেরত { OP, std::string(op) };
            }
        }

        // single-char operators
        যদি (std::string("+-*/%<>!").find(c) != std::string::npos) {
            pos++;
            ফেরত { OP, std::string(1, c) };
        }

        // unexpected
        pos++;
        ফেরত { END, "" };
    }

  ব্যক্তিগত:
    std::string_view src;
    size_t           pos;

    শূন্য skipWS() {
        যখন (pos < src.size() && std::isspace((unsigned অক্ষর) src[pos])) {
            pos++;
        }
    }
};

//==============================================================
// Expression Parser (recursive descent)
//==============================================================
শ্রেণী রাশিমালা_Parser {
  সর্বজনীন:
    রাশিমালা_Parser(std::string_view                                     রাশিমালা,
               ধ্রুবক std::unordered_map<std::string, std::string> & macros,
               std::unordered_set<std::string> &                    visiting) :
        lex(রাশিমালা),
        macros(macros),
        visiting(visiting) {
        advance();
    }

    পূর্ণসংখ্যা parse() { ফেরত parseLogicalOr(); }

  ব্যক্তিগত:
    রাশিমালা_Lexer                                            lex;
    রাশিমালা_Lexer::Tok                                       tok;
    ধ্রুবক std::unordered_map<std::string, std::string> & macros;
    std::unordered_set<std::string> &                    visiting;

    শূন্য advance() { tok = lex.পরবর্তী(); }

    যৌক্তিক acceptOp(ধ্রুবক std::string & s) {
        যদি (tok.kind == রাশিমালা_Lexer::OP && tok.text == s) {
            advance();
            ফেরত সত্য;
        }
        ফেরত মিথ্যা;
    }

    যৌক্তিক acceptKind(রাশিমালা_Lexer::Kind k) {
        যদি (tok.kind == k) {
            advance();
            ফেরত সত্য;
        }
        ফেরত মিথ্যা;
    }

    পূর্ণসংখ্যা parseLogicalOr() {
        পূর্ণসংখ্যা v = parseLogicalAnd();
        যখন (acceptOp("||")) {
            পূর্ণসংখ্যা rhs = parseLogicalAnd();
            v       = (v || rhs);
        }
        ফেরত v;
    }

    পূর্ণসংখ্যা parseLogicalAnd() {
        পূর্ণসংখ্যা v = parseEquality();
        যখন (acceptOp("&&")) {
            পূর্ণসংখ্যা rhs = parseEquality();
            v       = (v && rhs);
        }
        ফেরত v;
    }

    পূর্ণসংখ্যা parseEquality() {
        পূর্ণসংখ্যা v = parseRelational();
        জন্য (;;) {
            যদি (acceptOp("==")) {
                পূর্ণসংখ্যা rhs = parseRelational();
                v       = (v == rhs);
            } অন্যথা যদি (acceptOp("!=")) {
                পূর্ণসংখ্যা rhs = parseRelational();
                v       = (v != rhs);
            } অন্যথা {
                বিরতি;
            }
        }
        ফেরত v;
    }

    পূর্ণসংখ্যা parseRelational() {
        পূর্ণসংখ্যা v = parseShift();
        জন্য (;;) {
            যদি (acceptOp("<")) {
                পূর্ণসংখ্যা rhs = parseShift();
                v       = (v < rhs);
            } অন্যথা যদি (acceptOp(">")) {
                পূর্ণসংখ্যা rhs = parseShift();
                v       = (v > rhs);
            } অন্যথা যদি (acceptOp("<=")) {
                পূর্ণসংখ্যা rhs = parseShift();
                v       = (v <= rhs);
            } অন্যথা যদি (acceptOp(">=")) {
                পূর্ণসংখ্যা rhs = parseShift();
                v       = (v >= rhs);
            } অন্যথা {
                বিরতি;
            }
        }
        ফেরত v;
    }

    পূর্ণসংখ্যা parseShift() {
        পূর্ণসংখ্যা v = parseAdd();
        জন্য (;;) {
            যদি (acceptOp("<<")) {
                পূর্ণসংখ্যা rhs = parseAdd();
                v       = (v << rhs);
            } অন্যথা যদি (acceptOp(">>")) {
                পূর্ণসংখ্যা rhs = parseAdd();
                v       = (v >> rhs);
            } অন্যথা {
                বিরতি;
            }
        }
        ফেরত v;
    }

    পূর্ণসংখ্যা parseAdd() {
        পূর্ণসংখ্যা v = parseMult();
        জন্য (;;) {
            যদি (acceptOp("+")) {
                পূর্ণসংখ্যা rhs = parseMult();
                v       = (v + rhs);
            } অন্যথা যদি (acceptOp("-")) {
                পূর্ণসংখ্যা rhs = parseMult();
                v       = (v - rhs);
            } অন্যথা {
                বিরতি;
            }
        }
        ফেরত v;
    }

    পূর্ণসংখ্যা parseMult() {
        পূর্ণসংখ্যা v = parseUnary();
        জন্য (;;) {
            যদি (acceptOp("*")) {
                পূর্ণসংখ্যা rhs = parseUnary();
                v       = (v * rhs);
            } অন্যথা যদি (acceptOp("/")) {
                পূর্ণসংখ্যা rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v / rhs);
            } অন্যথা যদি (acceptOp("%")) {
                পূর্ণসংখ্যা rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v % rhs);
            } অন্যথা {
                বিরতি;
            }
        }
        ফেরত v;
    }

    পূর্ণসংখ্যা parseUnary() {
        যদি (acceptOp("!")) {
            ফেরত !parseUnary();
        }
        যদি (acceptOp("-")) {
            ফেরত -parseUnary();
        }
        যদি (acceptOp("+")) {
            ফেরত +parseUnary();
        }
        ফেরত parsePrimary();
    }

    পূর্ণসংখ্যা parsePrimary() {
        // '(' expr ')'
        যদি (acceptKind(রাশিমালা_Lexer::LPAREN)) {
            পূর্ণসংখ্যা v = parse();
            যদি (!acceptKind(রাশিমালা_Lexer::RPAREN)) {
                ছোড়ো std::runtime_error("missing ')'");
            }
            ফেরত v;
        }

        //সংখ্যা
        যদি (tok.kind == রাশিমালা_Lexer::সংখ্যা_2) {
            পূর্ণসংখ্যা v = std::stoi(tok.text);
            advance();
            ফেরত v;
        }

        // defined(identifier)
        যদি (tok.kind == রাশিমালা_Lexer::IDENT && tok.text == "defined") {
            advance();
            যদি (acceptKind(রাশিমালা_Lexer::LPAREN)) {
                যদি (tok.kind != রাশিমালা_Lexer::IDENT) {
                    ছোড়ো std::runtime_error("expected identifier in defined()");
                }
                std::string name = tok.text;
                advance();
                যদি (!acceptKind(রাশিমালা_Lexer::RPAREN)) {
                    ছোড়ো std::runtime_error("missing ) in defined()");
                }
                ফেরত macros.count(name) ? 1 : 0;
            } অন্যথা {
                // defined NAME
                যদি (tok.kind != রাশিমালা_Lexer::IDENT) {
                    ছোড়ো std::runtime_error("expected identifier in defined NAME");
                }
                std::string name = tok.text;
                advance();
                ফেরত macros.count(name) ? 1 : 0;
            }
        }

        // identifier -> treat as integer, if defined use its value else 0
        যদি (tok.kind == রাশিমালা_Lexer::IDENT) {
            std::string name = tok.text;
            advance();
            auto it = macros.find(name);
            যদি (it == macros.end()) {
                ফেরত 0;
            }
            যদি (it->second.empty()) {
                ফেরত 1;
            }
            ফেরত evalMacroExpression(name, it->second);
        }

        // unexpected
        ফেরত 0;
    }

    পূর্ণসংখ্যা evalMacroExpression(ধ্রুবক std::string & name, ধ্রুবক std::string & value) {
        যদি (visiting.count(name)) {
            ছোড়ো std::runtime_error("Recursive macro: " + name);
        }

        visiting.insert(name);
        রাশিমালা_Parser ep(value, macros, visiting);
        পূর্ণসংখ্যা        v = ep.parse();
        visiting.erase(name);
        ফেরত v;
    }
};

//==============================================================
// Preprocessor
//==============================================================
শ্রেণী Preprocessor {
  সর্বজনীন:
    explicit Preprocessor(Options opts = {}) : opts_(std::move(opts)) {
        // Treat empty include path as current directory
        যদি (opts_.include_path.empty()) {
            opts_.include_path = ".";
        }
        parseMacroDefinitions(opts_.macros);
    }

    std::string preprocess_file(ধ্রুবক std::string & filename, ধ্রুবক std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_স্তূপ;
        buildMacros(additional_macros, macros, predefined);

        std::string ফলাফল = processFile(filename, macros, predefined, include_স্তূপ, DirectiveMode::All);
        ফেরত ফলাফল;
    }

    std::string preprocess(ধ্রুবক std::string & contents, ধ্রুবক std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_স্তূপ;
        buildMacros(additional_macros, macros, predefined);

        std::string ফলাফল = processString(contents, macros, predefined, include_স্তূপ, DirectiveMode::All);
        ফেরত ফলাফল;
    }

    std::string preprocess_includes_file(ধ্রুবক std::string & filename) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_স্তূপ;
        std::string ফলাফল = processFile(filename, macros, predefined, include_স্তূপ, DirectiveMode::IncludesOnly);
        ফেরত ফলাফল;
    }

    std::string preprocess_includes(ধ্রুবক std::string & contents) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_স্তূপ;
        std::string ফলাফল = processString(contents, macros, predefined, include_স্তূপ, DirectiveMode::IncludesOnly);
        ফেরত ফলাফল;
    }

  ব্যক্তিগত:
    Options                                      opts_;
    std::unordered_map<std::string, std::string> global_macros;

    গণনা শ্রেণী DirectiveMode { All, IncludesOnly };

    কাঠামো Cond {
        যৌক্তিক parent_active;
        যৌক্তিক active;
        যৌক্তিক taken;
    };

    //----------------------------------------------------------
    // Parse macro definitions into global_macros
    //----------------------------------------------------------
    শূন্য parseMacroDefinitions(ধ্রুবক std::vector<std::string> & macro_defs) {
        জন্য (ধ্রুবক auto & def : macro_defs) {
            size_t eq_pos = def.find('=');
            যদি (eq_pos != std::string::npos) {
                // Format: NAME=VALUE
                std::string name    = trim(def.substr(0, eq_pos));
                std::string value   = trim(def.substr(eq_pos + 1));
                global_macros[name] = value;
            } অন্যথা {
                // Format: NAME
                std::string name    = trim(def);
                global_macros[name] = "";
            }
        }
    }

    //----------------------------------------------------------
    // Build combined macro map and predefined set for a preprocessing operation
    //----------------------------------------------------------
    শূন্য buildMacros(ধ্রুবক std::vector<std::string> &               additional_macros,
                     std::unordered_map<std::string, std::string> & macros,
                     std::unordered_set<std::string> &              predefined) {
        macros = global_macros;
        predefined.clear();

        জন্য (ধ্রুবক auto & [name, value] : global_macros) {
            predefined.insert(name);
        }

        জন্য (ধ্রুবক auto & def : additional_macros) {
            size_t      eq_pos = def.find('=');
            std::string name, value;
            যদি (eq_pos != std::string::npos) {
                name  = trim(def.substr(0, eq_pos));
                value = trim(def.substr(eq_pos + 1));
            } অন্যথা {
                name  = trim(def);
                value = "";
            }

            // Add to macros map (will override global if same name)
            macros[name] = value;
            predefined.insert(name);
        }
    }

    //----------------------------------------------------------
    // Helpers
    //----------------------------------------------------------
    std::string loadFile(ধ্রুবক std::string & fname) {
        std::ifstream f(fname);
        যদি (!f.is_open()) {
            ছোড়ো std::runtime_error("Could not open file: " + fname);
        }
        std::stringstream ss;
        ss << f.rdbuf();
        ফেরত ss.str();
    }

    যৌক্তিক condActive(ধ্রুবক std::vector<Cond> & cond) ধ্রুবক {
        যদি (cond.empty()) {
            ফেরত সত্য;
        }
        ফেরত cond.back().active;
    }

    //----------------------------------------------------------
    // Process a file
    //----------------------------------------------------------
    std::string processFile(ধ্রুবক std::string &                            name,
                            std::unordered_map<std::string, std::string> & macros,
                            ধ্রুবক std::unordered_set<std::string> &        predefined_macros,
                            std::unordered_set<std::string> &              include_স্তূপ,
                            DirectiveMode                                  mode) {
        যদি (include_স্তূপ.count(name)) {
            ছোড়ো std::runtime_error("Recursive include: " + name);
        }

        include_স্তূপ.insert(name);
        std::string shader_code = loadFile(name);
        std::string out         = processString(shader_code, macros, predefined_macros, include_স্তূপ, mode);
        include_স্তূপ.erase(name);
        ফেরত out;
    }

    std::string processIncludeFile(ধ্রুবক std::string &                            fname,
                                   std::unordered_map<std::string, std::string> & macros,
                                   ধ্রুবক std::unordered_set<std::string> &        predefined_macros,
                                   std::unordered_set<std::string> &              include_স্তূপ,
                                   DirectiveMode                                  mode) {
        std::string full_path = opts_.include_path + "/" + fname;
        ফেরত processFile(full_path, macros, predefined_macros, include_স্তূপ, mode);
    }

    //----------------------------------------------------------
    // Process text
    //----------------------------------------------------------
    std::string processString(ধ্রুবক std::string &                            shader_code,
                              std::unordered_map<std::string, std::string> & macros,
                              ধ্রুবক std::unordered_set<std::string> &        predefined_macros,
                              std::unordered_set<std::string> &              include_স্তূপ,
                              DirectiveMode                                  mode) {
        std::vector<Cond>  cond;  // Conditional stack for this shader
        std::stringstream  out;
        std::istringstream in(shader_code);
        std::string        সারি;

        যখন (std::getline(in, সারি)) {
            std::string logical = সারি;
            std::string t       = trim(logical);
            যদি (!t.empty() && t[0] == '#') {
                যখন (endsWithContinuation(logical)) {
                    stripContinuation(logical);
                    যদি (!std::getline(in, সারি)) {
                        বিরতি;
                    }
                    logical += "\n";
                    logical += সারি;
                }
                t = trim(logical);
            }

            যদি (!t.empty() && t[0] == '#') {
                যৌক্তিক handled = handleDirective(t, out, macros, predefined_macros, cond, include_স্তূপ, mode);
                যদি (mode == DirectiveMode::IncludesOnly && !handled) {
                    out << logical << "\n";
                }
            } অন্যথা {
                যদি (mode == DirectiveMode::IncludesOnly) {
                    out << logical << "\n";
                } অন্যথা যদি (condActive(cond)) {
                    // Expand macros in the line before outputting
                    std::string expanded = expandMacrosRecursive(logical, macros);
                    out << expanded << "\n";
                }
            }
        }

        যদি (mode == DirectiveMode::All && !cond.empty()) {
            ছোড়ো std::runtime_error("Unclosed #if directive");
        }

        ফেরত out.str();
    }

    //----------------------------------------------------------
    // Directive handler
    //----------------------------------------------------------
    যৌক্তিক handleDirective(ধ্রুবক std::string &                            t,
                         std::stringstream &                            out,
                         std::unordered_map<std::string, std::string> & macros,
                         ধ্রুবক std::unordered_set<std::string> &        predefined_macros,
                         std::vector<Cond> &                            cond,
                         std::unordered_set<std::string> &              include_স্তূপ,
                         DirectiveMode                                  mode) {
        // split into tokens
        std::string        body = t.substr(1);
        std::istringstream iss(body);
        std::string        cmd;
        iss >> cmd;

        যদি (cmd == "include") {
            যদি (mode == DirectiveMode::All && !condActive(cond)) {
                ফেরত সত্য;
            }
            std::string file;
            iss >> file;
            যদি (file.size() >= 2 && file.front() == '"' && file.back() == '"') {
                file = file.substr(1, file.size() - 2);
            }
            out << processIncludeFile(file, macros, predefined_macros, include_স্তূপ, mode);
            ফেরত সত্য;
        }

        যদি (mode == DirectiveMode::IncludesOnly) {
            ফেরত মিথ্যা;
        }

        যদি (cmd == "define") {
            যদি (!condActive(cond)) {
                ফেরত সত্য;
            }
            std::string name;
            iss >> name;
            // Don't override predefined macros from options
            যদি (predefined_macros.count(name)) {
                ফেরত সত্য;
            }
            std::string value = trim_মান(iss);
            macros[name]      = value;
            ফেরত সত্য;
        }

        যদি (cmd == "undef") {
            যদি (!condActive(cond)) {
                ফেরত সত্য;
            }
            std::string name;
            iss >> name;
            // Don't undef predefined macros from options
            যদি (predefined_macros.count(name)) {
                ফেরত সত্য;
            }
            macros.erase(name);
            ফেরত সত্য;
        }

        যদি (cmd == "ifdef") {
            std::string name;
            iss >> name;
            যৌক্তিক p = condActive(cond);
            যৌক্তিক v = macros.count(name);
            cond.push_back({ p, p && v, p && v });
            ফেরত সত্য;
        }

        যদি (cmd == "ifndef") {
            std::string name;
            iss >> name;
            যৌক্তিক p = condActive(cond);
            যৌক্তিক v = !macros.count(name);
            cond.push_back({ p, p && v, p && v });
            ফেরত সত্য;
        }

        যদি (cmd == "if") {
            std::string রাশিমালা = trim_মান(iss);
            যৌক্তিক        p    = condActive(cond);
            যৌক্তিক        v    = মিথ্যা;
            যদি (p) {
                std::unordered_set<std::string> visiting;
                রাশিমালা_Parser                      ep(রাশিমালা, macros, visiting);
                v = ep.parse() != 0;
            }
            cond.push_back({ p, p && v, p && v });
            ফেরত সত্য;
        }

        যদি (cmd == "elif") {
            std::string রাশিমালা = trim_মান(iss);

            যদি (cond.empty()) {
                ছোড়ো std::runtime_error("#elif without #if");
            }

            Cond & c = cond.back();
            যদি (!c.parent_active) {
                c.active = মিথ্যা;
                ফেরত সত্য;
            }

            যদি (c.taken) {
                c.active = মিথ্যা;
                ফেরত সত্য;
            }

            std::unordered_set<std::string> visiting;
            রাশিমালা_Parser                      ep(রাশিমালা, macros, visiting);
            যৌক্তিক                            v = ep.parse() != 0;
            c.active                          = v;
            যদি (v) {
                c.taken = সত্য;
            }
            ফেরত সত্য;
        }

        যদি (cmd == "else") {
            যদি (cond.empty()) {
                ছোড়ো std::runtime_error("#else without #if");
            }

            Cond & c = cond.back();
            যদি (!c.parent_active) {
                c.active = মিথ্যা;
                ফেরত সত্য;
            }
            যদি (c.taken) {
                c.active = মিথ্যা;
            } অন্যথা {
                c.active = সত্য;
                c.taken  = সত্য;
            }
            ফেরত সত্য;
        }

        যদি (cmd == "endif") {
            যদি (cond.empty()) {
                ছোড়ো std::runtime_error("#endif without #if");
            }
            cond.pop_back();
            ফেরত সত্য;
        }

        // Unknown directive
        ছোড়ো std::runtime_error("Unknown directive: #" + cmd);
    }
};

}  // namespace pre_wgsl

#endif  // PRE_WGSL_HPP
