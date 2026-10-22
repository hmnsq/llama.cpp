#ifndef PRE_WGSL_HPP
#define PRE_WGSL_HPP

#ضمن <cctype>
#ضمن <fstream>
#ضمن <sstream>
#ضمن <stdexcept>
#ضمن <string>
#ضمن <string_view>
#ضمن <unordered_map>
#ضمن <unordered_set>
#ضمن <vector>

نطاق pre_wgsl {

//==============================================================
// Options
//==============================================================
بنية Options {
    std::string              include_path = ".";
    std::vector<std::string> macros;
};

//==============================================================
// Utility: trim
//==============================================================
ساكن std::string trim(ثابت std::string & s) {
    size_t a = 0;
    כלעוד (a < s.size() && std::isspace((unsigned حرف) s[a])) {
        a++;
    }
    size_t b = s.size();
    כלעוד (b > a && std::isspace((unsigned حرف) s[b - 1])) {
        b--;
    }
    החזר s.substr(a, b - a);
}

ساكن std::string trim_value(std::istream & is) {
    std::ostringstream ss;
    ss << is.rdbuf();
    החזר trim(ss.str());
}

ساكن منطقي isIdentChar(حرف c) {
    החזר std::isalnum(static_cast<unsigned حرف>(c)) || c == '_';
}

ساكن منطقي endsWithContinuation(ثابت std::string & line) {
    size_t i = line.size();
    כלעוד (i > 0 && std::isspace((unsigned حرف) line[i - 1])) {
        i--;
    }
    החזר i > 0 && line[i - 1] == '\\';
}

ساكن فراغ stripContinuation(std::string & line) {
    size_t i = line.size();
    כלעוד (i > 0 && std::isspace((unsigned حرف) line[i - 1])) {
        i--;
    }
    אם (i > 0 && line[i - 1] == '\\') {
        line.erase(i - 1);
    }
}

ساكن std::string expandMacrosRecursiveInternal(ثابت std::string &                                  line,
                                                 ثابت std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting);

ساكن std::string expandMacroValue(ثابت std::string &                                  name,
                                    ثابت std::unordered_map<std::string, std::string> & macros,
                                    std::unordered_set<std::string> &                    visiting) {
    אם (visiting.count(name)) {
        ارم std::runtime_error("Recursive macro: " + name);
    }
    visiting.insert(name);

    auto it = macros.find(name);
    אם (it == macros.end()) {
        visiting.erase(name);
        החזר name;
    }

    ثابت std::string & value = it->second;
    אם (value.empty()) {
        visiting.erase(name);
        החזר "";
    }

    std::string expanded = expandMacrosRecursiveInternal(value, macros, visiting);
    visiting.erase(name);
    החזר expanded;
}

ساكن std::string expandMacrosRecursiveInternal(ثابت std::string &                                  line,
                                                 ثابت std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting) {
    std::string result;
    result.reserve(line.size());

    size_t i = 0;
    כלעוד (i < line.size()) {
        אם (isIdentChar(line[i])) {
            size_t start = i;
            כלעוד (i < line.size() && isIdentChar(line[i])) {
                i++;
            }
            std::string token = line.substr(start, i - start);

            auto it = macros.find(token);
            אם (it != macros.end()) {
                result += expandMacroValue(token, macros, visiting);
            } אחרת {
                result += token;
            }
        } אחרת {
            result += line[i];
            i++;
        }
    }

    החזר result;
}

ساكن std::string expandMacrosRecursive(ثابت std::string &                                  line,
                                         ثابت std::unordered_map<std::string, std::string> & macros) {
    std::unordered_set<std::string> visiting;
    החזר expandMacrosRecursiveInternal(line, macros, visiting);
}

//==============================================================
// Tokenizer for expressions in #if/#elif
//==============================================================
מחלקה ExprLexer {
  عام:
    تعداد Kind { END, IDENT, NUMBER, OP, LPAREN, RPAREN };

    بنية Tok {
        Kind        kind;
        std::string text;
    };

    explicit ExprLexer(std::string_view sv) : src(sv), pos(0) {}

    Tok next() {
        skipWS();
        אם (pos >= src.size()) {
            החזר { END, "" };
        }

        حرف c = src[pos];

        // number
        אם (std::isdigit((unsigned حرف) c)) {
            size_t start = pos;
            כלעוד (pos < src.size() && std::isdigit((unsigned حرف) src[pos])) {
                pos++;
            }
            החזר { NUMBER, std::string(src.substr(start, pos - start)) };
        }

        // identifier
        אם (std::isalpha((unsigned حرف) c) || c == '_') {
            size_t start = pos;
            כלעוד (pos < src.size() && (std::isalnum((unsigned حرف) src[pos]) || src[pos] == '_')) {
                pos++;
            }
            החזר { IDENT, std::string(src.substr(start, pos - start)) };
        }

        אם (c == '(') {
            pos++;
            החזר { LPAREN, "(" };
        }
        אם (c == ')') {
            pos++;
            החזר { RPAREN, ")" };
        }

        // multi-char operators
        ساكن ثابت حرف * two_ops[] = { "==", "!=", "<=", ">=", "&&", "||", "<<", ">>" };
        עבור (auto op : two_ops) {
            אם (src.substr(pos, 2) == op) {
                pos += 2;
                החזר { OP, std::string(op) };
            }
        }

        // single-char operators
        אם (std::string("+-*/%<>!").find(c) != std::string::npos) {
            pos++;
            החזר { OP, std::string(1, c) };
        }

        // unexpected
        pos++;
        החזר { END, "" };
    }

  خاص:
    std::string_view src;
    size_t           pos;

    فراغ skipWS() {
        כלעוד (pos < src.size() && std::isspace((unsigned حرف) src[pos])) {
            pos++;
        }
    }
};

//==============================================================
// Expression Parser (recursive descent)
//==============================================================
מחלקה ExprParser {
  عام:
    ExprParser(std::string_view                                     expr,
               ثابت std::unordered_map<std::string, std::string> & macros,
               std::unordered_set<std::string> &                    visiting) :
        lex(expr),
        macros(macros),
        visiting(visiting) {
        advance();
    }

    صحيح parse() { החזר parseLogicalOr(); }

  خاص:
    ExprLexer                                            lex;
    ExprLexer::Tok                                       tok;
    ثابت std::unordered_map<std::string, std::string> & macros;
    std::unordered_set<std::string> &                    visiting;

    فراغ advance() { tok = lex.next(); }

    منطقي acceptOp(ثابت std::string & s) {
        אם (tok.kind == ExprLexer::OP && tok.text == s) {
            advance();
            החזר אמת;
        }
        החזר שקר;
    }

    منطقي acceptKind(ExprLexer::Kind k) {
        אם (tok.kind == k) {
            advance();
            החזר אמת;
        }
        החזר שקר;
    }

    صحيح parseLogicalOr() {
        صحيح v = parseLogicalAnd();
        כלעוד (acceptOp("||")) {
            صحيح rhs = parseLogicalAnd();
            v       = (v || rhs);
        }
        החזר v;
    }

    صحيح parseLogicalAnd() {
        صحيح v = parseEquality();
        כלעוד (acceptOp("&&")) {
            صحيح rhs = parseEquality();
            v       = (v && rhs);
        }
        החזר v;
    }

    صحيح parseEquality() {
        صحيح v = parseRelational();
        עבור (;;) {
            אם (acceptOp("==")) {
                صحيح rhs = parseRelational();
                v       = (v == rhs);
            } אחרת אם (acceptOp("!=")) {
                صحيح rhs = parseRelational();
                v       = (v != rhs);
            } אחרת {
                اكسر;
            }
        }
        החזר v;
    }

    صحيح parseRelational() {
        صحيح v = parseShift();
        עבור (;;) {
            אם (acceptOp("<")) {
                صحيح rhs = parseShift();
                v       = (v < rhs);
            } אחרת אם (acceptOp(">")) {
                صحيح rhs = parseShift();
                v       = (v > rhs);
            } אחרת אם (acceptOp("<=")) {
                صحيح rhs = parseShift();
                v       = (v <= rhs);
            } אחרת אם (acceptOp(">=")) {
                صحيح rhs = parseShift();
                v       = (v >= rhs);
            } אחרת {
                اكسر;
            }
        }
        החזר v;
    }

    صحيح parseShift() {
        صحيح v = parseAdd();
        עבור (;;) {
            אם (acceptOp("<<")) {
                صحيح rhs = parseAdd();
                v       = (v << rhs);
            } אחרת אם (acceptOp(">>")) {
                صحيح rhs = parseAdd();
                v       = (v >> rhs);
            } אחרת {
                اكسر;
            }
        }
        החזר v;
    }

    صحيح parseAdd() {
        صحيح v = parseMult();
        עבור (;;) {
            אם (acceptOp("+")) {
                صحيح rhs = parseMult();
                v       = (v + rhs);
            } אחרת אם (acceptOp("-")) {
                صحيح rhs = parseMult();
                v       = (v - rhs);
            } אחרת {
                اكسر;
            }
        }
        החזר v;
    }

    صحيح parseMult() {
        صحيح v = parseUnary();
        עבור (;;) {
            אם (acceptOp("*")) {
                صحيح rhs = parseUnary();
                v       = (v * rhs);
            } אחרת אם (acceptOp("/")) {
                صحيح rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v / rhs);
            } אחרת אם (acceptOp("%")) {
                صحيح rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v % rhs);
            } אחרת {
                اكسر;
            }
        }
        החזר v;
    }

    صحيح parseUnary() {
        אם (acceptOp("!")) {
            החזר !parseUnary();
        }
        אם (acceptOp("-")) {
            החזר -parseUnary();
        }
        אם (acceptOp("+")) {
            החזר +parseUnary();
        }
        החזר parsePrimary();
    }

    صحيح parsePrimary() {
        // '(' expr ')'
        אם (acceptKind(ExprLexer::LPAREN)) {
            صحيح v = parse();
            אם (!acceptKind(ExprLexer::RPAREN)) {
                ارم std::runtime_error("missing ')'");
            }
            החזר v;
        }

        // number
        אם (tok.kind == ExprLexer::NUMBER) {
            صحيح v = std::stoi(tok.text);
            advance();
            החזר v;
        }

        // defined(identifier)
        אם (tok.kind == ExprLexer::IDENT && tok.text == "defined") {
            advance();
            אם (acceptKind(ExprLexer::LPAREN)) {
                אם (tok.kind != ExprLexer::IDENT) {
                    ارم std::runtime_error("expected identifier in defined()");
                }
                std::string name = tok.text;
                advance();
                אם (!acceptKind(ExprLexer::RPAREN)) {
                    ارم std::runtime_error("missing ) in defined()");
                }
                החזר macros.count(name) ? 1 : 0;
            } אחרת {
                // defined NAME
                אם (tok.kind != ExprLexer::IDENT) {
                    ارم std::runtime_error("expected identifier in defined NAME");
                }
                std::string name = tok.text;
                advance();
                החזר macros.count(name) ? 1 : 0;
            }
        }

        // identifier -> treat as integer, if defined use its value else 0
        אם (tok.kind == ExprLexer::IDENT) {
            std::string name = tok.text;
            advance();
            auto it = macros.find(name);
            אם (it == macros.end()) {
                החזר 0;
            }
            אם (it->second.empty()) {
                החזר 1;
            }
            החזר evalMacroExpression(name, it->second);
        }

        // unexpected
        החזר 0;
    }

    صحيح evalMacroExpression(ثابت std::string & name, ثابت std::string & value) {
        אם (visiting.count(name)) {
            ارم std::runtime_error("Recursive macro: " + name);
        }

        visiting.insert(name);
        ExprParser ep(value, macros, visiting);
        صحيح        v = ep.parse();
        visiting.erase(name);
        החזר v;
    }
};

//==============================================================
// Preprocessor
//==============================================================
מחלקה Preprocessor {
  عام:
    explicit Preprocessor(Options opts = {}) : opts_(std::move(opts)) {
        // Treat empty include path as current directory
        אם (opts_.include_path.empty()) {
            opts_.include_path = ".";
        }
        parseMacroDefinitions(opts_.macros);
    }

    std::string preprocess_file(ثابت std::string & filename, ثابت std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        buildMacros(additional_macros, macros, predefined);

        std::string result = processFile(filename, macros, predefined, include_stack, DirectiveMode::All);
        החזר result;
    }

    std::string preprocess(ثابت std::string & contents, ثابت std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        buildMacros(additional_macros, macros, predefined);

        std::string result = processString(contents, macros, predefined, include_stack, DirectiveMode::All);
        החזר result;
    }

    std::string preprocess_includes_file(ثابت std::string & filename) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        std::string result = processFile(filename, macros, predefined, include_stack, DirectiveMode::IncludesOnly);
        החזר result;
    }

    std::string preprocess_includes(ثابت std::string & contents) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        std::string result = processString(contents, macros, predefined, include_stack, DirectiveMode::IncludesOnly);
        החזר result;
    }

  خاص:
    Options                                      opts_;
    std::unordered_map<std::string, std::string> global_macros;

    تعداد מחלקה DirectiveMode { All, IncludesOnly };

    بنية Cond {
        منطقي parent_active;
        منطقي active;
        منطقي taken;
    };

    //----------------------------------------------------------
    // Parse macro definitions into global_macros
    //----------------------------------------------------------
    فراغ parseMacroDefinitions(ثابت std::vector<std::string> & macro_defs) {
        עבור (ثابت auto & def : macro_defs) {
            size_t eq_pos = def.find('=');
            אם (eq_pos != std::string::npos) {
                // Format: NAME=VALUE
                std::string name    = trim(def.substr(0, eq_pos));
                std::string value   = trim(def.substr(eq_pos + 1));
                global_macros[name] = value;
            } אחרת {
                // Format: NAME
                std::string name    = trim(def);
                global_macros[name] = "";
            }
        }
    }

    //----------------------------------------------------------
    // Build combined macro map and predefined set for a preprocessing operation
    //----------------------------------------------------------
    فراغ buildMacros(ثابت std::vector<std::string> &               additional_macros,
                     std::unordered_map<std::string, std::string> & macros,
                     std::unordered_set<std::string> &              predefined) {
        macros = global_macros;
        predefined.clear();

        עבור (ثابت auto & [name, value] : global_macros) {
            predefined.insert(name);
        }

        עבור (ثابت auto & def : additional_macros) {
            size_t      eq_pos = def.find('=');
            std::string name, value;
            אם (eq_pos != std::string::npos) {
                name  = trim(def.substr(0, eq_pos));
                value = trim(def.substr(eq_pos + 1));
            } אחרת {
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
    std::string loadFile(ثابت std::string & fname) {
        std::ifstream f(fname);
        אם (!f.is_open()) {
            ارم std::runtime_error("Could not open file: " + fname);
        }
        std::stringstream ss;
        ss << f.rdbuf();
        החזר ss.str();
    }

    منطقي condActive(ثابت std::vector<Cond> & cond) ثابت {
        אם (cond.empty()) {
            החזר אמת;
        }
        החזר cond.back().active;
    }

    //----------------------------------------------------------
    // Process a file
    //----------------------------------------------------------
    std::string processFile(ثابت std::string &                            name,
                            std::unordered_map<std::string, std::string> & macros,
                            ثابت std::unordered_set<std::string> &        predefined_macros,
                            std::unordered_set<std::string> &              include_stack,
                            DirectiveMode                                  mode) {
        אם (include_stack.count(name)) {
            ارم std::runtime_error("Recursive include: " + name);
        }

        include_stack.insert(name);
        std::string shader_code = loadFile(name);
        std::string out         = processString(shader_code, macros, predefined_macros, include_stack, mode);
        include_stack.erase(name);
        החזר out;
    }

    std::string processIncludeFile(ثابت std::string &                            fname,
                                   std::unordered_map<std::string, std::string> & macros,
                                   ثابت std::unordered_set<std::string> &        predefined_macros,
                                   std::unordered_set<std::string> &              include_stack,
                                   DirectiveMode                                  mode) {
        std::string full_path = opts_.include_path + "/" + fname;
        החזר processFile(full_path, macros, predefined_macros, include_stack, mode);
    }

    //----------------------------------------------------------
    // Process text
    //----------------------------------------------------------
    std::string processString(ثابت std::string &                            shader_code,
                              std::unordered_map<std::string, std::string> & macros,
                              ثابت std::unordered_set<std::string> &        predefined_macros,
                              std::unordered_set<std::string> &              include_stack,
                              DirectiveMode                                  mode) {
        std::vector<Cond>  cond;  // Conditional stack for this shader
        std::stringstream  out;
        std::istringstream in(shader_code);
        std::string        line;

        כלעוד (std::getline(in, line)) {
            std::string logical = line;
            std::string t       = trim(logical);
            אם (!t.empty() && t[0] == '#') {
                כלעוד (endsWithContinuation(logical)) {
                    stripContinuation(logical);
                    אם (!std::getline(in, line)) {
                        اكسر;
                    }
                    logical += "\n";
                    logical += line;
                }
                t = trim(logical);
            }

            אם (!t.empty() && t[0] == '#') {
                منطقي handled = handleDirective(t, out, macros, predefined_macros, cond, include_stack, mode);
                אם (mode == DirectiveMode::IncludesOnly && !handled) {
                    out << logical << "\n";
                }
            } אחרת {
                אם (mode == DirectiveMode::IncludesOnly) {
                    out << logical << "\n";
                } אחרת אם (condActive(cond)) {
                    // Expand macros in the line before outputting
                    std::string expanded = expandMacrosRecursive(logical, macros);
                    out << expanded << "\n";
                }
            }
        }

        אם (mode == DirectiveMode::All && !cond.empty()) {
            ارم std::runtime_error("Unclosed #if directive");
        }

        החזר out.str();
    }

    //----------------------------------------------------------
    // Directive handler
    //----------------------------------------------------------
    منطقي handleDirective(ثابت std::string &                            t,
                         std::stringstream &                            out,
                         std::unordered_map<std::string, std::string> & macros,
                         ثابت std::unordered_set<std::string> &        predefined_macros,
                         std::vector<Cond> &                            cond,
                         std::unordered_set<std::string> &              include_stack,
                         DirectiveMode                                  mode) {
        // split into tokens
        std::string        body = t.substr(1);
        std::istringstream iss(body);
        std::string        cmd;
        iss >> cmd;

        אם (cmd == "include") {
            אם (mode == DirectiveMode::All && !condActive(cond)) {
                החזר אמת;
            }
            std::string file;
            iss >> file;
            אם (file.size() >= 2 && file.front() == '"' && file.back() == '"') {
                file = file.substr(1, file.size() - 2);
            }
            out << processIncludeFile(file, macros, predefined_macros, include_stack, mode);
            החזר אמת;
        }

        אם (mode == DirectiveMode::IncludesOnly) {
            החזר שקר;
        }

        אם (cmd == "define") {
            אם (!condActive(cond)) {
                החזר אמת;
            }
            std::string name;
            iss >> name;
            // Don't override predefined macros from options
            אם (predefined_macros.count(name)) {
                החזר אמת;
            }
            std::string value = trim_value(iss);
            macros[name]      = value;
            החזר אמת;
        }

        אם (cmd == "undef") {
            אם (!condActive(cond)) {
                החזר אמת;
            }
            std::string name;
            iss >> name;
            // Don't undef predefined macros from options
            אם (predefined_macros.count(name)) {
                החזר אמת;
            }
            macros.erase(name);
            החזר אמת;
        }

        אם (cmd == "ifdef") {
            std::string name;
            iss >> name;
            منطقي p = condActive(cond);
            منطقي v = macros.count(name);
            cond.push_back({ p, p && v, p && v });
            החזר אמת;
        }

        אם (cmd == "ifndef") {
            std::string name;
            iss >> name;
            منطقي p = condActive(cond);
            منطقي v = !macros.count(name);
            cond.push_back({ p, p && v, p && v });
            החזר אמת;
        }

        אם (cmd == "if") {
            std::string expr = trim_value(iss);
            منطقي        p    = condActive(cond);
            منطقي        v    = שקר;
            אם (p) {
                std::unordered_set<std::string> visiting;
                ExprParser                      ep(expr, macros, visiting);
                v = ep.parse() != 0;
            }
            cond.push_back({ p, p && v, p && v });
            החזר אמת;
        }

        אם (cmd == "elif") {
            std::string expr = trim_value(iss);

            אם (cond.empty()) {
                ارم std::runtime_error("#elif without #if");
            }

            Cond & c = cond.back();
            אם (!c.parent_active) {
                c.active = שקר;
                החזר אמת;
            }

            אם (c.taken) {
                c.active = שקר;
                החזר אמת;
            }

            std::unordered_set<std::string> visiting;
            ExprParser                      ep(expr, macros, visiting);
            منطقي                            v = ep.parse() != 0;
            c.active                          = v;
            אם (v) {
                c.taken = אמת;
            }
            החזר אמת;
        }

        אם (cmd == "else") {
            אם (cond.empty()) {
                ارم std::runtime_error("#else without #if");
            }

            Cond & c = cond.back();
            אם (!c.parent_active) {
                c.active = שקר;
                החזר אמת;
            }
            אם (c.taken) {
                c.active = שקר;
            } אחרת {
                c.active = אמת;
                c.taken  = אמת;
            }
            החזר אמת;
        }

        אם (cmd == "endif") {
            אם (cond.empty()) {
                ارم std::runtime_error("#endif without #if");
            }
            cond.pop_back();
            החזר אמת;
        }

        // Unknown directive
        ارم std::runtime_error("Unknown directive: #" + cmd);
    }
};

}  // namespace pre_wgsl

#endif  // PRE_WGSL_HPP
