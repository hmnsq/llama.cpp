#ifndef PRE_WGSL_HPP
#define PRE_WGSL_HPP

#شامل <cctype>
#شامل <fstream>
#شامل <sstream>
#شامل <stdexcept>
#شامل <string>
#شامل <string_view>
#شامل <unordered_map>
#شامل <unordered_set>
#شامل <vector>

فضاینام pre_wgsl {

//==============================================================
// Options
//==============================================================
ساختار Options {
    std::string              include_path = ".";
    std::vector<std::string> macros;
};

//==============================================================
// Utility: trim
//==============================================================
ایستا std::string trim(ثابت std::string & s) {
    size_t a = 0;
    وقتی (a < s.size() && std::isspace((unsigned نویسه) s[a])) {
        a++;
    }
    size_t b = s.size();
    وقتی (b > a && std::isspace((unsigned نویسه) s[b - 1])) {
        b--;
    }
    بازگشت s.substr(a, b - a);
}

ایستا std::string trim_مقدار(std::istream & is) {
    std::ostringstream ss;
    ss << is.rdbuf();
    بازگشت trim(ss.str());
}

ایستا منطقی isIdentChar(نویسه c) {
    بازگشت std::isalnum(static_cast<unsigned نویسه>(c)) || c == '_';
}

ایستا منطقی endsWithContinuation(ثابت std::string & سطر) {
    size_t i = سطر.size();
    وقتی (i > 0 && std::isspace((unsigned نویسه) سطر[i - 1])) {
        i--;
    }
    بازگشت i > 0 && سطر[i - 1] == '\\';
}

ایستا تهی stripContinuation(std::string & سطر) {
    size_t i = سطر.size();
    وقتی (i > 0 && std::isspace((unsigned نویسه) سطر[i - 1])) {
        i--;
    }
    اگر (i > 0 && سطر[i - 1] == '\\') {
        سطر.erase(i - 1);
    }
}

ایستا std::string expandMacrosRecursiveInternal(ثابت std::string &                                  سطر,
                                                 ثابت std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting);

ایستا std::string expand_Macro_مقدار(ثابت std::string &                                  name,
                                    ثابت std::unordered_map<std::string, std::string> & macros,
                                    std::unordered_set<std::string> &                    visiting) {
    اگر (visiting.count(name)) {
        پرتابکن std::runtime_error("Recursive macro: " + name);
    }
    visiting.insert(name);

    auto it = macros.find(name);
    اگر (it == macros.end()) {
        visiting.erase(name);
        بازگشت name;
    }

    ثابت std::string & value = it->second;
    اگر (value.empty()) {
        visiting.erase(name);
        بازگشت "";
    }

    std::string expanded = expandMacrosRecursiveInternal(value, macros, visiting);
    visiting.erase(name);
    بازگشت expanded;
}

ایستا std::string expandMacrosRecursiveInternal(ثابت std::string &                                  سطر,
                                                 ثابت std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting) {
    std::string result;
    result.reserve(سطر.size());

    size_t i = 0;
    وقتی (i < سطر.size()) {
        اگر (isIdentChar(سطر[i])) {
            size_t start = i;
            وقتی (i < سطر.size() && isIdentChar(سطر[i])) {
                i++;
            }
            std::string token = سطر.substr(start, i - start);

            auto it = macros.find(token);
            اگر (it != macros.end()) {
                result += expand_Macro_مقدار(token, macros, visiting);
            } وگرنه {
                result += token;
            }
        } وگرنه {
            result += سطر[i];
            i++;
        }
    }

    بازگشت result;
}

ایستا std::string expandMacrosRecursive(ثابت std::string &                                  سطر,
                                         ثابت std::unordered_map<std::string, std::string> & macros) {
    std::unordered_set<std::string> visiting;
    بازگشت expandMacrosRecursiveInternal(سطر, macros, visiting);
}

//==============================================================
// Tokenizer for expressions in #if/#elif
//==============================================================
رده عبارت_Lexer {
  عمومی:
    شمارش Kind { END, IDENT, عدد_2, OP, LPAREN, RPAREN };

    ساختار Tok {
        Kind        kind;
        std::string text;
    };

    explicit عبارت_Lexer(std::string_view sv) : src(sv), pos(0) {}

    Tok next() {
        skipWS();
        اگر (pos >= src.size()) {
            بازگشت { END, "" };
        }

        نویسه c = src[pos];

        //عدد
        اگر (std::isdigit((unsigned نویسه) c)) {
            size_t start = pos;
            وقتی (pos < src.size() && std::isdigit((unsigned نویسه) src[pos])) {
                pos++;
            }
            بازگشت { عدد_2, std::string(src.substr(start, pos - start)) };
        }

        // identifier
        اگر (std::isalpha((unsigned نویسه) c) || c == '_') {
            size_t start = pos;
            وقتی (pos < src.size() && (std::isalnum((unsigned نویسه) src[pos]) || src[pos] == '_')) {
                pos++;
            }
            بازگشت { IDENT, std::string(src.substr(start, pos - start)) };
        }

        اگر (c == '(') {
            pos++;
            بازگشت { LPAREN, "(" };
        }
        اگر (c == ')') {
            pos++;
            بازگشت { RPAREN, ")" };
        }

        // multi-char operators
        ایستا ثابت نویسه * two_ops[] = { "==", "!=", "<=", ">=", "&&", "||", "<<", ">>" };
        برای (auto op : two_ops) {
            اگر (src.substr(pos, 2) == op) {
                pos += 2;
                بازگشت { OP, std::string(op) };
            }
        }

        // single-char operators
        اگر (std::string("+-*/%<>!").find(c) != std::string::npos) {
            pos++;
            بازگشت { OP, std::string(1, c) };
        }

        // unexpected
        pos++;
        بازگشت { END, "" };
    }

  خصوصی:
    std::string_view src;
    size_t           pos;

    تهی skipWS() {
        وقتی (pos < src.size() && std::isspace((unsigned نویسه) src[pos])) {
            pos++;
        }
    }
};

//==============================================================
// Expression Parser (recursive descent)
//==============================================================
رده عبارت_Parser {
  عمومی:
    عبارت_Parser(std::string_view                                     عبارت,
               ثابت std::unordered_map<std::string, std::string> & macros,
               std::unordered_set<std::string> &                    visiting) :
        lex(عبارت),
        macros(macros),
        visiting(visiting) {
        advance();
    }

    صحیح parse() { بازگشت parseLogicalOr(); }

  خصوصی:
    عبارت_Lexer                                            lex;
    عبارت_Lexer::Tok                                       tok;
    ثابت std::unordered_map<std::string, std::string> & macros;
    std::unordered_set<std::string> &                    visiting;

    تهی advance() { tok = lex.next(); }

    منطقی acceptOp(ثابت std::string & s) {
        اگر (tok.kind == عبارت_Lexer::OP && tok.text == s) {
            advance();
            بازگشت درست;
        }
        بازگشت نادرست;
    }

    منطقی acceptKind(عبارت_Lexer::Kind k) {
        اگر (tok.kind == k) {
            advance();
            بازگشت درست;
        }
        بازگشت نادرست;
    }

    صحیح parseLogicalOr() {
        صحیح v = parseLogicalAnd();
        وقتی (acceptOp("||")) {
            صحیح rhs = parseLogicalAnd();
            v       = (v || rhs);
        }
        بازگشت v;
    }

    صحیح parseLogicalAnd() {
        صحیح v = parseEquality();
        وقتی (acceptOp("&&")) {
            صحیح rhs = parseEquality();
            v       = (v && rhs);
        }
        بازگشت v;
    }

    صحیح parseEquality() {
        صحیح v = parseRelational();
        برای (;;) {
            اگر (acceptOp("==")) {
                صحیح rhs = parseRelational();
                v       = (v == rhs);
            } وگرنه اگر (acceptOp("!=")) {
                صحیح rhs = parseRelational();
                v       = (v != rhs);
            } وگرنه {
                بشکن;
            }
        }
        بازگشت v;
    }

    صحیح parseRelational() {
        صحیح v = parseShift();
        برای (;;) {
            اگر (acceptOp("<")) {
                صحیح rhs = parseShift();
                v       = (v < rhs);
            } وگرنه اگر (acceptOp(">")) {
                صحیح rhs = parseShift();
                v       = (v > rhs);
            } وگرنه اگر (acceptOp("<=")) {
                صحیح rhs = parseShift();
                v       = (v <= rhs);
            } وگرنه اگر (acceptOp(">=")) {
                صحیح rhs = parseShift();
                v       = (v >= rhs);
            } وگرنه {
                بشکن;
            }
        }
        بازگشت v;
    }

    صحیح parseShift() {
        صحیح v = parseAdd();
        برای (;;) {
            اگر (acceptOp("<<")) {
                صحیح rhs = parseAdd();
                v       = (v << rhs);
            } وگرنه اگر (acceptOp(">>")) {
                صحیح rhs = parseAdd();
                v       = (v >> rhs);
            } وگرنه {
                بشکن;
            }
        }
        بازگشت v;
    }

    صحیح parseAdd() {
        صحیح v = parseMult();
        برای (;;) {
            اگر (acceptOp("+")) {
                صحیح rhs = parseMult();
                v       = (v + rhs);
            } وگرنه اگر (acceptOp("-")) {
                صحیح rhs = parseMult();
                v       = (v - rhs);
            } وگرنه {
                بشکن;
            }
        }
        بازگشت v;
    }

    صحیح parseMult() {
        صحیح v = parseUnary();
        برای (;;) {
            اگر (acceptOp("*")) {
                صحیح rhs = parseUnary();
                v       = (v * rhs);
            } وگرنه اگر (acceptOp("/")) {
                صحیح rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v / rhs);
            } وگرنه اگر (acceptOp("%")) {
                صحیح rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v % rhs);
            } وگرنه {
                بشکن;
            }
        }
        بازگشت v;
    }

    صحیح parseUnary() {
        اگر (acceptOp("!")) {
            بازگشت !parseUnary();
        }
        اگر (acceptOp("-")) {
            بازگشت -parseUnary();
        }
        اگر (acceptOp("+")) {
            بازگشت +parseUnary();
        }
        بازگشت parsePrimary();
    }

    صحیح parsePrimary() {
        // '(' expr ')'
        اگر (acceptKind(عبارت_Lexer::LPAREN)) {
            صحیح v = parse();
            اگر (!acceptKind(عبارت_Lexer::RPAREN)) {
                پرتابکن std::runtime_error("missing ')'");
            }
            بازگشت v;
        }

        //عدد
        اگر (tok.kind == عبارت_Lexer::عدد_2) {
            صحیح v = std::stoi(tok.text);
            advance();
            بازگشت v;
        }

        // defined(identifier)
        اگر (tok.kind == عبارت_Lexer::IDENT && tok.text == "defined") {
            advance();
            اگر (acceptKind(عبارت_Lexer::LPAREN)) {
                اگر (tok.kind != عبارت_Lexer::IDENT) {
                    پرتابکن std::runtime_error("expected identifier in defined()");
                }
                std::string name = tok.text;
                advance();
                اگر (!acceptKind(عبارت_Lexer::RPAREN)) {
                    پرتابکن std::runtime_error("missing ) in defined()");
                }
                بازگشت macros.count(name) ? 1 : 0;
            } وگرنه {
                // defined NAME
                اگر (tok.kind != عبارت_Lexer::IDENT) {
                    پرتابکن std::runtime_error("expected identifier in defined NAME");
                }
                std::string name = tok.text;
                advance();
                بازگشت macros.count(name) ? 1 : 0;
            }
        }

        // identifier -> treat as integer, if defined use its value else 0
        اگر (tok.kind == عبارت_Lexer::IDENT) {
            std::string name = tok.text;
            advance();
            auto it = macros.find(name);
            اگر (it == macros.end()) {
                بازگشت 0;
            }
            اگر (it->second.empty()) {
                بازگشت 1;
            }
            بازگشت evalMacroExpression(name, it->second);
        }

        // unexpected
        بازگشت 0;
    }

    صحیح evalMacroExpression(ثابت std::string & name, ثابت std::string & value) {
        اگر (visiting.count(name)) {
            پرتابکن std::runtime_error("Recursive macro: " + name);
        }

        visiting.insert(name);
        عبارت_Parser ep(value, macros, visiting);
        صحیح        v = ep.parse();
        visiting.erase(name);
        بازگشت v;
    }
};

//==============================================================
// Preprocessor
//==============================================================
رده Preprocessor {
  عمومی:
    explicit Preprocessor(Options opts = {}) : opts_(std::move(opts)) {
        // Treat empty include path as current directory
        اگر (opts_.include_path.empty()) {
            opts_.include_path = ".";
        }
        parseMacroDefinitions(opts_.macros);
    }

    std::string preprocess_file(ثابت std::string & filename, ثابت std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_پشته;
        buildMacros(additional_macros, macros, predefined);

        std::string result = processFile(filename, macros, predefined, include_پشته, DirectiveMode::All);
        بازگشت result;
    }

    std::string preprocess(ثابت std::string & contents, ثابت std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_پشته;
        buildMacros(additional_macros, macros, predefined);

        std::string result = processString(contents, macros, predefined, include_پشته, DirectiveMode::All);
        بازگشت result;
    }

    std::string preprocess_includes_file(ثابت std::string & filename) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_پشته;
        std::string result = processFile(filename, macros, predefined, include_پشته, DirectiveMode::IncludesOnly);
        بازگشت result;
    }

    std::string preprocess_includes(ثابت std::string & contents) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_پشته;
        std::string result = processString(contents, macros, predefined, include_پشته, DirectiveMode::IncludesOnly);
        بازگشت result;
    }

  خصوصی:
    Options                                      opts_;
    std::unordered_map<std::string, std::string> global_macros;

    شمارش رده DirectiveMode { All, IncludesOnly };

    ساختار Cond {
        منطقی parent_active;
        منطقی active;
        منطقی taken;
    };

    //----------------------------------------------------------
    // Parse macro definitions into global_macros
    //----------------------------------------------------------
    تهی parseMacroDefinitions(ثابت std::vector<std::string> & macro_defs) {
        برای (ثابت auto & def : macro_defs) {
            size_t eq_pos = def.find('=');
            اگر (eq_pos != std::string::npos) {
                // Format: NAME=VALUE
                std::string name    = trim(def.substr(0, eq_pos));
                std::string value   = trim(def.substr(eq_pos + 1));
                global_macros[name] = value;
            } وگرنه {
                // Format: NAME
                std::string name    = trim(def);
                global_macros[name] = "";
            }
        }
    }

    //----------------------------------------------------------
    // Build combined macro map and predefined set for a preprocessing operation
    //----------------------------------------------------------
    تهی buildMacros(ثابت std::vector<std::string> &               additional_macros,
                     std::unordered_map<std::string, std::string> & macros,
                     std::unordered_set<std::string> &              predefined) {
        macros = global_macros;
        predefined.clear();

        برای (ثابت auto & [name, value] : global_macros) {
            predefined.insert(name);
        }

        برای (ثابت auto & def : additional_macros) {
            size_t      eq_pos = def.find('=');
            std::string name, value;
            اگر (eq_pos != std::string::npos) {
                name  = trim(def.substr(0, eq_pos));
                value = trim(def.substr(eq_pos + 1));
            } وگرنه {
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
        اگر (!f.is_open()) {
            پرتابکن std::runtime_error("Could not open file: " + fname);
        }
        std::stringstream ss;
        ss << f.rdbuf();
        بازگشت ss.str();
    }

    منطقی condActive(ثابت std::vector<Cond> & cond) ثابت {
        اگر (cond.empty()) {
            بازگشت درست;
        }
        بازگشت cond.back().active;
    }

    //----------------------------------------------------------
    // Process a file
    //----------------------------------------------------------
    std::string processFile(ثابت std::string &                            name,
                            std::unordered_map<std::string, std::string> & macros,
                            ثابت std::unordered_set<std::string> &        predefined_macros,
                            std::unordered_set<std::string> &              include_پشته,
                            DirectiveMode                                  mode) {
        اگر (include_پشته.count(name)) {
            پرتابکن std::runtime_error("Recursive include: " + name);
        }

        include_پشته.insert(name);
        std::string shader_code = loadFile(name);
        std::string out         = processString(shader_code, macros, predefined_macros, include_پشته, mode);
        include_پشته.erase(name);
        بازگشت out;
    }

    std::string processIncludeFile(ثابت std::string &                            fname,
                                   std::unordered_map<std::string, std::string> & macros,
                                   ثابت std::unordered_set<std::string> &        predefined_macros,
                                   std::unordered_set<std::string> &              include_پشته,
                                   DirectiveMode                                  mode) {
        std::string full_path = opts_.include_path + "/" + fname;
        بازگشت processFile(full_path, macros, predefined_macros, include_پشته, mode);
    }

    //----------------------------------------------------------
    // Process text
    //----------------------------------------------------------
    std::string processString(ثابت std::string &                            shader_code,
                              std::unordered_map<std::string, std::string> & macros,
                              ثابت std::unordered_set<std::string> &        predefined_macros,
                              std::unordered_set<std::string> &              include_پشته,
                              DirectiveMode                                  mode) {
        std::vector<Cond>  cond;  // Conditional stack for this shader
        std::stringstream  out;
        std::istringstream in(shader_code);
        std::string        سطر;

        وقتی (std::getline(in, سطر)) {
            std::string logical = سطر;
            std::string t       = trim(logical);
            اگر (!t.empty() && t[0] == '#') {
                وقتی (endsWithContinuation(logical)) {
                    stripContinuation(logical);
                    اگر (!std::getline(in, سطر)) {
                        بشکن;
                    }
                    logical += "\n";
                    logical += سطر;
                }
                t = trim(logical);
            }

            اگر (!t.empty() && t[0] == '#') {
                منطقی handled = handleDirective(t, out, macros, predefined_macros, cond, include_پشته, mode);
                اگر (mode == DirectiveMode::IncludesOnly && !handled) {
                    out << logical << "\n";
                }
            } وگرنه {
                اگر (mode == DirectiveMode::IncludesOnly) {
                    out << logical << "\n";
                } وگرنه اگر (condActive(cond)) {
                    // Expand macros in the line before outputting
                    std::string expanded = expandMacrosRecursive(logical, macros);
                    out << expanded << "\n";
                }
            }
        }

        اگر (mode == DirectiveMode::All && !cond.empty()) {
            پرتابکن std::runtime_error("Unclosed #if directive");
        }

        بازگشت out.str();
    }

    //----------------------------------------------------------
    // Directive handler
    //----------------------------------------------------------
    منطقی handleDirective(ثابت std::string &                            t,
                         std::stringstream &                            out,
                         std::unordered_map<std::string, std::string> & macros,
                         ثابت std::unordered_set<std::string> &        predefined_macros,
                         std::vector<Cond> &                            cond,
                         std::unordered_set<std::string> &              include_پشته,
                         DirectiveMode                                  mode) {
        // split into tokens
        std::string        body = t.substr(1);
        std::istringstream iss(body);
        std::string        cmd;
        iss >> cmd;

        اگر (cmd == "include") {
            اگر (mode == DirectiveMode::All && !condActive(cond)) {
                بازگشت درست;
            }
            std::string file;
            iss >> file;
            اگر (file.size() >= 2 && file.front() == '"' && file.back() == '"') {
                file = file.substr(1, file.size() - 2);
            }
            out << processIncludeFile(file, macros, predefined_macros, include_پشته, mode);
            بازگشت درست;
        }

        اگر (mode == DirectiveMode::IncludesOnly) {
            بازگشت نادرست;
        }

        اگر (cmd == "define") {
            اگر (!condActive(cond)) {
                بازگشت درست;
            }
            std::string name;
            iss >> name;
            // Don't override predefined macros from options
            اگر (predefined_macros.count(name)) {
                بازگشت درست;
            }
            std::string value = trim_مقدار(iss);
            macros[name]      = value;
            بازگشت درست;
        }

        اگر (cmd == "undef") {
            اگر (!condActive(cond)) {
                بازگشت درست;
            }
            std::string name;
            iss >> name;
            // Don't undef predefined macros from options
            اگر (predefined_macros.count(name)) {
                بازگشت درست;
            }
            macros.erase(name);
            بازگشت درست;
        }

        اگر (cmd == "ifdef") {
            std::string name;
            iss >> name;
            منطقی p = condActive(cond);
            منطقی v = macros.count(name);
            cond.push_back({ p, p && v, p && v });
            بازگشت درست;
        }

        اگر (cmd == "ifndef") {
            std::string name;
            iss >> name;
            منطقی p = condActive(cond);
            منطقی v = !macros.count(name);
            cond.push_back({ p, p && v, p && v });
            بازگشت درست;
        }

        اگر (cmd == "if") {
            std::string عبارت = trim_مقدار(iss);
            منطقی        p    = condActive(cond);
            منطقی        v    = نادرست;
            اگر (p) {
                std::unordered_set<std::string> visiting;
                عبارت_Parser                      ep(عبارت, macros, visiting);
                v = ep.parse() != 0;
            }
            cond.push_back({ p, p && v, p && v });
            بازگشت درست;
        }

        اگر (cmd == "elif") {
            std::string عبارت = trim_مقدار(iss);

            اگر (cond.empty()) {
                پرتابکن std::runtime_error("#elif without #if");
            }

            Cond & c = cond.back();
            اگر (!c.parent_active) {
                c.active = نادرست;
                بازگشت درست;
            }

            اگر (c.taken) {
                c.active = نادرست;
                بازگشت درست;
            }

            std::unordered_set<std::string> visiting;
            عبارت_Parser                      ep(عبارت, macros, visiting);
            منطقی                            v = ep.parse() != 0;
            c.active                          = v;
            اگر (v) {
                c.taken = درست;
            }
            بازگشت درست;
        }

        اگر (cmd == "else") {
            اگر (cond.empty()) {
                پرتابکن std::runtime_error("#else without #if");
            }

            Cond & c = cond.back();
            اگر (!c.parent_active) {
                c.active = نادرست;
                بازگشت درست;
            }
            اگر (c.taken) {
                c.active = نادرست;
            } وگرنه {
                c.active = درست;
                c.taken  = درست;
            }
            بازگشت درست;
        }

        اگر (cmd == "endif") {
            اگر (cond.empty()) {
                پرتابکن std::runtime_error("#endif without #if");
            }
            cond.pop_back();
            بازگشت درست;
        }

        // Unknown directive
        پرتابکن std::runtime_error("Unknown directive: #" + cmd);
    }
};

}  // namespace pre_wgsl

#endif  // PRE_WGSL_HPP
