#جيڪڏهن_وصف_نه_ٿيل PRE_WGSL_HPP
#وصف_ڪريو PRE_WGSL_HPP

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
    std::string              include_رستو = ".";
    std::vector<std::string> macros;
};

//==============================================================
// Utility: trim
//==============================================================
ساكن std::string trim(ثابت std::string & s) {
    size_t a = 0;
    طالما (a < s.size() && std::خال_آهي((بنانشان حرف) s[a])) {
        a++;
    }
    size_t b = s.size();
    طالما (b > a && std::خال_آهي((بنانشان حرف) s[b - 1])) {
        b--;
    }
    أرجع s.substr(a, b - a);
}

ساكن std::string trim_قدر(std::istream & is) {
    std::ostringstream ss;
    ss << is.rdbuf();
    أرجع trim(ss.str());
}

ساكن منطقي isIdentChar(حرف c) {
    أرجع std::اکر_يا_انگ(static_cast<بنانشان حرف>(c)) || c == '_';
}

ساكن منطقي پڇاڙيون_With_Continuation(ثابت std::string & سٽ) {
    size_t i = سٽ.size();
    طالما (i > 0 && std::خال_آهي((بنانشان حرف) سٽ[i - 1])) {
        i--;
    }
    أرجع i > 0 && سٽ[i - 1] == '\\';
}

ساكن فراغ stripContinuation(std::string & سٽ) {
    size_t i = سٽ.size();
    طالما (i > 0 && std::خال_آهي((بنانشان حرف) سٽ[i - 1])) {
        i--;
    }
    إذا (i > 0 && سٽ[i - 1] == '\\') {
        سٽ.erase(i - 1);
    }
}

ساكن std::string expand_Macros_Recursive_اندروني(ثابت std::string &                                  سٽ,
                                                 ثابت std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting);

ساكن std::string expand_Macro_قدر(ثابت std::string &                                  name,
                                    ثابت std::unordered_map<std::string, std::string> & macros,
                                    std::unordered_set<std::string> &                    visiting) {
    إذا (visiting.count(name)) {
        ارم std::runtime_error("Recursive macro: " + name);
    }
    visiting.insert(name);

    خودڪار it = macros.find(name);
    إذا (it == macros.end()) {
        visiting.erase(name);
        أرجع name;
    }

    ثابت std::string & value = it->second;
    إذا (value.خالي_جايون()) {
        visiting.erase(name);
        أرجع "";
    }

    std::string expanded = expand_Macros_Recursive_اندروني(value, macros, visiting);
    visiting.erase(name);
    أرجع expanded;
}

ساكن std::string expand_Macros_Recursive_اندروني(ثابت std::string &                                  سٽ,
                                                 ثابت std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting) {
    std::string نتيجو;
    نتيجو.reserve(سٽ.size());

    size_t i = 0;
    طالما (i < سٽ.size()) {
        إذا (isIdentChar(سٽ[i])) {
            size_t start = i;
            طالما (i < سٽ.size() && isIdentChar(سٽ[i])) {
                i++;
            }
            std::string token = سٽ.substr(start, i - start);

            خودڪار it = macros.find(token);
            إذا (it != macros.end()) {
                نتيجو += expand_Macro_قدر(token, macros, visiting);
            } وإلا {
                نتيجو += token;
            }
        } وإلا {
            نتيجو += سٽ[i];
            i++;
        }
    }

    أرجع نتيجو;
}

ساكن std::string expandMacrosRecursive(ثابت std::string &                                  سٽ,
                                         ثابت std::unordered_map<std::string, std::string> & macros) {
    std::unordered_set<std::string> visiting;
    أرجع expand_Macros_Recursive_اندروني(سٽ, macros, visiting);
}

//==============================================================
// Tokenizer for expressions in #if/#elif
//==============================================================
صنف اظهار_Lexer {
  عام:
    تعداد قسم_2 { پڇاڙي_3, IDENT, عدد_2, OP, LPAREN, RPAREN };

    بنية Tok {
        قسم_2        قسم;
        std::string متن_2;
    };

    واضح اظهار_Lexer(std::string_view sv) : src(sv), pos(0) {}

    Tok ايندڙ() {
        skipWS();
        إذا (pos >= src.size()) {
            أرجع { پڇاڙي_3, "" };
        }

        حرف c = src[pos];

        //عدد
        إذا (std::انگ_آهي((بنانشان حرف) c)) {
            size_t start = pos;
            طالما (pos < src.size() && std::انگ_آهي((بنانشان حرف) src[pos])) {
                pos++;
            }
            أرجع { عدد_2, std::string(src.substr(start, pos - start)) };
        }

        // identifier
        إذا (std::اکر_آهي((بنانشان حرف) c) || c == '_') {
            size_t start = pos;
            طالما (pos < src.size() && (std::اکر_يا_انگ((بنانشان حرف) src[pos]) || src[pos] == '_')) {
                pos++;
            }
            أرجع { IDENT, std::string(src.substr(start, pos - start)) };
        }

        إذا (c == '(') {
            pos++;
            أرجع { LPAREN, "(" };
        }
        إذا (c == ')') {
            pos++;
            أرجع { RPAREN, ")" };
        }

        // multi-char operators
        ساكن ثابت حرف * two_ops[] = { "==", "!=", "<=", ">=", "&&", "||", "<<", ">>" };
        لكل (خودڪار op : two_ops) {
            إذا (src.substr(pos, 2) == op) {
                pos += 2;
                أرجع { OP, std::string(op) };
            }
        }

        // single-char operators
        إذا (std::string("+-*/%<>!").find(c) != std::string::npos) {
            pos++;
            أرجع { OP, std::string(1, c) };
        }

        // unexpected
        pos++;
        أرجع { پڇاڙي_3, "" };
    }

  خاص:
    std::string_view src;
    size_t           pos;

    فراغ skipWS() {
        طالما (pos < src.size() && std::خال_آهي((بنانشان حرف) src[pos])) {
            pos++;
        }
    }
};

//==============================================================
// Expression Parser (recursive descent)
//==============================================================
صنف اظهار_Parser {
  عام:
    اظهار_Parser(std::string_view                                     اظهار,
               ثابت std::unordered_map<std::string, std::string> & macros,
               std::unordered_set<std::string> &                    visiting) :
        lex(اظهار),
        macros(macros),
        visiting(visiting) {
        advance();
    }

    صحيح parse() { أرجع parseLogicalOr(); }

  خاص:
    اظهار_Lexer                                            lex;
    اظهار_Lexer::Tok                                       tok;
    ثابت std::unordered_map<std::string, std::string> & macros;
    std::unordered_set<std::string> &                    visiting;

    فراغ advance() { tok = lex.ايندڙ(); }

    منطقي acceptOp(ثابت std::string & s) {
        إذا (tok.قسم == اظهار_Lexer::OP && tok.متن_2 == s) {
            advance();
            أرجع سچ;
        }
        أرجع خطأ;
    }

    منطقي accept_قسم(اظهار_Lexer::قسم_2 k) {
        إذا (tok.قسم == k) {
            advance();
            أرجع سچ;
        }
        أرجع خطأ;
    }

    صحيح parseLogicalOr() {
        صحيح v = parseLogicalAnd();
        طالما (acceptOp("||")) {
            صحيح rhs = parseLogicalAnd();
            v       = (v || rhs);
        }
        أرجع v;
    }

    صحيح parseLogicalAnd() {
        صحيح v = parseEquality();
        طالما (acceptOp("&&")) {
            صحيح rhs = parseEquality();
            v       = (v && rhs);
        }
        أرجع v;
    }

    صحيح parseEquality() {
        صحيح v = parseRelational();
        لكل (;;) {
            إذا (acceptOp("==")) {
                صحيح rhs = parseRelational();
                v       = (v == rhs);
            } وإلا إذا (acceptOp("!=")) {
                صحيح rhs = parseRelational();
                v       = (v != rhs);
            } وإلا {
                اكسر;
            }
        }
        أرجع v;
    }

    صحيح parseRelational() {
        صحيح v = parse_سرڪايو();
        لكل (;;) {
            إذا (acceptOp("<")) {
                صحيح rhs = parse_سرڪايو();
                v       = (v < rhs);
            } وإلا إذا (acceptOp(">")) {
                صحيح rhs = parse_سرڪايو();
                v       = (v > rhs);
            } وإلا إذا (acceptOp("<=")) {
                صحيح rhs = parse_سرڪايو();
                v       = (v <= rhs);
            } وإلا إذا (acceptOp(">=")) {
                صحيح rhs = parse_سرڪايو();
                v       = (v >= rhs);
            } وإلا {
                اكسر;
            }
        }
        أرجع v;
    }

    صحيح parse_سرڪايو() {
        صحيح v = parse_جوڙيو();
        لكل (;;) {
            إذا (acceptOp("<<")) {
                صحيح rhs = parse_جوڙيو();
                v       = (v << rhs);
            } وإلا إذا (acceptOp(">>")) {
                صحيح rhs = parse_جوڙيو();
                v       = (v >> rhs);
            } وإلا {
                اكسر;
            }
        }
        أرجع v;
    }

    صحيح parse_جوڙيو() {
        صحيح v = parseMult();
        لكل (;;) {
            إذا (acceptOp("+")) {
                صحيح rhs = parseMult();
                v       = (v + rhs);
            } وإلا إذا (acceptOp("-")) {
                صحيح rhs = parseMult();
                v       = (v - rhs);
            } وإلا {
                اكسر;
            }
        }
        أرجع v;
    }

    صحيح parseMult() {
        صحيح v = parseUnary();
        لكل (;;) {
            إذا (acceptOp("*")) {
                صحيح rhs = parseUnary();
                v       = (v * rhs);
            } وإلا إذا (acceptOp("/")) {
                صحيح rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v / rhs);
            } وإلا إذا (acceptOp("%")) {
                صحيح rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v % rhs);
            } وإلا {
                اكسر;
            }
        }
        أرجع v;
    }

    صحيح parseUnary() {
        إذا (acceptOp("!")) {
            أرجع !parseUnary();
        }
        إذا (acceptOp("-")) {
            أرجع -parseUnary();
        }
        إذا (acceptOp("+")) {
            أرجع +parseUnary();
        }
        أرجع parsePrimary();
    }

    صحيح parsePrimary() {
        // '(' expr ')'
        إذا (accept_قسم(اظهار_Lexer::LPAREN)) {
            صحيح v = parse();
            إذا (!accept_قسم(اظهار_Lexer::RPAREN)) {
                ارم std::runtime_error("missing ')'");
            }
            أرجع v;
        }

        //عدد
        إذا (tok.قسم == اظهار_Lexer::عدد_2) {
            صحيح v = std::stoi(tok.متن_2);
            advance();
            أرجع v;
        }

        // defined(identifier)
        إذا (tok.قسم == اظهار_Lexer::IDENT && tok.متن_2 == "defined") {
            advance();
            إذا (accept_قسم(اظهار_Lexer::LPAREN)) {
                إذا (tok.قسم != اظهار_Lexer::IDENT) {
                    ارم std::runtime_error("expected identifier in defined()");
                }
                std::string name = tok.متن_2;
                advance();
                إذا (!accept_قسم(اظهار_Lexer::RPAREN)) {
                    ارم std::runtime_error("missing ) in defined()");
                }
                أرجع macros.count(name) ? 1 : 0;
            } وإلا {
                // defined NAME
                إذا (tok.قسم != اظهار_Lexer::IDENT) {
                    ارم std::runtime_error("expected identifier in defined NAME");
                }
                std::string name = tok.متن_2;
                advance();
                أرجع macros.count(name) ? 1 : 0;
            }
        }

        // identifier -> treat as integer, if defined use its value else 0
        إذا (tok.قسم == اظهار_Lexer::IDENT) {
            std::string name = tok.متن_2;
            advance();
            خودڪار it = macros.find(name);
            إذا (it == macros.end()) {
                أرجع 0;
            }
            إذا (it->second.خالي_جايون()) {
                أرجع 1;
            }
            أرجع eval_Macro_اظهار(name, it->second);
        }

        // unexpected
        أرجع 0;
    }

    صحيح eval_Macro_اظهار(ثابت std::string & name, ثابت std::string & value) {
        إذا (visiting.count(name)) {
            ارم std::runtime_error("Recursive macro: " + name);
        }

        visiting.insert(name);
        اظهار_Parser ep(value, macros, visiting);
        صحيح        v = ep.parse();
        visiting.erase(name);
        أرجع v;
    }
};

//==============================================================
// Preprocessor
//==============================================================
صنف Preprocessor {
  عام:
    واضح Preprocessor(Options opts = {}) : opts_(std::move(opts)) {
        // Treat empty include path as current directory
        إذا (opts_.include_رستو.خالي_جايون()) {
            opts_.include_رستو = ".";
        }
        parseMacroDefinitions(opts_.macros);
    }

    std::string preprocess_file(ثابت std::string & filename, ثابت std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_ڍير;
        buildMacros(additional_macros, macros, predefined);

        std::string نتيجو = processFile(filename, macros, predefined, include_ڍير, Directive_گهڻائي::All);
        أرجع نتيجو;
    }

    std::string preprocess(ثابت std::string & contents, ثابت std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_ڍير;
        buildMacros(additional_macros, macros, predefined);

        std::string نتيجو = processString(contents, macros, predefined, include_ڍير, Directive_گهڻائي::All);
        أرجع نتيجو;
    }

    std::string preprocess_includes_file(ثابت std::string & filename) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_ڍير;
        std::string نتيجو = processFile(filename, macros, predefined, include_ڍير, Directive_گهڻائي::IncludesOnly);
        أرجع نتيجو;
    }

    std::string preprocess_includes(ثابت std::string & contents) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_ڍير;
        std::string نتيجو = processString(contents, macros, predefined, include_ڍير, Directive_گهڻائي::IncludesOnly);
        أرجع نتيجو;
    }

  خاص:
    Options                                      opts_;
    std::unordered_map<std::string, std::string> global_macros;

    تعداد صنف Directive_گهڻائي { All, IncludesOnly };

    بنية Cond {
        منطقي والد_سرگرم;
        منطقي سرگرم;
        منطقي ورتل;
    };

    //----------------------------------------------------------
    // Parse macro definitions into global_macros
    //----------------------------------------------------------
    فراغ parseMacroDefinitions(ثابت std::vector<std::string> & macro_defs) {
        لكل (ثابت خودڪار & def : macro_defs) {
            size_t eq_pos = def.find('=');
            إذا (eq_pos != std::string::npos) {
                // Format: NAME=VALUE
                std::string name    = trim(def.substr(0, eq_pos));
                std::string value   = trim(def.substr(eq_pos + 1));
                global_macros[name] = value;
            } وإلا {
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

        لكل (ثابت خودڪار & [name, value] : global_macros) {
            predefined.insert(name);
        }

        لكل (ثابت خودڪار & def : additional_macros) {
            size_t      eq_pos = def.find('=');
            std::string name, value;
            إذا (eq_pos != std::string::npos) {
                name  = trim(def.substr(0, eq_pos));
                value = trim(def.substr(eq_pos + 1));
            } وإلا {
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
    std::string ڀريو_File(ثابت std::string & fname) {
        std::ifstream f(fname);
        إذا (!f.is_open()) {
            ارم std::runtime_error("Could not open file: " + fname);
        }
        std::stringstream ss;
        ss << f.rdbuf();
        أرجع ss.str();
    }

    منطقي cond_سرگرم(ثابت std::vector<Cond> & cond) ثابت {
        إذا (cond.خالي_جايون()) {
            أرجع سچ;
        }
        أرجع cond.back().سرگرم;
    }

    //----------------------------------------------------------
    // Process a file
    //----------------------------------------------------------
    std::string processFile(ثابت std::string &                            name,
                            std::unordered_map<std::string, std::string> & macros,
                            ثابت std::unordered_set<std::string> &        predefined_macros,
                            std::unordered_set<std::string> &              include_ڍير,
                            Directive_گهڻائي                                  mode) {
        إذا (include_ڍير.count(name)) {
            ارم std::runtime_error("Recursive include: " + name);
        }

        include_ڍير.insert(name);
        std::string shader_ڪوڊ = ڀريو_File(name);
        std::string ٻاهريون         = processString(shader_ڪوڊ, macros, predefined_macros, include_ڍير, mode);
        include_ڍير.erase(name);
        أرجع ٻاهريون;
    }

    std::string processIncludeFile(ثابت std::string &                            fname,
                                   std::unordered_map<std::string, std::string> & macros,
                                   ثابت std::unordered_set<std::string> &        predefined_macros,
                                   std::unordered_set<std::string> &              include_ڍير,
                                   Directive_گهڻائي                                  mode) {
        std::string ڀريل_رستو = opts_.include_رستو + "/" + fname;
        أرجع processFile(ڀريل_رستو, macros, predefined_macros, include_ڍير, mode);
    }

    //----------------------------------------------------------
    // Process text
    //----------------------------------------------------------
    std::string processString(ثابت std::string &                            shader_ڪوڊ,
                              std::unordered_map<std::string, std::string> & macros,
                              ثابت std::unordered_set<std::string> &        predefined_macros,
                              std::unordered_set<std::string> &              include_ڍير,
                              Directive_گهڻائي                                  mode) {
        std::vector<Cond>  cond;  // Conditional stack for this shader
        std::stringstream  ٻاهريون;
        std::istringstream اندريون(shader_ڪوڊ);
        std::string        سٽ;

        طالما (std::getline(اندريون, سٽ)) {
            std::string logical = سٽ;
            std::string t       = trim(logical);
            إذا (!t.خالي_جايون() && t[0] == '#') {
                طالما (پڇاڙيون_With_Continuation(logical)) {
                    stripContinuation(logical);
                    إذا (!std::getline(اندريون, سٽ)) {
                        اكسر;
                    }
                    logical += "\n";
                    logical += سٽ;
                }
                t = trim(logical);
            }

            إذا (!t.خالي_جايون() && t[0] == '#') {
                منطقي handled = handleDirective(t, ٻاهريون, macros, predefined_macros, cond, include_ڍير, mode);
                إذا (mode == Directive_گهڻائي::IncludesOnly && !handled) {
                    ٻاهريون << logical << "\n";
                }
            } وإلا {
                إذا (mode == Directive_گهڻائي::IncludesOnly) {
                    ٻاهريون << logical << "\n";
                } وإلا إذا (cond_سرگرم(cond)) {
                    // Expand macros in the line before outputting
                    std::string expanded = expandMacrosRecursive(logical, macros);
                    ٻاهريون << expanded << "\n";
                }
            }
        }

        إذا (mode == Directive_گهڻائي::All && !cond.خالي_جايون()) {
            ارم std::runtime_error("Unclosed #if directive");
        }

        أرجع ٻاهريون.str();
    }

    //----------------------------------------------------------
    // Directive handler
    //----------------------------------------------------------
    منطقي handleDirective(ثابت std::string &                            t,
                         std::stringstream &                            ٻاهريون,
                         std::unordered_map<std::string, std::string> & macros,
                         ثابت std::unordered_set<std::string> &        predefined_macros,
                         std::vector<Cond> &                            cond,
                         std::unordered_set<std::string> &              include_ڍير,
                         Directive_گهڻائي                                  mode) {
        // split into tokens
        std::string        body = t.substr(1);
        std::istringstream iss(body);
        std::string        cmd;
        iss >> cmd;

        إذا (cmd == "include") {
            إذا (mode == Directive_گهڻائي::All && !cond_سرگرم(cond)) {
                أرجع سچ;
            }
            std::string file;
            iss >> file;
            إذا (file.size() >= 2 && file.front() == '"' && file.back() == '"') {
                file = file.substr(1, file.size() - 2);
            }
            ٻاهريون << processIncludeFile(file, macros, predefined_macros, include_ڍير, mode);
            أرجع سچ;
        }

        إذا (mode == Directive_گهڻائي::IncludesOnly) {
            أرجع خطأ;
        }

        إذا (cmd == "define") {
            إذا (!cond_سرگرم(cond)) {
                أرجع سچ;
            }
            std::string name;
            iss >> name;
            // Don't override predefined macros from options
            إذا (predefined_macros.count(name)) {
                أرجع سچ;
            }
            std::string value = trim_قدر(iss);
            macros[name]      = value;
            أرجع سچ;
        }

        إذا (cmd == "undef") {
            إذا (!cond_سرگرم(cond)) {
                أرجع سچ;
            }
            std::string name;
            iss >> name;
            // Don't undef predefined macros from options
            إذا (predefined_macros.count(name)) {
                أرجع سچ;
            }
            macros.erase(name);
            أرجع سچ;
        }

        إذا (cmd == "ifdef") {
            std::string name;
            iss >> name;
            منطقي p = cond_سرگرم(cond);
            منطقي v = macros.count(name);
            cond.push_back({ p, p && v, p && v });
            أرجع سچ;
        }

        إذا (cmd == "ifndef") {
            std::string name;
            iss >> name;
            منطقي p = cond_سرگرم(cond);
            منطقي v = !macros.count(name);
            cond.push_back({ p, p && v, p && v });
            أرجع سچ;
        }

        إذا (cmd == "if") {
            std::string اظهار = trim_قدر(iss);
            منطقي        p    = cond_سرگرم(cond);
            منطقي        v    = خطأ;
            إذا (p) {
                std::unordered_set<std::string> visiting;
                اظهار_Parser                      ep(اظهار, macros, visiting);
                v = ep.parse() != 0;
            }
            cond.push_back({ p, p && v, p && v });
            أرجع سچ;
        }

        إذا (cmd == "elif") {
            std::string اظهار = trim_قدر(iss);

            إذا (cond.خالي_جايون()) {
                ارم std::runtime_error("#elif without #if");
            }

            Cond & c = cond.back();
            إذا (!c.والد_سرگرم) {
                c.سرگرم = خطأ;
                أرجع سچ;
            }

            إذا (c.ورتل) {
                c.سرگرم = خطأ;
                أرجع سچ;
            }

            std::unordered_set<std::string> visiting;
            اظهار_Parser                      ep(اظهار, macros, visiting);
            منطقي                            v = ep.parse() != 0;
            c.سرگرم                          = v;
            إذا (v) {
                c.ورتل = سچ;
            }
            أرجع سچ;
        }

        إذا (cmd == "else") {
            إذا (cond.خالي_جايون()) {
                ارم std::runtime_error("#else without #if");
            }

            Cond & c = cond.back();
            إذا (!c.والد_سرگرم) {
                c.سرگرم = خطأ;
                أرجع سچ;
            }
            إذا (c.ورتل) {
                c.سرگرم = خطأ;
            } وإلا {
                c.سرگرم = سچ;
                c.ورتل  = سچ;
            }
            أرجع سچ;
        }

        إذا (cmd == "endif") {
            إذا (cond.خالي_جايون()) {
                ارم std::runtime_error("#endif without #if");
            }
            cond.pop_back();
            أرجع سچ;
        }

        // Unknown directive
        ارم std::runtime_error("Unknown directive: #" + cmd);
    }
};

}  // namespace pre_wgsl

#جيڪڏهن_پورو  // PRE_WGSL_HPP
