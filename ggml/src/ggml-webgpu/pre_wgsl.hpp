#اگر_تعریف_نشده PRE_WGSL_HPP
#تعریف_کن PRE_WGSL_HPP

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
    std::string              include_مسیر = ".";
    std::vector<std::string> macros;
};

//==============================================================
// Utility: trim
//==============================================================
ساكن std::string trim(ثابت std::string & s) {
    size_t a = 0;
    طالما (a < s.size() && std::آیا_فاصله((بیعلامت حرف) s[a])) {
        a++;
    }
    size_t b = s.size();
    طالما (b > a && std::آیا_فاصله((بیعلامت حرف) s[b - 1])) {
        b--;
    }
    أرجع s.substr(a, b - a);
}

ساكن std::string trim_مقدار(std::istream & is) {
    std::ostringstream ss;
    ss << is.rdbuf();
    أرجع trim(ss.str());
}

ساكن منطقي isIdentChar(حرف c) {
    أرجع std::آیا_حرف_یا_رقم(static_cast<بیعلامت حرف>(c)) || c == '_';
}

ساكن منطقي دوسر_With_Continuation(ثابت std::string & سطر) {
    size_t i = سطر.size();
    طالما (i > 0 && std::آیا_فاصله((بیعلامت حرف) سطر[i - 1])) {
        i--;
    }
    أرجع i > 0 && سطر[i - 1] == '\\';
}

ساكن فراغ stripContinuation(std::string & سطر) {
    size_t i = سطر.size();
    طالما (i > 0 && std::آیا_فاصله((بیعلامت حرف) سطر[i - 1])) {
        i--;
    }
    إذا (i > 0 && سطر[i - 1] == '\\') {
        سطر.erase(i - 1);
    }
}

ساكن std::string expand_Macros_Recursive_داخلی(ثابت std::string &                                  سطر,
                                                 ثابت std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting);

ساكن std::string expand_Macro_مقدار(ثابت std::string &                                  name,
                                    ثابت std::unordered_map<std::string, std::string> & macros,
                                    std::unordered_set<std::string> &                    visiting) {
    إذا (visiting.count(name)) {
        ارم std::runtime_error("Recursive macro: " + name);
    }
    visiting.insert(name);

    خودکار it = macros.find(name);
    إذا (it == macros.end()) {
        visiting.erase(name);
        أرجع name;
    }

    ثابت std::string & value = it->second;
    إذا (value.خالی()) {
        visiting.erase(name);
        أرجع "";
    }

    std::string expanded = expand_Macros_Recursive_داخلی(value, macros, visiting);
    visiting.erase(name);
    أرجع expanded;
}

ساكن std::string expand_Macros_Recursive_داخلی(ثابت std::string &                                  سطر,
                                                 ثابت std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting) {
    std::string نتیجه;
    نتیجه.reserve(سطر.size());

    size_t i = 0;
    طالما (i < سطر.size()) {
        إذا (isIdentChar(سطر[i])) {
            size_t start = i;
            طالما (i < سطر.size() && isIdentChar(سطر[i])) {
                i++;
            }
            std::string token = سطر.substr(start, i - start);

            خودکار it = macros.find(token);
            إذا (it != macros.end()) {
                نتیجه += expand_Macro_مقدار(token, macros, visiting);
            } وإلا {
                نتیجه += token;
            }
        } وإلا {
            نتیجه += سطر[i];
            i++;
        }
    }

    أرجع نتیجه;
}

ساكن std::string expandMacrosRecursive(ثابت std::string &                                  سطر,
                                         ثابت std::unordered_map<std::string, std::string> & macros) {
    std::unordered_set<std::string> visiting;
    أرجع expand_Macros_Recursive_داخلی(سطر, macros, visiting);
}

//==============================================================
// Tokenizer for expressions in #if/#elif
//==============================================================
صنف عبارت_Lexer {
  عام:
    تعداد نوع_3 { انتها_3, IDENT, عدد_2, OP, LPAREN, RPAREN };

    بنية Tok {
        نوع_3        نوع_2;
        std::string متن_2;
    };

    صریح عبارت_Lexer(std::string_view sv) : src(sv), pos(0) {}

    Tok بعدی() {
        skipWS();
        إذا (pos >= src.size()) {
            أرجع { انتها_3, "" };
        }

        حرف c = src[pos];

        //عدد
        إذا (std::آیا_رقم((بیعلامت حرف) c)) {
            size_t start = pos;
            طالما (pos < src.size() && std::آیا_رقم((بیعلامت حرف) src[pos])) {
                pos++;
            }
            أرجع { عدد_2, std::string(src.substr(start, pos - start)) };
        }

        // identifier
        إذا (std::آیا_حرف((بیعلامت حرف) c) || c == '_') {
            size_t start = pos;
            طالما (pos < src.size() && (std::آیا_حرف_یا_رقم((بیعلامت حرف) src[pos]) || src[pos] == '_')) {
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
        لكل (خودکار op : two_ops) {
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
        أرجع { انتها_3, "" };
    }

  خاص:
    std::string_view src;
    size_t           pos;

    فراغ skipWS() {
        طالما (pos < src.size() && std::آیا_فاصله((بیعلامت حرف) src[pos])) {
            pos++;
        }
    }
};

//==============================================================
// Expression Parser (recursive descent)
//==============================================================
صنف عبارت_Parser {
  عام:
    عبارت_Parser(std::string_view                                     عبارت,
               ثابت std::unordered_map<std::string, std::string> & macros,
               std::unordered_set<std::string> &                    visiting) :
        lex(عبارت),
        macros(macros),
        visiting(visiting) {
        advance();
    }

    صحيح parse() { أرجع parseLogicalOr(); }

  خاص:
    عبارت_Lexer                                            lex;
    عبارت_Lexer::Tok                                       tok;
    ثابت std::unordered_map<std::string, std::string> & macros;
    std::unordered_set<std::string> &                    visiting;

    فراغ advance() { tok = lex.بعدی(); }

    منطقي acceptOp(ثابت std::string & s) {
        إذا (tok.نوع_2 == عبارت_Lexer::OP && tok.متن_2 == s) {
            advance();
            أرجع درست;
        }
        أرجع خطأ;
    }

    منطقي accept_نوع(عبارت_Lexer::نوع_3 k) {
        إذا (tok.نوع_2 == k) {
            advance();
            أرجع درست;
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
        صحيح v = parse_لغزش();
        لكل (;;) {
            إذا (acceptOp("<")) {
                صحيح rhs = parse_لغزش();
                v       = (v < rhs);
            } وإلا إذا (acceptOp(">")) {
                صحيح rhs = parse_لغزش();
                v       = (v > rhs);
            } وإلا إذا (acceptOp("<=")) {
                صحيح rhs = parse_لغزش();
                v       = (v <= rhs);
            } وإلا إذا (acceptOp(">=")) {
                صحيح rhs = parse_لغزش();
                v       = (v >= rhs);
            } وإلا {
                اكسر;
            }
        }
        أرجع v;
    }

    صحيح parse_لغزش() {
        صحيح v = parse_جمع_کن();
        لكل (;;) {
            إذا (acceptOp("<<")) {
                صحيح rhs = parse_جمع_کن();
                v       = (v << rhs);
            } وإلا إذا (acceptOp(">>")) {
                صحيح rhs = parse_جمع_کن();
                v       = (v >> rhs);
            } وإلا {
                اكسر;
            }
        }
        أرجع v;
    }

    صحيح parse_جمع_کن() {
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
        إذا (accept_نوع(عبارت_Lexer::LPAREN)) {
            صحيح v = parse();
            إذا (!accept_نوع(عبارت_Lexer::RPAREN)) {
                ارم std::runtime_error("missing ')'");
            }
            أرجع v;
        }

        //عدد
        إذا (tok.نوع_2 == عبارت_Lexer::عدد_2) {
            صحيح v = std::stoi(tok.متن_2);
            advance();
            أرجع v;
        }

        // defined(identifier)
        إذا (tok.نوع_2 == عبارت_Lexer::IDENT && tok.متن_2 == "defined") {
            advance();
            إذا (accept_نوع(عبارت_Lexer::LPAREN)) {
                إذا (tok.نوع_2 != عبارت_Lexer::IDENT) {
                    ارم std::runtime_error("expected identifier in defined()");
                }
                std::string name = tok.متن_2;
                advance();
                إذا (!accept_نوع(عبارت_Lexer::RPAREN)) {
                    ارم std::runtime_error("missing ) in defined()");
                }
                أرجع macros.count(name) ? 1 : 0;
            } وإلا {
                // defined NAME
                إذا (tok.نوع_2 != عبارت_Lexer::IDENT) {
                    ارم std::runtime_error("expected identifier in defined NAME");
                }
                std::string name = tok.متن_2;
                advance();
                أرجع macros.count(name) ? 1 : 0;
            }
        }

        // identifier -> treat as integer, if defined use its value else 0
        إذا (tok.نوع_2 == عبارت_Lexer::IDENT) {
            std::string name = tok.متن_2;
            advance();
            خودکار it = macros.find(name);
            إذا (it == macros.end()) {
                أرجع 0;
            }
            إذا (it->second.خالی()) {
                أرجع 1;
            }
            أرجع eval_Macro_عبارت(name, it->second);
        }

        // unexpected
        أرجع 0;
    }

    صحيح eval_Macro_عبارت(ثابت std::string & name, ثابت std::string & value) {
        إذا (visiting.count(name)) {
            ارم std::runtime_error("Recursive macro: " + name);
        }

        visiting.insert(name);
        عبارت_Parser ep(value, macros, visiting);
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
    صریح Preprocessor(Options opts = {}) : opts_(std::move(opts)) {
        // Treat empty include path as current directory
        إذا (opts_.include_مسیر.خالی()) {
            opts_.include_مسیر = ".";
        }
        parseMacroDefinitions(opts_.macros);
    }

    std::string preprocess_file(ثابت std::string & filename, ثابت std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_پشته;
        buildMacros(additional_macros, macros, predefined);

        std::string نتیجه = processFile(filename, macros, predefined, include_پشته, Directive_نما::All);
        أرجع نتیجه;
    }

    std::string preprocess(ثابت std::string & contents, ثابت std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_پشته;
        buildMacros(additional_macros, macros, predefined);

        std::string نتیجه = processString(contents, macros, predefined, include_پشته, Directive_نما::All);
        أرجع نتیجه;
    }

    std::string preprocess_includes_file(ثابت std::string & filename) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_پشته;
        std::string نتیجه = processFile(filename, macros, predefined, include_پشته, Directive_نما::IncludesOnly);
        أرجع نتیجه;
    }

    std::string preprocess_includes(ثابت std::string & contents) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_پشته;
        std::string نتیجه = processString(contents, macros, predefined, include_پشته, Directive_نما::IncludesOnly);
        أرجع نتیجه;
    }

  خاص:
    Options                                      opts_;
    std::unordered_map<std::string, std::string> global_macros;

    تعداد صنف Directive_نما { All, IncludesOnly };

    بنية Cond {
        منطقي والد_فعال;
        منطقي فعال;
        منطقي گرفته;
    };

    //----------------------------------------------------------
    // Parse macro definitions into global_macros
    //----------------------------------------------------------
    فراغ parseMacroDefinitions(ثابت std::vector<std::string> & macro_defs) {
        لكل (ثابت خودکار & def : macro_defs) {
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

        لكل (ثابت خودکار & [name, value] : global_macros) {
            predefined.insert(name);
        }

        لكل (ثابت خودکار & def : additional_macros) {
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
    std::string بارگیری_File(ثابت std::string & fname) {
        std::ifstream f(fname);
        إذا (!f.is_open()) {
            ارم std::runtime_error("Could not open file: " + fname);
        }
        std::stringstream ss;
        ss << f.rdbuf();
        أرجع ss.str();
    }

    منطقي cond_فعال(ثابت std::vector<Cond> & cond) ثابت {
        إذا (cond.خالی()) {
            أرجع درست;
        }
        أرجع cond.back().فعال;
    }

    //----------------------------------------------------------
    // Process a file
    //----------------------------------------------------------
    std::string processFile(ثابت std::string &                            name,
                            std::unordered_map<std::string, std::string> & macros,
                            ثابت std::unordered_set<std::string> &        predefined_macros,
                            std::unordered_set<std::string> &              include_پشته,
                            Directive_نما                                  mode) {
        إذا (include_پشته.count(name)) {
            ارم std::runtime_error("Recursive include: " + name);
        }

        include_پشته.insert(name);
        std::string shader_کد = بارگیری_File(name);
        std::string بیرون         = processString(shader_کد, macros, predefined_macros, include_پشته, mode);
        include_پشته.erase(name);
        أرجع بیرون;
    }

    std::string processIncludeFile(ثابت std::string &                            fname,
                                   std::unordered_map<std::string, std::string> & macros,
                                   ثابت std::unordered_set<std::string> &        predefined_macros,
                                   std::unordered_set<std::string> &              include_پشته,
                                   Directive_نما                                  mode) {
        std::string پر_مسیر = opts_.include_مسیر + "/" + fname;
        أرجع processFile(پر_مسیر, macros, predefined_macros, include_پشته, mode);
    }

    //----------------------------------------------------------
    // Process text
    //----------------------------------------------------------
    std::string processString(ثابت std::string &                            shader_کد,
                              std::unordered_map<std::string, std::string> & macros,
                              ثابت std::unordered_set<std::string> &        predefined_macros,
                              std::unordered_set<std::string> &              include_پشته,
                              Directive_نما                                  mode) {
        std::vector<Cond>  cond;  // Conditional stack for this shader
        std::stringstream  بیرون;
        std::istringstream درون(shader_کد);
        std::string        سطر;

        طالما (std::getline(درون, سطر)) {
            std::string logical = سطر;
            std::string t       = trim(logical);
            إذا (!t.خالی() && t[0] == '#') {
                طالما (دوسر_With_Continuation(logical)) {
                    stripContinuation(logical);
                    إذا (!std::getline(درون, سطر)) {
                        اكسر;
                    }
                    logical += "\n";
                    logical += سطر;
                }
                t = trim(logical);
            }

            إذا (!t.خالی() && t[0] == '#') {
                منطقي handled = handleDirective(t, بیرون, macros, predefined_macros, cond, include_پشته, mode);
                إذا (mode == Directive_نما::IncludesOnly && !handled) {
                    بیرون << logical << "\n";
                }
            } وإلا {
                إذا (mode == Directive_نما::IncludesOnly) {
                    بیرون << logical << "\n";
                } وإلا إذا (cond_فعال(cond)) {
                    // Expand macros in the line before outputting
                    std::string expanded = expandMacrosRecursive(logical, macros);
                    بیرون << expanded << "\n";
                }
            }
        }

        إذا (mode == Directive_نما::All && !cond.خالی()) {
            ارم std::runtime_error("Unclosed #if directive");
        }

        أرجع بیرون.str();
    }

    //----------------------------------------------------------
    // Directive handler
    //----------------------------------------------------------
    منطقي handleDirective(ثابت std::string &                            t,
                         std::stringstream &                            بیرون,
                         std::unordered_map<std::string, std::string> & macros,
                         ثابت std::unordered_set<std::string> &        predefined_macros,
                         std::vector<Cond> &                            cond,
                         std::unordered_set<std::string> &              include_پشته,
                         Directive_نما                                  mode) {
        // split into tokens
        std::string        body = t.substr(1);
        std::istringstream iss(body);
        std::string        cmd;
        iss >> cmd;

        إذا (cmd == "include") {
            إذا (mode == Directive_نما::All && !cond_فعال(cond)) {
                أرجع درست;
            }
            std::string file;
            iss >> file;
            إذا (file.size() >= 2 && file.front() == '"' && file.back() == '"') {
                file = file.substr(1, file.size() - 2);
            }
            بیرون << processIncludeFile(file, macros, predefined_macros, include_پشته, mode);
            أرجع درست;
        }

        إذا (mode == Directive_نما::IncludesOnly) {
            أرجع خطأ;
        }

        إذا (cmd == "define") {
            إذا (!cond_فعال(cond)) {
                أرجع درست;
            }
            std::string name;
            iss >> name;
            // Don't override predefined macros from options
            إذا (predefined_macros.count(name)) {
                أرجع درست;
            }
            std::string value = trim_مقدار(iss);
            macros[name]      = value;
            أرجع درست;
        }

        إذا (cmd == "undef") {
            إذا (!cond_فعال(cond)) {
                أرجع درست;
            }
            std::string name;
            iss >> name;
            // Don't undef predefined macros from options
            إذا (predefined_macros.count(name)) {
                أرجع درست;
            }
            macros.erase(name);
            أرجع درست;
        }

        إذا (cmd == "ifdef") {
            std::string name;
            iss >> name;
            منطقي p = cond_فعال(cond);
            منطقي v = macros.count(name);
            cond.push_back({ p, p && v, p && v });
            أرجع درست;
        }

        إذا (cmd == "ifndef") {
            std::string name;
            iss >> name;
            منطقي p = cond_فعال(cond);
            منطقي v = !macros.count(name);
            cond.push_back({ p, p && v, p && v });
            أرجع درست;
        }

        إذا (cmd == "if") {
            std::string عبارت = trim_مقدار(iss);
            منطقي        p    = cond_فعال(cond);
            منطقي        v    = خطأ;
            إذا (p) {
                std::unordered_set<std::string> visiting;
                عبارت_Parser                      ep(عبارت, macros, visiting);
                v = ep.parse() != 0;
            }
            cond.push_back({ p, p && v, p && v });
            أرجع درست;
        }

        إذا (cmd == "elif") {
            std::string عبارت = trim_مقدار(iss);

            إذا (cond.خالی()) {
                ارم std::runtime_error("#elif without #if");
            }

            Cond & c = cond.back();
            إذا (!c.والد_فعال) {
                c.فعال = خطأ;
                أرجع درست;
            }

            إذا (c.گرفته) {
                c.فعال = خطأ;
                أرجع درست;
            }

            std::unordered_set<std::string> visiting;
            عبارت_Parser                      ep(عبارت, macros, visiting);
            منطقي                            v = ep.parse() != 0;
            c.فعال                          = v;
            إذا (v) {
                c.گرفته = درست;
            }
            أرجع درست;
        }

        إذا (cmd == "else") {
            إذا (cond.خالی()) {
                ارم std::runtime_error("#else without #if");
            }

            Cond & c = cond.back();
            إذا (!c.والد_فعال) {
                c.فعال = خطأ;
                أرجع درست;
            }
            إذا (c.گرفته) {
                c.فعال = خطأ;
            } وإلا {
                c.فعال = درست;
                c.گرفته  = درست;
            }
            أرجع درست;
        }

        إذا (cmd == "endif") {
            إذا (cond.خالی()) {
                ارم std::runtime_error("#endif without #if");
            }
            cond.pop_back();
            أرجع درست;
        }

        // Unknown directive
        ارم std::runtime_error("Unknown directive: #" + cmd);
    }
};

}  // namespace pre_wgsl

#پایان_اگر  // PRE_WGSL_HPP
