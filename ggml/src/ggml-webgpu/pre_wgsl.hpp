#که_نه_تعریف_شوی PRE_WGSL_HPP
#تعریف_کړه PRE_WGSL_HPP

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
    std::string              include_لار = ".";
    std::vector<std::string> macros;
};

//==============================================================
// Utility: trim
//==============================================================
ساكن std::string trim(ثابت std::string & s) {
    size_t a = 0;
    طالما (a < s.size() && std::که_تشه((بینښه حرف) s[a])) {
        a++;
    }
    size_t b = s.size();
    طالما (b > a && std::که_تشه((بینښه حرف) s[b - 1])) {
        b--;
    }
    أرجع s.substr(a, b - a);
}

ساكن std::string trim_ارزښت(std::istream & is) {
    std::ostringstream ss;
    ss << is.rdbuf();
    أرجع trim(ss.str());
}

ساكن منطقي isIdentChar(حرف c) {
    أرجع std::که_توری_یا_عدد(static_cast<بینښه حرف>(c)) || c == '_';
}

ساكن منطقي پایونه_With_Continuation(ثابت std::string & کرښه) {
    size_t i = کرښه.size();
    طالما (i > 0 && std::که_تشه((بینښه حرف) کرښه[i - 1])) {
        i--;
    }
    أرجع i > 0 && کرښه[i - 1] == '\\';
}

ساكن فراغ stripContinuation(std::string & کرښه) {
    size_t i = کرښه.size();
    طالما (i > 0 && std::که_تشه((بینښه حرف) کرښه[i - 1])) {
        i--;
    }
    إذا (i > 0 && کرښه[i - 1] == '\\') {
        کرښه.erase(i - 1);
    }
}

ساكن std::string expand_Macros_Recursive_دننی(ثابت std::string &                                  کرښه,
                                                 ثابت std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting);

ساكن std::string expand_Macro_ارزښت(ثابت std::string &                                  name,
                                    ثابت std::unordered_map<std::string, std::string> & macros,
                                    std::unordered_set<std::string> &                    visiting) {
    إذا (visiting.count(name)) {
        ارم std::runtime_error("Recursive macro: " + name);
    }
    visiting.insert(name);

    خپلکاری it = macros.find(name);
    إذا (it == macros.end()) {
        visiting.erase(name);
        أرجع name;
    }

    ثابت std::string & value = it->second;
    إذا (value.تشه()) {
        visiting.erase(name);
        أرجع "";
    }

    std::string expanded = expand_Macros_Recursive_دننی(value, macros, visiting);
    visiting.erase(name);
    أرجع expanded;
}

ساكن std::string expand_Macros_Recursive_دننی(ثابت std::string &                                  کرښه,
                                                 ثابت std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting) {
    std::string پایله;
    پایله.reserve(کرښه.size());

    size_t i = 0;
    طالما (i < کرښه.size()) {
        إذا (isIdentChar(کرښه[i])) {
            size_t start = i;
            طالما (i < کرښه.size() && isIdentChar(کرښه[i])) {
                i++;
            }
            std::string token = کرښه.substr(start, i - start);

            خپلکاری it = macros.find(token);
            إذا (it != macros.end()) {
                پایله += expand_Macro_ارزښت(token, macros, visiting);
            } وإلا {
                پایله += token;
            }
        } وإلا {
            پایله += کرښه[i];
            i++;
        }
    }

    أرجع پایله;
}

ساكن std::string expandMacrosRecursive(ثابت std::string &                                  کرښه,
                                         ثابت std::unordered_map<std::string, std::string> & macros) {
    std::unordered_set<std::string> visiting;
    أرجع expand_Macros_Recursive_دننی(کرښه, macros, visiting);
}

//==============================================================
// Tokenizer for expressions in #if/#elif
//==============================================================
صنف بیان_Lexer {
  عام:
    تعداد ډول_2 { پای_ټکی_3, IDENT, شمېره_2, OP, LPAREN, RPAREN };

    بنية Tok {
        ډول_2        ډول;
        std::string متن_2;
    };

    څرګند بیان_Lexer(std::string_view sv) : src(sv), pos(0) {}

    Tok بل() {
        skipWS();
        إذا (pos >= src.size()) {
            أرجع { پای_ټکی_3, "" };
        }

        حرف c = src[pos];

        //شمېره
        إذا (std::که_عدد((بینښه حرف) c)) {
            size_t start = pos;
            طالما (pos < src.size() && std::که_عدد((بینښه حرف) src[pos])) {
                pos++;
            }
            أرجع { شمېره_2, std::string(src.substr(start, pos - start)) };
        }

        // identifier
        إذا (std::که_توری((بینښه حرف) c) || c == '_') {
            size_t start = pos;
            طالما (pos < src.size() && (std::که_توری_یا_عدد((بینښه حرف) src[pos]) || src[pos] == '_')) {
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
        لكل (خپلکاری op : two_ops) {
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
        أرجع { پای_ټکی_3, "" };
    }

  خاص:
    std::string_view src;
    size_t           pos;

    فراغ skipWS() {
        طالما (pos < src.size() && std::که_تشه((بینښه حرف) src[pos])) {
            pos++;
        }
    }
};

//==============================================================
// Expression Parser (recursive descent)
//==============================================================
صنف بیان_Parser {
  عام:
    بیان_Parser(std::string_view                                     بیان,
               ثابت std::unordered_map<std::string, std::string> & macros,
               std::unordered_set<std::string> &                    visiting) :
        lex(بیان),
        macros(macros),
        visiting(visiting) {
        advance();
    }

    صحيح parse() { أرجع parseLogicalOr(); }

  خاص:
    بیان_Lexer                                            lex;
    بیان_Lexer::Tok                                       tok;
    ثابت std::unordered_map<std::string, std::string> & macros;
    std::unordered_set<std::string> &                    visiting;

    فراغ advance() { tok = lex.بل(); }

    منطقي acceptOp(ثابت std::string & s) {
        إذا (tok.ډول == بیان_Lexer::OP && tok.متن_2 == s) {
            advance();
            أرجع ریښتیا;
        }
        أرجع خطأ;
    }

    منطقي accept_ډول(بیان_Lexer::ډول_2 k) {
        إذا (tok.ډول == k) {
            advance();
            أرجع ریښتیا;
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
        صحيح v = parse_ښويدنه();
        لكل (;;) {
            إذا (acceptOp("<")) {
                صحيح rhs = parse_ښويدنه();
                v       = (v < rhs);
            } وإلا إذا (acceptOp(">")) {
                صحيح rhs = parse_ښويدنه();
                v       = (v > rhs);
            } وإلا إذا (acceptOp("<=")) {
                صحيح rhs = parse_ښويدنه();
                v       = (v <= rhs);
            } وإلا إذا (acceptOp(">=")) {
                صحيح rhs = parse_ښويدنه();
                v       = (v >= rhs);
            } وإلا {
                اكسر;
            }
        }
        أرجع v;
    }

    صحيح parse_ښويدنه() {
        صحيح v = parse_جمع_کړه();
        لكل (;;) {
            إذا (acceptOp("<<")) {
                صحيح rhs = parse_جمع_کړه();
                v       = (v << rhs);
            } وإلا إذا (acceptOp(">>")) {
                صحيح rhs = parse_جمع_کړه();
                v       = (v >> rhs);
            } وإلا {
                اكسر;
            }
        }
        أرجع v;
    }

    صحيح parse_جمع_کړه() {
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
        إذا (accept_ډول(بیان_Lexer::LPAREN)) {
            صحيح v = parse();
            إذا (!accept_ډول(بیان_Lexer::RPAREN)) {
                ارم std::runtime_error("missing ')'");
            }
            أرجع v;
        }

        //شمېره
        إذا (tok.ډول == بیان_Lexer::شمېره_2) {
            صحيح v = std::stoi(tok.متن_2);
            advance();
            أرجع v;
        }

        // defined(identifier)
        إذا (tok.ډول == بیان_Lexer::IDENT && tok.متن_2 == "defined") {
            advance();
            إذا (accept_ډول(بیان_Lexer::LPAREN)) {
                إذا (tok.ډول != بیان_Lexer::IDENT) {
                    ارم std::runtime_error("expected identifier in defined()");
                }
                std::string name = tok.متن_2;
                advance();
                إذا (!accept_ډول(بیان_Lexer::RPAREN)) {
                    ارم std::runtime_error("missing ) in defined()");
                }
                أرجع macros.count(name) ? 1 : 0;
            } وإلا {
                // defined NAME
                إذا (tok.ډول != بیان_Lexer::IDENT) {
                    ارم std::runtime_error("expected identifier in defined NAME");
                }
                std::string name = tok.متن_2;
                advance();
                أرجع macros.count(name) ? 1 : 0;
            }
        }

        // identifier -> treat as integer, if defined use its value else 0
        إذا (tok.ډول == بیان_Lexer::IDENT) {
            std::string name = tok.متن_2;
            advance();
            خپلکاری it = macros.find(name);
            إذا (it == macros.end()) {
                أرجع 0;
            }
            إذا (it->second.تشه()) {
                أرجع 1;
            }
            أرجع eval_Macro_افاده(name, it->second);
        }

        // unexpected
        أرجع 0;
    }

    صحيح eval_Macro_افاده(ثابت std::string & name, ثابت std::string & value) {
        إذا (visiting.count(name)) {
            ارم std::runtime_error("Recursive macro: " + name);
        }

        visiting.insert(name);
        بیان_Parser ep(value, macros, visiting);
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
    څرګند Preprocessor(Options opts = {}) : opts_(std::move(opts)) {
        // Treat empty include path as current directory
        إذا (opts_.include_لار.تشه()) {
            opts_.include_لار = ".";
        }
        parseMacroDefinitions(opts_.macros);
    }

    std::string preprocess_file(ثابت std::string & filename, ثابت std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_ډېری;
        buildMacros(additional_macros, macros, predefined);

        std::string پایله = processFile(filename, macros, predefined, include_ډېری, Directive_ډېرپېښ::All);
        أرجع پایله;
    }

    std::string preprocess(ثابت std::string & contents, ثابت std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_ډېری;
        buildMacros(additional_macros, macros, predefined);

        std::string پایله = processString(contents, macros, predefined, include_ډېری, Directive_ډېرپېښ::All);
        أرجع پایله;
    }

    std::string preprocess_includes_file(ثابت std::string & filename) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_ډېری;
        std::string پایله = processFile(filename, macros, predefined, include_ډېری, Directive_ډېرپېښ::IncludesOnly);
        أرجع پایله;
    }

    std::string preprocess_includes(ثابت std::string & contents) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_ډېری;
        std::string پایله = processString(contents, macros, predefined, include_ډېری, Directive_ډېرپېښ::IncludesOnly);
        أرجع پایله;
    }

  خاص:
    Options                                      opts_;
    std::unordered_map<std::string, std::string> global_macros;

    تعداد صنف Directive_ډېرپېښ { All, IncludesOnly };

    بنية Cond {
        منطقي پلار_فعال;
        منطقي فعال;
        منطقي اخیستل_شوی;
    };

    //----------------------------------------------------------
    // Parse macro definitions into global_macros
    //----------------------------------------------------------
    فراغ parseMacroDefinitions(ثابت std::vector<std::string> & macro_defs) {
        لكل (ثابت خپلکاری & def : macro_defs) {
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

        لكل (ثابت خپلکاری & [name, value] : global_macros) {
            predefined.insert(name);
        }

        لكل (ثابت خپلکاری & def : additional_macros) {
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
    std::string بارول_File(ثابت std::string & fname) {
        std::ifstream f(fname);
        إذا (!f.is_open()) {
            ارم std::runtime_error("Could not open file: " + fname);
        }
        std::stringstream ss;
        ss << f.rdbuf();
        أرجع ss.str();
    }

    منطقي cond_فعال(ثابت std::vector<Cond> & cond) ثابت {
        إذا (cond.تشه()) {
            أرجع ریښتیا;
        }
        أرجع cond.back().فعال;
    }

    //----------------------------------------------------------
    // Process a file
    //----------------------------------------------------------
    std::string processFile(ثابت std::string &                            name,
                            std::unordered_map<std::string, std::string> & macros,
                            ثابت std::unordered_set<std::string> &        predefined_macros,
                            std::unordered_set<std::string> &              include_ډېری,
                            Directive_ډېرپېښ                                  mode) {
        إذا (include_ډېری.count(name)) {
            ارم std::runtime_error("Recursive include: " + name);
        }

        include_ډېری.insert(name);
        std::string shader_کوډ = بارول_File(name);
        std::string بهر         = processString(shader_کوډ, macros, predefined_macros, include_ډېری, mode);
        include_ډېری.erase(name);
        أرجع بهر;
    }

    std::string processIncludeFile(ثابت std::string &                            fname,
                                   std::unordered_map<std::string, std::string> & macros,
                                   ثابت std::unordered_set<std::string> &        predefined_macros,
                                   std::unordered_set<std::string> &              include_ډېری,
                                   Directive_ډېرپېښ                                  mode) {
        std::string ډک_لار = opts_.include_لار + "/" + fname;
        أرجع processFile(ډک_لار, macros, predefined_macros, include_ډېری, mode);
    }

    //----------------------------------------------------------
    // Process text
    //----------------------------------------------------------
    std::string processString(ثابت std::string &                            shader_کوډ,
                              std::unordered_map<std::string, std::string> & macros,
                              ثابت std::unordered_set<std::string> &        predefined_macros,
                              std::unordered_set<std::string> &              include_ډېری,
                              Directive_ډېرپېښ                                  mode) {
        std::vector<Cond>  cond;  // Conditional stack for this shader
        std::stringstream  بهر;
        std::istringstream دننه(shader_کوډ);
        std::string        کرښه;

        طالما (std::getline(دننه, کرښه)) {
            std::string logical = کرښه;
            std::string t       = trim(logical);
            إذا (!t.تشه() && t[0] == '#') {
                طالما (پایونه_With_Continuation(logical)) {
                    stripContinuation(logical);
                    إذا (!std::getline(دننه, کرښه)) {
                        اكسر;
                    }
                    logical += "\n";
                    logical += کرښه;
                }
                t = trim(logical);
            }

            إذا (!t.تشه() && t[0] == '#') {
                منطقي handled = handleDirective(t, بهر, macros, predefined_macros, cond, include_ډېری, mode);
                إذا (mode == Directive_ډېرپېښ::IncludesOnly && !handled) {
                    بهر << logical << "\n";
                }
            } وإلا {
                إذا (mode == Directive_ډېرپېښ::IncludesOnly) {
                    بهر << logical << "\n";
                } وإلا إذا (cond_فعال(cond)) {
                    // Expand macros in the line before outputting
                    std::string expanded = expandMacrosRecursive(logical, macros);
                    بهر << expanded << "\n";
                }
            }
        }

        إذا (mode == Directive_ډېرپېښ::All && !cond.تشه()) {
            ارم std::runtime_error("Unclosed #if directive");
        }

        أرجع بهر.str();
    }

    //----------------------------------------------------------
    // Directive handler
    //----------------------------------------------------------
    منطقي handleDirective(ثابت std::string &                            t,
                         std::stringstream &                            بهر,
                         std::unordered_map<std::string, std::string> & macros,
                         ثابت std::unordered_set<std::string> &        predefined_macros,
                         std::vector<Cond> &                            cond,
                         std::unordered_set<std::string> &              include_ډېری,
                         Directive_ډېرپېښ                                  mode) {
        // split into tokens
        std::string        body = t.substr(1);
        std::istringstream iss(body);
        std::string        cmd;
        iss >> cmd;

        إذا (cmd == "include") {
            إذا (mode == Directive_ډېرپېښ::All && !cond_فعال(cond)) {
                أرجع ریښتیا;
            }
            std::string file;
            iss >> file;
            إذا (file.size() >= 2 && file.front() == '"' && file.back() == '"') {
                file = file.substr(1, file.size() - 2);
            }
            بهر << processIncludeFile(file, macros, predefined_macros, include_ډېری, mode);
            أرجع ریښتیا;
        }

        إذا (mode == Directive_ډېرپېښ::IncludesOnly) {
            أرجع خطأ;
        }

        إذا (cmd == "define") {
            إذا (!cond_فعال(cond)) {
                أرجع ریښتیا;
            }
            std::string name;
            iss >> name;
            // Don't override predefined macros from options
            إذا (predefined_macros.count(name)) {
                أرجع ریښتیا;
            }
            std::string value = trim_ارزښت(iss);
            macros[name]      = value;
            أرجع ریښتیا;
        }

        إذا (cmd == "undef") {
            إذا (!cond_فعال(cond)) {
                أرجع ریښتیا;
            }
            std::string name;
            iss >> name;
            // Don't undef predefined macros from options
            إذا (predefined_macros.count(name)) {
                أرجع ریښتیا;
            }
            macros.erase(name);
            أرجع ریښتیا;
        }

        إذا (cmd == "ifdef") {
            std::string name;
            iss >> name;
            منطقي p = cond_فعال(cond);
            منطقي v = macros.count(name);
            cond.push_back({ p, p && v, p && v });
            أرجع ریښتیا;
        }

        إذا (cmd == "ifndef") {
            std::string name;
            iss >> name;
            منطقي p = cond_فعال(cond);
            منطقي v = !macros.count(name);
            cond.push_back({ p, p && v, p && v });
            أرجع ریښتیا;
        }

        إذا (cmd == "if") {
            std::string بیان = trim_ارزښت(iss);
            منطقي        p    = cond_فعال(cond);
            منطقي        v    = خطأ;
            إذا (p) {
                std::unordered_set<std::string> visiting;
                بیان_Parser                      ep(بیان, macros, visiting);
                v = ep.parse() != 0;
            }
            cond.push_back({ p, p && v, p && v });
            أرجع ریښتیا;
        }

        إذا (cmd == "elif") {
            std::string بیان = trim_ارزښت(iss);

            إذا (cond.تشه()) {
                ارم std::runtime_error("#elif without #if");
            }

            Cond & c = cond.back();
            إذا (!c.پلار_فعال) {
                c.فعال = خطأ;
                أرجع ریښتیا;
            }

            إذا (c.اخیستل_شوی) {
                c.فعال = خطأ;
                أرجع ریښتیا;
            }

            std::unordered_set<std::string> visiting;
            بیان_Parser                      ep(بیان, macros, visiting);
            منطقي                            v = ep.parse() != 0;
            c.فعال                          = v;
            إذا (v) {
                c.اخیستل_شوی = ریښتیا;
            }
            أرجع ریښتیا;
        }

        إذا (cmd == "else") {
            إذا (cond.تشه()) {
                ارم std::runtime_error("#else without #if");
            }

            Cond & c = cond.back();
            إذا (!c.پلار_فعال) {
                c.فعال = خطأ;
                أرجع ریښتیا;
            }
            إذا (c.اخیستل_شوی) {
                c.فعال = خطأ;
            } وإلا {
                c.فعال = ریښتیا;
                c.اخیستل_شوی  = ریښتیا;
            }
            أرجع ریښتیا;
        }

        إذا (cmd == "endif") {
            إذا (cond.تشه()) {
                ارم std::runtime_error("#endif without #if");
            }
            cond.pop_back();
            أرجع ریښتیا;
        }

        // Unknown directive
        ارم std::runtime_error("Unknown directive: #" + cmd);
    }
};

}  // namespace pre_wgsl

#د_که_پای  // PRE_WGSL_HPP
