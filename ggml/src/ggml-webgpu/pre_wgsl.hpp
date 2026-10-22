#اگر_غیر_تعریف_شدہ PRE_WGSL_HPP
#تعریف_کرو PRE_WGSL_HPP

#شامل <cctype>
#شامل <fstream>
#شامل <sstream>
#شامل <stdexcept>
#شامل <string>
#شامل <string_view>
#شامل <unordered_map>
#شامل <unordered_set>
#شامل <vector>

نامخلا pre_wgsl {

//==============================================================
// Options
//==============================================================
ساخت Options {
    std::string              include_راستہ = ".";
    std::vector<std::string> macros;
};

//==============================================================
// Utility: trim
//==============================================================
ساکن std::string trim(ثابت std::string & s) {
    size_t a = 0;
    جب (a < s.size() && std::خلا_ہے((غیرعلامتی حرف) s[a])) {
        a++;
    }
    size_t b = s.size();
    جب (b > a && std::خلا_ہے((غیرعلامتی حرف) s[b - 1])) {
        b--;
    }
    واپس s.substr(a, b - a);
}

ساکن std::string trim_قدر(std::istream & is) {
    std::ostringstream ss;
    ss << is.rdbuf();
    واپس trim(ss.str());
}

ساکن منطقی isIdentChar(حرف c) {
    واپس std::حرف_یا_ہندسہ(static_cast<غیرعلامتی حرف>(c)) || c == '_';
}

ساکن منطقی سرے_With_Continuation(ثابت std::string & سطر) {
    size_t i = سطر.size();
    جب (i > 0 && std::خلا_ہے((غیرعلامتی حرف) سطر[i - 1])) {
        i--;
    }
    واپس i > 0 && سطر[i - 1] == '\\';
}

ساکن خالی stripContinuation(std::string & سطر) {
    size_t i = سطر.size();
    جب (i > 0 && std::خلا_ہے((غیرعلامتی حرف) سطر[i - 1])) {
        i--;
    }
    اگر (i > 0 && سطر[i - 1] == '\\') {
        سطر.erase(i - 1);
    }
}

ساکن std::string expand_Macros_Recursive_اندرونی(ثابت std::string &                                  سطر,
                                                 ثابت std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting);

ساکن std::string expand_Macro_قدر(ثابت std::string &                                  name,
                                    ثابت std::unordered_map<std::string, std::string> & macros,
                                    std::unordered_set<std::string> &                    visiting) {
    اگر (visiting.count(name)) {
        پھینکو std::runtime_error("Recursive macro: " + name);
    }
    visiting.insert(name);

    خودکار it = macros.find(name);
    اگر (it == macros.end()) {
        visiting.erase(name);
        واپس name;
    }

    ثابت std::string & value = it->second;
    اگر (value.خالی_خانے()) {
        visiting.erase(name);
        واپس "";
    }

    std::string expanded = expand_Macros_Recursive_اندرونی(value, macros, visiting);
    visiting.erase(name);
    واپس expanded;
}

ساکن std::string expand_Macros_Recursive_اندرونی(ثابت std::string &                                  سطر,
                                                 ثابت std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting) {
    std::string نتیجہ;
    نتیجہ.reserve(سطر.size());

    size_t i = 0;
    جب (i < سطر.size()) {
        اگر (isIdentChar(سطر[i])) {
            size_t start = i;
            جب (i < سطر.size() && isIdentChar(سطر[i])) {
                i++;
            }
            std::string token = سطر.substr(start, i - start);

            خودکار it = macros.find(token);
            اگر (it != macros.end()) {
                نتیجہ += expand_Macro_قدر(token, macros, visiting);
            } ورنہ {
                نتیجہ += token;
            }
        } ورنہ {
            نتیجہ += سطر[i];
            i++;
        }
    }

    واپس نتیجہ;
}

ساکن std::string expandMacrosRecursive(ثابت std::string &                                  سطر,
                                         ثابت std::unordered_map<std::string, std::string> & macros) {
    std::unordered_set<std::string> visiting;
    واپس expand_Macros_Recursive_اندرونی(سطر, macros, visiting);
}

//==============================================================
// Tokenizer for expressions in #if/#elif
//==============================================================
جماعت اظہاریہ_Lexer {
  عوامی:
    شمار قسم_3 { انتہا_3, IDENT, نمبر_2, OP, LPAREN, RPAREN };

    ساخت Tok {
        قسم_3        قسم_2;
        std::string متن_2;
    };

    صریح اظہاریہ_Lexer(std::string_view sv) : src(sv), pos(0) {}

    Tok اگلا() {
        skipWS();
        اگر (pos >= src.size()) {
            واپس { انتہا_3, "" };
        }

        حرف c = src[pos];

        //نمبر
        اگر (std::ہندسہ_ہے((غیرعلامتی حرف) c)) {
            size_t start = pos;
            جب (pos < src.size() && std::ہندسہ_ہے((غیرعلامتی حرف) src[pos])) {
                pos++;
            }
            واپس { نمبر_2, std::string(src.substr(start, pos - start)) };
        }

        // identifier
        اگر (std::حرف_ہے((غیرعلامتی حرف) c) || c == '_') {
            size_t start = pos;
            جب (pos < src.size() && (std::حرف_یا_ہندسہ((غیرعلامتی حرف) src[pos]) || src[pos] == '_')) {
                pos++;
            }
            واپس { IDENT, std::string(src.substr(start, pos - start)) };
        }

        اگر (c == '(') {
            pos++;
            واپس { LPAREN, "(" };
        }
        اگر (c == ')') {
            pos++;
            واپس { RPAREN, ")" };
        }

        // multi-char operators
        ساکن ثابت حرف * two_ops[] = { "==", "!=", "<=", ">=", "&&", "||", "<<", ">>" };
        برائے (خودکار op : two_ops) {
            اگر (src.substr(pos, 2) == op) {
                pos += 2;
                واپس { OP, std::string(op) };
            }
        }

        // single-char operators
        اگر (std::string("+-*/%<>!").find(c) != std::string::npos) {
            pos++;
            واپس { OP, std::string(1, c) };
        }

        // unexpected
        pos++;
        واپس { انتہا_3, "" };
    }

  نجی:
    std::string_view src;
    size_t           pos;

    خالی skipWS() {
        جب (pos < src.size() && std::خلا_ہے((غیرعلامتی حرف) src[pos])) {
            pos++;
        }
    }
};

//==============================================================
// Expression Parser (recursive descent)
//==============================================================
جماعت اظہاریہ_Parser {
  عوامی:
    اظہاریہ_Parser(std::string_view                                     اظہاریہ,
               ثابت std::unordered_map<std::string, std::string> & macros,
               std::unordered_set<std::string> &                    visiting) :
        lex(اظہاریہ),
        macros(macros),
        visiting(visiting) {
        advance();
    }

    صحیح parse() { واپس parseLogicalOr(); }

  نجی:
    اظہاریہ_Lexer                                            lex;
    اظہاریہ_Lexer::Tok                                       tok;
    ثابت std::unordered_map<std::string, std::string> & macros;
    std::unordered_set<std::string> &                    visiting;

    خالی advance() { tok = lex.اگلا(); }

    منطقی acceptOp(ثابت std::string & s) {
        اگر (tok.قسم_2 == اظہاریہ_Lexer::OP && tok.متن_2 == s) {
            advance();
            واپس سچ;
        }
        واپس جھوٹ;
    }

    منطقی accept_قسم(اظہاریہ_Lexer::قسم_3 k) {
        اگر (tok.قسم_2 == k) {
            advance();
            واپس سچ;
        }
        واپس جھوٹ;
    }

    صحیح parseLogicalOr() {
        صحیح v = parseLogicalAnd();
        جب (acceptOp("||")) {
            صحیح rhs = parseLogicalAnd();
            v       = (v || rhs);
        }
        واپس v;
    }

    صحیح parseLogicalAnd() {
        صحیح v = parseEquality();
        جب (acceptOp("&&")) {
            صحیح rhs = parseEquality();
            v       = (v && rhs);
        }
        واپس v;
    }

    صحیح parseEquality() {
        صحیح v = parseRelational();
        برائے (;;) {
            اگر (acceptOp("==")) {
                صحیح rhs = parseRelational();
                v       = (v == rhs);
            } ورنہ اگر (acceptOp("!=")) {
                صحیح rhs = parseRelational();
                v       = (v != rhs);
            } ورنہ {
                توڑو;
            }
        }
        واپس v;
    }

    صحیح parseRelational() {
        صحیح v = parse_سرکاؤ();
        برائے (;;) {
            اگر (acceptOp("<")) {
                صحیح rhs = parse_سرکاؤ();
                v       = (v < rhs);
            } ورنہ اگر (acceptOp(">")) {
                صحیح rhs = parse_سرکاؤ();
                v       = (v > rhs);
            } ورنہ اگر (acceptOp("<=")) {
                صحیح rhs = parse_سرکاؤ();
                v       = (v <= rhs);
            } ورنہ اگر (acceptOp(">=")) {
                صحیح rhs = parse_سرکاؤ();
                v       = (v >= rhs);
            } ورنہ {
                توڑو;
            }
        }
        واپس v;
    }

    صحیح parse_سرکاؤ() {
        صحیح v = parse_جوڑو();
        برائے (;;) {
            اگر (acceptOp("<<")) {
                صحیح rhs = parse_جوڑو();
                v       = (v << rhs);
            } ورنہ اگر (acceptOp(">>")) {
                صحیح rhs = parse_جوڑو();
                v       = (v >> rhs);
            } ورنہ {
                توڑو;
            }
        }
        واپس v;
    }

    صحیح parse_جوڑو() {
        صحیح v = parseMult();
        برائے (;;) {
            اگر (acceptOp("+")) {
                صحیح rhs = parseMult();
                v       = (v + rhs);
            } ورنہ اگر (acceptOp("-")) {
                صحیح rhs = parseMult();
                v       = (v - rhs);
            } ورنہ {
                توڑو;
            }
        }
        واپس v;
    }

    صحیح parseMult() {
        صحیح v = parseUnary();
        برائے (;;) {
            اگر (acceptOp("*")) {
                صحیح rhs = parseUnary();
                v       = (v * rhs);
            } ورنہ اگر (acceptOp("/")) {
                صحیح rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v / rhs);
            } ورنہ اگر (acceptOp("%")) {
                صحیح rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v % rhs);
            } ورنہ {
                توڑو;
            }
        }
        واپس v;
    }

    صحیح parseUnary() {
        اگر (acceptOp("!")) {
            واپس !parseUnary();
        }
        اگر (acceptOp("-")) {
            واپس -parseUnary();
        }
        اگر (acceptOp("+")) {
            واپس +parseUnary();
        }
        واپس parsePrimary();
    }

    صحیح parsePrimary() {
        // '(' expr ')'
        اگر (accept_قسم(اظہاریہ_Lexer::LPAREN)) {
            صحیح v = parse();
            اگر (!accept_قسم(اظہاریہ_Lexer::RPAREN)) {
                پھینکو std::runtime_error("missing ')'");
            }
            واپس v;
        }

        //نمبر
        اگر (tok.قسم_2 == اظہاریہ_Lexer::نمبر_2) {
            صحیح v = std::stoi(tok.متن_2);
            advance();
            واپس v;
        }

        // defined(identifier)
        اگر (tok.قسم_2 == اظہاریہ_Lexer::IDENT && tok.متن_2 == "defined") {
            advance();
            اگر (accept_قسم(اظہاریہ_Lexer::LPAREN)) {
                اگر (tok.قسم_2 != اظہاریہ_Lexer::IDENT) {
                    پھینکو std::runtime_error("expected identifier in defined()");
                }
                std::string name = tok.متن_2;
                advance();
                اگر (!accept_قسم(اظہاریہ_Lexer::RPAREN)) {
                    پھینکو std::runtime_error("missing ) in defined()");
                }
                واپس macros.count(name) ? 1 : 0;
            } ورنہ {
                // defined NAME
                اگر (tok.قسم_2 != اظہاریہ_Lexer::IDENT) {
                    پھینکو std::runtime_error("expected identifier in defined NAME");
                }
                std::string name = tok.متن_2;
                advance();
                واپس macros.count(name) ? 1 : 0;
            }
        }

        // identifier -> treat as integer, if defined use its value else 0
        اگر (tok.قسم_2 == اظہاریہ_Lexer::IDENT) {
            std::string name = tok.متن_2;
            advance();
            خودکار it = macros.find(name);
            اگر (it == macros.end()) {
                واپس 0;
            }
            اگر (it->second.خالی_خانے()) {
                واپس 1;
            }
            واپس eval_Macro_عبارت(name, it->second);
        }

        // unexpected
        واپس 0;
    }

    صحیح eval_Macro_عبارت(ثابت std::string & name, ثابت std::string & value) {
        اگر (visiting.count(name)) {
            پھینکو std::runtime_error("Recursive macro: " + name);
        }

        visiting.insert(name);
        اظہاریہ_Parser ep(value, macros, visiting);
        صحیح        v = ep.parse();
        visiting.erase(name);
        واپس v;
    }
};

//==============================================================
// Preprocessor
//==============================================================
جماعت Preprocessor {
  عوامی:
    صریح Preprocessor(Options opts = {}) : opts_(std::move(opts)) {
        // Treat empty include path as current directory
        اگر (opts_.include_راستہ.خالی_خانے()) {
            opts_.include_راستہ = ".";
        }
        parseMacroDefinitions(opts_.macros);
    }

    std::string preprocess_file(ثابت std::string & filename, ثابت std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_ڈھیر;
        buildMacros(additional_macros, macros, predefined);

        std::string نتیجہ = processFile(filename, macros, predefined, include_ڈھیر, Directive_منوال::All);
        واپس نتیجہ;
    }

    std::string preprocess(ثابت std::string & contents, ثابت std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_ڈھیر;
        buildMacros(additional_macros, macros, predefined);

        std::string نتیجہ = processString(contents, macros, predefined, include_ڈھیر, Directive_منوال::All);
        واپس نتیجہ;
    }

    std::string preprocess_includes_file(ثابت std::string & filename) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_ڈھیر;
        std::string نتیجہ = processFile(filename, macros, predefined, include_ڈھیر, Directive_منوال::IncludesOnly);
        واپس نتیجہ;
    }

    std::string preprocess_includes(ثابت std::string & contents) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_ڈھیر;
        std::string نتیجہ = processString(contents, macros, predefined, include_ڈھیر, Directive_منوال::IncludesOnly);
        واپس نتیجہ;
    }

  نجی:
    Options                                      opts_;
    std::unordered_map<std::string, std::string> global_macros;

    شمار جماعت Directive_منوال { All, IncludesOnly };

    ساخت Cond {
        منطقی والد_فعال;
        منطقی فعال;
        منطقی لیا_گیا;
    };

    //----------------------------------------------------------
    // Parse macro definitions into global_macros
    //----------------------------------------------------------
    خالی parseMacroDefinitions(ثابت std::vector<std::string> & macro_defs) {
        برائے (ثابت خودکار & def : macro_defs) {
            size_t eq_pos = def.find('=');
            اگر (eq_pos != std::string::npos) {
                // Format: NAME=VALUE
                std::string name    = trim(def.substr(0, eq_pos));
                std::string value   = trim(def.substr(eq_pos + 1));
                global_macros[name] = value;
            } ورنہ {
                // Format: NAME
                std::string name    = trim(def);
                global_macros[name] = "";
            }
        }
    }

    //----------------------------------------------------------
    // Build combined macro map and predefined set for a preprocessing operation
    //----------------------------------------------------------
    خالی buildMacros(ثابت std::vector<std::string> &               additional_macros,
                     std::unordered_map<std::string, std::string> & macros,
                     std::unordered_set<std::string> &              predefined) {
        macros = global_macros;
        predefined.clear();

        برائے (ثابت خودکار & [name, value] : global_macros) {
            predefined.insert(name);
        }

        برائے (ثابت خودکار & def : additional_macros) {
            size_t      eq_pos = def.find('=');
            std::string name, value;
            اگر (eq_pos != std::string::npos) {
                name  = trim(def.substr(0, eq_pos));
                value = trim(def.substr(eq_pos + 1));
            } ورنہ {
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
    std::string بھرو_File(ثابت std::string & fname) {
        std::ifstream f(fname);
        اگر (!f.is_open()) {
            پھینکو std::runtime_error("Could not open file: " + fname);
        }
        std::stringstream ss;
        ss << f.rdbuf();
        واپس ss.str();
    }

    منطقی cond_فعال(ثابت std::vector<Cond> & cond) ثابت {
        اگر (cond.خالی_خانے()) {
            واپس سچ;
        }
        واپس cond.back().فعال;
    }

    //----------------------------------------------------------
    // Process a file
    //----------------------------------------------------------
    std::string processFile(ثابت std::string &                            name,
                            std::unordered_map<std::string, std::string> & macros,
                            ثابت std::unordered_set<std::string> &        predefined_macros,
                            std::unordered_set<std::string> &              include_ڈھیر,
                            Directive_منوال                                  mode) {
        اگر (include_ڈھیر.count(name)) {
            پھینکو std::runtime_error("Recursive include: " + name);
        }

        include_ڈھیر.insert(name);
        std::string shader_رمز = بھرو_File(name);
        std::string برآمد         = processString(shader_رمز, macros, predefined_macros, include_ڈھیر, mode);
        include_ڈھیر.erase(name);
        واپس برآمد;
    }

    std::string processIncludeFile(ثابت std::string &                            fname,
                                   std::unordered_map<std::string, std::string> & macros,
                                   ثابت std::unordered_set<std::string> &        predefined_macros,
                                   std::unordered_set<std::string> &              include_ڈھیر,
                                   Directive_منوال                                  mode) {
        std::string بھرا_راستہ = opts_.include_راستہ + "/" + fname;
        واپس processFile(بھرا_راستہ, macros, predefined_macros, include_ڈھیر, mode);
    }

    //----------------------------------------------------------
    // Process text
    //----------------------------------------------------------
    std::string processString(ثابت std::string &                            shader_رمز,
                              std::unordered_map<std::string, std::string> & macros,
                              ثابت std::unordered_set<std::string> &        predefined_macros,
                              std::unordered_set<std::string> &              include_ڈھیر,
                              Directive_منوال                                  mode) {
        std::vector<Cond>  cond;  // Conditional stack for this shader
        std::stringstream  برآمد;
        std::istringstream اندر(shader_رمز);
        std::string        سطر;

        جب (std::getline(اندر, سطر)) {
            std::string logical = سطر;
            std::string t       = trim(logical);
            اگر (!t.خالی_خانے() && t[0] == '#') {
                جب (سرے_With_Continuation(logical)) {
                    stripContinuation(logical);
                    اگر (!std::getline(اندر, سطر)) {
                        توڑو;
                    }
                    logical += "\n";
                    logical += سطر;
                }
                t = trim(logical);
            }

            اگر (!t.خالی_خانے() && t[0] == '#') {
                منطقی handled = handleDirective(t, برآمد, macros, predefined_macros, cond, include_ڈھیر, mode);
                اگر (mode == Directive_منوال::IncludesOnly && !handled) {
                    برآمد << logical << "\n";
                }
            } ورنہ {
                اگر (mode == Directive_منوال::IncludesOnly) {
                    برآمد << logical << "\n";
                } ورنہ اگر (cond_فعال(cond)) {
                    // Expand macros in the line before outputting
                    std::string expanded = expandMacrosRecursive(logical, macros);
                    برآمد << expanded << "\n";
                }
            }
        }

        اگر (mode == Directive_منوال::All && !cond.خالی_خانے()) {
            پھینکو std::runtime_error("Unclosed #if directive");
        }

        واپس برآمد.str();
    }

    //----------------------------------------------------------
    // Directive handler
    //----------------------------------------------------------
    منطقی handleDirective(ثابت std::string &                            t,
                         std::stringstream &                            برآمد,
                         std::unordered_map<std::string, std::string> & macros,
                         ثابت std::unordered_set<std::string> &        predefined_macros,
                         std::vector<Cond> &                            cond,
                         std::unordered_set<std::string> &              include_ڈھیر,
                         Directive_منوال                                  mode) {
        // split into tokens
        std::string        body = t.substr(1);
        std::istringstream iss(body);
        std::string        cmd;
        iss >> cmd;

        اگر (cmd == "include") {
            اگر (mode == Directive_منوال::All && !cond_فعال(cond)) {
                واپس سچ;
            }
            std::string file;
            iss >> file;
            اگر (file.size() >= 2 && file.front() == '"' && file.back() == '"') {
                file = file.substr(1, file.size() - 2);
            }
            برآمد << processIncludeFile(file, macros, predefined_macros, include_ڈھیر, mode);
            واپس سچ;
        }

        اگر (mode == Directive_منوال::IncludesOnly) {
            واپس جھوٹ;
        }

        اگر (cmd == "define") {
            اگر (!cond_فعال(cond)) {
                واپس سچ;
            }
            std::string name;
            iss >> name;
            // Don't override predefined macros from options
            اگر (predefined_macros.count(name)) {
                واپس سچ;
            }
            std::string value = trim_قدر(iss);
            macros[name]      = value;
            واپس سچ;
        }

        اگر (cmd == "undef") {
            اگر (!cond_فعال(cond)) {
                واپس سچ;
            }
            std::string name;
            iss >> name;
            // Don't undef predefined macros from options
            اگر (predefined_macros.count(name)) {
                واپس سچ;
            }
            macros.erase(name);
            واپس سچ;
        }

        اگر (cmd == "ifdef") {
            std::string name;
            iss >> name;
            منطقی p = cond_فعال(cond);
            منطقی v = macros.count(name);
            cond.push_back({ p, p && v, p && v });
            واپس سچ;
        }

        اگر (cmd == "ifndef") {
            std::string name;
            iss >> name;
            منطقی p = cond_فعال(cond);
            منطقی v = !macros.count(name);
            cond.push_back({ p, p && v, p && v });
            واپس سچ;
        }

        اگر (cmd == "if") {
            std::string اظہاریہ = trim_قدر(iss);
            منطقی        p    = cond_فعال(cond);
            منطقی        v    = جھوٹ;
            اگر (p) {
                std::unordered_set<std::string> visiting;
                اظہاریہ_Parser                      ep(اظہاریہ, macros, visiting);
                v = ep.parse() != 0;
            }
            cond.push_back({ p, p && v, p && v });
            واپس سچ;
        }

        اگر (cmd == "elif") {
            std::string اظہاریہ = trim_قدر(iss);

            اگر (cond.خالی_خانے()) {
                پھینکو std::runtime_error("#elif without #if");
            }

            Cond & c = cond.back();
            اگر (!c.والد_فعال) {
                c.فعال = جھوٹ;
                واپس سچ;
            }

            اگر (c.لیا_گیا) {
                c.فعال = جھوٹ;
                واپس سچ;
            }

            std::unordered_set<std::string> visiting;
            اظہاریہ_Parser                      ep(اظہاریہ, macros, visiting);
            منطقی                            v = ep.parse() != 0;
            c.فعال                          = v;
            اگر (v) {
                c.لیا_گیا = سچ;
            }
            واپس سچ;
        }

        اگر (cmd == "else") {
            اگر (cond.خالی_خانے()) {
                پھینکو std::runtime_error("#else without #if");
            }

            Cond & c = cond.back();
            اگر (!c.والد_فعال) {
                c.فعال = جھوٹ;
                واپس سچ;
            }
            اگر (c.لیا_گیا) {
                c.فعال = جھوٹ;
            } ورنہ {
                c.فعال = سچ;
                c.لیا_گیا  = سچ;
            }
            واپس سچ;
        }

        اگر (cmd == "endif") {
            اگر (cond.خالی_خانے()) {
                پھینکو std::runtime_error("#endif without #if");
            }
            cond.pop_back();
            واپس سچ;
        }

        // Unknown directive
        پھینکو std::runtime_error("Unknown directive: #" + cmd);
    }
};

}  // namespace pre_wgsl

#اگر_ختم  // PRE_WGSL_HPP
