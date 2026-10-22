#ifndef PRE_WGSL_HPP
#define PRE_WGSL_HPP

#অন্তর্ভাৱ <cctype>
#অন্তর্ভাৱ <fstream>
#অন্তর্ভাৱ <sstream>
#অন্তর্ভাৱ <stdexcept>
#অন্তর্ভাৱ <string>
#অন্তর্ভাৱ <string_view>
#অন্তর্ভাৱ <unordered_map>
#অন্তর্ভাৱ <unordered_set>
#অন্তর্ভাৱ <vector>

নামাকাশ pre_wgsl {

//==============================================================
// Options
//==============================================================
সংরচনা Options {
    std::string              include_path = ".";
    std::vector<std::string> macros;
};

//==============================================================
// Utility: trim
//==============================================================
স্থির std::string trim(ধ্রুৱ std::string & s) {
    size_t a = 0;
    যেতিয়ালৈ (a < s.size() && std::isspace((unsigned ৱর্ণ) s[a])) {
        a++;
    }
    size_t b = s.size();
    যেতিয়ালৈ (b > a && std::isspace((unsigned ৱর্ণ) s[b - 1])) {
        b--;
    }
    ঘূৰাই s.substr(a, b - a);
}

স্থির std::string trim_মান(std::istream & is) {
    std::ostringstream ss;
    ss << is.rdbuf();
    ঘূৰাই trim(ss.str());
}

স্থির তার্কিক isIdentChar(ৱর্ণ c) {
    ঘূৰাই std::isalnum(static_cast<unsigned ৱর্ণ>(c)) || c == '_';
}

স্থির তার্কিক endsWithContinuation(ধ্রুৱ std::string & সাৰি) {
    size_t i = সাৰি.size();
    যেতিয়ালৈ (i > 0 && std::isspace((unsigned ৱর্ণ) সাৰি[i - 1])) {
        i--;
    }
    ঘূৰাই i > 0 && সাৰি[i - 1] == '\\';
}

স্থির শূন্য stripContinuation(std::string & সাৰি) {
    size_t i = সাৰি.size();
    যেতিয়ালৈ (i > 0 && std::isspace((unsigned ৱর্ণ) সাৰি[i - 1])) {
        i--;
    }
    যদি (i > 0 && সাৰি[i - 1] == '\\') {
        সাৰি.erase(i - 1);
    }
}

স্থির std::string expandMacrosRecursiveInternal(ধ্রুৱ std::string &                                  সাৰি,
                                                 ধ্রুৱ std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting);

স্থির std::string expand_Macro_মান(ধ্রুৱ std::string &                                  name,
                                    ধ্রুৱ std::unordered_map<std::string, std::string> & macros,
                                    std::unordered_set<std::string> &                    visiting) {
    যদি (visiting.count(name)) {
        ক্ষিপ std::runtime_error("Recursive macro: " + name);
    }
    visiting.insert(name);

    auto it = macros.find(name);
    যদি (it == macros.end()) {
        visiting.erase(name);
        ঘূৰাই name;
    }

    ধ্রুৱ std::string & value = it->second;
    যদি (value.empty()) {
        visiting.erase(name);
        ঘূৰাই "";
    }

    std::string expanded = expandMacrosRecursiveInternal(value, macros, visiting);
    visiting.erase(name);
    ঘূৰাই expanded;
}

স্থির std::string expandMacrosRecursiveInternal(ধ্রুৱ std::string &                                  সাৰি,
                                                 ধ্রুৱ std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting) {
    std::string result;
    result.reserve(সাৰি.size());

    size_t i = 0;
    যেতিয়ালৈ (i < সাৰি.size()) {
        যদি (isIdentChar(সাৰি[i])) {
            size_t start = i;
            যেতিয়ালৈ (i < সাৰি.size() && isIdentChar(সাৰি[i])) {
                i++;
            }
            std::string token = সাৰি.substr(start, i - start);

            auto it = macros.find(token);
            যদি (it != macros.end()) {
                result += expand_Macro_মান(token, macros, visiting);
            } অন্যথা {
                result += token;
            }
        } অন্যথা {
            result += সাৰি[i];
            i++;
        }
    }

    ঘূৰাই result;
}

স্থির std::string expandMacrosRecursive(ধ্রুৱ std::string &                                  সাৰি,
                                         ধ্রুৱ std::unordered_map<std::string, std::string> & macros) {
    std::unordered_set<std::string> visiting;
    ঘূৰাই expandMacrosRecursiveInternal(সাৰি, macros, visiting);
}

//==============================================================
// Tokenizer for expressions in #if/#elif
//==============================================================
শ্ৰেণী ৰাশিমালা_Lexer {
  সার্ৱজনিক:
    গণনা Kind { END, IDENT, সংখ্যা_2, OP, LPAREN, RPAREN };

    সংরচনা Tok {
        Kind        kind;
        std::string text;
    };

    explicit ৰাশিমালা_Lexer(std::string_view sv) : src(sv), pos(0) {}

    Tok next() {
        skipWS();
        যদি (pos >= src.size()) {
            ঘূৰাই { END, "" };
        }

        ৱর্ণ c = src[pos];

        //সংখ্যা
        যদি (std::isdigit((unsigned ৱর্ণ) c)) {
            size_t start = pos;
            যেতিয়ালৈ (pos < src.size() && std::isdigit((unsigned ৱর্ণ) src[pos])) {
                pos++;
            }
            ঘূৰাই { সংখ্যা_2, std::string(src.substr(start, pos - start)) };
        }

        // identifier
        যদি (std::isalpha((unsigned ৱর্ণ) c) || c == '_') {
            size_t start = pos;
            যেতিয়ালৈ (pos < src.size() && (std::isalnum((unsigned ৱর্ণ) src[pos]) || src[pos] == '_')) {
                pos++;
            }
            ঘূৰাই { IDENT, std::string(src.substr(start, pos - start)) };
        }

        যদি (c == '(') {
            pos++;
            ঘূৰাই { LPAREN, "(" };
        }
        যদি (c == ')') {
            pos++;
            ঘূৰাই { RPAREN, ")" };
        }

        // multi-char operators
        স্থির ধ্রুৱ ৱর্ণ * two_ops[] = { "==", "!=", "<=", ">=", "&&", "||", "<<", ">>" };
        বাবে (auto op : two_ops) {
            যদি (src.substr(pos, 2) == op) {
                pos += 2;
                ঘূৰাই { OP, std::string(op) };
            }
        }

        // single-char operators
        যদি (std::string("+-*/%<>!").find(c) != std::string::npos) {
            pos++;
            ঘূৰাই { OP, std::string(1, c) };
        }

        // unexpected
        pos++;
        ঘূৰাই { END, "" };
    }

  গুপ্ত:
    std::string_view src;
    size_t           pos;

    শূন্য skipWS() {
        যেতিয়ালৈ (pos < src.size() && std::isspace((unsigned ৱর্ণ) src[pos])) {
            pos++;
        }
    }
};

//==============================================================
// Expression Parser (recursive descent)
//==============================================================
শ্ৰেণী ৰাশিমালা_Parser {
  সার্ৱজনিক:
    ৰাশিমালা_Parser(std::string_view                                     ৰাশিমালা,
               ধ্রুৱ std::unordered_map<std::string, std::string> & macros,
               std::unordered_set<std::string> &                    visiting) :
        lex(ৰাশিমালা),
        macros(macros),
        visiting(visiting) {
        advance();
    }

    পূর্ণাঙ্ক parse() { ঘূৰাই parseLogicalOr(); }

  গুপ্ত:
    ৰাশিমালা_Lexer                                            lex;
    ৰাশিমালা_Lexer::Tok                                       tok;
    ধ্রুৱ std::unordered_map<std::string, std::string> & macros;
    std::unordered_set<std::string> &                    visiting;

    শূন্য advance() { tok = lex.next(); }

    তার্কিক acceptOp(ধ্রুৱ std::string & s) {
        যদি (tok.kind == ৰাশিমালা_Lexer::OP && tok.text == s) {
            advance();
            ঘূৰাই সঁচা;
        }
        ঘূৰাই মিছা;
    }

    তার্কিক acceptKind(ৰাশিমালা_Lexer::Kind k) {
        যদি (tok.kind == k) {
            advance();
            ঘূৰাই সঁচা;
        }
        ঘূৰাই মিছা;
    }

    পূর্ণাঙ্ক parseLogicalOr() {
        পূর্ণাঙ্ক v = parseLogicalAnd();
        যেতিয়ালৈ (acceptOp("||")) {
            পূর্ণাঙ্ক rhs = parseLogicalAnd();
            v       = (v || rhs);
        }
        ঘূৰাই v;
    }

    পূর্ণাঙ্ক parseLogicalAnd() {
        পূর্ণাঙ্ক v = parseEquality();
        যেতিয়ালৈ (acceptOp("&&")) {
            পূর্ণাঙ্ক rhs = parseEquality();
            v       = (v && rhs);
        }
        ঘূৰাই v;
    }

    পূর্ণাঙ্ক parseEquality() {
        পূর্ণাঙ্ক v = parseRelational();
        বাবে (;;) {
            যদি (acceptOp("==")) {
                পূর্ণাঙ্ক rhs = parseRelational();
                v       = (v == rhs);
            } অন্যথা যদি (acceptOp("!=")) {
                পূর্ণাঙ্ক rhs = parseRelational();
                v       = (v != rhs);
            } অন্যথা {
                ভঙ্গ;
            }
        }
        ঘূৰাই v;
    }

    পূর্ণাঙ্ক parseRelational() {
        পূর্ণাঙ্ক v = parseShift();
        বাবে (;;) {
            যদি (acceptOp("<")) {
                পূর্ণাঙ্ক rhs = parseShift();
                v       = (v < rhs);
            } অন্যথা যদি (acceptOp(">")) {
                পূর্ণাঙ্ক rhs = parseShift();
                v       = (v > rhs);
            } অন্যথা যদি (acceptOp("<=")) {
                পূর্ণাঙ্ক rhs = parseShift();
                v       = (v <= rhs);
            } অন্যথা যদি (acceptOp(">=")) {
                পূর্ণাঙ্ক rhs = parseShift();
                v       = (v >= rhs);
            } অন্যথা {
                ভঙ্গ;
            }
        }
        ঘূৰাই v;
    }

    পূর্ণাঙ্ক parseShift() {
        পূর্ণাঙ্ক v = parseAdd();
        বাবে (;;) {
            যদি (acceptOp("<<")) {
                পূর্ণাঙ্ক rhs = parseAdd();
                v       = (v << rhs);
            } অন্যথা যদি (acceptOp(">>")) {
                পূর্ণাঙ্ক rhs = parseAdd();
                v       = (v >> rhs);
            } অন্যথা {
                ভঙ্গ;
            }
        }
        ঘূৰাই v;
    }

    পূর্ণাঙ্ক parseAdd() {
        পূর্ণাঙ্ক v = parseMult();
        বাবে (;;) {
            যদি (acceptOp("+")) {
                পূর্ণাঙ্ক rhs = parseMult();
                v       = (v + rhs);
            } অন্যথা যদি (acceptOp("-")) {
                পূর্ণাঙ্ক rhs = parseMult();
                v       = (v - rhs);
            } অন্যথা {
                ভঙ্গ;
            }
        }
        ঘূৰাই v;
    }

    পূর্ণাঙ্ক parseMult() {
        পূর্ণাঙ্ক v = parseUnary();
        বাবে (;;) {
            যদি (acceptOp("*")) {
                পূর্ণাঙ্ক rhs = parseUnary();
                v       = (v * rhs);
            } অন্যথা যদি (acceptOp("/")) {
                পূর্ণাঙ্ক rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v / rhs);
            } অন্যথা যদি (acceptOp("%")) {
                পূর্ণাঙ্ক rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v % rhs);
            } অন্যথা {
                ভঙ্গ;
            }
        }
        ঘূৰাই v;
    }

    পূর্ণাঙ্ক parseUnary() {
        যদি (acceptOp("!")) {
            ঘূৰাই !parseUnary();
        }
        যদি (acceptOp("-")) {
            ঘূৰাই -parseUnary();
        }
        যদি (acceptOp("+")) {
            ঘূৰাই +parseUnary();
        }
        ঘূৰাই parsePrimary();
    }

    পূর্ণাঙ্ক parsePrimary() {
        // '(' expr ')'
        যদি (acceptKind(ৰাশিমালা_Lexer::LPAREN)) {
            পূর্ণাঙ্ক v = parse();
            যদি (!acceptKind(ৰাশিমালা_Lexer::RPAREN)) {
                ক্ষিপ std::runtime_error("missing ')'");
            }
            ঘূৰাই v;
        }

        //সংখ্যা
        যদি (tok.kind == ৰাশিমালা_Lexer::সংখ্যা_2) {
            পূর্ণাঙ্ক v = std::stoi(tok.text);
            advance();
            ঘূৰাই v;
        }

        // defined(identifier)
        যদি (tok.kind == ৰাশিমালা_Lexer::IDENT && tok.text == "defined") {
            advance();
            যদি (acceptKind(ৰাশিমালা_Lexer::LPAREN)) {
                যদি (tok.kind != ৰাশিমালা_Lexer::IDENT) {
                    ক্ষিপ std::runtime_error("expected identifier in defined()");
                }
                std::string name = tok.text;
                advance();
                যদি (!acceptKind(ৰাশিমালা_Lexer::RPAREN)) {
                    ক্ষিপ std::runtime_error("missing ) in defined()");
                }
                ঘূৰাই macros.count(name) ? 1 : 0;
            } অন্যথা {
                // defined NAME
                যদি (tok.kind != ৰাশিমালা_Lexer::IDENT) {
                    ক্ষিপ std::runtime_error("expected identifier in defined NAME");
                }
                std::string name = tok.text;
                advance();
                ঘূৰাই macros.count(name) ? 1 : 0;
            }
        }

        // identifier -> treat as integer, if defined use its value else 0
        যদি (tok.kind == ৰাশিমালা_Lexer::IDENT) {
            std::string name = tok.text;
            advance();
            auto it = macros.find(name);
            যদি (it == macros.end()) {
                ঘূৰাই 0;
            }
            যদি (it->second.empty()) {
                ঘূৰাই 1;
            }
            ঘূৰাই evalMacroExpression(name, it->second);
        }

        // unexpected
        ঘূৰাই 0;
    }

    পূর্ণাঙ্ক evalMacroExpression(ধ্রুৱ std::string & name, ধ্রুৱ std::string & value) {
        যদি (visiting.count(name)) {
            ক্ষিপ std::runtime_error("Recursive macro: " + name);
        }

        visiting.insert(name);
        ৰাশিমালা_Parser ep(value, macros, visiting);
        পূর্ণাঙ্ক        v = ep.parse();
        visiting.erase(name);
        ঘূৰাই v;
    }
};

//==============================================================
// Preprocessor
//==============================================================
শ্ৰেণী Preprocessor {
  সার্ৱজনিক:
    explicit Preprocessor(Options opts = {}) : opts_(std::move(opts)) {
        // Treat empty include path as current directory
        যদি (opts_.include_path.empty()) {
            opts_.include_path = ".";
        }
        parseMacroDefinitions(opts_.macros);
    }

    std::string preprocess_file(ধ্রুৱ std::string & filename, ধ্রুৱ std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_স্তূপ;
        buildMacros(additional_macros, macros, predefined);

        std::string result = processFile(filename, macros, predefined, include_স্তূপ, DirectiveMode::All);
        ঘূৰাই result;
    }

    std::string preprocess(ধ্রুৱ std::string & contents, ধ্রুৱ std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_স্তূপ;
        buildMacros(additional_macros, macros, predefined);

        std::string result = processString(contents, macros, predefined, include_স্তূপ, DirectiveMode::All);
        ঘূৰাই result;
    }

    std::string preprocess_includes_file(ধ্রুৱ std::string & filename) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_স্তূপ;
        std::string result = processFile(filename, macros, predefined, include_স্তূপ, DirectiveMode::IncludesOnly);
        ঘূৰাই result;
    }

    std::string preprocess_includes(ধ্রুৱ std::string & contents) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_স্তূপ;
        std::string result = processString(contents, macros, predefined, include_স্তূপ, DirectiveMode::IncludesOnly);
        ঘূৰাই result;
    }

  গুপ্ত:
    Options                                      opts_;
    std::unordered_map<std::string, std::string> global_macros;

    গণনা শ্ৰেণী DirectiveMode { All, IncludesOnly };

    সংরচনা Cond {
        তার্কিক parent_active;
        তার্কিক active;
        তার্কিক taken;
    };

    //----------------------------------------------------------
    // Parse macro definitions into global_macros
    //----------------------------------------------------------
    শূন্য parseMacroDefinitions(ধ্রুৱ std::vector<std::string> & macro_defs) {
        বাবে (ধ্রুৱ auto & def : macro_defs) {
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
    শূন্য buildMacros(ধ্রুৱ std::vector<std::string> &               additional_macros,
                     std::unordered_map<std::string, std::string> & macros,
                     std::unordered_set<std::string> &              predefined) {
        macros = global_macros;
        predefined.clear();

        বাবে (ধ্রুৱ auto & [name, value] : global_macros) {
            predefined.insert(name);
        }

        বাবে (ধ্রুৱ auto & def : additional_macros) {
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
    std::string loadFile(ধ্রুৱ std::string & fname) {
        std::ifstream f(fname);
        যদি (!f.is_open()) {
            ক্ষিপ std::runtime_error("Could not open file: " + fname);
        }
        std::stringstream ss;
        ss << f.rdbuf();
        ঘূৰাই ss.str();
    }

    তার্কিক condActive(ধ্রুৱ std::vector<Cond> & cond) ধ্রুৱ {
        যদি (cond.empty()) {
            ঘূৰাই সঁচা;
        }
        ঘূৰাই cond.back().active;
    }

    //----------------------------------------------------------
    // Process a file
    //----------------------------------------------------------
    std::string processFile(ধ্রুৱ std::string &                            name,
                            std::unordered_map<std::string, std::string> & macros,
                            ধ্রুৱ std::unordered_set<std::string> &        predefined_macros,
                            std::unordered_set<std::string> &              include_স্তূপ,
                            DirectiveMode                                  mode) {
        যদি (include_স্তূপ.count(name)) {
            ক্ষিপ std::runtime_error("Recursive include: " + name);
        }

        include_স্তূপ.insert(name);
        std::string shader_code = loadFile(name);
        std::string out         = processString(shader_code, macros, predefined_macros, include_স্তূপ, mode);
        include_স্তূপ.erase(name);
        ঘূৰাই out;
    }

    std::string processIncludeFile(ধ্রুৱ std::string &                            fname,
                                   std::unordered_map<std::string, std::string> & macros,
                                   ধ্রুৱ std::unordered_set<std::string> &        predefined_macros,
                                   std::unordered_set<std::string> &              include_স্তূপ,
                                   DirectiveMode                                  mode) {
        std::string full_path = opts_.include_path + "/" + fname;
        ঘূৰাই processFile(full_path, macros, predefined_macros, include_স্তূপ, mode);
    }

    //----------------------------------------------------------
    // Process text
    //----------------------------------------------------------
    std::string processString(ধ্রুৱ std::string &                            shader_code,
                              std::unordered_map<std::string, std::string> & macros,
                              ধ্রুৱ std::unordered_set<std::string> &        predefined_macros,
                              std::unordered_set<std::string> &              include_স্তূপ,
                              DirectiveMode                                  mode) {
        std::vector<Cond>  cond;  // Conditional stack for this shader
        std::stringstream  out;
        std::istringstream in(shader_code);
        std::string        সাৰি;

        যেতিয়ালৈ (std::getline(in, সাৰি)) {
            std::string logical = সাৰি;
            std::string t       = trim(logical);
            যদি (!t.empty() && t[0] == '#') {
                যেতিয়ালৈ (endsWithContinuation(logical)) {
                    stripContinuation(logical);
                    যদি (!std::getline(in, সাৰি)) {
                        ভঙ্গ;
                    }
                    logical += "\n";
                    logical += সাৰি;
                }
                t = trim(logical);
            }

            যদি (!t.empty() && t[0] == '#') {
                তার্কিক handled = handleDirective(t, out, macros, predefined_macros, cond, include_স্তূপ, mode);
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
            ক্ষিপ std::runtime_error("Unclosed #if directive");
        }

        ঘূৰাই out.str();
    }

    //----------------------------------------------------------
    // Directive handler
    //----------------------------------------------------------
    তার্কিক handleDirective(ধ্রুৱ std::string &                            t,
                         std::stringstream &                            out,
                         std::unordered_map<std::string, std::string> & macros,
                         ধ্রুৱ std::unordered_set<std::string> &        predefined_macros,
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
                ঘূৰাই সঁচা;
            }
            std::string file;
            iss >> file;
            যদি (file.size() >= 2 && file.front() == '"' && file.back() == '"') {
                file = file.substr(1, file.size() - 2);
            }
            out << processIncludeFile(file, macros, predefined_macros, include_স্তূপ, mode);
            ঘূৰাই সঁচা;
        }

        যদি (mode == DirectiveMode::IncludesOnly) {
            ঘূৰাই মিছা;
        }

        যদি (cmd == "define") {
            যদি (!condActive(cond)) {
                ঘূৰাই সঁচা;
            }
            std::string name;
            iss >> name;
            // Don't override predefined macros from options
            যদি (predefined_macros.count(name)) {
                ঘূৰাই সঁচা;
            }
            std::string value = trim_মান(iss);
            macros[name]      = value;
            ঘূৰাই সঁচা;
        }

        যদি (cmd == "undef") {
            যদি (!condActive(cond)) {
                ঘূৰাই সঁচা;
            }
            std::string name;
            iss >> name;
            // Don't undef predefined macros from options
            যদি (predefined_macros.count(name)) {
                ঘূৰাই সঁচা;
            }
            macros.erase(name);
            ঘূৰাই সঁচা;
        }

        যদি (cmd == "ifdef") {
            std::string name;
            iss >> name;
            তার্কিক p = condActive(cond);
            তার্কিক v = macros.count(name);
            cond.push_back({ p, p && v, p && v });
            ঘূৰাই সঁচা;
        }

        যদি (cmd == "ifndef") {
            std::string name;
            iss >> name;
            তার্কিক p = condActive(cond);
            তার্কিক v = !macros.count(name);
            cond.push_back({ p, p && v, p && v });
            ঘূৰাই সঁচা;
        }

        যদি (cmd == "if") {
            std::string ৰাশিমালা = trim_মান(iss);
            তার্কিক        p    = condActive(cond);
            তার্কিক        v    = মিছা;
            যদি (p) {
                std::unordered_set<std::string> visiting;
                ৰাশিমালা_Parser                      ep(ৰাশিমালা, macros, visiting);
                v = ep.parse() != 0;
            }
            cond.push_back({ p, p && v, p && v });
            ঘূৰাই সঁচা;
        }

        যদি (cmd == "elif") {
            std::string ৰাশিমালা = trim_মান(iss);

            যদি (cond.empty()) {
                ক্ষিপ std::runtime_error("#elif without #if");
            }

            Cond & c = cond.back();
            যদি (!c.parent_active) {
                c.active = মিছা;
                ঘূৰাই সঁচা;
            }

            যদি (c.taken) {
                c.active = মিছা;
                ঘূৰাই সঁচা;
            }

            std::unordered_set<std::string> visiting;
            ৰাশিমালা_Parser                      ep(ৰাশিমালা, macros, visiting);
            তার্কিক                            v = ep.parse() != 0;
            c.active                          = v;
            যদি (v) {
                c.taken = সঁচা;
            }
            ঘূৰাই সঁচা;
        }

        যদি (cmd == "else") {
            যদি (cond.empty()) {
                ক্ষিপ std::runtime_error("#else without #if");
            }

            Cond & c = cond.back();
            যদি (!c.parent_active) {
                c.active = মিছা;
                ঘূৰাই সঁচা;
            }
            যদি (c.taken) {
                c.active = মিছা;
            } অন্যথা {
                c.active = সঁচা;
                c.taken  = সঁচা;
            }
            ঘূৰাই সঁচা;
        }

        যদি (cmd == "endif") {
            যদি (cond.empty()) {
                ক্ষিপ std::runtime_error("#endif without #if");
            }
            cond.pop_back();
            ঘূৰাই সঁচা;
        }

        // Unknown directive
        ক্ষিপ std::runtime_error("Unknown directive: #" + cmd);
    }
};

}  // namespace pre_wgsl

#endif  // PRE_WGSL_HPP
