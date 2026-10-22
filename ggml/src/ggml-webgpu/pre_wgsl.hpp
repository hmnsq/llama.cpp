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
সংরচনা Options {
    std::string              include_path = ".";
    std::vector<std::string> macros;
};

//==============================================================
// Utility: trim
//==============================================================
স্থির std::string trim(ধ্রুব std::string & s) {
    size_t a = 0;
    যাবত্ (a < s.size() && std::isspace((unsigned বর্ণ) s[a])) {
        a++;
    }
    size_t b = s.size();
    যাবত্ (b > a && std::isspace((unsigned বর্ণ) s[b - 1])) {
        b--;
    }
    প্রত্যাগম s.substr(a, b - a);
}

স্থির std::string trim_মান(std::istream & is) {
    std::ostringstream ss;
    ss << is.rdbuf();
    প্রত্যাগম trim(ss.str());
}

স্থির তার্কিক isIdentChar(বর্ণ c) {
    প্রত্যাগম std::isalnum(static_cast<unsigned বর্ণ>(c)) || c == '_';
}

স্থির তার্কিক endsWithContinuation(ধ্রুব std::string & সারি) {
    size_t i = সারি.size();
    যাবত্ (i > 0 && std::isspace((unsigned বর্ণ) সারি[i - 1])) {
        i--;
    }
    প্রত্যাগম i > 0 && সারি[i - 1] == '\\';
}

স্থির শূন্য stripContinuation(std::string & সারি) {
    size_t i = সারি.size();
    যাবত্ (i > 0 && std::isspace((unsigned বর্ণ) সারি[i - 1])) {
        i--;
    }
    যদি (i > 0 && সারি[i - 1] == '\\') {
        সারি.erase(i - 1);
    }
}

স্থির std::string expandMacrosRecursiveInternal(ধ্রুব std::string &                                  সারি,
                                                 ধ্রুব std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting);

স্থির std::string expand_Macro_মান(ধ্রুব std::string &                                  name,
                                    ধ্রুব std::unordered_map<std::string, std::string> & macros,
                                    std::unordered_set<std::string> &                    visiting) {
    যদি (visiting.count(name)) {
        ক্ষিপ std::runtime_error("Recursive macro: " + name);
    }
    visiting.insert(name);

    auto it = macros.find(name);
    যদি (it == macros.end()) {
        visiting.erase(name);
        প্রত্যাগম name;
    }

    ধ্রুব std::string & value = it->second;
    যদি (value.empty()) {
        visiting.erase(name);
        প্রত্যাগম "";
    }

    std::string expanded = expandMacrosRecursiveInternal(value, macros, visiting);
    visiting.erase(name);
    প্রত্যাগম expanded;
}

স্থির std::string expandMacrosRecursiveInternal(ধ্রুব std::string &                                  সারি,
                                                 ধ্রুব std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting) {
    std::string result;
    result.reserve(সারি.size());

    size_t i = 0;
    যাবত্ (i < সারি.size()) {
        যদি (isIdentChar(সারি[i])) {
            size_t start = i;
            যাবত্ (i < সারি.size() && isIdentChar(সারি[i])) {
                i++;
            }
            std::string token = সারি.substr(start, i - start);

            auto it = macros.find(token);
            যদি (it != macros.end()) {
                result += expand_Macro_মান(token, macros, visiting);
            } অন্যথা {
                result += token;
            }
        } অন্যথা {
            result += সারি[i];
            i++;
        }
    }

    প্রত্যাগম result;
}

স্থির std::string expandMacrosRecursive(ধ্রুব std::string &                                  সারি,
                                         ধ্রুব std::unordered_map<std::string, std::string> & macros) {
    std::unordered_set<std::string> visiting;
    প্রত্যাগম expandMacrosRecursiveInternal(সারি, macros, visiting);
}

//==============================================================
// Tokenizer for expressions in #if/#elif
//==============================================================
বর্গ রাশিমালা_Lexer {
  সার্বজনিক:
    গণনা Kind { END, IDENT, সংখ্যা_2, OP, LPAREN, RPAREN };

    সংরচনা Tok {
        Kind        kind;
        std::string text;
    };

    explicit রাশিমালা_Lexer(std::string_view sv) : src(sv), pos(0) {}

    Tok next() {
        skipWS();
        যদি (pos >= src.size()) {
            প্রত্যাগম { END, "" };
        }

        বর্ণ c = src[pos];

        //সংখ্যা
        যদি (std::isdigit((unsigned বর্ণ) c)) {
            size_t start = pos;
            যাবত্ (pos < src.size() && std::isdigit((unsigned বর্ণ) src[pos])) {
                pos++;
            }
            প্রত্যাগম { সংখ্যা_2, std::string(src.substr(start, pos - start)) };
        }

        // identifier
        যদি (std::isalpha((unsigned বর্ণ) c) || c == '_') {
            size_t start = pos;
            যাবত্ (pos < src.size() && (std::isalnum((unsigned বর্ণ) src[pos]) || src[pos] == '_')) {
                pos++;
            }
            প্রত্যাগম { IDENT, std::string(src.substr(start, pos - start)) };
        }

        যদি (c == '(') {
            pos++;
            প্রত্যাগম { LPAREN, "(" };
        }
        যদি (c == ')') {
            pos++;
            প্রত্যাগম { RPAREN, ")" };
        }

        // multi-char operators
        স্থির ধ্রুব বর্ণ * two_ops[] = { "==", "!=", "<=", ">=", "&&", "||", "<<", ">>" };
        প্রতি (auto op : two_ops) {
            যদি (src.substr(pos, 2) == op) {
                pos += 2;
                প্রত্যাগম { OP, std::string(op) };
            }
        }

        // single-char operators
        যদি (std::string("+-*/%<>!").find(c) != std::string::npos) {
            pos++;
            প্রত্যাগম { OP, std::string(1, c) };
        }

        // unexpected
        pos++;
        প্রত্যাগম { END, "" };
    }

  গুপ্ত:
    std::string_view src;
    size_t           pos;

    শূন্য skipWS() {
        যাবত্ (pos < src.size() && std::isspace((unsigned বর্ণ) src[pos])) {
            pos++;
        }
    }
};

//==============================================================
// Expression Parser (recursive descent)
//==============================================================
বর্গ রাশিমালা_Parser {
  সার্বজনিক:
    রাশিমালা_Parser(std::string_view                                     রাশিমালা,
               ধ্রুব std::unordered_map<std::string, std::string> & macros,
               std::unordered_set<std::string> &                    visiting) :
        lex(রাশিমালা),
        macros(macros),
        visiting(visiting) {
        advance();
    }

    পূর্ণাঙ্ক parse() { প্রত্যাগম parseLogicalOr(); }

  গুপ্ত:
    রাশিমালা_Lexer                                            lex;
    রাশিমালা_Lexer::Tok                                       tok;
    ধ্রুব std::unordered_map<std::string, std::string> & macros;
    std::unordered_set<std::string> &                    visiting;

    শূন্য advance() { tok = lex.next(); }

    তার্কিক acceptOp(ধ্রুব std::string & s) {
        যদি (tok.kind == রাশিমালা_Lexer::OP && tok.text == s) {
            advance();
            প্রত্যাগম সত্য;
        }
        প্রত্যাগম অসত্য;
    }

    তার্কিক acceptKind(রাশিমালা_Lexer::Kind k) {
        যদি (tok.kind == k) {
            advance();
            প্রত্যাগম সত্য;
        }
        প্রত্যাগম অসত্য;
    }

    পূর্ণাঙ্ক parseLogicalOr() {
        পূর্ণাঙ্ক v = parseLogicalAnd();
        যাবত্ (acceptOp("||")) {
            পূর্ণাঙ্ক rhs = parseLogicalAnd();
            v       = (v || rhs);
        }
        প্রত্যাগম v;
    }

    পূর্ণাঙ্ক parseLogicalAnd() {
        পূর্ণাঙ্ক v = parseEquality();
        যাবত্ (acceptOp("&&")) {
            পূর্ণাঙ্ক rhs = parseEquality();
            v       = (v && rhs);
        }
        প্রত্যাগম v;
    }

    পূর্ণাঙ্ক parseEquality() {
        পূর্ণাঙ্ক v = parseRelational();
        প্রতি (;;) {
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
        প্রত্যাগম v;
    }

    পূর্ণাঙ্ক parseRelational() {
        পূর্ণাঙ্ক v = parseShift();
        প্রতি (;;) {
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
        প্রত্যাগম v;
    }

    পূর্ণাঙ্ক parseShift() {
        পূর্ণাঙ্ক v = parseAdd();
        প্রতি (;;) {
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
        প্রত্যাগম v;
    }

    পূর্ণাঙ্ক parseAdd() {
        পূর্ণাঙ্ক v = parseMult();
        প্রতি (;;) {
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
        প্রত্যাগম v;
    }

    পূর্ণাঙ্ক parseMult() {
        পূর্ণাঙ্ক v = parseUnary();
        প্রতি (;;) {
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
        প্রত্যাগম v;
    }

    পূর্ণাঙ্ক parseUnary() {
        যদি (acceptOp("!")) {
            প্রত্যাগম !parseUnary();
        }
        যদি (acceptOp("-")) {
            প্রত্যাগম -parseUnary();
        }
        যদি (acceptOp("+")) {
            প্রত্যাগম +parseUnary();
        }
        প্রত্যাগম parsePrimary();
    }

    পূর্ণাঙ্ক parsePrimary() {
        // '(' expr ')'
        যদি (acceptKind(রাশিমালা_Lexer::LPAREN)) {
            পূর্ণাঙ্ক v = parse();
            যদি (!acceptKind(রাশিমালা_Lexer::RPAREN)) {
                ক্ষিপ std::runtime_error("missing ')'");
            }
            প্রত্যাগম v;
        }

        //সংখ্যা
        যদি (tok.kind == রাশিমালা_Lexer::সংখ্যা_2) {
            পূর্ণাঙ্ক v = std::stoi(tok.text);
            advance();
            প্রত্যাগম v;
        }

        // defined(identifier)
        যদি (tok.kind == রাশিমালা_Lexer::IDENT && tok.text == "defined") {
            advance();
            যদি (acceptKind(রাশিমালা_Lexer::LPAREN)) {
                যদি (tok.kind != রাশিমালা_Lexer::IDENT) {
                    ক্ষিপ std::runtime_error("expected identifier in defined()");
                }
                std::string name = tok.text;
                advance();
                যদি (!acceptKind(রাশিমালা_Lexer::RPAREN)) {
                    ক্ষিপ std::runtime_error("missing ) in defined()");
                }
                প্রত্যাগম macros.count(name) ? 1 : 0;
            } অন্যথা {
                // defined NAME
                যদি (tok.kind != রাশিমালা_Lexer::IDENT) {
                    ক্ষিপ std::runtime_error("expected identifier in defined NAME");
                }
                std::string name = tok.text;
                advance();
                প্রত্যাগম macros.count(name) ? 1 : 0;
            }
        }

        // identifier -> treat as integer, if defined use its value else 0
        যদি (tok.kind == রাশিমালা_Lexer::IDENT) {
            std::string name = tok.text;
            advance();
            auto it = macros.find(name);
            যদি (it == macros.end()) {
                প্রত্যাগম 0;
            }
            যদি (it->second.empty()) {
                প্রত্যাগম 1;
            }
            প্রত্যাগম evalMacroExpression(name, it->second);
        }

        // unexpected
        প্রত্যাগম 0;
    }

    পূর্ণাঙ্ক evalMacroExpression(ধ্রুব std::string & name, ধ্রুব std::string & value) {
        যদি (visiting.count(name)) {
            ক্ষিপ std::runtime_error("Recursive macro: " + name);
        }

        visiting.insert(name);
        রাশিমালা_Parser ep(value, macros, visiting);
        পূর্ণাঙ্ক        v = ep.parse();
        visiting.erase(name);
        প্রত্যাগম v;
    }
};

//==============================================================
// Preprocessor
//==============================================================
বর্গ Preprocessor {
  সার্বজনিক:
    explicit Preprocessor(Options opts = {}) : opts_(std::move(opts)) {
        // Treat empty include path as current directory
        যদি (opts_.include_path.empty()) {
            opts_.include_path = ".";
        }
        parseMacroDefinitions(opts_.macros);
    }

    std::string preprocess_file(ধ্রুব std::string & filename, ধ্রুব std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_স্তূপ;
        buildMacros(additional_macros, macros, predefined);

        std::string result = processFile(filename, macros, predefined, include_স্তূপ, DirectiveMode::All);
        প্রত্যাগম result;
    }

    std::string preprocess(ধ্রুব std::string & contents, ধ্রুব std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_স্তূপ;
        buildMacros(additional_macros, macros, predefined);

        std::string result = processString(contents, macros, predefined, include_স্তূপ, DirectiveMode::All);
        প্রত্যাগম result;
    }

    std::string preprocess_includes_file(ধ্রুব std::string & filename) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_স্তূপ;
        std::string result = processFile(filename, macros, predefined, include_স্তূপ, DirectiveMode::IncludesOnly);
        প্রত্যাগম result;
    }

    std::string preprocess_includes(ধ্রুব std::string & contents) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_স্তূপ;
        std::string result = processString(contents, macros, predefined, include_স্তূপ, DirectiveMode::IncludesOnly);
        প্রত্যাগম result;
    }

  গুপ্ত:
    Options                                      opts_;
    std::unordered_map<std::string, std::string> global_macros;

    গণনা বর্গ DirectiveMode { All, IncludesOnly };

    সংরচনা Cond {
        তার্কিক parent_active;
        তার্কিক active;
        তার্কিক taken;
    };

    //----------------------------------------------------------
    // Parse macro definitions into global_macros
    //----------------------------------------------------------
    শূন্য parseMacroDefinitions(ধ্রুব std::vector<std::string> & macro_defs) {
        প্রতি (ধ্রুব auto & def : macro_defs) {
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
    শূন্য buildMacros(ধ্রুব std::vector<std::string> &               additional_macros,
                     std::unordered_map<std::string, std::string> & macros,
                     std::unordered_set<std::string> &              predefined) {
        macros = global_macros;
        predefined.clear();

        প্রতি (ধ্রুব auto & [name, value] : global_macros) {
            predefined.insert(name);
        }

        প্রতি (ধ্রুব auto & def : additional_macros) {
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
    std::string loadFile(ধ্রুব std::string & fname) {
        std::ifstream f(fname);
        যদি (!f.is_open()) {
            ক্ষিপ std::runtime_error("Could not open file: " + fname);
        }
        std::stringstream ss;
        ss << f.rdbuf();
        প্রত্যাগম ss.str();
    }

    তার্কিক condActive(ধ্রুব std::vector<Cond> & cond) ধ্রুব {
        যদি (cond.empty()) {
            প্রত্যাগম সত্য;
        }
        প্রত্যাগম cond.back().active;
    }

    //----------------------------------------------------------
    // Process a file
    //----------------------------------------------------------
    std::string processFile(ধ্রুব std::string &                            name,
                            std::unordered_map<std::string, std::string> & macros,
                            ধ্রুব std::unordered_set<std::string> &        predefined_macros,
                            std::unordered_set<std::string> &              include_স্তূপ,
                            DirectiveMode                                  mode) {
        যদি (include_স্তূপ.count(name)) {
            ক্ষিপ std::runtime_error("Recursive include: " + name);
        }

        include_স্তূপ.insert(name);
        std::string shader_code = loadFile(name);
        std::string out         = processString(shader_code, macros, predefined_macros, include_স্তূপ, mode);
        include_স্তূপ.erase(name);
        প্রত্যাগম out;
    }

    std::string processIncludeFile(ধ্রুব std::string &                            fname,
                                   std::unordered_map<std::string, std::string> & macros,
                                   ধ্রুব std::unordered_set<std::string> &        predefined_macros,
                                   std::unordered_set<std::string> &              include_স্তূপ,
                                   DirectiveMode                                  mode) {
        std::string full_path = opts_.include_path + "/" + fname;
        প্রত্যাগম processFile(full_path, macros, predefined_macros, include_স্তূপ, mode);
    }

    //----------------------------------------------------------
    // Process text
    //----------------------------------------------------------
    std::string processString(ধ্রুব std::string &                            shader_code,
                              std::unordered_map<std::string, std::string> & macros,
                              ধ্রুব std::unordered_set<std::string> &        predefined_macros,
                              std::unordered_set<std::string> &              include_স্তূপ,
                              DirectiveMode                                  mode) {
        std::vector<Cond>  cond;  // Conditional stack for this shader
        std::stringstream  out;
        std::istringstream in(shader_code);
        std::string        সারি;

        যাবত্ (std::getline(in, সারি)) {
            std::string logical = সারি;
            std::string t       = trim(logical);
            যদি (!t.empty() && t[0] == '#') {
                যাবত্ (endsWithContinuation(logical)) {
                    stripContinuation(logical);
                    যদি (!std::getline(in, সারি)) {
                        ভঙ্গ;
                    }
                    logical += "\n";
                    logical += সারি;
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

        প্রত্যাগম out.str();
    }

    //----------------------------------------------------------
    // Directive handler
    //----------------------------------------------------------
    তার্কিক handleDirective(ধ্রুব std::string &                            t,
                         std::stringstream &                            out,
                         std::unordered_map<std::string, std::string> & macros,
                         ধ্রুব std::unordered_set<std::string> &        predefined_macros,
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
                প্রত্যাগম সত্য;
            }
            std::string file;
            iss >> file;
            যদি (file.size() >= 2 && file.front() == '"' && file.back() == '"') {
                file = file.substr(1, file.size() - 2);
            }
            out << processIncludeFile(file, macros, predefined_macros, include_স্তূপ, mode);
            প্রত্যাগম সত্য;
        }

        যদি (mode == DirectiveMode::IncludesOnly) {
            প্রত্যাগম অসত্য;
        }

        যদি (cmd == "define") {
            যদি (!condActive(cond)) {
                প্রত্যাগম সত্য;
            }
            std::string name;
            iss >> name;
            // Don't override predefined macros from options
            যদি (predefined_macros.count(name)) {
                প্রত্যাগম সত্য;
            }
            std::string value = trim_মান(iss);
            macros[name]      = value;
            প্রত্যাগম সত্য;
        }

        যদি (cmd == "undef") {
            যদি (!condActive(cond)) {
                প্রত্যাগম সত্য;
            }
            std::string name;
            iss >> name;
            // Don't undef predefined macros from options
            যদি (predefined_macros.count(name)) {
                প্রত্যাগম সত্য;
            }
            macros.erase(name);
            প্রত্যাগম সত্য;
        }

        যদি (cmd == "ifdef") {
            std::string name;
            iss >> name;
            তার্কিক p = condActive(cond);
            তার্কিক v = macros.count(name);
            cond.push_back({ p, p && v, p && v });
            প্রত্যাগম সত্য;
        }

        যদি (cmd == "ifndef") {
            std::string name;
            iss >> name;
            তার্কিক p = condActive(cond);
            তার্কিক v = !macros.count(name);
            cond.push_back({ p, p && v, p && v });
            প্রত্যাগম সত্য;
        }

        যদি (cmd == "if") {
            std::string রাশিমালা = trim_মান(iss);
            তার্কিক        p    = condActive(cond);
            তার্কিক        v    = অসত্য;
            যদি (p) {
                std::unordered_set<std::string> visiting;
                রাশিমালা_Parser                      ep(রাশিমালা, macros, visiting);
                v = ep.parse() != 0;
            }
            cond.push_back({ p, p && v, p && v });
            প্রত্যাগম সত্য;
        }

        যদি (cmd == "elif") {
            std::string রাশিমালা = trim_মান(iss);

            যদি (cond.empty()) {
                ক্ষিপ std::runtime_error("#elif without #if");
            }

            Cond & c = cond.back();
            যদি (!c.parent_active) {
                c.active = অসত্য;
                প্রত্যাগম সত্য;
            }

            যদি (c.taken) {
                c.active = অসত্য;
                প্রত্যাগম সত্য;
            }

            std::unordered_set<std::string> visiting;
            রাশিমালা_Parser                      ep(রাশিমালা, macros, visiting);
            তার্কিক                            v = ep.parse() != 0;
            c.active                          = v;
            যদি (v) {
                c.taken = সত্য;
            }
            প্রত্যাগম সত্য;
        }

        যদি (cmd == "else") {
            যদি (cond.empty()) {
                ক্ষিপ std::runtime_error("#else without #if");
            }

            Cond & c = cond.back();
            যদি (!c.parent_active) {
                c.active = অসত্য;
                প্রত্যাগম সত্য;
            }
            যদি (c.taken) {
                c.active = অসত্য;
            } অন্যথা {
                c.active = সত্য;
                c.taken  = সত্য;
            }
            প্রত্যাগম সত্য;
        }

        যদি (cmd == "endif") {
            যদি (cond.empty()) {
                ক্ষিপ std::runtime_error("#endif without #if");
            }
            cond.pop_back();
            প্রত্যাগম সত্য;
        }

        // Unknown directive
        ক্ষিপ std::runtime_error("Unknown directive: #" + cmd);
    }
};

}  // namespace pre_wgsl

#endif  // PRE_WGSL_HPP
