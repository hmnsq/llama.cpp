#ifndef PRE_WGSL_HPP
#define PRE_WGSL_HPP

#ଅନ୍ତର୍ଭାଵ <cctype>
#ଅନ୍ତର୍ଭାଵ <fstream>
#ଅନ୍ତର୍ଭାଵ <sstream>
#ଅନ୍ତର୍ଭାଵ <stdexcept>
#ଅନ୍ତର୍ଭାଵ <string>
#ଅନ୍ତର୍ଭାଵ <string_view>
#ଅନ୍ତର୍ଭାଵ <unordered_map>
#ଅନ୍ତର୍ଭାଵ <unordered_set>
#ଅନ୍ତର୍ଭାଵ <vector>

ନାମାକାଶ pre_wgsl {

//==============================================================
// Options
//==============================================================
ସଂରଚନା Options {
    std::string              include_path = ".";
    std::vector<std::string> macros;
};

//==============================================================
// Utility: trim
//==============================================================
ସ୍ଥିର std::string trim(ଅଚଳ std::string & s) {
    size_t a = 0;
    ଯେତେବେଳେ (a < s.size() && std::isspace((unsigned ଅକ୍ଷର) s[a])) {
        a++;
    }
    size_t b = s.size();
    ଯେତେବେଳେ (b > a && std::isspace((unsigned ଅକ୍ଷର) s[b - 1])) {
        b--;
    }
    ଫେରନ୍ତୁ s.substr(a, b - a);
}

ସ୍ଥିର std::string trim_ମୂଲ୍ୟ(std::istream & is) {
    std::ostringstream ss;
    ss << is.rdbuf();
    ଫେରନ୍ତୁ trim(ss.str());
}

ସ୍ଥିର ତାର୍କିକ isIdentChar(ଅକ୍ଷର c) {
    ଫେରନ୍ତୁ std::isalnum(static_cast<unsigned ଅକ୍ଷର>(c)) || c == '_';
}

ସ୍ଥିର ତାର୍କିକ endsWithContinuation(ଅଚଳ std::string & ଧାଡ଼ି) {
    size_t i = ଧାଡ଼ି.size();
    ଯେତେବେଳେ (i > 0 && std::isspace((unsigned ଅକ୍ଷର) ଧାଡ଼ି[i - 1])) {
        i--;
    }
    ଫେରନ୍ତୁ i > 0 && ଧାଡ଼ି[i - 1] == '\\';
}

ସ୍ଥିର ଶୂନ୍ୟ stripContinuation(std::string & ଧାଡ଼ି) {
    size_t i = ଧାଡ଼ି.size();
    ଯେତେବେଳେ (i > 0 && std::isspace((unsigned ଅକ୍ଷର) ଧାଡ଼ି[i - 1])) {
        i--;
    }
    ଯଦି (i > 0 && ଧାଡ଼ି[i - 1] == '\\') {
        ଧାଡ଼ି.erase(i - 1);
    }
}

ସ୍ଥିର std::string expandMacrosRecursiveInternal(ଅଚଳ std::string &                                  ଧାଡ଼ି,
                                                 ଅଚଳ std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting);

ସ୍ଥିର std::string expand_Macro_ମୂଲ୍ୟ(ଅଚଳ std::string &                                  name,
                                    ଅଚଳ std::unordered_map<std::string, std::string> & macros,
                                    std::unordered_set<std::string> &                    visiting) {
    ଯଦି (visiting.count(name)) {
        ଫିଙ୍ଗ std::runtime_error("Recursive macro: " + name);
    }
    visiting.insert(name);

    auto it = macros.find(name);
    ଯଦି (it == macros.end()) {
        visiting.erase(name);
        ଫେରନ୍ତୁ name;
    }

    ଅଚଳ std::string & value = it->second;
    ଯଦି (value.empty()) {
        visiting.erase(name);
        ଫେରନ୍ତୁ "";
    }

    std::string expanded = expandMacrosRecursiveInternal(value, macros, visiting);
    visiting.erase(name);
    ଫେରନ୍ତୁ expanded;
}

ସ୍ଥିର std::string expandMacrosRecursiveInternal(ଅଚଳ std::string &                                  ଧାଡ଼ି,
                                                 ଅଚଳ std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting) {
    std::string result;
    result.reserve(ଧାଡ଼ି.size());

    size_t i = 0;
    ଯେତେବେଳେ (i < ଧାଡ଼ି.size()) {
        ଯଦି (isIdentChar(ଧାଡ଼ି[i])) {
            size_t start = i;
            ଯେତେବେଳେ (i < ଧାଡ଼ି.size() && isIdentChar(ଧାଡ଼ି[i])) {
                i++;
            }
            std::string token = ଧାଡ଼ି.substr(start, i - start);

            auto it = macros.find(token);
            ଯଦି (it != macros.end()) {
                result += expand_Macro_ମୂଲ୍ୟ(token, macros, visiting);
            } ଅନ୍ୟଥା {
                result += token;
            }
        } ଅନ୍ୟଥା {
            result += ଧାଡ଼ି[i];
            i++;
        }
    }

    ଫେରନ୍ତୁ result;
}

ସ୍ଥିର std::string expandMacrosRecursive(ଅଚଳ std::string &                                  ଧାଡ଼ି,
                                         ଅଚଳ std::unordered_map<std::string, std::string> & macros) {
    std::unordered_set<std::string> visiting;
    ଫେରନ୍ତୁ expandMacrosRecursiveInternal(ଧାଡ଼ି, macros, visiting);
}

//==============================================================
// Tokenizer for expressions in #if/#elif
//==============================================================
ଶ୍ରେଣୀ ପଦସମଷ୍ଟି_Lexer {
  ସର୍ବସାଧାରଣ:
    ଗଣନା Kind { END, IDENT, ସଂଖ୍ୟା_2, OP, LPAREN, RPAREN };

    ସଂରଚନା Tok {
        Kind        kind;
        std::string text;
    };

    explicit ପଦସମଷ୍ଟି_Lexer(std::string_view sv) : src(sv), pos(0) {}

    Tok next() {
        skipWS();
        ଯଦି (pos >= src.size()) {
            ଫେରନ୍ତୁ { END, "" };
        }

        ଅକ୍ଷର c = src[pos];

        //ସଂଖ୍ୟା
        ଯଦି (std::isdigit((unsigned ଅକ୍ଷର) c)) {
            size_t start = pos;
            ଯେତେବେଳେ (pos < src.size() && std::isdigit((unsigned ଅକ୍ଷର) src[pos])) {
                pos++;
            }
            ଫେରନ୍ତୁ { ସଂଖ୍ୟା_2, std::string(src.substr(start, pos - start)) };
        }

        // identifier
        ଯଦି (std::isalpha((unsigned ଅକ୍ଷର) c) || c == '_') {
            size_t start = pos;
            ଯେତେବେଳେ (pos < src.size() && (std::isalnum((unsigned ଅକ୍ଷର) src[pos]) || src[pos] == '_')) {
                pos++;
            }
            ଫେରନ୍ତୁ { IDENT, std::string(src.substr(start, pos - start)) };
        }

        ଯଦି (c == '(') {
            pos++;
            ଫେରନ୍ତୁ { LPAREN, "(" };
        }
        ଯଦି (c == ')') {
            pos++;
            ଫେରନ୍ତୁ { RPAREN, ")" };
        }

        // multi-char operators
        ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * two_ops[] = { "==", "!=", "<=", ">=", "&&", "||", "<<", ">>" };
        ପାଇଁ (auto op : two_ops) {
            ଯଦି (src.substr(pos, 2) == op) {
                pos += 2;
                ଫେରନ୍ତୁ { OP, std::string(op) };
            }
        }

        // single-char operators
        ଯଦି (std::string("+-*/%<>!").find(c) != std::string::npos) {
            pos++;
            ଫେରନ୍ତୁ { OP, std::string(1, c) };
        }

        // unexpected
        pos++;
        ଫେରନ୍ତୁ { END, "" };
    }

  ବ୍ୟକ୍ତିଗତ:
    std::string_view src;
    size_t           pos;

    ଶୂନ୍ୟ skipWS() {
        ଯେତେବେଳେ (pos < src.size() && std::isspace((unsigned ଅକ୍ଷର) src[pos])) {
            pos++;
        }
    }
};

//==============================================================
// Expression Parser (recursive descent)
//==============================================================
ଶ୍ରେଣୀ ପଦସମଷ୍ଟି_Parser {
  ସର୍ବସାଧାରଣ:
    ପଦସମଷ୍ଟି_Parser(std::string_view                                     ପଦସମଷ୍ଟି,
               ଅଚଳ std::unordered_map<std::string, std::string> & macros,
               std::unordered_set<std::string> &                    visiting) :
        lex(ପଦସମଷ୍ଟି),
        macros(macros),
        visiting(visiting) {
        advance();
    }

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା parse() { ଫେରନ୍ତୁ parseLogicalOr(); }

  ବ୍ୟକ୍ତିଗତ:
    ପଦସମଷ୍ଟି_Lexer                                            lex;
    ପଦସମଷ୍ଟି_Lexer::Tok                                       tok;
    ଅଚଳ std::unordered_map<std::string, std::string> & macros;
    std::unordered_set<std::string> &                    visiting;

    ଶୂନ୍ୟ advance() { tok = lex.next(); }

    ତାର୍କିକ acceptOp(ଅଚଳ std::string & s) {
        ଯଦି (tok.kind == ପଦସମଷ୍ଟି_Lexer::OP && tok.text == s) {
            advance();
            ଫେରନ୍ତୁ ସତ୍ୟ;
        }
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ତାର୍କିକ acceptKind(ପଦସମଷ୍ଟି_Lexer::Kind k) {
        ଯଦି (tok.kind == k) {
            advance();
            ଫେରନ୍ତୁ ସତ୍ୟ;
        }
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା parseLogicalOr() {
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା v = parseLogicalAnd();
        ଯେତେବେଳେ (acceptOp("||")) {
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା rhs = parseLogicalAnd();
            v       = (v || rhs);
        }
        ଫେରନ୍ତୁ v;
    }

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା parseLogicalAnd() {
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା v = parseEquality();
        ଯେତେବେଳେ (acceptOp("&&")) {
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା rhs = parseEquality();
            v       = (v && rhs);
        }
        ଫେରନ୍ତୁ v;
    }

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା parseEquality() {
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା v = parseRelational();
        ପାଇଁ (;;) {
            ଯଦି (acceptOp("==")) {
                ପୂର୍ଣ୍ଣସଂଖ୍ୟା rhs = parseRelational();
                v       = (v == rhs);
            } ଅନ୍ୟଥା ଯଦି (acceptOp("!=")) {
                ପୂର୍ଣ୍ଣସଂଖ୍ୟା rhs = parseRelational();
                v       = (v != rhs);
            } ଅନ୍ୟଥା {
                ଭାଙ୍ଗ;
            }
        }
        ଫେରନ୍ତୁ v;
    }

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା parseRelational() {
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା v = parseShift();
        ପାଇଁ (;;) {
            ଯଦି (acceptOp("<")) {
                ପୂର୍ଣ୍ଣସଂଖ୍ୟା rhs = parseShift();
                v       = (v < rhs);
            } ଅନ୍ୟଥା ଯଦି (acceptOp(">")) {
                ପୂର୍ଣ୍ଣସଂଖ୍ୟା rhs = parseShift();
                v       = (v > rhs);
            } ଅନ୍ୟଥା ଯଦି (acceptOp("<=")) {
                ପୂର୍ଣ୍ଣସଂଖ୍ୟା rhs = parseShift();
                v       = (v <= rhs);
            } ଅନ୍ୟଥା ଯଦି (acceptOp(">=")) {
                ପୂର୍ଣ୍ଣସଂଖ୍ୟା rhs = parseShift();
                v       = (v >= rhs);
            } ଅନ୍ୟଥା {
                ଭାଙ୍ଗ;
            }
        }
        ଫେରନ୍ତୁ v;
    }

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା parseShift() {
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା v = parseAdd();
        ପାଇଁ (;;) {
            ଯଦି (acceptOp("<<")) {
                ପୂର୍ଣ୍ଣସଂଖ୍ୟା rhs = parseAdd();
                v       = (v << rhs);
            } ଅନ୍ୟଥା ଯଦି (acceptOp(">>")) {
                ପୂର୍ଣ୍ଣସଂଖ୍ୟା rhs = parseAdd();
                v       = (v >> rhs);
            } ଅନ୍ୟଥା {
                ଭାଙ୍ଗ;
            }
        }
        ଫେରନ୍ତୁ v;
    }

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା parseAdd() {
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା v = parseMult();
        ପାଇଁ (;;) {
            ଯଦି (acceptOp("+")) {
                ପୂର୍ଣ୍ଣସଂଖ୍ୟା rhs = parseMult();
                v       = (v + rhs);
            } ଅନ୍ୟଥା ଯଦି (acceptOp("-")) {
                ପୂର୍ଣ୍ଣସଂଖ୍ୟା rhs = parseMult();
                v       = (v - rhs);
            } ଅନ୍ୟଥା {
                ଭାଙ୍ଗ;
            }
        }
        ଫେରନ୍ତୁ v;
    }

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା parseMult() {
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା v = parseUnary();
        ପାଇଁ (;;) {
            ଯଦି (acceptOp("*")) {
                ପୂର୍ଣ୍ଣସଂଖ୍ୟା rhs = parseUnary();
                v       = (v * rhs);
            } ଅନ୍ୟଥା ଯଦି (acceptOp("/")) {
                ପୂର୍ଣ୍ଣସଂଖ୍ୟା rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v / rhs);
            } ଅନ୍ୟଥା ଯଦି (acceptOp("%")) {
                ପୂର୍ଣ୍ଣସଂଖ୍ୟା rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v % rhs);
            } ଅନ୍ୟଥା {
                ଭାଙ୍ଗ;
            }
        }
        ଫେରନ୍ତୁ v;
    }

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା parseUnary() {
        ଯଦି (acceptOp("!")) {
            ଫେରନ୍ତୁ !parseUnary();
        }
        ଯଦି (acceptOp("-")) {
            ଫେରନ୍ତୁ -parseUnary();
        }
        ଯଦି (acceptOp("+")) {
            ଫେରନ୍ତୁ +parseUnary();
        }
        ଫେରନ୍ତୁ parsePrimary();
    }

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା parsePrimary() {
        // '(' expr ')'
        ଯଦି (acceptKind(ପଦସମଷ୍ଟି_Lexer::LPAREN)) {
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା v = parse();
            ଯଦି (!acceptKind(ପଦସମଷ୍ଟି_Lexer::RPAREN)) {
                ଫିଙ୍ଗ std::runtime_error("missing ')'");
            }
            ଫେରନ୍ତୁ v;
        }

        //ସଂଖ୍ୟା
        ଯଦି (tok.kind == ପଦସମଷ୍ଟି_Lexer::ସଂଖ୍ୟା_2) {
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା v = std::stoi(tok.text);
            advance();
            ଫେରନ୍ତୁ v;
        }

        // defined(identifier)
        ଯଦି (tok.kind == ପଦସମଷ୍ଟି_Lexer::IDENT && tok.text == "defined") {
            advance();
            ଯଦି (acceptKind(ପଦସମଷ୍ଟି_Lexer::LPAREN)) {
                ଯଦି (tok.kind != ପଦସମଷ୍ଟି_Lexer::IDENT) {
                    ଫିଙ୍ଗ std::runtime_error("expected identifier in defined()");
                }
                std::string name = tok.text;
                advance();
                ଯଦି (!acceptKind(ପଦସମଷ୍ଟି_Lexer::RPAREN)) {
                    ଫିଙ୍ଗ std::runtime_error("missing ) in defined()");
                }
                ଫେରନ୍ତୁ macros.count(name) ? 1 : 0;
            } ଅନ୍ୟଥା {
                // defined NAME
                ଯଦି (tok.kind != ପଦସମଷ୍ଟି_Lexer::IDENT) {
                    ଫିଙ୍ଗ std::runtime_error("expected identifier in defined NAME");
                }
                std::string name = tok.text;
                advance();
                ଫେରନ୍ତୁ macros.count(name) ? 1 : 0;
            }
        }

        // identifier -> treat as integer, if defined use its value else 0
        ଯଦି (tok.kind == ପଦସମଷ୍ଟି_Lexer::IDENT) {
            std::string name = tok.text;
            advance();
            auto it = macros.find(name);
            ଯଦି (it == macros.end()) {
                ଫେରନ୍ତୁ 0;
            }
            ଯଦି (it->second.empty()) {
                ଫେରନ୍ତୁ 1;
            }
            ଫେରନ୍ତୁ evalMacroExpression(name, it->second);
        }

        // unexpected
        ଫେରନ୍ତୁ 0;
    }

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା evalMacroExpression(ଅଚଳ std::string & name, ଅଚଳ std::string & value) {
        ଯଦି (visiting.count(name)) {
            ଫିଙ୍ଗ std::runtime_error("Recursive macro: " + name);
        }

        visiting.insert(name);
        ପଦସମଷ୍ଟି_Parser ep(value, macros, visiting);
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା        v = ep.parse();
        visiting.erase(name);
        ଫେରନ୍ତୁ v;
    }
};

//==============================================================
// Preprocessor
//==============================================================
ଶ୍ରେଣୀ Preprocessor {
  ସର୍ବସାଧାରଣ:
    explicit Preprocessor(Options opts = {}) : opts_(std::move(opts)) {
        // Treat empty include path as current directory
        ଯଦି (opts_.include_path.empty()) {
            opts_.include_path = ".";
        }
        parseMacroDefinitions(opts_.macros);
    }

    std::string preprocess_file(ଅଚଳ std::string & filename, ଅଚଳ std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_ଗଦା;
        buildMacros(additional_macros, macros, predefined);

        std::string result = processFile(filename, macros, predefined, include_ଗଦା, DirectiveMode::All);
        ଫେରନ୍ତୁ result;
    }

    std::string preprocess(ଅଚଳ std::string & contents, ଅଚଳ std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_ଗଦା;
        buildMacros(additional_macros, macros, predefined);

        std::string result = processString(contents, macros, predefined, include_ଗଦା, DirectiveMode::All);
        ଫେରନ୍ତୁ result;
    }

    std::string preprocess_includes_file(ଅଚଳ std::string & filename) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_ଗଦା;
        std::string result = processFile(filename, macros, predefined, include_ଗଦା, DirectiveMode::IncludesOnly);
        ଫେରନ୍ତୁ result;
    }

    std::string preprocess_includes(ଅଚଳ std::string & contents) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_ଗଦା;
        std::string result = processString(contents, macros, predefined, include_ଗଦା, DirectiveMode::IncludesOnly);
        ଫେରନ୍ତୁ result;
    }

  ବ୍ୟକ୍ତିଗତ:
    Options                                      opts_;
    std::unordered_map<std::string, std::string> global_macros;

    ଗଣନା ଶ୍ରେଣୀ DirectiveMode { All, IncludesOnly };

    ସଂରଚନା Cond {
        ତାର୍କିକ parent_active;
        ତାର୍କିକ active;
        ତାର୍କିକ taken;
    };

    //----------------------------------------------------------
    // Parse macro definitions into global_macros
    //----------------------------------------------------------
    ଶୂନ୍ୟ parseMacroDefinitions(ଅଚଳ std::vector<std::string> & macro_defs) {
        ପାଇଁ (ଅଚଳ auto & def : macro_defs) {
            size_t eq_pos = def.find('=');
            ଯଦି (eq_pos != std::string::npos) {
                // Format: NAME=VALUE
                std::string name    = trim(def.substr(0, eq_pos));
                std::string value   = trim(def.substr(eq_pos + 1));
                global_macros[name] = value;
            } ଅନ୍ୟଥା {
                // Format: NAME
                std::string name    = trim(def);
                global_macros[name] = "";
            }
        }
    }

    //----------------------------------------------------------
    // Build combined macro map and predefined set for a preprocessing operation
    //----------------------------------------------------------
    ଶୂନ୍ୟ buildMacros(ଅଚଳ std::vector<std::string> &               additional_macros,
                     std::unordered_map<std::string, std::string> & macros,
                     std::unordered_set<std::string> &              predefined) {
        macros = global_macros;
        predefined.clear();

        ପାଇଁ (ଅଚଳ auto & [name, value] : global_macros) {
            predefined.insert(name);
        }

        ପାଇଁ (ଅଚଳ auto & def : additional_macros) {
            size_t      eq_pos = def.find('=');
            std::string name, value;
            ଯଦି (eq_pos != std::string::npos) {
                name  = trim(def.substr(0, eq_pos));
                value = trim(def.substr(eq_pos + 1));
            } ଅନ୍ୟଥା {
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
    std::string loadFile(ଅଚଳ std::string & fname) {
        std::ifstream f(fname);
        ଯଦି (!f.is_open()) {
            ଫିଙ୍ଗ std::runtime_error("Could not open file: " + fname);
        }
        std::stringstream ss;
        ss << f.rdbuf();
        ଫେରନ୍ତୁ ss.str();
    }

    ତାର୍କିକ condActive(ଅଚଳ std::vector<Cond> & cond) ଅଚଳ {
        ଯଦି (cond.empty()) {
            ଫେରନ୍ତୁ ସତ୍ୟ;
        }
        ଫେରନ୍ତୁ cond.back().active;
    }

    //----------------------------------------------------------
    // Process a file
    //----------------------------------------------------------
    std::string processFile(ଅଚଳ std::string &                            name,
                            std::unordered_map<std::string, std::string> & macros,
                            ଅଚଳ std::unordered_set<std::string> &        predefined_macros,
                            std::unordered_set<std::string> &              include_ଗଦା,
                            DirectiveMode                                  mode) {
        ଯଦି (include_ଗଦା.count(name)) {
            ଫିଙ୍ଗ std::runtime_error("Recursive include: " + name);
        }

        include_ଗଦା.insert(name);
        std::string shader_code = loadFile(name);
        std::string out         = processString(shader_code, macros, predefined_macros, include_ଗଦା, mode);
        include_ଗଦା.erase(name);
        ଫେରନ୍ତୁ out;
    }

    std::string processIncludeFile(ଅଚଳ std::string &                            fname,
                                   std::unordered_map<std::string, std::string> & macros,
                                   ଅଚଳ std::unordered_set<std::string> &        predefined_macros,
                                   std::unordered_set<std::string> &              include_ଗଦା,
                                   DirectiveMode                                  mode) {
        std::string full_path = opts_.include_path + "/" + fname;
        ଫେରନ୍ତୁ processFile(full_path, macros, predefined_macros, include_ଗଦା, mode);
    }

    //----------------------------------------------------------
    // Process text
    //----------------------------------------------------------
    std::string processString(ଅଚଳ std::string &                            shader_code,
                              std::unordered_map<std::string, std::string> & macros,
                              ଅଚଳ std::unordered_set<std::string> &        predefined_macros,
                              std::unordered_set<std::string> &              include_ଗଦା,
                              DirectiveMode                                  mode) {
        std::vector<Cond>  cond;  // Conditional stack for this shader
        std::stringstream  out;
        std::istringstream in(shader_code);
        std::string        ଧାଡ଼ି;

        ଯେତେବେଳେ (std::getline(in, ଧାଡ଼ି)) {
            std::string logical = ଧାଡ଼ି;
            std::string t       = trim(logical);
            ଯଦି (!t.empty() && t[0] == '#') {
                ଯେତେବେଳେ (endsWithContinuation(logical)) {
                    stripContinuation(logical);
                    ଯଦି (!std::getline(in, ଧାଡ଼ି)) {
                        ଭାଙ୍ଗ;
                    }
                    logical += "\n";
                    logical += ଧାଡ଼ି;
                }
                t = trim(logical);
            }

            ଯଦି (!t.empty() && t[0] == '#') {
                ତାର୍କିକ handled = handleDirective(t, out, macros, predefined_macros, cond, include_ଗଦା, mode);
                ଯଦି (mode == DirectiveMode::IncludesOnly && !handled) {
                    out << logical << "\n";
                }
            } ଅନ୍ୟଥା {
                ଯଦି (mode == DirectiveMode::IncludesOnly) {
                    out << logical << "\n";
                } ଅନ୍ୟଥା ଯଦି (condActive(cond)) {
                    // Expand macros in the line before outputting
                    std::string expanded = expandMacrosRecursive(logical, macros);
                    out << expanded << "\n";
                }
            }
        }

        ଯଦି (mode == DirectiveMode::All && !cond.empty()) {
            ଫିଙ୍ଗ std::runtime_error("Unclosed #if directive");
        }

        ଫେରନ୍ତୁ out.str();
    }

    //----------------------------------------------------------
    // Directive handler
    //----------------------------------------------------------
    ତାର୍କିକ handleDirective(ଅଚଳ std::string &                            t,
                         std::stringstream &                            out,
                         std::unordered_map<std::string, std::string> & macros,
                         ଅଚଳ std::unordered_set<std::string> &        predefined_macros,
                         std::vector<Cond> &                            cond,
                         std::unordered_set<std::string> &              include_ଗଦା,
                         DirectiveMode                                  mode) {
        // split into tokens
        std::string        body = t.substr(1);
        std::istringstream iss(body);
        std::string        cmd;
        iss >> cmd;

        ଯଦି (cmd == "include") {
            ଯଦି (mode == DirectiveMode::All && !condActive(cond)) {
                ଫେରନ୍ତୁ ସତ୍ୟ;
            }
            std::string file;
            iss >> file;
            ଯଦି (file.size() >= 2 && file.front() == '"' && file.back() == '"') {
                file = file.substr(1, file.size() - 2);
            }
            out << processIncludeFile(file, macros, predefined_macros, include_ଗଦା, mode);
            ଫେରନ୍ତୁ ସତ୍ୟ;
        }

        ଯଦି (mode == DirectiveMode::IncludesOnly) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        ଯଦି (cmd == "define") {
            ଯଦି (!condActive(cond)) {
                ଫେରନ୍ତୁ ସତ୍ୟ;
            }
            std::string name;
            iss >> name;
            // Don't override predefined macros from options
            ଯଦି (predefined_macros.count(name)) {
                ଫେରନ୍ତୁ ସତ୍ୟ;
            }
            std::string value = trim_ମୂଲ୍ୟ(iss);
            macros[name]      = value;
            ଫେରନ୍ତୁ ସତ୍ୟ;
        }

        ଯଦି (cmd == "undef") {
            ଯଦି (!condActive(cond)) {
                ଫେରନ୍ତୁ ସତ୍ୟ;
            }
            std::string name;
            iss >> name;
            // Don't undef predefined macros from options
            ଯଦି (predefined_macros.count(name)) {
                ଫେରନ୍ତୁ ସତ୍ୟ;
            }
            macros.erase(name);
            ଫେରନ୍ତୁ ସତ୍ୟ;
        }

        ଯଦି (cmd == "ifdef") {
            std::string name;
            iss >> name;
            ତାର୍କିକ p = condActive(cond);
            ତାର୍କିକ v = macros.count(name);
            cond.push_back({ p, p && v, p && v });
            ଫେରନ୍ତୁ ସତ୍ୟ;
        }

        ଯଦି (cmd == "ifndef") {
            std::string name;
            iss >> name;
            ତାର୍କିକ p = condActive(cond);
            ତାର୍କିକ v = !macros.count(name);
            cond.push_back({ p, p && v, p && v });
            ଫେରନ୍ତୁ ସତ୍ୟ;
        }

        ଯଦି (cmd == "if") {
            std::string ପଦସମଷ୍ଟି = trim_ମୂଲ୍ୟ(iss);
            ତାର୍କିକ        p    = condActive(cond);
            ତାର୍କିକ        v    = ମିଥ୍ୟା;
            ଯଦି (p) {
                std::unordered_set<std::string> visiting;
                ପଦସମଷ୍ଟି_Parser                      ep(ପଦସମଷ୍ଟି, macros, visiting);
                v = ep.parse() != 0;
            }
            cond.push_back({ p, p && v, p && v });
            ଫେରନ୍ତୁ ସତ୍ୟ;
        }

        ଯଦି (cmd == "elif") {
            std::string ପଦସମଷ୍ଟି = trim_ମୂଲ୍ୟ(iss);

            ଯଦି (cond.empty()) {
                ଫିଙ୍ଗ std::runtime_error("#elif without #if");
            }

            Cond & c = cond.back();
            ଯଦି (!c.parent_active) {
                c.active = ମିଥ୍ୟା;
                ଫେରନ୍ତୁ ସତ୍ୟ;
            }

            ଯଦି (c.taken) {
                c.active = ମିଥ୍ୟା;
                ଫେରନ୍ତୁ ସତ୍ୟ;
            }

            std::unordered_set<std::string> visiting;
            ପଦସମଷ୍ଟି_Parser                      ep(ପଦସମଷ୍ଟି, macros, visiting);
            ତାର୍କିକ                            v = ep.parse() != 0;
            c.active                          = v;
            ଯଦି (v) {
                c.taken = ସତ୍ୟ;
            }
            ଫେରନ୍ତୁ ସତ୍ୟ;
        }

        ଯଦି (cmd == "else") {
            ଯଦି (cond.empty()) {
                ଫିଙ୍ଗ std::runtime_error("#else without #if");
            }

            Cond & c = cond.back();
            ଯଦି (!c.parent_active) {
                c.active = ମିଥ୍ୟା;
                ଫେରନ୍ତୁ ସତ୍ୟ;
            }
            ଯଦି (c.taken) {
                c.active = ମିଥ୍ୟା;
            } ଅନ୍ୟଥା {
                c.active = ସତ୍ୟ;
                c.taken  = ସତ୍ୟ;
            }
            ଫେରନ୍ତୁ ସତ୍ୟ;
        }

        ଯଦି (cmd == "endif") {
            ଯଦି (cond.empty()) {
                ଫିଙ୍ଗ std::runtime_error("#endif without #if");
            }
            cond.pop_back();
            ଫେରନ୍ତୁ ସତ୍ୟ;
        }

        // Unknown directive
        ଫିଙ୍ଗ std::runtime_error("Unknown directive: #" + cmd);
    }
};

}  // namespace pre_wgsl

#endif  // PRE_WGSL_HPP
