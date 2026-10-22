#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ PRE_WGSL_HPP
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ PRE_WGSL_HPP

#ਅਨ੍ਤਰ੍ਭਾਵ <cctype>
#ਅਨ੍ਤਰ੍ਭਾਵ <fstream>
#ਅਨ੍ਤਰ੍ਭਾਵ <sstream>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdexcept>
#ਅਨ੍ਤਰ੍ਭਾਵ <string>
#ਅਨ੍ਤਰ੍ਭਾਵ <string_view>
#ਅਨ੍ਤਰ੍ਭਾਵ <unordered_map>
#ਅਨ੍ਤਰ੍ਭਾਵ <unordered_set>
#ਅਨ੍ਤਰ੍ਭਾਵ <vector>

ਨਾਮਾਕਾਸ਼ pre_wgsl {

//==============================================================
// Options
//==============================================================
ਬਣਤਰ Options {
    std::string              include_ਰਸਤਾ = ".";
    std::vector<std::string> macros;
};

//==============================================================
// Utility: trim
//==============================================================
ਸਥਿਰ std::string trim(ਅਚਲ std::string & s) {
    size_t a = 0;
    ਜਦੋਂ (a < s.size() && std::ਖਾਲੀ_ਥਾਂ_ਹੈ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ) s[a])) {
        a++;
    }
    size_t b = s.size();
    ਜਦੋਂ (b > a && std::ਖਾਲੀ_ਥਾਂ_ਹੈ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ) s[b - 1])) {
        b--;
    }
    ਵਾਪਸ s.substr(a, b - a);
}

ਸਥਿਰ std::string trim_ਮੁੱਲ(std::istream & is) {
    std::ostringstream ss;
    ss << is.rdbuf();
    ਵਾਪਸ trim(ss.str());
}

ਸਥਿਰ ਤਾਰਕਿਕ isIdentChar(ਅੱਖਰ c) {
    ਵਾਪਸ std::ਅੱਖਰ_ਜਾਂ_ਅੰਕ(static_cast<ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ>(c)) || c == '_';
}

ਸਥਿਰ ਤਾਰਕਿਕ ਸਿਰੇ_With_Continuation(ਅਚਲ std::string & ਸਤਰ) {
    size_t i = ਸਤਰ.size();
    ਜਦੋਂ (i > 0 && std::ਖਾਲੀ_ਥਾਂ_ਹੈ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ) ਸਤਰ[i - 1])) {
        i--;
    }
    ਵਾਪਸ i > 0 && ਸਤਰ[i - 1] == '\\';
}

ਸਥਿਰ ਖਾਲੀ stripContinuation(std::string & ਸਤਰ) {
    size_t i = ਸਤਰ.size();
    ਜਦੋਂ (i > 0 && std::ਖਾਲੀ_ਥਾਂ_ਹੈ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ) ਸਤਰ[i - 1])) {
        i--;
    }
    ਜੇ (i > 0 && ਸਤਰ[i - 1] == '\\') {
        ਸਤਰ.erase(i - 1);
    }
}

ਸਥਿਰ std::string expand_Macros_Recursive_ਅੰਦਰੂਨੀ(ਅਚਲ std::string &                                  ਸਤਰ,
                                                 ਅਚਲ std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting);

ਸਥਿਰ std::string expand_Macro_ਮੁੱਲ(ਅਚਲ std::string &                                  name,
                                    ਅਚਲ std::unordered_map<std::string, std::string> & macros,
                                    std::unordered_set<std::string> &                    visiting) {
    ਜੇ (visiting.count(name)) {
        ਸੁੱਟੋ std::runtime_error("Recursive macro: " + name);
    }
    visiting.insert(name);

    ਸਵੈਚਾਲਿਤ it = macros.find(name);
    ਜੇ (it == macros.end()) {
        visiting.erase(name);
        ਵਾਪਸ name;
    }

    ਅਚਲ std::string & value = it->second;
    ਜੇ (value.ਖਾਲੀ_ਥਾਵਾਂ()) {
        visiting.erase(name);
        ਵਾਪਸ "";
    }

    std::string expanded = expand_Macros_Recursive_ਅੰਦਰੂਨੀ(value, macros, visiting);
    visiting.erase(name);
    ਵਾਪਸ expanded;
}

ਸਥਿਰ std::string expand_Macros_Recursive_ਅੰਦਰੂਨੀ(ਅਚਲ std::string &                                  ਸਤਰ,
                                                 ਅਚਲ std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting) {
    std::string ਨਤੀਜਾ;
    ਨਤੀਜਾ.reserve(ਸਤਰ.size());

    size_t i = 0;
    ਜਦੋਂ (i < ਸਤਰ.size()) {
        ਜੇ (isIdentChar(ਸਤਰ[i])) {
            size_t start = i;
            ਜਦੋਂ (i < ਸਤਰ.size() && isIdentChar(ਸਤਰ[i])) {
                i++;
            }
            std::string token = ਸਤਰ.substr(start, i - start);

            ਸਵੈਚਾਲਿਤ it = macros.find(token);
            ਜੇ (it != macros.end()) {
                ਨਤੀਜਾ += expand_Macro_ਮੁੱਲ(token, macros, visiting);
            } ਨਹੀਂਤਾਂ {
                ਨਤੀਜਾ += token;
            }
        } ਨਹੀਂਤਾਂ {
            ਨਤੀਜਾ += ਸਤਰ[i];
            i++;
        }
    }

    ਵਾਪਸ ਨਤੀਜਾ;
}

ਸਥਿਰ std::string expandMacrosRecursive(ਅਚਲ std::string &                                  ਸਤਰ,
                                         ਅਚਲ std::unordered_map<std::string, std::string> & macros) {
    std::unordered_set<std::string> visiting;
    ਵਾਪਸ expand_Macros_Recursive_ਅੰਦਰੂਨੀ(ਸਤਰ, macros, visiting);
}

//==============================================================
// Tokenizer for expressions in #if/#elif
//==============================================================
ਵਰਗ ਵਿਅੰਜਕ_Lexer {
  ਜਨਤਕ:
    ਗਿਣਤੀ ਕਿਸਮ_3 { ਸਿਰਾ_3, IDENT, ਸੰਖਿਆ_2, OP, LPAREN, RPAREN };

    ਬਣਤਰ Tok {
        ਕਿਸਮ_3        ਕਿਸਮ_2;
        std::string ਪਾਠ_2;
    };

    ਸਪਸ਼ਟ ਵਿਅੰਜਕ_Lexer(std::string_view sv) : src(sv), pos(0) {}

    Tok ਅਗਲਾ() {
        skipWS();
        ਜੇ (pos >= src.size()) {
            ਵਾਪਸ { ਸਿਰਾ_3, "" };
        }

        ਅੱਖਰ c = src[pos];

        //ਸੰਖਿਆ
        ਜੇ (std::ਅੰਕ_ਹੈ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ) c)) {
            size_t start = pos;
            ਜਦੋਂ (pos < src.size() && std::ਅੰਕ_ਹੈ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ) src[pos])) {
                pos++;
            }
            ਵਾਪਸ { ਸੰਖਿਆ_2, std::string(src.substr(start, pos - start)) };
        }

        // identifier
        ਜੇ (std::ਅੱਖਰ_ਹੈ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ) c) || c == '_') {
            size_t start = pos;
            ਜਦੋਂ (pos < src.size() && (std::ਅੱਖਰ_ਜਾਂ_ਅੰਕ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ) src[pos]) || src[pos] == '_')) {
                pos++;
            }
            ਵਾਪਸ { IDENT, std::string(src.substr(start, pos - start)) };
        }

        ਜੇ (c == '(') {
            pos++;
            ਵਾਪਸ { LPAREN, "(" };
        }
        ਜੇ (c == ')') {
            pos++;
            ਵਾਪਸ { RPAREN, ")" };
        }

        // multi-char operators
        ਸਥਿਰ ਅਚਲ ਅੱਖਰ * two_ops[] = { "==", "!=", "<=", ">=", "&&", "||", "<<", ">>" };
        ਲਈ (ਸਵੈਚਾਲਿਤ op : two_ops) {
            ਜੇ (src.substr(pos, 2) == op) {
                pos += 2;
                ਵਾਪਸ { OP, std::string(op) };
            }
        }

        // single-char operators
        ਜੇ (std::string("+-*/%<>!").find(c) != std::string::npos) {
            pos++;
            ਵਾਪਸ { OP, std::string(1, c) };
        }

        // unexpected
        pos++;
        ਵਾਪਸ { ਸਿਰਾ_3, "" };
    }

  ਨਿਜੀ:
    std::string_view src;
    size_t           pos;

    ਖਾਲੀ skipWS() {
        ਜਦੋਂ (pos < src.size() && std::ਖਾਲੀ_ਥਾਂ_ਹੈ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ) src[pos])) {
            pos++;
        }
    }
};

//==============================================================
// Expression Parser (recursive descent)
//==============================================================
ਵਰਗ ਵਿਅੰਜਕ_Parser {
  ਜਨਤਕ:
    ਵਿਅੰਜਕ_Parser(std::string_view                                     ਵਿਅੰਜਕ,
               ਅਚਲ std::unordered_map<std::string, std::string> & macros,
               std::unordered_set<std::string> &                    visiting) :
        lex(ਵਿਅੰਜਕ),
        macros(macros),
        visiting(visiting) {
        advance();
    }

    ਪੂਰਨ parse() { ਵਾਪਸ parseLogicalOr(); }

  ਨਿਜੀ:
    ਵਿਅੰਜਕ_Lexer                                            lex;
    ਵਿਅੰਜਕ_Lexer::Tok                                       tok;
    ਅਚਲ std::unordered_map<std::string, std::string> & macros;
    std::unordered_set<std::string> &                    visiting;

    ਖਾਲੀ advance() { tok = lex.ਅਗਲਾ(); }

    ਤਾਰਕਿਕ acceptOp(ਅਚਲ std::string & s) {
        ਜੇ (tok.ਕਿਸਮ_2 == ਵਿਅੰਜਕ_Lexer::OP && tok.ਪਾਠ_2 == s) {
            advance();
            ਵਾਪਸ ਸੱਚ;
        }
        ਵਾਪਸ ਝੂਠ;
    }

    ਤਾਰਕਿਕ accept_ਕਿਸਮ(ਵਿਅੰਜਕ_Lexer::ਕਿਸਮ_3 k) {
        ਜੇ (tok.ਕਿਸਮ_2 == k) {
            advance();
            ਵਾਪਸ ਸੱਚ;
        }
        ਵਾਪਸ ਝੂਠ;
    }

    ਪੂਰਨ parseLogicalOr() {
        ਪੂਰਨ v = parseLogicalAnd();
        ਜਦੋਂ (acceptOp("||")) {
            ਪੂਰਨ rhs = parseLogicalAnd();
            v       = (v || rhs);
        }
        ਵਾਪਸ v;
    }

    ਪੂਰਨ parseLogicalAnd() {
        ਪੂਰਨ v = parseEquality();
        ਜਦੋਂ (acceptOp("&&")) {
            ਪੂਰਨ rhs = parseEquality();
            v       = (v && rhs);
        }
        ਵਾਪਸ v;
    }

    ਪੂਰਨ parseEquality() {
        ਪੂਰਨ v = parseRelational();
        ਲਈ (;;) {
            ਜੇ (acceptOp("==")) {
                ਪੂਰਨ rhs = parseRelational();
                v       = (v == rhs);
            } ਨਹੀਂਤਾਂ ਜੇ (acceptOp("!=")) {
                ਪੂਰਨ rhs = parseRelational();
                v       = (v != rhs);
            } ਨਹੀਂਤਾਂ {
                ਰੋਕੋ;
            }
        }
        ਵਾਪਸ v;
    }

    ਪੂਰਨ parseRelational() {
        ਪੂਰਨ v = parse_ਖਿਸਕਾਓ();
        ਲਈ (;;) {
            ਜੇ (acceptOp("<")) {
                ਪੂਰਨ rhs = parse_ਖਿਸਕਾਓ();
                v       = (v < rhs);
            } ਨਹੀਂਤਾਂ ਜੇ (acceptOp(">")) {
                ਪੂਰਨ rhs = parse_ਖਿਸਕਾਓ();
                v       = (v > rhs);
            } ਨਹੀਂਤਾਂ ਜੇ (acceptOp("<=")) {
                ਪੂਰਨ rhs = parse_ਖਿਸਕਾਓ();
                v       = (v <= rhs);
            } ਨਹੀਂਤਾਂ ਜੇ (acceptOp(">=")) {
                ਪੂਰਨ rhs = parse_ਖਿਸਕਾਓ();
                v       = (v >= rhs);
            } ਨਹੀਂਤਾਂ {
                ਰੋਕੋ;
            }
        }
        ਵਾਪਸ v;
    }

    ਪੂਰਨ parse_ਖਿਸਕਾਓ() {
        ਪੂਰਨ v = parse_ਜੋੜੋ();
        ਲਈ (;;) {
            ਜੇ (acceptOp("<<")) {
                ਪੂਰਨ rhs = parse_ਜੋੜੋ();
                v       = (v << rhs);
            } ਨਹੀਂਤਾਂ ਜੇ (acceptOp(">>")) {
                ਪੂਰਨ rhs = parse_ਜੋੜੋ();
                v       = (v >> rhs);
            } ਨਹੀਂਤਾਂ {
                ਰੋਕੋ;
            }
        }
        ਵਾਪਸ v;
    }

    ਪੂਰਨ parse_ਜੋੜੋ() {
        ਪੂਰਨ v = parseMult();
        ਲਈ (;;) {
            ਜੇ (acceptOp("+")) {
                ਪੂਰਨ rhs = parseMult();
                v       = (v + rhs);
            } ਨਹੀਂਤਾਂ ਜੇ (acceptOp("-")) {
                ਪੂਰਨ rhs = parseMult();
                v       = (v - rhs);
            } ਨਹੀਂਤਾਂ {
                ਰੋਕੋ;
            }
        }
        ਵਾਪਸ v;
    }

    ਪੂਰਨ parseMult() {
        ਪੂਰਨ v = parseUnary();
        ਲਈ (;;) {
            ਜੇ (acceptOp("*")) {
                ਪੂਰਨ rhs = parseUnary();
                v       = (v * rhs);
            } ਨਹੀਂਤਾਂ ਜੇ (acceptOp("/")) {
                ਪੂਰਨ rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v / rhs);
            } ਨਹੀਂਤਾਂ ਜੇ (acceptOp("%")) {
                ਪੂਰਨ rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v % rhs);
            } ਨਹੀਂਤਾਂ {
                ਰੋਕੋ;
            }
        }
        ਵਾਪਸ v;
    }

    ਪੂਰਨ parseUnary() {
        ਜੇ (acceptOp("!")) {
            ਵਾਪਸ !parseUnary();
        }
        ਜੇ (acceptOp("-")) {
            ਵਾਪਸ -parseUnary();
        }
        ਜੇ (acceptOp("+")) {
            ਵਾਪਸ +parseUnary();
        }
        ਵਾਪਸ parsePrimary();
    }

    ਪੂਰਨ parsePrimary() {
        // '(' expr ')'
        ਜੇ (accept_ਕਿਸਮ(ਵਿਅੰਜਕ_Lexer::LPAREN)) {
            ਪੂਰਨ v = parse();
            ਜੇ (!accept_ਕਿਸਮ(ਵਿਅੰਜਕ_Lexer::RPAREN)) {
                ਸੁੱਟੋ std::runtime_error("missing ')'");
            }
            ਵਾਪਸ v;
        }

        //ਸੰਖਿਆ
        ਜੇ (tok.ਕਿਸਮ_2 == ਵਿਅੰਜਕ_Lexer::ਸੰਖਿਆ_2) {
            ਪੂਰਨ v = std::stoi(tok.ਪਾਠ_2);
            advance();
            ਵਾਪਸ v;
        }

        // defined(identifier)
        ਜੇ (tok.ਕਿਸਮ_2 == ਵਿਅੰਜਕ_Lexer::IDENT && tok.ਪਾਠ_2 == "defined") {
            advance();
            ਜੇ (accept_ਕਿਸਮ(ਵਿਅੰਜਕ_Lexer::LPAREN)) {
                ਜੇ (tok.ਕਿਸਮ_2 != ਵਿਅੰਜਕ_Lexer::IDENT) {
                    ਸੁੱਟੋ std::runtime_error("expected identifier in defined()");
                }
                std::string name = tok.ਪਾਠ_2;
                advance();
                ਜੇ (!accept_ਕਿਸਮ(ਵਿਅੰਜਕ_Lexer::RPAREN)) {
                    ਸੁੱਟੋ std::runtime_error("missing ) in defined()");
                }
                ਵਾਪਸ macros.count(name) ? 1 : 0;
            } ਨਹੀਂਤਾਂ {
                // defined NAME
                ਜੇ (tok.ਕਿਸਮ_2 != ਵਿਅੰਜਕ_Lexer::IDENT) {
                    ਸੁੱਟੋ std::runtime_error("expected identifier in defined NAME");
                }
                std::string name = tok.ਪਾਠ_2;
                advance();
                ਵਾਪਸ macros.count(name) ? 1 : 0;
            }
        }

        // identifier -> treat as integer, if defined use its value else 0
        ਜੇ (tok.ਕਿਸਮ_2 == ਵਿਅੰਜਕ_Lexer::IDENT) {
            std::string name = tok.ਪਾਠ_2;
            advance();
            ਸਵੈਚਾਲਿਤ it = macros.find(name);
            ਜੇ (it == macros.end()) {
                ਵਾਪਸ 0;
            }
            ਜੇ (it->second.ਖਾਲੀ_ਥਾਵਾਂ()) {
                ਵਾਪਸ 1;
            }
            ਵਾਪਸ eval_Macro_ਵਿਅੰਜਕ(name, it->second);
        }

        // unexpected
        ਵਾਪਸ 0;
    }

    ਪੂਰਨ eval_Macro_ਵਿਅੰਜਕ(ਅਚਲ std::string & name, ਅਚਲ std::string & value) {
        ਜੇ (visiting.count(name)) {
            ਸੁੱਟੋ std::runtime_error("Recursive macro: " + name);
        }

        visiting.insert(name);
        ਵਿਅੰਜਕ_Parser ep(value, macros, visiting);
        ਪੂਰਨ        v = ep.parse();
        visiting.erase(name);
        ਵਾਪਸ v;
    }
};

//==============================================================
// Preprocessor
//==============================================================
ਵਰਗ Preprocessor {
  ਜਨਤਕ:
    ਸਪਸ਼ਟ Preprocessor(Options opts = {}) : opts_(std::move(opts)) {
        // Treat empty include path as current directory
        ਜੇ (opts_.include_ਰਸਤਾ.ਖਾਲੀ_ਥਾਵਾਂ()) {
            opts_.include_ਰਸਤਾ = ".";
        }
        parseMacroDefinitions(opts_.macros);
    }

    std::string preprocess_file(ਅਚਲ std::string & filename, ਅਚਲ std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_ਢੇਰ;
        buildMacros(additional_macros, macros, predefined);

        std::string ਨਤੀਜਾ = processFile(filename, macros, predefined, include_ਢੇਰ, Directive_ਬਹੁਲਕ::All);
        ਵਾਪਸ ਨਤੀਜਾ;
    }

    std::string preprocess(ਅਚਲ std::string & contents, ਅਚਲ std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_ਢੇਰ;
        buildMacros(additional_macros, macros, predefined);

        std::string ਨਤੀਜਾ = processString(contents, macros, predefined, include_ਢੇਰ, Directive_ਬਹੁਲਕ::All);
        ਵਾਪਸ ਨਤੀਜਾ;
    }

    std::string preprocess_includes_file(ਅਚਲ std::string & filename) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_ਢੇਰ;
        std::string ਨਤੀਜਾ = processFile(filename, macros, predefined, include_ਢੇਰ, Directive_ਬਹੁਲਕ::IncludesOnly);
        ਵਾਪਸ ਨਤੀਜਾ;
    }

    std::string preprocess_includes(ਅਚਲ std::string & contents) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_ਢੇਰ;
        std::string ਨਤੀਜਾ = processString(contents, macros, predefined, include_ਢੇਰ, Directive_ਬਹੁਲਕ::IncludesOnly);
        ਵਾਪਸ ਨਤੀਜਾ;
    }

  ਨਿਜੀ:
    Options                                      opts_;
    std::unordered_map<std::string, std::string> global_macros;

    ਗਿਣਤੀ ਵਰਗ Directive_ਬਹੁਲਕ { All, IncludesOnly };

    ਬਣਤਰ Cond {
        ਤਾਰਕਿਕ ਮਾਪਾ_ਸਰਗਰਮ;
        ਤਾਰਕਿਕ ਸਰਗਰਮ;
        ਤਾਰਕਿਕ ਲਿਆ;
    };

    //----------------------------------------------------------
    // Parse macro definitions into global_macros
    //----------------------------------------------------------
    ਖਾਲੀ parseMacroDefinitions(ਅਚਲ std::vector<std::string> & macro_defs) {
        ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & def : macro_defs) {
            size_t eq_pos = def.find('=');
            ਜੇ (eq_pos != std::string::npos) {
                // Format: NAME=VALUE
                std::string name    = trim(def.substr(0, eq_pos));
                std::string value   = trim(def.substr(eq_pos + 1));
                global_macros[name] = value;
            } ਨਹੀਂਤਾਂ {
                // Format: NAME
                std::string name    = trim(def);
                global_macros[name] = "";
            }
        }
    }

    //----------------------------------------------------------
    // Build combined macro map and predefined set for a preprocessing operation
    //----------------------------------------------------------
    ਖਾਲੀ buildMacros(ਅਚਲ std::vector<std::string> &               additional_macros,
                     std::unordered_map<std::string, std::string> & macros,
                     std::unordered_set<std::string> &              predefined) {
        macros = global_macros;
        predefined.clear();

        ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & [name, value] : global_macros) {
            predefined.insert(name);
        }

        ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & def : additional_macros) {
            size_t      eq_pos = def.find('=');
            std::string name, value;
            ਜੇ (eq_pos != std::string::npos) {
                name  = trim(def.substr(0, eq_pos));
                value = trim(def.substr(eq_pos + 1));
            } ਨਹੀਂਤਾਂ {
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
    std::string ਲੱਦੋ_File(ਅਚਲ std::string & fname) {
        std::ifstream f(fname);
        ਜੇ (!f.is_open()) {
            ਸੁੱਟੋ std::runtime_error("Could not open file: " + fname);
        }
        std::stringstream ss;
        ss << f.rdbuf();
        ਵਾਪਸ ss.str();
    }

    ਤਾਰਕਿਕ cond_ਸਰਗਰਮ(ਅਚਲ std::vector<Cond> & cond) ਅਚਲ {
        ਜੇ (cond.ਖਾਲੀ_ਥਾਵਾਂ()) {
            ਵਾਪਸ ਸੱਚ;
        }
        ਵਾਪਸ cond.back().ਸਰਗਰਮ;
    }

    //----------------------------------------------------------
    // Process a file
    //----------------------------------------------------------
    std::string processFile(ਅਚਲ std::string &                            name,
                            std::unordered_map<std::string, std::string> & macros,
                            ਅਚਲ std::unordered_set<std::string> &        predefined_macros,
                            std::unordered_set<std::string> &              include_ਢੇਰ,
                            Directive_ਬਹੁਲਕ                                  mode) {
        ਜੇ (include_ਢੇਰ.count(name)) {
            ਸੁੱਟੋ std::runtime_error("Recursive include: " + name);
        }

        include_ਢੇਰ.insert(name);
        std::string shader_ਕੋਡ = ਲੱਦੋ_File(name);
        std::string ਬਾਹਰਲਾ         = processString(shader_ਕੋਡ, macros, predefined_macros, include_ਢੇਰ, mode);
        include_ਢੇਰ.erase(name);
        ਵਾਪਸ ਬਾਹਰਲਾ;
    }

    std::string processIncludeFile(ਅਚਲ std::string &                            fname,
                                   std::unordered_map<std::string, std::string> & macros,
                                   ਅਚਲ std::unordered_set<std::string> &        predefined_macros,
                                   std::unordered_set<std::string> &              include_ਢੇਰ,
                                   Directive_ਬਹੁਲਕ                                  mode) {
        std::string ਭਰਿਆ_ਰਸਤਾ = opts_.include_ਰਸਤਾ + "/" + fname;
        ਵਾਪਸ processFile(ਭਰਿਆ_ਰਸਤਾ, macros, predefined_macros, include_ਢੇਰ, mode);
    }

    //----------------------------------------------------------
    // Process text
    //----------------------------------------------------------
    std::string processString(ਅਚਲ std::string &                            shader_ਕੋਡ,
                              std::unordered_map<std::string, std::string> & macros,
                              ਅਚਲ std::unordered_set<std::string> &        predefined_macros,
                              std::unordered_set<std::string> &              include_ਢੇਰ,
                              Directive_ਬਹੁਲਕ                                  mode) {
        std::vector<Cond>  cond;  // Conditional stack for this shader
        std::stringstream  ਬਾਹਰਲਾ;
        std::istringstream ਵਿੱਚ(shader_ਕੋਡ);
        std::string        ਸਤਰ;

        ਜਦੋਂ (std::getline(ਵਿੱਚ, ਸਤਰ)) {
            std::string logical = ਸਤਰ;
            std::string t       = trim(logical);
            ਜੇ (!t.ਖਾਲੀ_ਥਾਵਾਂ() && t[0] == '#') {
                ਜਦੋਂ (ਸਿਰੇ_With_Continuation(logical)) {
                    stripContinuation(logical);
                    ਜੇ (!std::getline(ਵਿੱਚ, ਸਤਰ)) {
                        ਰੋਕੋ;
                    }
                    logical += "\n";
                    logical += ਸਤਰ;
                }
                t = trim(logical);
            }

            ਜੇ (!t.ਖਾਲੀ_ਥਾਵਾਂ() && t[0] == '#') {
                ਤਾਰਕਿਕ handled = handleDirective(t, ਬਾਹਰਲਾ, macros, predefined_macros, cond, include_ਢੇਰ, mode);
                ਜੇ (mode == Directive_ਬਹੁਲਕ::IncludesOnly && !handled) {
                    ਬਾਹਰਲਾ << logical << "\n";
                }
            } ਨਹੀਂਤਾਂ {
                ਜੇ (mode == Directive_ਬਹੁਲਕ::IncludesOnly) {
                    ਬਾਹਰਲਾ << logical << "\n";
                } ਨਹੀਂਤਾਂ ਜੇ (cond_ਸਰਗਰਮ(cond)) {
                    // Expand macros in the line before outputting
                    std::string expanded = expandMacrosRecursive(logical, macros);
                    ਬਾਹਰਲਾ << expanded << "\n";
                }
            }
        }

        ਜੇ (mode == Directive_ਬਹੁਲਕ::All && !cond.ਖਾਲੀ_ਥਾਵਾਂ()) {
            ਸੁੱਟੋ std::runtime_error("Unclosed #if directive");
        }

        ਵਾਪਸ ਬਾਹਰਲਾ.str();
    }

    //----------------------------------------------------------
    // Directive handler
    //----------------------------------------------------------
    ਤਾਰਕਿਕ handleDirective(ਅਚਲ std::string &                            t,
                         std::stringstream &                            ਬਾਹਰਲਾ,
                         std::unordered_map<std::string, std::string> & macros,
                         ਅਚਲ std::unordered_set<std::string> &        predefined_macros,
                         std::vector<Cond> &                            cond,
                         std::unordered_set<std::string> &              include_ਢੇਰ,
                         Directive_ਬਹੁਲਕ                                  mode) {
        // split into tokens
        std::string        body = t.substr(1);
        std::istringstream iss(body);
        std::string        cmd;
        iss >> cmd;

        ਜੇ (cmd == "include") {
            ਜੇ (mode == Directive_ਬਹੁਲਕ::All && !cond_ਸਰਗਰਮ(cond)) {
                ਵਾਪਸ ਸੱਚ;
            }
            std::string file;
            iss >> file;
            ਜੇ (file.size() >= 2 && file.front() == '"' && file.back() == '"') {
                file = file.substr(1, file.size() - 2);
            }
            ਬਾਹਰਲਾ << processIncludeFile(file, macros, predefined_macros, include_ਢੇਰ, mode);
            ਵਾਪਸ ਸੱਚ;
        }

        ਜੇ (mode == Directive_ਬਹੁਲਕ::IncludesOnly) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਜੇ (cmd == "define") {
            ਜੇ (!cond_ਸਰਗਰਮ(cond)) {
                ਵਾਪਸ ਸੱਚ;
            }
            std::string name;
            iss >> name;
            // Don't override predefined macros from options
            ਜੇ (predefined_macros.count(name)) {
                ਵਾਪਸ ਸੱਚ;
            }
            std::string value = trim_ਮੁੱਲ(iss);
            macros[name]      = value;
            ਵਾਪਸ ਸੱਚ;
        }

        ਜੇ (cmd == "undef") {
            ਜੇ (!cond_ਸਰਗਰਮ(cond)) {
                ਵਾਪਸ ਸੱਚ;
            }
            std::string name;
            iss >> name;
            // Don't undef predefined macros from options
            ਜੇ (predefined_macros.count(name)) {
                ਵਾਪਸ ਸੱਚ;
            }
            macros.erase(name);
            ਵਾਪਸ ਸੱਚ;
        }

        ਜੇ (cmd == "ifdef") {
            std::string name;
            iss >> name;
            ਤਾਰਕਿਕ p = cond_ਸਰਗਰਮ(cond);
            ਤਾਰਕਿਕ v = macros.count(name);
            cond.push_back({ p, p && v, p && v });
            ਵਾਪਸ ਸੱਚ;
        }

        ਜੇ (cmd == "ifndef") {
            std::string name;
            iss >> name;
            ਤਾਰਕਿਕ p = cond_ਸਰਗਰਮ(cond);
            ਤਾਰਕਿਕ v = !macros.count(name);
            cond.push_back({ p, p && v, p && v });
            ਵਾਪਸ ਸੱਚ;
        }

        ਜੇ (cmd == "if") {
            std::string ਵਿਅੰਜਕ = trim_ਮੁੱਲ(iss);
            ਤਾਰਕਿਕ        p    = cond_ਸਰਗਰਮ(cond);
            ਤਾਰਕਿਕ        v    = ਝੂਠ;
            ਜੇ (p) {
                std::unordered_set<std::string> visiting;
                ਵਿਅੰਜਕ_Parser                      ep(ਵਿਅੰਜਕ, macros, visiting);
                v = ep.parse() != 0;
            }
            cond.push_back({ p, p && v, p && v });
            ਵਾਪਸ ਸੱਚ;
        }

        ਜੇ (cmd == "elif") {
            std::string ਵਿਅੰਜਕ = trim_ਮੁੱਲ(iss);

            ਜੇ (cond.ਖਾਲੀ_ਥਾਵਾਂ()) {
                ਸੁੱਟੋ std::runtime_error("#elif without #if");
            }

            Cond & c = cond.back();
            ਜੇ (!c.ਮਾਪਾ_ਸਰਗਰਮ) {
                c.ਸਰਗਰਮ = ਝੂਠ;
                ਵਾਪਸ ਸੱਚ;
            }

            ਜੇ (c.ਲਿਆ) {
                c.ਸਰਗਰਮ = ਝੂਠ;
                ਵਾਪਸ ਸੱਚ;
            }

            std::unordered_set<std::string> visiting;
            ਵਿਅੰਜਕ_Parser                      ep(ਵਿਅੰਜਕ, macros, visiting);
            ਤਾਰਕਿਕ                            v = ep.parse() != 0;
            c.ਸਰਗਰਮ                          = v;
            ਜੇ (v) {
                c.ਲਿਆ = ਸੱਚ;
            }
            ਵਾਪਸ ਸੱਚ;
        }

        ਜੇ (cmd == "else") {
            ਜੇ (cond.ਖਾਲੀ_ਥਾਵਾਂ()) {
                ਸੁੱਟੋ std::runtime_error("#else without #if");
            }

            Cond & c = cond.back();
            ਜੇ (!c.ਮਾਪਾ_ਸਰਗਰਮ) {
                c.ਸਰਗਰਮ = ਝੂਠ;
                ਵਾਪਸ ਸੱਚ;
            }
            ਜੇ (c.ਲਿਆ) {
                c.ਸਰਗਰਮ = ਝੂਠ;
            } ਨਹੀਂਤਾਂ {
                c.ਸਰਗਰਮ = ਸੱਚ;
                c.ਲਿਆ  = ਸੱਚ;
            }
            ਵਾਪਸ ਸੱਚ;
        }

        ਜੇ (cmd == "endif") {
            ਜੇ (cond.ਖਾਲੀ_ਥਾਵਾਂ()) {
                ਸੁੱਟੋ std::runtime_error("#endif without #if");
            }
            cond.pop_back();
            ਵਾਪਸ ਸੱਚ;
        }

        // Unknown directive
        ਸੁੱਟੋ std::runtime_error("Unknown directive: #" + cmd);
    }
};

}  // namespace pre_wgsl

#ਜੇ_ਅੰਤ  // PRE_WGSL_HPP
