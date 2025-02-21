#define LLAMA_API_INTERNAL
#include "sampling.h"
#include <random>
/**
 * 初始化llama采样上下文。
 *
 * @param params llama采样参数，用于配置采样过程。
 * @return 返回一个初始化后的llama采样上下文结构体指针。如果解析语法失败或缺少"root"符号，则返回nullptr。
 */
struct llama_sampling_context * llama_sampling_init(const struct llama_sampling_params & params) {
    struct llama_sampling_context * result = new llama_sampling_context();

    result->params  = params;
    result->grammar = nullptr;

    // if there is a grammar, parse it
    if (!params.grammar.empty()) {
        result->parsed_grammar = grammar_parser::parse(params.grammar.c_str());

        // will be empty (default) if there are parse errors
        if (result->parsed_grammar.rules.empty()) {
            fprintf(stderr, "%s: failed to parse grammar\n", __func__);
            delete result;
            return nullptr;
        }

        // Ensure that there is a "root" node.
        if (result->parsed_grammar.symbol_ids.find("root") == result->parsed_grammar.symbol_ids.end()) {
            fprintf(stderr, "%s: grammar does not contain a 'root' symbol\n", __func__);
            delete result;
            return nullptr;
        }

        std::vector<const llama_grammar_element *> grammar_rules(result->parsed_grammar.c_rules());

        struct llama_grammar * grammar = llama_grammar_init(
                grammar_rules.data(),
                grammar_rules.size(), result->parsed_grammar.symbol_ids.at("root"));
        if (grammar == nullptr) {
            throw std::runtime_error("Failed to initialize llama_grammar");
        }
        result->grammar = grammar;
    }

    result->prev.resize(params.n_prev);

    result->n_valid = 0;

    llama_sampling_set_rng_seed(result, params.seed);

    return result;
}
/**
 * 释放llama_sampling_context结构体所占用的内存。
 *
 * @param ctx 指向需要被释放的llama_sampling_context结构体的指针。如果该结构体中的grammar不为空，则先释放grammar占用的内存，然后释放ctx自身占用的内存。
 */
void llama_sampling_free(struct llama_sampling_context * ctx) {
    if (ctx->grammar != NULL) {
        llama_grammar_free(ctx->grammar);
    }

    delete ctx;
}
/**
 * 重置llama采样上下文。
 *
 * @param ctx 指向要重置的llama_sampling_context的指针。
*/
void llama_sampling_reset(llama_sampling_context * ctx) {
    if (ctx->grammar != NULL) {
        llama_grammar_free(ctx->grammar);
        ctx->grammar = NULL;
    }

    if (!ctx->parsed_grammar.rules.empty()) {
        std::vector<const llama_grammar_element *> grammar_rules(ctx->parsed_grammar.c_rules());

        struct llama_grammar * grammar = llama_grammar_init(
                grammar_rules.data(),
                grammar_rules.size(), ctx->parsed_grammar.symbol_ids.at("root"));
        if (grammar == nullptr) {
            throw std::runtime_error("Failed to initialize llama_grammar");
        }
        ctx->grammar = grammar;
    }

    std::fill(ctx->prev.begin(), ctx->prev.end(), 0);
    ctx->cur.clear();
    ctx->n_valid = 0;
}
/**
 * 设置llama采样的随机数生成器的种子。
 *
 * @param ctx 指向llama_sampling_context结构的指针，该结构包含随机数生成器的状态。
 * @param seed 要设置的种子值。如果seed等于LLAMA_DEFAULT_SEED，则使用std::random_device生成一个真正的随机种子。
 */
void llama_sampling_set_rng_seed(struct llama_sampling_context * ctx, uint32_t seed) {
    if (seed == LLAMA_DEFAULT_SEED) {
        seed = std::random_device{}();
    }
    ctx->rng.seed(seed);
}
/**
 * 将源llama_sampling_context的内容复制到目标llama_sampling_context中。
 *
 * @param src 源llama_sampling_context，其内容将被复制到dst中。
 * @param dst 目标llama_sampling_context，其内容将被覆盖为src的内容。
 */
void llama_sampling_cp(llama_sampling_context * src, llama_sampling_context * dst) {
    if (dst->grammar) {
        llama_grammar_free(dst->grammar);
        dst->grammar = nullptr;
    }

    if (src->grammar) {
        dst->grammar = llama_grammar_copy(src->grammar);
    }

    dst->prev = src->prev;
}
/**
 * 获取llama_sampling_context结构体中的prev向量的最后一个元素。
 *
 * @param ctx 指向llama_sampling_context结构体的指针，该结构体中包含一个prev向量。
 * @return 返回prev向量的最后一个元素。
 */
llama_token llama_sampling_last(llama_sampling_context * ctx) {
    return ctx->prev.back();
}
/**
 * 从给定的llama_sampling_context中获取前n个token，并将其转换为字符串。
 *
 * @param ctx_sampling 指向llama_sampling_context的指针，该结构体包含了需要采样的前一个token序列。
 * @param ctx_main 指向llama_context的指针，该结构体包含了所有token的信息。
 * @param n 需要获取的token数量。如果大于前一个token序列的大小，则返回整个序列。
 * @return 返回一个字符串，其中包含了从ctx_sampling的前一个token序列中取出的前n个token转换后的结果。
 */
std::string llama_sampling_prev_str(llama_sampling_context * ctx_sampling, llama_context * ctx_main, int n) {
    const int size = ctx_sampling->prev.size();

    n = std::min(n, size);

    std::string result;

    for (int i = size - n; i < size; i++) {
        result += llama_token_to_piece(ctx_main, ctx_sampling->prev[i]);
    }

    return result;
}
/**
 * 将llama_sampling_params参数转化为字符串格式。
 *
 * @param params 需要转化的llama_sampling_params参数。
 * @return 返回转化后的字符串，包含了所有参数的值。
 */
std::string llama_sampling_print(const llama_sampling_params & params) {
    char result[1024];

    snprintf(result, sizeof(result),
            "\trepeat_last_n = %d, repeat_penalty = %.3f, frequency_penalty = %.3f, presence_penalty = %.3f\n"
            "\ttop_k = %d, tfs_z = %.3f, top_p = %.3f, min_p = %.3f, typical_p = %.3f, temp = %.3f\n"
            "\tmirostat = %d, mirostat_lr = %.3f, mirostat_ent = %.3f",
            params.penalty_last_n, params.penalty_repeat, params.penalty_freq, params.penalty_present,
            params.top_k, params.tfs_z, params.top_p, params.min_p, params.typical_p, params.temp,
            params.mirostat, params.mirostat_eta, params.mirostat_tau);

    return std::string(result);
}
/**
 * 使用给定的采样参数，生成并返回一个字符串，该字符串描述了采样顺序。
 *
 * @param params llama_sampling_params类型的对象，包含了采样过程中所需的所有参数。
 * @return 返回一个字符串，描述了采样的顺序。如果mirostat参数为0，则按照samplers_sequence中的顺序进行采样；否则，采样顺序为"-> mirostat "。
 */
std::string llama_sampling_order_print(const llama_sampling_params & params) {
    std::string result = "CFG -> Penalties ";
    if (params.mirostat == 0) {
        for (auto sampler_type : params.samplers_sequence) {
            const auto sampler_type_name = llama_sampling_type_to_str(sampler_type);
            if (!sampler_type_name.empty()) {
                result += "-> " + sampler_type_name + " ";
            }
        }
    } else {
        result += "-> mirostat ";
    }

    return result;
}
/**
 * 将llama采样器类型转换为字符串。
 *
 * @param sampler_type 需要转换的llama采样器类型。
 * @return 返回对应的字符串表示，如果输入的类型不在预定义的范围内，则返回空字符串。
 */
std::string llama_sampling_type_to_str(llama_sampler_type sampler_type) {
    switch (sampler_type) {
        case llama_sampler_type::TOP_K:       return "top_k";
        case llama_sampler_type::TFS_Z:       return "tfs_z";
        case llama_sampler_type::TYPICAL_P:   return "typical_p";
        case llama_sampler_type::TOP_P:       return "top_p";
        case llama_sampler_type::MIN_P:       return "min_p";
        case llama_sampler_type::TEMPERATURE: return "temperature";
        default : return "";
    }
}
/**
 * 根据给定的名称列表，返回对应的采样器类型列表。
 *
 * @param names 需要转换的采样器名称列表。
 * @param allow_alt_names 是否允许使用别名作为采样器名称。
 * @return 返回一个包含对应采样器类型的向量。
 */
std::vector<llama_sampler_type> llama_sampling_types_from_names(const std::vector<std::string> & names, bool allow_alt_names) {
    std::unordered_map<std::string, llama_sampler_type> sampler_canonical_name_map {
        {"top_k",       llama_sampler_type::TOP_K},
        {"top_p",       llama_sampler_type::TOP_P},
        {"typical_p",   llama_sampler_type::TYPICAL_P},
        {"min_p",       llama_sampler_type::MIN_P},
        {"tfs_z",       llama_sampler_type::TFS_Z},
        {"temperature", llama_sampler_type::TEMPERATURE}
    };

    // since samplers names are written multiple ways
    // make it ready for both system names and input names
    std::unordered_map<std::string, llama_sampler_type> sampler_alt_name_map {
        {"top-k",       llama_sampler_type::TOP_K},
        {"top-p",       llama_sampler_type::TOP_P},
        {"nucleus",     llama_sampler_type::TOP_P},
        {"typical-p",   llama_sampler_type::TYPICAL_P},
        {"typical",     llama_sampler_type::TYPICAL_P},
        {"min-p",       llama_sampler_type::MIN_P},
        {"tfs-z",       llama_sampler_type::TFS_Z},
        {"tfs",         llama_sampler_type::TFS_Z},
        {"temp",        llama_sampler_type::TEMPERATURE}
    };

    std::vector<llama_sampler_type> sampler_types;
    sampler_types.reserve(names.size());
    for (const auto & name : names)
    {
        auto sampler_item = sampler_canonical_name_map.find(name);
        if (sampler_item != sampler_canonical_name_map.end())
        {
            sampler_types.push_back(sampler_item->second);
        }
        else
        {
            if (allow_alt_names)
            {
                sampler_item = sampler_alt_name_map.find(name);
                if (sampler_item != sampler_alt_name_map.end())
                {
                    sampler_types.push_back(sampler_item->second);
                }
            }
        }
    }
    return sampler_types;
}
/**
 * 从给定的字符串中解析出对应的采样类型。
 *
 * @param names_string 包含采样类型字符的字符串，每个字符对应一种采样类型。
 * @return 返回一个向量，其中包含了从输入字符串中解析出的采样类型。如果输入字符串中的某个字符在映射表中找不到对应的采样类型，则忽略该字符。
 */
std::vector<llama_sampler_type> llama_sampling_types_from_chars(const std::string & names_string) {
    std::unordered_map<char, llama_sampler_type> sampler_name_map {
        {'k', llama_sampler_type::TOP_K},
        {'p', llama_sampler_type::TOP_P},
        {'y', llama_sampler_type::TYPICAL_P},
        {'m', llama_sampler_type::MIN_P},
        {'f', llama_sampler_type::TFS_Z},
        {'t', llama_sampler_type::TEMPERATURE}
    };

    std::vector<llama_sampler_type> sampler_types;
    sampler_types.reserve(names_string.size());
    for (const auto & c : names_string) {
        const auto sampler_item = sampler_name_map.find(c);
        if (sampler_item != sampler_name_map.end()) {
            sampler_types.push_back(sampler_item->second);
        }
    }
    return sampler_types;
}
/**
 * 对给定的上下文和采样参数，使用指定的采样器类型进行采样。
 *
 * @param ctx_main 主上下文，用于执行采样操作。
 * @param params 采样参数，包括温度、动态温度范围、动态温度指数、top_k、top_p、min_p、tfs_z以及典型概率等参数。
 * @param cur_p 当前待采样的数据数组。
 * @param min_keep 最小保留数，即在采样过程中，至少保留的数量。
 */
static void sampler_queue(
                   struct llama_context * ctx_main,
            const llama_sampling_params & params,
                 llama_token_data_array & cur_p,
                                 size_t   min_keep) {
    const float         temp              = params.temp;
    const float         dynatemp_range    = params.dynatemp_range;
    const float         dynatemp_exponent = params.dynatemp_exponent;
    const int32_t       top_k             = params.top_k;
    const float         top_p             = params.top_p;
    const float         min_p             = params.min_p;
    const float         tfs_z             = params.tfs_z;
    const float         typical_p         = params.typical_p;
    const std::vector<llama_sampler_type> & samplers_sequence = params.samplers_sequence;

    for (auto sampler_type : samplers_sequence) {
        switch (sampler_type) {
            case llama_sampler_type::TOP_K    : llama_sample_top_k    (ctx_main, &cur_p, top_k,     min_keep); break;
            case llama_sampler_type::TFS_Z    : llama_sample_tail_free(ctx_main, &cur_p, tfs_z,     min_keep); break;
            case llama_sampler_type::TYPICAL_P: llama_sample_typical  (ctx_main, &cur_p, typical_p, min_keep); break;
            case llama_sampler_type::TOP_P    : llama_sample_top_p    (ctx_main, &cur_p, top_p,     min_keep); break;
            case llama_sampler_type::MIN_P    : llama_sample_min_p    (ctx_main, &cur_p, min_p,     min_keep); break;
            case llama_sampler_type::TEMPERATURE:
                if (dynatemp_range > 0) {
                    float dynatemp_min = std::max(0.0f, temp - dynatemp_range);
                    float dynatemp_max = std::max(0.0f, temp + dynatemp_range);
                    llama_sample_entropy(ctx_main, &cur_p, dynatemp_min, dynatemp_max, dynatemp_exponent);
                } else {
                    llama_sample_temp(ctx_main, &cur_p, temp);
                }
                break;
            default : break;
        }
    }
}
/**
 * 实现llama采样的样本生成。
 *
 * @param ctx_sampling 采样上下文。
 * @param ctx_main 主上下文。
 * @param ctx_cfg 配置上下文。
 * @param idx 索引。
 * @param is_resampling 是否进行重采样。
 * @return 返回生成的llama token。
 */
static llama_token llama_sampling_sample_impl(
                  struct llama_sampling_context * ctx_sampling,
                  struct llama_context * ctx_main,
                  struct llama_context * ctx_cfg,
                  const int idx,
                  bool is_resampling) {
    const llama_sampling_params & params = ctx_sampling->params;

    const float   temp            = params.temp;
    const int     mirostat        = params.mirostat;
    const float   mirostat_tau    = params.mirostat_tau;
    const float   mirostat_eta    = params.mirostat_eta;

    std::vector<float> original_logits;
    auto cur_p = llama_sampling_prepare(ctx_sampling, ctx_main, ctx_cfg, idx, /* apply_grammar= */ is_resampling, &original_logits);
    if (ctx_sampling->grammar != NULL && !is_resampling) {
        GGML_ASSERT(!original_logits.empty());
    }
    llama_token id = 0;

    if (temp < 0.0) {
        // greedy sampling, with probs
        llama_sample_softmax(ctx_main, &cur_p);
        id = cur_p.data[0].id;
    } else if (temp == 0.0) {
        // greedy sampling, no probs
        id = llama_sample_token_greedy(ctx_main, &cur_p);
    } else {
        if (mirostat == 1) {
            const int mirostat_m = 100;
            llama_sample_temp(ctx_main, &cur_p, temp);
            id = llama_sample_token_mirostat(ctx_main, &cur_p, mirostat_tau, mirostat_eta, mirostat_m, &ctx_sampling->mirostat_mu);
        } else if (mirostat == 2) {
            llama_sample_temp(ctx_main, &cur_p, temp);
            id = llama_sample_token_mirostat_v2(ctx_main, &cur_p, mirostat_tau, mirostat_eta, &ctx_sampling->mirostat_mu);
        } else {
            // temperature sampling
            size_t min_keep = std::max(1, params.min_keep);

            sampler_queue(ctx_main, params, cur_p, min_keep);

            id = llama_sample_token_with_rng(ctx_main, &cur_p, ctx_sampling->rng);

            //{
            //    const int n_top = 10;
            //    LOG("top %d candidates:\n", n_top);

            //    for (int i = 0; i < n_top; i++) {
            //        const llama_token id = cur_p.data[i].id;
            //        (void)id; // To avoid a warning that id is unused when logging is disabled.
            //        LOG(" - %5d: '%12s' (%.3f)\n", id, llama_token_to_piece(ctx_main, id).c_str(), cur_p.data[i].p);
            //    }
            //}

            //LOG("sampled token: %5d: '%s'\n", id, llama_token_to_piece(ctx_main, id).c_str());
        }
    }

    if (ctx_sampling->grammar != NULL && !is_resampling) {
        // Get a pointer to the logits
        float * logits = llama_get_logits_ith(ctx_main, idx);

        // Create an array with a single token data element for the sampled id
        llama_token_data single_token_data = {id, logits[id], 0.0f};
        llama_token_data_array single_token_data_array = { &single_token_data, 1, false };

        // Apply grammar constraints to the single token
        llama_grammar_sample(ctx_sampling->grammar, ctx_main, &single_token_data_array);

        // Check if the token is valid according to the grammar by seeing if its logit has been set to -INFINITY
        bool is_valid = single_token_data_array.data[0].logit != -INFINITY;

        // If the token is not valid according to the grammar, perform resampling
        if (!is_valid) {
            LOG("Resampling because token %d: '%s' does not meet grammar rules\n", id, llama_token_to_piece(ctx_main, id).c_str());

            // Restore logits from the copy
            std::copy(original_logits.begin(), original_logits.end(), logits);

            return llama_sampling_sample_impl(ctx_sampling, ctx_main, ctx_cfg, idx, /* is_resampling= */ true);
        }
    }

    ctx_sampling->n_valid = temp == 0.0f ? 0 : cur_p.size;

    return id;
}
/**
 * 准备进行采样的实现函数。该函数主要负责获取模型的logits，应用惩罚参数，以及在需要时应用语法检查。
 *
 * @param ctx_sampling 采样上下文，包含采样所需的所有参数和状态信息。
 * @param ctx_main 主模型上下文，用于获取模型和logits。
 * @param ctx_cfg 配置模型上下文，如果提供，将用于应用引导（guidance）参数。
 * @param idx 当前处理的token索引。
 * @param apply_grammar 是否应用语法检查。
 * @param original_logits 如果提供了原始logits，且不应用语法检查，则将原始logits复制到original_logits中。
 * @return 返回一个llama_token_data_array对象，其中包含了当前token的id、logit值和概率。
 */
static llama_token_data_array llama_sampling_prepare_impl(
                  struct llama_sampling_context * ctx_sampling,
                  struct llama_context * ctx_main,
                  struct llama_context * ctx_cfg,
                  const int idx,
                  bool apply_grammar,
                  std::vector<float> * original_logits) {
    const llama_sampling_params & params = ctx_sampling->params;

    const int n_vocab = llama_n_vocab(llama_get_model(ctx_main));

    const int32_t penalty_last_n  = params.penalty_last_n < 0 ? params.n_prev : params.penalty_last_n;
    const float   penalty_repeat  = params.penalty_repeat;
    const float   penalty_freq    = params.penalty_freq;
    const float   penalty_present = params.penalty_present;

    const bool    penalize_nl     = params.penalize_nl;

    auto & prev = ctx_sampling->prev;
    auto & cur  = ctx_sampling->cur;

    // Get a pointer to the logits
    float * logits = llama_get_logits_ith(ctx_main, idx);

    if (ctx_sampling->grammar != NULL && !apply_grammar) {
        GGML_ASSERT(original_logits != NULL);
        // Only make a copy of the original logits if we are not applying grammar checks, not sure if I actually have to do this.
        *original_logits = {logits, logits + n_vocab};
    }

    // apply params.logit_bias map
    for (auto it = params.logit_bias.begin(); it != params.logit_bias.end(); it++) {
        logits[it->first] += it->second;
    }

    if (ctx_cfg) {
        float * logits_guidance = llama_get_logits_ith(ctx_cfg, idx);
        llama_sample_apply_guidance(ctx_main, logits, logits_guidance, params.cfg_scale);
    }

    cur.resize(n_vocab);

    for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
        cur[token_id] = llama_token_data{token_id, logits[token_id], 0.0f};
    }

    llama_token_data_array cur_p = { cur.data(), cur.size(), false };

    // apply penalties
    const auto& penalty_tokens = params.use_penalty_prompt_tokens ? params.penalty_prompt_tokens : prev;
    const int penalty_tokens_used_size = std::min((int)penalty_tokens.size(), penalty_last_n);
    if (penalty_tokens_used_size) {
        const float nl_logit = logits[llama_token_nl(llama_get_model(ctx_main))];

        llama_sample_repetition_penalties(ctx_main, &cur_p,
                penalty_tokens.data() + penalty_tokens.size() - penalty_tokens_used_size,
                penalty_tokens_used_size, penalty_repeat, penalty_freq, penalty_present);

        if (!penalize_nl) {
            for (size_t idx = 0; idx < cur_p.size; idx++) {
                if (cur_p.data[idx].id == llama_token_nl(llama_get_model(ctx_main))) {
                    cur_p.data[idx].logit = nl_logit;
                    break;
                }
            }
        }
    }

    // apply grammar checks before sampling logic
    if (apply_grammar && ctx_sampling->grammar != NULL) {
        llama_grammar_sample(ctx_sampling->grammar, ctx_main, &cur_p);
    }

    return cur_p;
}
/**
 * 使用给定的上下文和索引，调用llama_sampling_sample_impl函数进行采样。
 *
 * @param ctx_sampling 用于采样的llama_sampling_context结构体指针。
 * @param ctx_main 主llama_context结构体指针。
 * @param ctx_cfg 配置llama_context结构体指针。
 * @param idx 采样的索引值。
 * @return 返回llama_token类型的采样结果。
 */
llama_token llama_sampling_sample(
                  struct llama_sampling_context * ctx_sampling,
                  struct llama_context * ctx_main,
                  struct llama_context * ctx_cfg,
                  const int idx) {
    // Call the implementation function with is_resampling set to false by default
    return llama_sampling_sample_impl(ctx_sampling, ctx_main, ctx_cfg, idx, /* is_resampling= */ false);
}
/**
 * 准备llama采样所需的数据。
 *
 * @param ctx_sampling 指向llama采样上下文的指针。
 * @param ctx_main 指向主llama上下文的指针。
 * @param ctx_cfg 指向配置llama上下文的指针。
 * @param idx 索引值，用于指定要处理的数据。
 * @param apply_grammar 是否应用语法规则。
 * @param original_logits 指向原始logits的向量指针。
 * @return 返回一个llama_token_data_array类型的对象，该对象包含了采样所需的所有数据。
 */
llama_token_data_array llama_sampling_prepare(
                  struct llama_sampling_context * ctx_sampling,
                  struct llama_context * ctx_main,
                  struct llama_context * ctx_cfg,
                  const int idx,
                  bool apply_grammar,
                  std::vector<float> * original_logits) {
    return llama_sampling_prepare_impl(ctx_sampling,ctx_main, ctx_cfg, idx, apply_grammar, original_logits);
}

/**
 * 对给定的llama采样上下文进行采样接受操作。
 *
 * @param ctx_sampling 指向要进行采样接受操作的llama采样上下文的指针。
 * @param ctx_main 指向主llama上下文的指针，用于在应用语法时使用。
 * @param id 要添加到采样上下文中的新token。
 * @param apply_grammar 如果为真，则在采样上下文中应用语法规则。
 */
void llama_sampling_accept(
        struct llama_sampling_context * ctx_sampling,
        struct llama_context * ctx_main,
        llama_token id,
        bool apply_grammar) {
    ctx_sampling->prev.erase(ctx_sampling->prev.begin());
    ctx_sampling->prev.push_back(id);

    if (ctx_sampling->grammar != NULL && apply_grammar) {
        llama_grammar_accept_token(ctx_sampling->grammar, ctx_main, id);
    }
}
