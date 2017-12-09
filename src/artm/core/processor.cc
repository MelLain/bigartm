// Copyright 2017, Additive Regularization of Topic Models.

#include "artm/core/processor.h"

#include <stdlib.h>

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <set>

#include "boost/exception/diagnostic_information.hpp"
#include "boost/lexical_cast.hpp"
#include "boost/uuid/uuid_generators.hpp"
#include "boost/uuid/uuid_io.hpp"
#include "boost/functional/hash.hpp"

#include "glog/logging.h"

#include "artm/regularizer_interface.h"
#include "artm/score_calculator_interface.h"

#include "artm/core/protobuf_helpers.h"
#include "artm/core/call_on_destruction.h"
#include "artm/core/cuckoo_watch.h"
#include "artm/core/helpers.h"
#include "artm/core/batch_manager.h"
#include "artm/core/cache_manager.h"
#include "artm/core/score_manager.h"
#include "artm/core/phi_matrix.h"
#include "artm/core/phi_matrix_operations.h"
#include "artm/core/instance.h"

#include "artm/utility/blas.h"

namespace util = artm::utility;
namespace fs = boost::filesystem;
using ::util::CsrMatrix;
using ::util::LocalThetaMatrix;
using ::util::LocalPhiMatrix;

const float kProcessorEps = 1e-16f;

namespace artm {
namespace core {

namespace {
std::vector<Token> GetBatchTokens(const Batch& batch) {
  std::vector<Token> retval;
  std::unordered_map<ClassId, std::set<TransactionType> > batch_tts;

  for (const auto& ptt : batch.transaction_type()) {
    std::vector<ClassId> class_ids = std::vector<ClassId>(ptt.value().begin(), ptt.value().end());
    auto tt = TransactionType(class_ids);
    for (const ClassId& class_id : class_ids) {
      batch_tts[class_id].insert(tt);
    }
  }

  for (int token_index = 0; token_index < batch.token_size(); ++token_index) {
    std::string keyword = batch.token(token_index);
    ClassId class_id = batch.class_id(token_index);
    auto iter = batch_tts.find(class_id);
    if (iter == batch_tts.end()) {
      LOG(WARNING) << "Token " << keyword << " with class_id " << class_id
        << " doesn't match any transaction from batch transaction_type and will be skipped";
      continue;
    }
    for (const auto& tt : iter->second) {
      retval.push_back(Token(class_id, keyword, tt));
    }
  }
  return retval;
}

std::pair<std::map<std::vector<int>, int>, std::map<ClassId, std::set<TransactionType>>>
GetBatchTransactions(const Batch& batch) {
  std::map<std::vector<int>, int> transactions;
  std::map<ClassId, std::set<TransactionType>> class_id_to_tt;

  for (int item_index = 0; item_index < batch.item_size(); ++item_index) {
    const Item& item = batch.item(item_index);

    for (int index = 0; index < item.transaction_token_ids_size(); ++index) {
      const auto& token_ids = item.transaction_token_ids(index).value();
      const auto elem = std::vector<int>(token_ids.begin(), token_ids.end());
      auto iter = transactions.find(elem);
      if (iter == transactions.end()) {
        transactions.insert(std::make_pair(elem, transactions.size()));
      }

      std::vector<ClassId> class_ids;
      for (const int token_id : token_ids) {
        class_ids.push_back(batch.class_id(token_id));
      }
      TransactionType tt(class_ids);
      for (const ClassId& class_id : class_ids) {
        class_id_to_tt[class_id].emplace(tt);
      }
    }
  }
  return std::make_pair(transactions, class_id_to_tt);
}

}

class RegularizeThetaAgentCollection : public RegularizeThetaAgent {
 private:
  std::vector<std::shared_ptr<RegularizeThetaAgent>> agents_;

 public:
  void AddAgent(std::shared_ptr<RegularizeThetaAgent> agent) {
    if (agent != nullptr) {
      agents_.push_back(agent);
    }
  }

  bool empty() const { return agents_.empty(); }

  virtual void Apply(int item_index, int inner_iter, int topics_size, const float* n_td, float* r_td) const {
    for (auto& agent : agents_) {
      agent->Apply(item_index, inner_iter, topics_size, n_td, r_td);
    }
  }

  virtual void Apply(int inner_iter, const LocalThetaMatrix<float>& n_td, LocalThetaMatrix<float>* r_td) const {
    for (auto& agent : agents_) {
      agent->Apply(inner_iter, n_td, r_td);
    }
  }
};

class RegularizePtdwAgentCollection : public RegularizePtdwAgent {
 private:
  std::vector<std::shared_ptr<RegularizePtdwAgent>> agents_;

 public:
  void AddAgent(std::shared_ptr<RegularizePtdwAgent> agent) {
    if (agent != nullptr) {
      agents_.push_back(agent);
    }
  }

  bool empty() const { return agents_.empty(); }

  virtual void Apply(int item_index, int inner_iter, LocalPhiMatrix<float>* ptdw) const {
    for (auto& agent : agents_) {
      agent->Apply(item_index, inner_iter, ptdw);
    }
  }
};

class NormalizeThetaAgent : public RegularizeThetaAgent {
 public:
  virtual void Apply(int item_index, int inner_iter, int topics_size, const float* n_td, float* r_td) const {
    float sum = 0.0f;
    for (int topic_index = 0; topic_index < topics_size; ++topic_index) {
      float val = n_td[topic_index] + r_td[topic_index];
      if (val > 0) {
        sum += val;
      }
    }

    float sum_inv = sum > 0.0f ? (1.0f / sum) : 0.0f;
    for (int topic_index = 0; topic_index < topics_size; ++topic_index) {
      float val = sum_inv * (n_td[topic_index] + r_td[topic_index]);
      if (val < 1e-16f) {
        val = 0.0f;
      }

      // Hack-hack, write normalized values back to n_td
      const_cast<float*>(n_td)[topic_index] = val;
    }
  }
};

static void CreateThetaCacheEntry(ThetaMatrix* new_cache_entry_ptr,
                                  LocalThetaMatrix<float>* theta_matrix,
                                  const Batch& batch,
                                  const PhiMatrix& p_wt,
                                  const ProcessBatchesArgs& args) {
  if (new_cache_entry_ptr == nullptr) {
    return;
  }

  const int topic_size = p_wt.topic_size();
  for (int item_index = 0; item_index < batch.item_size(); ++item_index) {
    const Item& item = batch.item(item_index);
    new_cache_entry_ptr->add_item_id(item.id());
    new_cache_entry_ptr->add_item_title(item.has_title() ? item.title() : std::string());
    new_cache_entry_ptr->add_item_weights();
  }

  if (!args.has_predict_transaction_type()) {
    for (int item_index = 0; item_index < batch.item_size(); ++item_index) {
      for (int topic_index = 0; topic_index < topic_size; ++topic_index) {
        new_cache_entry_ptr->mutable_item_weights(item_index)->add_value((*theta_matrix)(topic_index, item_index));
      }
    }
  } else {
    new_cache_entry_ptr->clear_topic_name();
    for (int token_index = 0; token_index < p_wt.token_size(); token_index++) {
      const Token& token = p_wt.token(token_index);
      if (token.transaction_type != TransactionType({ args.predict_transaction_type() })) {
        continue;
      }

      new_cache_entry_ptr->add_topic_name(token.keyword);
      for (int item_index = 0; item_index < batch.item_size(); ++item_index) {
        float weight = 0.0;
        for (int topic_index = 0; topic_index < topic_size; ++topic_index) {
          weight += (*theta_matrix)(topic_index, item_index) * p_wt.get(token_index, topic_index);
        }
        new_cache_entry_ptr->mutable_item_weights(item_index)->add_value(weight);
      }
    }
  }
}

static void CreatePtdwCacheEntry(ThetaMatrix* new_cache_entry_ptr,
                                 LocalPhiMatrix<float>* ptdw_matrix,
                                 const Batch& batch,
                                 int item_index,
                                 int topic_size) {
  if (new_cache_entry_ptr == nullptr) {
    return;
  }

  const Item& item = batch.item(item_index);
  for (int token_index = 0; token_index < ptdw_matrix->num_tokens(); ++token_index) {
    new_cache_entry_ptr->add_item_id(item.id());
    new_cache_entry_ptr->add_item_title(item.has_title() ? item.title() : std::string());
    auto non_zero_topic_values = new_cache_entry_ptr->add_item_weights();
    auto non_zero_topic_indices = new_cache_entry_ptr->add_topic_indices();

    for (int topic_index = 0; topic_index < topic_size; ++topic_index) {
      float value = ptdw_matrix->operator()(token_index, topic_index);
      if (std::fabs(value) > kProcessorEps) {
        // store not-null values p(t|d,w) for given d and w
        non_zero_topic_values->add_value(value);
        // store indices of these not-null values
        non_zero_topic_indices->add_value(topic_index);
      }
    }
  }
}

class NwtWriteAdapter {
 public:
  virtual void Store(int batch_token_id, int pwt_token_id, const std::vector<float>& nwt_vector) = 0;
  virtual ~NwtWriteAdapter() { }
};

class PhiMatrixWriter : public NwtWriteAdapter {
 public:
  explicit PhiMatrixWriter(PhiMatrix* n_wt) : n_wt_(n_wt) { }

  virtual void Store(int batch_token_id, int pwt_token_id, const std::vector<float>& nwt_vector) {
    assert(nwt_vector.size() == n_wt_->topic_size());
    n_wt_->increase(pwt_token_id, nwt_vector);
  }

 private:
  PhiMatrix* n_wt_;
};

Processor::Processor(Instance* instance)
    : instance_(instance),
      is_stopping(false),
      thread_() {
  // Keep this at the last action in constructor.
  // http://stackoverflow.com/questions/15751618/initialize-boost-thread-in-object-constructor
  boost::thread t(&Processor::ThreadFunction, this);
  thread_.swap(t);
}

Processor::~Processor() {
  is_stopping = true;
  if (thread_.joinable()) {
    thread_.join();
  }
}

static std::shared_ptr<LocalThetaMatrix<float>>
InitializeTheta(int topic_size, const Batch& batch, const ProcessBatchesArgs& args, const ThetaMatrix* cache) {
  auto Theta = std::make_shared<LocalThetaMatrix<float>>(topic_size, batch.item_size());

  Theta->InitializeZeros();

  for (int item_index = 0; item_index < batch.item_size(); ++item_index) {
    int index_of_item = -1;
    if ((cache != nullptr) && args.reuse_theta()) {
      index_of_item = repeated_field_index_of(cache->item_title(),
        batch.item(item_index).title());
    }

    if ((index_of_item != -1) && args.reuse_theta()) {
      const FloatArray& old_thetas = cache->item_weights(index_of_item);
      for (int topic_index = 0; topic_index < topic_size; ++topic_index) {
        (*Theta)(topic_index, item_index) = old_thetas.value(topic_index);
      }
    } else {
      if (args.use_random_theta()) {
        size_t seed = 0;
        boost::hash_combine(seed, std::hash<std::string>()(batch.id()));
        boost::hash_combine(seed, std::hash<int>()(item_index));
        std::vector<float> theta_values = Helpers::GenerateRandomVector(topic_size, seed);
        for (int iTopic = 0; iTopic < topic_size; ++iTopic) {
          (*Theta)(iTopic, item_index) = theta_values[iTopic];
        }
      } else {
        const float default_theta = 1.0f / topic_size;
        for (int iTopic = 0; iTopic < topic_size; ++iTopic) {
          (*Theta)(iTopic, item_index) = default_theta;
        }
      }
    }
  }

  return Theta;
}

static std::shared_ptr<LocalPhiMatrix<float>>
InitializePhi(const Batch& batch,
              const ::artm::core::PhiMatrix& p_wt) {
  std::vector<Token> tokens = GetBatchTokens(batch);
  bool phi_is_empty = true;
  int topic_size = p_wt.topic_size();
  auto phi_matrix = std::make_shared<LocalPhiMatrix<float>>(tokens.size(), topic_size);
  phi_matrix->InitializeZeros();

  for (int token_index = 0; token_index < tokens.size(); ++token_index) {
    Token token = tokens[token_index];
    int p_wt_token_index = p_wt.token_index(token);

    if (p_wt_token_index != ::artm::core::PhiMatrix::kUndefIndex) {
      phi_is_empty = false;
      for (int topic_index = 0; topic_index < topic_size; ++topic_index) {
        float value = p_wt.get(p_wt_token_index, topic_index);
        if (value < kProcessorEps) {
          // Reset small values to 0.0 to avoid performance hit.
          // http://en.wikipedia.org/wiki/Denormal_number#Performance_issues
          // http://stackoverflow.com/questions/13964606/inconsistent-multiplication-performance-with-floats
          value = 0.0f;
        }
        (*phi_matrix)(token_index, topic_index) = value;
      }
    }
  }

  if (phi_is_empty) {
    return nullptr;
  }

  return phi_matrix;
}

static void
CreateRegularizerAgents(const Batch& batch, const ProcessBatchesArgs& args, Instance* instance,
                        RegularizeThetaAgentCollection* theta_agents, RegularizePtdwAgentCollection* ptdw_agents) {
  for (int reg_index = 0; reg_index < args.regularizer_name_size(); ++reg_index) {
    auto& reg_name = args.regularizer_name(reg_index);
    float tau = args.regularizer_tau(reg_index);
    auto regularizer = instance->regularizers()->get(reg_name);
    if (regularizer == nullptr) {
      LOG(ERROR) << "Theta Regularizer with name <" << reg_name << "> does not exist.";
      continue;
    }

    if (theta_agents != nullptr) {
      theta_agents->AddAgent(regularizer->CreateRegularizeThetaAgent(batch, args, tau));
    }

    if (ptdw_agents != nullptr) {
      ptdw_agents->AddAgent(regularizer->CreateRegularizePtdwAgent(batch, args, tau));
    }
  }

  if (theta_agents != nullptr) {
    theta_agents->AddAgent(std::make_shared<NormalizeThetaAgent>());
  }
}

static std::shared_ptr<CsrMatrix<float>>
InitializeSparseNdw(const Batch& batch, const ProcessBatchesArgs& args) {
  std::vector<float> n_dw_val;
  std::vector<int> n_dw_row_ptr;
  std::vector<int> n_dw_col_ind;

  bool use_classes = false;
  std::map<ClassId, float> class_id_to_weight;
  if (args.class_id_size() != 0) {
    use_classes = true;
    for (int i = 0; i < args.class_id_size(); ++i) {
      class_id_to_weight.insert(std::make_pair(args.class_id(i), args.class_weight(i)));
    }
  }

  // For sparse case
  for (int item_index = 0; item_index < batch.item_size(); ++item_index) {
    n_dw_row_ptr.push_back(static_cast<int>(n_dw_val.size()));
    const Item& item = batch.item(item_index);
    for (int token_index = 0; token_index < item.transaction_token_ids_size(); ++token_index) {
      int token_id = item.transaction_token_ids(token_index).value(0);

      float class_weight = 1.0f;
      if (use_classes) {
        ClassId class_id = batch.class_id(token_id);
        auto iter = class_id_to_weight.find(class_id);
        class_weight = (iter == class_id_to_weight.end()) ? 0.0f : iter->second;
      }

      const float token_weight = item.token_weight(token_index);
      n_dw_val.push_back(class_weight * token_weight);
      n_dw_col_ind.push_back(token_id);
    }
  }
  n_dw_row_ptr.push_back(static_cast<int>(n_dw_val.size()));
  return std::make_shared<CsrMatrix<float>>(batch.token_size(), &n_dw_val, &n_dw_row_ptr, &n_dw_col_ind);
}


static std::shared_ptr<CsrMatrix<float>>
InitializeSparseNdw_New(const Batch& batch, const ProcessBatchesArgs& args) {
  std::vector<float> n_dw_val;
  std::vector<int> n_dw_row_ptr;
  std::vector<int> n_dw_col_ind;

  bool use_classes = false;
  std::map<ClassId, float> class_id_to_weight;
  if (args.class_id_size() != 0) {
    use_classes = true;
    for (int i = 0; i < args.class_id_size(); ++i) {
      class_id_to_weight.insert(std::make_pair(args.class_id(i), args.class_weight(i)));
    }
  }

  std::vector<Token> tokens = GetBatchTokens(batch);
  std::map<Token, int> token_to_index;
  int idx = 0;
  for (auto& token : tokens) {
    token_to_index.emplace(std::make_pair(token, idx++));
  }

  // For sparse case
  for (int item_index = 0; item_index < batch.item_size(); ++item_index) {
    n_dw_row_ptr.push_back(static_cast<int>(n_dw_val.size()));
    const Item& item = batch.item(item_index);
    for (int token_index = 0; token_index < item.transaction_token_ids_size(); ++token_index) {
      std::vector<ClassId> class_ids;
      for (int token_id : item.transaction_token_ids(token_index).value()) {
        class_ids.push_back(batch.class_id(token_id));
      }

      for (int token_id : item.transaction_token_ids(token_index).value()) {
        float class_weight = 1.0f;
        if (use_classes) {
          ClassId class_id = batch.class_id(token_id);
          auto iter = class_id_to_weight.find(class_id);
          class_weight = (iter == class_id_to_weight.end()) ? 0.0f : iter->second;
        }
        TransactionType tt(class_ids);
        const float token_weight = item.token_weight(token_index);
        n_dw_val.push_back(class_weight * token_weight);
        n_dw_col_ind.push_back(token_to_index.find(Token(batch.class_id(token_id), batch.token(token_id), tt))->second);
      }
    }
  }
  n_dw_row_ptr.push_back(static_cast<int>(n_dw_val.size()));
  return std::make_shared<CsrMatrix<float>>(batch.token_size(), &n_dw_val, &n_dw_row_ptr, &n_dw_col_ind);
}

static int InitializeSparseTransactionNdw(const Batch& batch, const ::artm::core::PhiMatrix& p_wt,
                                          const ProcessBatchesArgs& args,
                                          CsrMatrix<float>** sparse_ndw,
                                          CsrMatrix<int>** sparse_nmw) {
  bool use_transactions = false;
  std::map<TransactionType, float> tt_to_weight;
  if (args.transaction_type_size() != 0) {
    use_transactions = true;
    for (int i = 0; i < args.transaction_type_size(); ++i) {
      tt_to_weight.insert(std::make_pair(TransactionType(args.transaction_type(i).value()),
                                         args.transaction_weight(i)));
    }
  }

  auto result = GetBatchTransactions(batch);
  std::map<std::vector<int>, int>& transaction_to_index = result.first;
  std::map<ClassId, std::set<TransactionType>>& class_id_to_tt = result.second;

  // create doc x transactions matrix (value == weight of transaction)
  std::vector<float> n_dw_val;
  std::vector<int> n_dw_row_ptr;
  std::vector<int> n_dw_col_ind;

  int max_doc_len = 0;
  for (int item_index = 0; item_index < batch.item_size(); ++item_index) {
    n_dw_row_ptr.push_back(static_cast<int>(n_dw_val.size()));
    const Item& item = batch.item(item_index);
    int cur_doc_len = 0;
    for (int index = 0; index < item.transaction_token_ids_size(); ++index) {
      float transaction_weight = 1.0f;
      cur_doc_len += item.transaction_token_ids(index).value().size();
      if (use_transactions) {
        std::vector<ClassId> class_ids;
        for (const int idx : item.transaction_token_ids(index).value()) {
          class_ids.push_back(batch.class_id(idx));
        }
        auto iter = tt_to_weight.find(TransactionType(class_ids));
        transaction_weight = (iter == tt_to_weight.end()) ? 0.0f : iter->second;
      }

      const float token_weight = item.token_weight(index);
      n_dw_val.push_back(transaction_weight * token_weight);
      const auto& token_ids = item.transaction_token_ids(index).value();
      const auto elem = std::vector<int>(token_ids.begin(), token_ids.end());
      n_dw_col_ind.push_back(transaction_to_index[elem]);
    }
    max_doc_len = max_doc_len < cur_doc_len ? cur_doc_len : max_doc_len;
  }

  n_dw_row_ptr.push_back(static_cast<int>(n_dw_val.size()));

  // create modality (class_id) x transactions matrix (value == index in phi matrix)
  std::vector<std::vector<Token>> transactions(transaction_to_index.size(), {});
  for (const auto& elem : transaction_to_index) {
    std::vector<Token> transaction;
    for (const int e : elem.first) {
      transaction.push_back(Token(batch.class_id(e), batch.token(e)));
    }
    transactions[elem.second] = transaction;
  }

  std::vector<int> n_mw_val;
  std::vector<int> n_mw_row_ptr;
  std::vector<int> n_mw_col_ind;

  for (const auto& class_id_tts : class_id_to_tt) {
    n_mw_row_ptr.push_back(static_cast<int>(n_mw_val.size()));
    for (int transaction_index = 0; transaction_index < transactions.size(); ++transaction_index) {
      const auto& transaction = transactions[transaction_index];
      std::vector<ClassId> class_ids;
      int index = -1;
      for (int i = 0; i < transaction.size(); ++i) {
        if (transaction[i].class_id == class_id_tts.first) {
          index = i;
        }
        class_ids.push_back(transaction[i].class_id);
      }

      if (index == -1) {
        continue;
      }

      const TransactionType tt(class_ids);
      int token_index = p_wt.token_index(Token(transaction[index].class_id, transaction[index].keyword, tt));
      n_mw_val.push_back(token_index);
      n_mw_col_ind.push_back(transaction_index);
    }
  }

  n_mw_row_ptr.push_back(static_cast<int>(n_mw_val.size()));

  *sparse_ndw = new CsrMatrix<float>(transactions.size(), &n_dw_val, &n_dw_row_ptr, &n_dw_col_ind);
  *sparse_nmw = new CsrMatrix<int>(transactions.size(), &n_mw_val, &n_mw_row_ptr, &n_mw_col_ind);

  return max_doc_len;
}






// change
static void
InferThetaAndUpdateNwtSparse(const ProcessBatchesArgs& args, const Batch& batch, float batch_weight,
                             const ::artm::core::PhiMatrix& p_wt,
                             const RegularizeThetaAgentCollection& theta_agents,
                             LocalThetaMatrix<float>* theta_matrix,
                             NwtWriteAdapter* nwt_writer, util::Blas* blas,
                             ThetaMatrix* new_cache_entry_ptr = nullptr) {
  LocalThetaMatrix<float> n_td(theta_matrix->num_topics(), theta_matrix->num_items());
  const int num_topics = p_wt.topic_size();
  const int docs_count = theta_matrix->num_items();
  const int tokens_count = batch.token_size();

  CsrMatrix<float>* sparse_ndw = nullptr;

  auto result = GetBatchTransactions(batch);
  std::map<std::vector<int>, int>& transaction_to_index = result.first;
  std::map<ClassId, std::set<TransactionType>>& class_id_to_tt = result.second;
  std::vector<std::vector<Token>> transactions(transaction_to_index.size(), {});
  std::map<Token, int> token_to_local_index;
  int token_index_local = 0;
  for (const auto& elem : transaction_to_index) {
    std::vector<Token> transaction;
    std::vector<ClassId> class_ids;
    for (const int e : elem.first) {
      class_ids.push_back(batch.class_id(e));
    }
    for (const int e : elem.first) {
      auto token = Token(batch.class_id(e), batch.token(e), TransactionType(class_ids));
      token_to_local_index.emplace(std::make_pair(token, token_index_local++));
      transaction.push_back(token);
    }
    transactions[elem.second] = transaction;
  }

  if (args.opt_for_avx()) {
  // This version is about 40% faster than the second alternative below.
  // Both versions return 100% equal results.
  // Speedup is due to several factors:
  // 1. explicit loops instead of blas->saxpy and blas->sdot
  //    makes compiler generate AVX instructions (vectorized 128-bit float-point operations)
  // 2. better memory usage (reduced bandwith to DRAM and more sequential accesss)

  CsrMatrix<int>* sparse_nmw = nullptr;
  auto temp = InitializeSparseNdw(batch, args);
  auto sparse_ndw_2 = temp.get();
  int max_local_token_size = InitializeSparseTransactionNdw(batch, p_wt, args, &sparse_ndw, &sparse_nmw);

  LocalPhiMatrix<float> local_phi(max_local_token_size, num_topics);
  LocalThetaMatrix<float> r_td(num_topics, 1);
  std::vector<float> helper_vector(num_topics, 0.0f);

  for (int d = 0; d < docs_count; ++d) {
    float* ntd_ptr = &n_td(0, d);
    float* theta_ptr = &(*theta_matrix)(0, d);  // NOLINT

    const int begin_index = sparse_ndw->row_ptr()[d];
    const int end_index = sparse_ndw->row_ptr()[d + 1];
    local_phi.InitializeZeros();
    bool item_has_tokens = false;
    for (int i = begin_index; i < end_index; ++i) {
      auto& transaction = transactions[i - begin_index];
      for (const auto& token : transaction) {
        if (p_wt.token_index(token) == ::artm::core::PhiMatrix::kUndefIndex) {
          continue;
        }

        auto iter = token_to_local_index.find(token);
        if (iter == token_to_local_index.end()) {
          continue;
        }

        item_has_tokens = true;
        float* local_phi_ptr = &local_phi(iter->second, 0);
        p_wt.get(p_wt.token_index(token), &helper_vector);
        for (int k = 0; k < num_topics; ++k) {
          local_phi_ptr[k] = helper_vector[k];
        }
      }
    }

    if (!item_has_tokens) {
      continue;  // continue to the next item
    }

    for (int inner_iter = 0; inner_iter < args.num_document_passes(); ++inner_iter) {
      for (int k = 0; k < num_topics; ++k) {
        ntd_ptr[k] = 0.0f;
      }

      for (int i = begin_index; i < end_index; ++i) {
        std::vector<float> values(num_topics, 1.0f);
        auto& transaction = transactions[i - begin_index];
        for (const auto& token : transaction) {
          auto iter = token_to_local_index.find(token);
          if (iter == token_to_local_index.end()) {
            continue;
          }
          const float* phi_ptr = &local_phi(iter->second, 0);
          for (int k = 0; k < num_topics; ++k) {
            values[k] *= phi_ptr[k];
          }
        }

        float p_dw_val = 0;
        for (int k = 0; k < num_topics; ++k) {
          p_dw_val += values[k] * theta_ptr[k];
        }
        if (p_dw_val == 0) {
          continue;
        }

        const float alpha = sparse_ndw->val()[i] / p_dw_val;
        for (int k = 0; k < num_topics; ++k) {
          ntd_ptr[k] += alpha * values[k];
        }
      }

      for (int k = 0; k < num_topics; ++k) {
        theta_ptr[k] *= ntd_ptr[k];
      }

      r_td.InitializeZeros();
      theta_agents.Apply(d, inner_iter, num_topics, theta_ptr, r_td.get_data());
    }
  }
  } else {
    auto temp = InitializeSparseNdw(batch, args);
    sparse_ndw = temp.get();
    
  std::shared_ptr<LocalPhiMatrix<float>> phi_matrix_ptr = InitializePhi(batch, p_wt);
  if (phi_matrix_ptr == nullptr) {
    return;
  }
  const LocalPhiMatrix<float>& phi_matrix = *phi_matrix_ptr;
  for (int inner_iter = 0; inner_iter < args.num_document_passes(); ++inner_iter) {
    // helper_td will represent either n_td or r_td, depending on the context - see code below
    LocalThetaMatrix<float> helper_td(theta_matrix->num_topics(), theta_matrix->num_items());
    helper_td.InitializeZeros();

    for (int d = 0; d < docs_count; ++d) {
      for (int i = sparse_ndw->row_ptr()[d]; i < sparse_ndw->row_ptr()[d + 1]; ++i) {
        int w = sparse_ndw->col_ind()[i];
        float p_dw_val = blas->sdot(num_topics, &phi_matrix(w, 0), 1, &(*theta_matrix)(0, d), 1);  // NOLINT
        if (p_dw_val == 0) {
          continue;
        }
        blas->saxpy(num_topics, sparse_ndw->val()[i] / p_dw_val, &phi_matrix(w, 0), 1, &helper_td(0, d), 1);
      }
    }

    AssignDenseMatrixByProduct(*theta_matrix, helper_td, theta_matrix);

    helper_td.InitializeZeros();  // from now this represents r_td
    theta_agents.Apply(inner_iter, *theta_matrix, &helper_td);
  }
  }

  CreateThetaCacheEntry(new_cache_entry_ptr, theta_matrix, batch, p_wt, args);

  if (nwt_writer == nullptr) {
    return;
  }

  // rewrite this fragment using formules
  auto temp = InitializeSparseNdw_New(batch, args);
  sparse_ndw = temp.get();
  CsrMatrix<float> sparse_nwd(*sparse_ndw);
  sparse_nwd.Transpose(blas);

  std::vector<float> p_wt_local(num_topics, 0.0f);
  std::vector<float> n_wt_local(num_topics, 0.0f);

  auto tokens = GetBatchTokens(batch);
  for (int w = 0; w < tokens.size(); ++w) {
    auto r = tokens[w];
    auto t = p_wt.token_index(r);

    if (t == -1) {
      continue;
    }

    p_wt.get(t, &p_wt_local);

    for (int i = sparse_nwd.row_ptr()[w]; i < sparse_nwd.row_ptr()[w + 1]; ++i) {
      int d = sparse_nwd.col_ind()[i];
      float p_wd_val = blas->sdot(num_topics, &p_wt_local[0], 1, &(*theta_matrix)(0, d), 1);  // NOLINT
      if (p_wd_val == 0) {
        continue;
      }
      blas->saxpy(num_topics, sparse_nwd.val()[i] / p_wd_val,
        &(*theta_matrix)(0, d), 1, &n_wt_local[0], 1);  // NOLINT
    }

    std::vector<float> values(num_topics, 0.0f);
    for (int topic_index = 0; topic_index < num_topics; ++topic_index) {
      values[topic_index] = p_wt_local[topic_index] * n_wt_local[topic_index];
      n_wt_local[topic_index] = 0.0f;
    }

    for (float& value : values) {
      value *= batch_weight;
    }
    nwt_writer->Store(w, p_wt.token_index(tokens[w]), values);
  }
}







static void
InferPtdwAndUpdateNwtSparse(const ProcessBatchesArgs& args, const Batch& batch, float batch_weight,
                            const ::artm::core::PhiMatrix& p_wt,
                            const RegularizeThetaAgentCollection& theta_agents,
                            const RegularizePtdwAgentCollection& ptdw_agents,
                            LocalThetaMatrix<float>* theta_matrix,
                            NwtWriteAdapter* nwt_writer, util::Blas* blas,
                            ThetaMatrix* new_cache_entry_ptr = nullptr,
                            ThetaMatrix* new_ptdw_cache_entry_ptr = nullptr) {
  LocalThetaMatrix<float> n_td(theta_matrix->num_topics(), theta_matrix->num_items());
  LocalThetaMatrix<float> r_td(theta_matrix->num_topics(), 1);

  const int num_topics = p_wt.topic_size();
  const int docs_count = theta_matrix->num_items();

  std::shared_ptr<CsrMatrix<float>> sparse_ndw = InitializeSparseNdw(batch, args);

  std::vector<int> token_id(batch.token_size(), -1);
  for (int token_index = 0; token_index < batch.token_size(); ++token_index) {
    token_id[token_index] = p_wt.token_index(Token(batch.class_id(token_index), batch.token(token_index)));
  }

  for (int d = 0; d < docs_count; ++d) {
    float* ntd_ptr = &n_td(0, d);
    float* theta_ptr = &(*theta_matrix)(0, d);  // NOLINT

    const int begin_index = sparse_ndw->row_ptr()[d];
    const int end_index = sparse_ndw->row_ptr()[d + 1];
    const int local_token_size = end_index - begin_index;
    LocalPhiMatrix<float> local_phi(local_token_size, num_topics);
    LocalPhiMatrix<float> local_ptdw(local_token_size, num_topics);
    local_phi.InitializeZeros();
    bool item_has_tokens = false;
    for (int i = begin_index; i < end_index; ++i) {
      int w = sparse_ndw->col_ind()[i];
      if (token_id[w] == ::artm::core::PhiMatrix::kUndefIndex) {
        continue;
      }
      item_has_tokens = true;
      float* local_phi_ptr = &local_phi(i - begin_index, 0);
      for (int k = 0; k < num_topics; ++k) {
        local_phi_ptr[k] = p_wt.get(token_id[w], k);
      }
    }

    if (!item_has_tokens) {
      continue;  // continue to the next item
    }

    for (int inner_iter = 0; inner_iter <= args.num_document_passes(); ++inner_iter) {
      const bool last_iteration = (inner_iter == args.num_document_passes());
      for (int i = begin_index; i < end_index; ++i) {
        const float* phi_ptr = &local_phi(i - begin_index, 0);
        float* ptdw_ptr = &local_ptdw(i - begin_index, 0);

        float p_dw_val = 0.0f;
        for (int k = 0; k < num_topics; ++k) {
          float p_tdw_val = phi_ptr[k] * theta_ptr[k];
          ptdw_ptr[k] = p_tdw_val;
          p_dw_val += p_tdw_val;
        }

        if (p_dw_val == 0) {
          continue;
        }
        const float Z = 1.0f / p_dw_val;
        for (int k = 0; k < num_topics; ++k) {
          ptdw_ptr[k] *= Z;
        }
      }

      ptdw_agents.Apply(d, inner_iter, &local_ptdw);

      if (!last_iteration) {  // update theta matrix (except for the last iteration)
        for (int k = 0; k < num_topics; ++k) {
          ntd_ptr[k] = 0.0f;
        }
        for (int i = begin_index; i < end_index; ++i) {
          const float n_dw = sparse_ndw->val()[i];
          const float* ptdw_ptr = &local_ptdw(i - begin_index, 0);
          for (int k = 0; k < num_topics; ++k) {
            ntd_ptr[k] += n_dw * ptdw_ptr[k];
          }
        }

        for (int k = 0; k < num_topics; ++k) {
          theta_ptr[k] = ntd_ptr[k];
        }

        r_td.InitializeZeros();
        theta_agents.Apply(d, inner_iter, num_topics, theta_ptr, r_td.get_data());
      } else {  // update n_wt matrix (on the last iteration)
        if (nwt_writer != nullptr) {
          std::vector<float> values(num_topics, 0.0f);
          for (int i = begin_index; i < end_index; ++i) {
            const float n_dw = batch_weight * sparse_ndw->val()[i];
            const float* ptdw_ptr = &local_ptdw(i - begin_index, 0);

            for (int k = 0; k < num_topics; ++k) {
              values[k] = ptdw_ptr[k] * n_dw;
            }

            int w = sparse_ndw->col_ind()[i];
            nwt_writer->Store(w, token_id[w], values);
          }
        }
      }
    }
    CreatePtdwCacheEntry(new_ptdw_cache_entry_ptr, &local_ptdw, batch, d, num_topics);
  }
  CreateThetaCacheEntry(new_cache_entry_ptr, theta_matrix, batch, p_wt, args);
}







static std::shared_ptr<Score>
CalcScores(ScoreCalculatorInterface* score_calc, const Batch& batch,
           const PhiMatrix& p_wt, const ProcessBatchesArgs& args, const LocalThetaMatrix<float>& theta_matrix) {
  if (!score_calc->is_cumulative()) {
    return nullptr;
  }

  std::vector<Token> token_dict;
  for (int token_index = 0; token_index < batch.token_size(); ++token_index) {
    token_dict.push_back(Token(batch.class_id(token_index), batch.token(token_index)));
  }

  std::shared_ptr<Score> score = score_calc->CreateScore();
  for (int item_index = 0; item_index < batch.item_size(); ++item_index) {
    const Item& item = batch.item(item_index);

    std::vector<float> theta_vec;
    assert(theta_matrix.num_topics() == p_wt.topic_size());
    for (int topic_index = 0; topic_index < theta_matrix.num_topics(); ++topic_index) {
      theta_vec.push_back(theta_matrix(topic_index, item_index));
    }

    score_calc->AppendScore(item, token_dict, p_wt, args, theta_vec, score.get());
  }

  score_calc->AppendScore(batch, p_wt, args, score.get());

  return score;
}





void Processor::ThreadFunction() {
  try {
    int total_processed_batches = 0;  // counter

    // Do not log performance measurements below kTimeLoggingThreshold milliseconds
    const int kTimeLoggingThreshold = 0;

    Helpers::SetThreadName(-1, "Processor thread");
    LOG(INFO) << "Processor thread started";
    int pop_retries = 0;
    const int pop_retries_max = 20;

    util::Blas* blas = util::Blas::builtin();

    for (;;) {
      if (is_stopping) {
        LOG(INFO) << "Processor thread stopped";
        LOG(INFO) << "Total number of processed batches: " << total_processed_batches;
        break;
      }

      std::shared_ptr<ProcessorInput> part;
      if (!instance_->processor_queue()->try_pop(&part)) {
        pop_retries++;
        LOG_IF(INFO, pop_retries == pop_retries_max) << "No data in processing queue, waiting...";

        boost::this_thread::sleep(boost::posix_time::milliseconds(kIdleLoopFrequency));

        continue;
      }

      LOG_IF(INFO, pop_retries >= pop_retries_max) << "Processing queue has data, processing started";
      pop_retries = 0;

      // CuckooWatch logs time from now to destruction
      const std::string batch_name = part->has_batch_filename() ? part->batch_filename() : part->batch().id();
      CuckooWatch cuckoo(std::string("ProcessBatch(") + batch_name + std::string(")"));
      total_processed_batches++;

      call_on_destruction c([&]() {  // NOLINT
        if (part->batch_manager() != nullptr) {
          part->batch_manager()->Callback(part->task_id());
        }
      });

      Batch batch;
      {
        CuckooWatch cuckoo2("LoadMessage", &cuckoo, kTimeLoggingThreshold);
        if (part->has_batch_filename()) {
          auto mem_batch = instance_->batches()->get(part->batch_filename());
          if (mem_batch != nullptr) {
            batch.CopyFrom(*mem_batch);
          } else {
            try {
              ::artm::core::Helpers::LoadMessage(part->batch_filename(), &batch);
            } catch (std::exception& ex) {
              LOG(ERROR) << ex.what() << ", the batch will be skipped.";
              continue;
            }
          }
        } else {  // part->has_batch_filename()
          batch.CopyFrom(part->batch());
        }
      }

      std::shared_ptr<MasterModelConfig> master_config = instance_->config();

      const ModelName& model_name = part->model_name();
      const ProcessBatchesArgs& args = part->args();
      {
        if (args.class_id_size() != args.class_weight_size()) {
          BOOST_THROW_EXCEPTION(InternalError(
              "model.class_id_size() != model.class_weight_size()"));
        }

        std::shared_ptr<const PhiMatrix> phi_matrix = instance_->GetPhiMatrix(model_name);
        if (phi_matrix == nullptr) {
          LOG(ERROR) << "Model " << model_name << " does not exist.";
          continue;
        }
        const PhiMatrix& p_wt = *phi_matrix;

        if (batch.token_size() == 0) {
          continue;
        }

        std::shared_ptr<const PhiMatrix> nwt_target;
        if (part->has_nwt_target_name()) {
          nwt_target = instance_->GetPhiMatrix(part->nwt_target_name());
          if (nwt_target == nullptr) {
            LOG(ERROR) << "Model " << part->nwt_target_name() << " does not exist.";
            continue;
          }

          if (!PhiMatrixOperations::HasEqualShape(*nwt_target, p_wt)) {
            LOG(ERROR) << "Models " << part->nwt_target_name() << " and "
                       << model_name << " have inconsistent shapes.";
            continue;
          }
        }

        std::stringstream model_description;
        if (part->has_nwt_target_name()) {
          model_description << part->nwt_target_name();
        } else {
          model_description << &p_wt;
        }
        VLOG(0) << "Processor: start processing batch " << batch.id() << " into model " << model_description.str();

        std::shared_ptr<ThetaMatrix> cache;
        if (part->has_reuse_theta_cache_manager()) {
          cache = part->reuse_theta_cache_manager()->FindCacheEntry(batch);
        }
        std::shared_ptr<LocalThetaMatrix<float>> theta_matrix =
          InitializeTheta(p_wt.topic_size(), batch, args, cache.get());

        if (p_wt.token_size() == 0) {
          LOG(INFO) << "Phi is empty, calculations for the model " + model_name +
            "would not be processed on this iteration";
          continue;
        }

        std::shared_ptr<NwtWriteAdapter> nwt_writer;
        if (nwt_target != nullptr) {
          nwt_writer = std::make_shared<PhiMatrixWriter>(const_cast<PhiMatrix*>(nwt_target.get()));
        }

        std::shared_ptr<ThetaMatrix> new_cache_entry_ptr(nullptr);
        if (part->has_cache_manager()) {
          new_cache_entry_ptr.reset(new ThetaMatrix());
        }

        std::shared_ptr<ThetaMatrix> new_ptdw_cache_entry_ptr(nullptr);
        if (part->has_ptdw_cache_manager()) {
          new_ptdw_cache_entry_ptr.reset(new ThetaMatrix());
        }

        if (new_cache_entry_ptr != nullptr) {
          new_cache_entry_ptr->mutable_topic_name()->CopyFrom(p_wt.topic_name());
        }

        if (new_ptdw_cache_entry_ptr != nullptr) {
          new_ptdw_cache_entry_ptr->mutable_topic_name()->CopyFrom(p_wt.topic_name());
        }

        {
          RegularizeThetaAgentCollection theta_agents;
          RegularizePtdwAgentCollection ptdw_agents;
          CreateRegularizerAgents(batch, args, instance_, &theta_agents, &ptdw_agents);

          if (ptdw_agents.empty() && !part->has_ptdw_cache_manager()) {
            CuckooWatch cuckoo2("InferThetaAndUpdateNwtSparse", &cuckoo, kTimeLoggingThreshold);
            InferThetaAndUpdateNwtSparse(args, batch, part->batch_weight(),
                                         p_wt, theta_agents, theta_matrix.get(), nwt_writer.get(),
                                         blas, new_cache_entry_ptr.get());
          } else {
            CuckooWatch cuckoo2("InferPtdwAndUpdateNwtSparse", &cuckoo, kTimeLoggingThreshold);
            InferPtdwAndUpdateNwtSparse(args, batch, part->batch_weight(),
                                        p_wt, theta_agents, ptdw_agents, theta_matrix.get(), nwt_writer.get(),
                                        blas, new_cache_entry_ptr.get(),
                                        new_ptdw_cache_entry_ptr.get());
          }
        }

        if (new_cache_entry_ptr != nullptr) {
          part->cache_manager()->UpdateCacheEntry(batch.id(), *new_cache_entry_ptr);
        }

        if (new_ptdw_cache_entry_ptr != nullptr) {
          part->ptdw_cache_manager()->UpdateCacheEntry(batch.id(), *new_ptdw_cache_entry_ptr);
        }

        for (int score_index = 0; score_index < master_config->score_config_size(); ++score_index) {
          const ScoreName& score_name = master_config->score_config(score_index).name();

          auto score_calc = instance_->scores_calculators()->get(score_name);
          if (score_calc == nullptr) {
            LOG(ERROR) << "Unable to find score calculator '" << score_name << "', referenced by "
              << "model " << p_wt.model_name() << ".";
            continue;
          }

          if (!score_calc->is_cumulative()) {
            continue;
          }

          CuckooWatch cuckoo2("CalculateScore(" + score_name + ")", &cuckoo, kTimeLoggingThreshold);

          auto score_value = CalcScores(score_calc.get(), batch, p_wt, args, *theta_matrix);
          if (score_value != nullptr) {
            instance_->score_manager()->Append(score_name, score_value->SerializeAsString());
            if (part->score_manager() != nullptr) {
              part->score_manager()->Append(score_name, score_value->SerializeAsString());
            }
          }
        }

        VLOG(0) << "Processor: complete processing batch " << batch.id() << " into model " << model_description.str();
      }
    }
  }
  catch (...) {
    LOG(FATAL) << boost::current_exception_diagnostic_information();
  }
}

}  // namespace core
}  // namespace artm
