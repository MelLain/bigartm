// Copyright 2017, Additive Regularization of Topic Models.

#pragma once

#include <string>
#include <vector>
#include <unordered_map>

#include "artm/core/common.h"
#include "artm/core/token.h"

namespace artm {
namespace core {

// Phi matrix is an interface (abstract class without methods).
// It represents a single-precision matrix with two dimentions (tokens and topics).
class PhiMatrix {
 public:
  static const int kUndefIndex = -1;

  virtual int token_size() const = 0;
  virtual int topic_size() const = 0;
  virtual google::protobuf::RepeatedPtrField<std::string> topic_name() const = 0;
  virtual const std::string& topic_name(int topic_id) const = 0;
  virtual void set_topic_name(int topic_id, const std::string& topic_name) = 0;
  virtual ModelName model_name() const = 0;
  virtual int64_t ByteSize() const = 0;

  virtual const Token& token(int index) const = 0;
  virtual bool has_token(const Token& token) const = 0;
  virtual int token_index(const Token& token) const = 0;

  virtual float get(int token_id, int topic_id) const = 0;
  virtual void get(int token_id, std::vector<float>* buffer) const = 0;
  virtual void set(int token_id, int topic_id, float value) = 0;
  virtual void increase(int token_id, int topic_id, float increment) = 0;
  virtual void increase(int token_id, const std::vector<float>& increment) = 0;  // must be thread-safe

  virtual void Clear() = 0;
  virtual int AddToken(const Token& token) = 0;

  virtual const TransactionType* GetTransactionByIndex(int index) const = 0;
  virtual int GetIndexByTransaction(const TransactionType& transaction_type) const = 0;
  virtual int GetOrAddTransactionType(const TransactionType& transaction_type) = 0;

  virtual const std::unordered_map<int, TransactionType>& GetIndexToTransactionType() const = 0;
  virtual const std::unordered_map<TransactionType, int, TransactionTypeHasher>&
  GetTransactionTypeToIndex() const = 0;

  virtual bool UseTransactions() const = 0;

  virtual void Reshape(const PhiMatrix& phi_matrix) = 0;

  virtual std::shared_ptr<PhiMatrix> Duplicate() const = 0;
  virtual ~PhiMatrix() { }
};

}  // namespace core
}  // namespace artm
