
// Copyright 2017, Additive Regularization of Topic Models.

#pragma once

#include <string>
#include <vector>

#include "boost/functional/hash.hpp"

namespace artm {
namespace core {

typedef std::string ClassId;
typedef std::vector<ClassId> TransactionType;
const int NoAnyTransactionType = -1;
const int NoSuchTransactionType = -2;
const std::string DefaultClass = "@default_class";
const std::string DocumentsClass = "@documents_class";

// Token is a triple of keyword, its class_id (also known as tokens' modality) and id of the transaction.
// Pay attention to the order of the arguments in the constructor.
// For historical reasons ClassId goes first, followed by the keyword and transaction id.
struct Token {
  Token(const ClassId& _class_id, const std::string& _keyword)
      : keyword(_keyword), class_id(_class_id), transaction_id(NoAnyTransactionType)
      , hash_(calcHash(_class_id, _keyword, NoAnyTransactionType)) { }

  Token(const ClassId& _class_id, const std::string& _keyword, int _transaction_id)
      : keyword(_keyword), class_id(_class_id), transaction_id(_transaction_id)
      , hash_(calcHash(_class_id, _keyword, _transaction_id)) { }

  Token& operator=(const Token& rhs) {
    if (this != &rhs) {
      const_cast<std::string&>(keyword) = rhs.keyword;
      const_cast<ClassId&>(class_id) = rhs.class_id;
      const_cast<int&>(transaction_id) = rhs.transaction_id;
      const_cast<size_t&>(hash_) = rhs.hash_;
    }

    return *this;
  }

  bool operator<(const Token& token) const {
    if (keyword != token.keyword) {
      return keyword < token.keyword;
    }

    if (class_id != token.class_id) {
      return class_id < token.class_id;
    }

    return transaction_id < token.transaction_id;
  }

  bool operator==(const Token& token) const {
    if (keyword == token.keyword && class_id == token.class_id
        && transaction_id == token.transaction_id) {
      return true;
    }
    return false;
  }

  bool operator!=(const Token& token) const {
    return !(*this == token);
  }

  size_t hash() const { return hash_; }

  const std::string keyword;
  const ClassId class_id;
  const int transaction_id;

 private:
  friend struct TokenHasher;
  const size_t hash_;

  static size_t calcHash(const ClassId& class_id, const std::string& keyword, int transaction_id) {
    size_t hash = 0;
    boost::hash_combine<std::string>(hash, keyword);
    boost::hash_combine<std::string>(hash, class_id);
    boost::hash_combine<std::string>(hash, std::to_string(transaction_id));
    return hash;
  }
};

struct TokenHasher {
  size_t operator()(const Token& token) const {
    return token.hash_;
  }
};

struct TransactionTypeHasher {
  size_t operator()(const TransactionType& transaction_type) const {
    size_t hash = 0;
    for (const ClassId& class_id : transaction_type) {
      boost::hash_combine<std::string>(hash, class_id);
    }
    return hash;
  }
};

}  // namespace core
}  // namespace artm
