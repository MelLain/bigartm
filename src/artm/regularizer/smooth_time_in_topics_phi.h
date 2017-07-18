/* Copyright 2017, Additive Regularization of Topic Models.

   Author: Murat Apishev (great-mel@yandex.ru)

   This class proceeds smoothing of tokens in Phi using nearest values.

   More preferable for time stamps tokens. Requires to be used with sorted dictionary
   (e.g. the tokens should follow in some order, chronological, for instance).

   The formula of M-step is
   
   p_wt \propto n_wt + tau * p_wt * (sign(p_{w-1,t} - p_wt) + sign(p_{w+1,t} - p_wt)),
   
   The parameters of the regularizer:
   - topic_names (the names of topics to regularize, empty == all)
   - class_ids (class ids to regularize, empty == all)

   Note: w runs from 1 to number of tokens - 1, e.g. ignores first and last tokens.
*/

#ifndef SRC_ARTM_REGULARIZER_SMOOTH_TIME_IN_TOPICS_PHI_H_
#define SRC_ARTM_REGULARIZER_SMOOTH_TIME_IN_TOPICS_PHI_H_

#include <memory>
#include <string>

#include "artm/regularizer_interface.h"

namespace artm {
namespace regularizer {

class SmoothTimeInTopicsPhi : public RegularizerInterface {
 public:
  explicit SmoothTimeInTopicsPhi(const SmoothTimeInTopicsPhiConfig& config) : config_(config) { }

  virtual bool RegularizePhi(const ::artm::core::PhiMatrix& p_wt,
                             const ::artm::core::PhiMatrix& n_wt,
                             ::artm::core::PhiMatrix* result);

  virtual google::protobuf::RepeatedPtrField<std::string> topics_to_regularize();
  virtual google::protobuf::RepeatedPtrField<std::string> class_ids_to_regularize();

  virtual bool Reconfigure(const RegularizerConfig& config);

 private:
  SmoothTimeInTopicsPhiConfig config_;
};

}  // namespace regularizer
}  // namespace artm

#endif  // SRC_ARTM_REGULARIZER_SMOOTH_TIME_IN_TOPICS_PHI_H_
