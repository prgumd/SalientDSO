#ifndef SALIENCY_UTIL
#define SALIENCY_UTIL

#include <vector>
#include <unordered_map>
#include "Eigen/Core"
#include "opencv2/opencv.hpp"

#include "util/ImageAndExposure.h"

using namespace cv;
using namespace dso;

namespace SaliencyUtil {

struct SaliencyContainer {
  SaliencyContainer() : saliency_(NULL), segmentation_(NULL) {}
  SaliencyContainer(ImageAndExposure* saliency, ImageAndExposure* segmentation)
    : saliency_(saliency), segmentation_(segmentation) {}

  ~SaliencyContainer() {}

  // member
  // Saliency for each pixel.
  ImageAndExposure* saliency_;

  // Segmentation label for each pixel.
  ImageAndExposure* segmentation_;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

class SaliencySmoother {
 public:
  SaliencySmoother();
  ~SaliencySmoother();

  // Smooth saliency by detecting blob and decide to sample much evenly or biasly.
  float SmoothByBlob(float* saliency, int width, int height);

  // Down weighted or up weighted saliency of each pixel based on segmentation
  // and predefined mapping.
  void SmoothBySegmentation(float* saliency, unsigned char* segmentation,
    int width, int height,
    const float* weights_map, const int label_max=149);

  // Calculate saliency weight for each patch by finding the dominant
  // segmentation class and using it's probability vector of class to
  // calculate P. The final saliency weight would be sum_i(Si * P^i),
  // i is class.
  void SmoothBySegmentationPatch(int* saliency_ths, float* saliency,
    Eigen::Matrix<float, Eigen::Dynamic, 150> segmentation_prob, int width,
    int height, int patch_size,
    const std::unordered_map<unsigned char, float>& weights_map);

 private:
  // Given saliency prediction, do blob detection or segmentation weighting.
  float PredictSmoothingFactor(float* saliency, int width, int height);
  // Base on the blob detection, decide the smoothing factor.
  float DecideFactor(const std::vector<KeyPoint>&);

  // Compute median of saliency for each class. Replace the saliency of every
  // pixel with the median of the class that it belongs to.
  void ReplaceWithClassMedian(float* saliency, unsigned char* segmentation,
    int width, int height, const int label_max=149);


  SimpleBlobDetector::Params params_;
  Ptr<SimpleBlobDetector> detector_;
};

} // namespace SaliencyUtil

#endif // SALIENCY_UTIL