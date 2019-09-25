#include "SaliencyUtil.h"

using namespace cv;

namespace SaliencyUtil {

SaliencySmoother::SaliencySmoother() {
  // Setup SimpleBlobDetector parameters.

  // Change thresholds
  params_.minThreshold = 0;
  params_.maxThreshold = 200;

  // Filter by Area.
  params_.filterByArea = true;
  params_.minArea = 600;

  // Filter by Circularity
  // params_.filterByCircularity = true;
  // params_.minCircularity = 0.1;

  // Filter by Convexity
  // params_.filterByConvexity = true;
  // params_.minConvexity = 0.87;

  // Filter by Inertia
  params_.filterByInertia = true;
  params_.minInertiaRatio = 0.001;

  // Set up detector with params
  Ptr<SimpleBlobDetector> detector_ = SimpleBlobDetector::create(params_);
}

SaliencySmoother::~SaliencySmoother() {
  
}

float SaliencySmoother::SmoothByBlob(float* saliency, int width, int height) {
  return PredictSmoothingFactor(saliency, width, height);
}


// TODO
void SaliencySmoother::SmoothBySegmentation(float* saliency,
    unsigned char* segmentation, int width, int height,
    const float* weights_map, const int label_max) {

  ReplaceWithClassMedian(saliency, segmentation, width, height, label_max);

  for (int h = 0; h < height; h++) {
    for (int w = 0; w < width; w++) {
      int index = width * h + w;
      int label = segmentation[index] - 0;
      // printf("%d\n", label);
      // Out of range.
      if (label > label_max) {
        continue;
      }
      saliency[index] *= weights_map[label];
    }
  }
}

// TODO
void SaliencySmoother::SmoothBySegmentationPatch(int* saliency_ths,
    float* saliency,
    Eigen::Matrix<float, Eigen::Dynamic, 150> segmentation_prob, int width,
    int height, int patch_size,
    const std::unordered_map<unsigned char, float>& weights_map) {
  assert(false);
}

float SaliencySmoother::PredictSmoothingFactor(float* saliency, int width, int height) {

  // Create Mat
  Mat saliency_mat = Mat(height, width, CV_32F, *saliency);
  Mat uchar_saliency_mat;
  saliency_mat.convertTo(uchar_saliency_mat, CV_8UC1);

  // Storage for blobs
  std::vector<KeyPoint> keypoints;  

  // Detect blobs
  detector_->detect(uchar_saliency_mat, keypoints);

  // Draw detected blobs as red circles.
  // DrawMatchesFlags::DRAW_RICH_KEYPOINTS flag ensures
  // the size of the circle corresponds to the size of blob

  // Mat im_with_keypoints;
  // drawKeypoints( saliency_mat, keypoints, im_with_keypoints, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );

  // Show blobs
  // imshow("keypoints", im_with_keypoints );
  // waitKey(0);

  return DecideFactor(keypoints);
}

// TODO
// Decide rules.
float SaliencySmoother::DecideFactor(const std::vector<KeyPoint>& kps) {
  int blobs_num = kps.size();

  // Accumulate diameter of all blobs.
  float accu_size = 0.0;

  for (const KeyPoint& kp : kps) {
    accu_size += kp.size;
  }

  float factor = 1.0;

  return factor;
}

void SaliencySmoother::ReplaceWithClassMedian(float* saliency,
    unsigned char* segmentation, int width, int height, const int label_max) {
  int** class_hist = new int* [label_max + 1];
  for (int i = 0; i <= label_max; i++) {
    class_hist[i] = NULL;
  }

  // Compute histogram.
  for (int h = 0; h < height; h++) {
    for (int w = 0; w < width; w++) {
      int index = width * h + w;
      int label = segmentation[index] - 0;
      
      // Out of range.
      if (label > label_max) {
        continue;
      }

      if (class_hist[label] == NULL) {
        class_hist[label] = new int [257];
        memset(class_hist[label], 0, sizeof(int) * 257);
      }
      class_hist[label][static_cast<int>(saliency[index])]++;
      class_hist[label][256]++;
    }
  }

  // For exist class, compute it's median and use class_hist[i][0] to store it.
  for (int i = 0; i <= label_max; i++) {
    if (class_hist[i] == NULL) {
      continue;
    }
    int median = 255;
    int all_count = class_hist[i][256] * 0.5 + 0.5f;

    for (int j = 0; j < 256; j++) {
      all_count -= class_hist[i][j];
      if (all_count < 0) {
        median = j;
        break;
      }
    }

    class_hist[i][0] = median;
  }

  // Compute histogram.
  for (int h = 0; h < height; h++) {
    for (int w = 0; w < width; w++) {
      int index = width * h + w;
      int label = segmentation[index] - 0;
      
      // Out of range.
      if (label > label_max) {
        continue;
      }

      saliency[index] = class_hist[label][0];
    }
  }

  // Clean up.
  for (int i = 0; i <= label_max; i++) {
    if (class_hist[i] == NULL) {
      continue;
    }

    delete [] class_hist[i];
  }

  delete [] class_hist;
}

} // namespace SaliencyUtil
