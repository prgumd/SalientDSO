/**
* This file is part of DSO.
* 
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/


#include "FullSystem/PixelSelector2.h"
 
// 

#include <algorithm>
#include <vector>

#include "util/NumType.h"
#include "IOWrapper/ImageDisplay.h"
#include "util/globalCalib.h"
#include "FullSystem/HessianBlocks.h"
#include "util/globalFuncs.h"
#include "util/settings.h"
#include "util/FrameShell.h"

namespace dso
{


PixelSelector::PixelSelector(int w, int h)
{
  patch_size_ = setting_patch_size;
  randomPattern = new unsigned char[w*h];
  std::srand(3141592);  // want to be deterministic.
  for(int i=0;i<w*h;i++) randomPattern[i] = rand() & 0xFF;

  currentPotential=3;


  gradHist = new int[100*(1+w/patch_size_)*(1+h/patch_size_)];
  ths = new float[(w/patch_size_)*(h/patch_size_)+100];
  thsSmoothed = new float[(w/patch_size_)*(h/patch_size_)+100];

  allowFast=false;
  gradHistFrame=0;

  // Saliency
  if (USE_SALIENCY_SAMPLING) {
    saliency_hist_ = new int [257];
    saliency_ths_ = new int [(w/patch_size_)*(h/patch_size_)+100];
    saliency_ths_smoothed_ = new float [(w/patch_size_)*(h/patch_size_)+100];
    std::random_device rd;
    gen_ = new std::mt19937(rd());
    saliency_random_ = NULL;
    saliency_smoother_ = new SaliencyUtil::SaliencySmoother();
  }
  max_saliency_ = 0.0f;
  min_saliency_ = 256.0f;
}

PixelSelector::~PixelSelector()
{
  delete[] randomPattern;
  delete[] gradHist;
  delete[] ths;
  delete[] thsSmoothed;

  // Saliency
  if (USE_SALIENCY_SAMPLING) {
    delete [] saliency_hist_;
    delete [] saliency_ths_;
    delete [] saliency_ths_smoothed_;
    delete gen_;
    delete saliency_random_;
    delete saliency_smoother_;
  }
}

int computeHistQuantil(int* hist, float below)
{
  int th = hist[0]*below+0.5f;
  for(int i=0;i<90;i++)
  {
    th -= hist[i+1];
    if(th<0) return i;
  }
  return 90;
}


void PixelSelector::makeHists(const FrameHessian* const fh)
{
  gradHistFrame = fh;
  float * mapmax0 = fh->absSquaredGrad[0];

  int w = wG[0];
  int h = hG[0];

  int p_w = w/patch_size_;
  int p_h = h/patch_size_;
  thsStep = p_w;

  for(int y=0;y<p_h;y++)
    for(int x=0;x<p_w;x++)
    {
      float* map0 = mapmax0+patch_size_*x+patch_size_*y*w;
      int* hist0 = gradHist;// + 50*(x+y*p_w);
      memset(hist0,0,sizeof(int)*50);

      for(int j=0;j<patch_size_;j++) for(int i=0;i<patch_size_;i++)
      {
        int it = i+patch_size_*x;
        int jt = j+patch_size_*y;
        if(it>w-2 || jt>h-2 || it<1 || jt<1) continue;
        int g = sqrtf(map0[i+j*w]);
        if(g>48) g=48;
        hist0[g+1]++;
        hist0[0]++;
      }

      ths[x+y*p_w] = computeHistQuantil(hist0,setting_minGradHistCut) + setting_minGradHistAdd;
    }

  for(int y=0;y<p_h;y++)
    for(int x=0;x<p_w;x++)
    {
      float sum=0,num=0;
      if(x>0)
      {
        if(y>0)   {num++;   sum+=ths[x-1+(y-1)*p_w];}
        if(y<p_h-1) {num++;   sum+=ths[x-1+(y+1)*p_w];}
        num++; sum+=ths[x-1+(y)*p_w];
      }

      if(x<p_w-1)
      {
        if(y>0)   {num++;   sum+=ths[x+1+(y-1)*p_w];}
        if(y<p_h-1) {num++;   sum+=ths[x+1+(y+1)*p_w];}
        num++; sum+=ths[x+1+(y)*p_w];
      }

      if(y>0)   {num++;   sum+=ths[x+(y-1)*p_w];}
      if(y<p_h-1) {num++;   sum+=ths[x+(y+1)*p_w];}
      num++; sum+=ths[x+y*p_w];

      thsSmoothed[x+y*p_w] = (sum/num) * (sum/num);

    }

  if (USE_SALIENCY_SAMPLING && fh->saliency_ != NULL) {
    makeSaliencyHists(fh);
    setUpSaliencyRandomSampler();
  }
}

int PixelSelector::makeMaps(
    const FrameHessian* const fh,
    float* map_out, float density, int recursionsLeft, bool plot, float thFactor)
{
  float numHave=0;
  float numWant=density;
  float quotia;
  int idealPotential = currentPotential;


//  if(setting_pixelSelectionUseFast>0 && allowFast)
//  {
//    memset(map_out, 0, sizeof(float)*wG[0]*hG[0]);
//    std::vector<cv::KeyPoint> pts;
//    cv::Mat img8u(hG[0],wG[0],CV_8U);
//    for(int i=0;i<wG[0]*hG[0];i++)
//    {
//      float v = fh->dI[i][0]*0.8;
//      img8u.at<uchar>(i) = (!std::isfinite(v) || v>255) ? 255 : v;
//    }
//    cv::FAST(img8u, pts, setting_pixelSelectionUseFast, true);
//    for(unsigned int i=0;i<pts.size();i++)
//    {
//      int x = pts[i].pt.x+0.5;
//      int y = pts[i].pt.y+0.5;
//      map_out[x+y*wG[0]]=1;
//      numHave++;
//    }
//
//    printf("FAST selection: got %f / %f!\n", numHave, numWant);
//    quotia = numWant / numHave;
//  }
//  else
  {




    // the number of selected pixels behaves approximately as
    // K / (pot+1)^2, where K is a scene-dependent constant.
    // we will allow sub-selecting pixels by up to a quotia of 0.25, otherwise we will re-select.
    // Do only once.
    if(fh != gradHistFrame) makeHists(fh);

    Eigen::Vector3i n;
    // select!
    if (USE_SALIENCY_SAMPLING) {
      // Saliency.
      n = this->saliencySelect(fh, map_out,currentPotential, thFactor);
    } else {
      n = this->select(fh, map_out,currentPotential, thFactor);
    }

    // sub-select!
    numHave = n[0]+n[1]+n[2];
    // printf("Number of sampled points: %f\n", numHave);
    quotia = numWant / numHave;

    // by default we want to over-sample by 40% just to be sure.
    float K = numHave * (currentPotential+1) * (currentPotential+1);
    idealPotential = sqrtf(K/numWant)-1;  // round down.
    // if (idealPotential > patch_size_) {
    //   idealPotential = patch_size_;
    // }
    if(idealPotential<1) idealPotential=1;

    if( recursionsLeft>0 && quotia > 1.25 && currentPotential>1)
    {
      //re-sample to get more points!
      // potential needs to be smaller
      if(idealPotential>=currentPotential)
        idealPotential = currentPotential-1;

      // printf("PixelSelector: have %.2f%%, need %.2f%%. RESAMPLE with pot %d -> %d.\n",
      //    100*numHave/(float)(wG[0]*hG[0]),
      //    100*numWant/(float)(wG[0]*hG[0]),
      //    currentPotential,
      //    idealPotential);
      currentPotential = idealPotential;
      return makeMaps(fh,map_out, density, recursionsLeft-1, plot,thFactor);
    }
    else if(recursionsLeft>0 && quotia < 0.25)
    {
      // re-sample to get less points!

      if(idealPotential<=currentPotential)
        idealPotential = currentPotential+1;

     // printf("PixelSelector: have %.2f%%, need %.2f%%. RESAMPLE with pot %d -> %d.\n",
     //     100*numHave/(float)(wG[0]*hG[0]),
     //     100*numWant/(float)(wG[0]*hG[0]),
     //     currentPotential,
     //     idealPotential);
      currentPotential = idealPotential;
      return makeMaps(fh,map_out, density, recursionsLeft-1, plot,thFactor);

    }
  }

  int numHaveSub = numHave;
  if(quotia < 0.95)
  {
    int wh=wG[0]*hG[0];
    int rn=0;
    unsigned char charTH = 255*quotia;
    for(int i=0;i<wh;i++)
    {
      if(map_out[i] != 0)
      {
        if(randomPattern[rn] > charTH )
        {
          map_out[i]=0;
          numHaveSub--;
        }
        rn++;
      }
    }
  }

 // printf("PixelSelector: have %.2f%%, need %.2f%%. KEEPCURR with pot %d -> %d. Subsampled to %.2f%%\n",
 //     100*numHave/(float)(wG[0]*hG[0]),
 //     100*numWant/(float)(wG[0]*hG[0]),
 //     currentPotential,
 //     idealPotential,
 //     100*numHaveSub/(float)(wG[0]*hG[0]));
  currentPotential = idealPotential;


  if(plot)
  {
    int w = wG[0];
    int h = hG[0];

    MinimalImageB3 img(w,h);

    for(int i=0;i<w*h;i++)
    {
      float c = fh->dI[i][0]*0.7;
      // float c = fh->saliency_[i];
      if(c>255) c=255;
      img.at(i) = Vec3b(c,c,c);
    }
    // IOWrap::displayImage("Selector Image", &img);

    for(int y=0; y<h;y++)
      for(int x=0;x<w;x++)
      {
        int i=x+y*w;
        if(map_out[i] == 1)
          img.setPixelCirc(x,y,Vec3b(0,255,0));
        else if(map_out[i] == 2)
          img.setPixelCirc(x,y,Vec3b(255,0,0));
        else if(map_out[i] == 4)
          img.setPixelCirc(x,y,Vec3b(0,0,255));
      }
    IOWrap::displayImage("Selector Pixels", &img);

    printf("Image ID: %d\n", fh->shell->incoming_id);
  }

  return numHaveSub;
}



Eigen::Vector3i PixelSelector::select(const FrameHessian* const fh,
    float* map_out, int pot, float thFactor)
{

  Eigen::Vector3f const * const map0 = fh->dI;

  float * mapmax0 = fh->absSquaredGrad[0];
  float * mapmax1 = fh->absSquaredGrad[1];
  float * mapmax2 = fh->absSquaredGrad[2];


  int w = wG[0];
  int w1 = wG[1];
  int w2 = wG[2];
  int h = hG[0];


  const Vec2f directions[16] = {
           Vec2f(0,    1.0000),
           Vec2f(0.3827,    0.9239),
           Vec2f(0.1951,    0.9808),
           Vec2f(0.9239,    0.3827),
           Vec2f(0.7071,    0.7071),
           Vec2f(0.3827,   -0.9239),
           Vec2f(0.8315,    0.5556),
           Vec2f(0.8315,   -0.5556),
           Vec2f(0.5556,   -0.8315),
           Vec2f(0.9808,    0.1951),
           Vec2f(0.9239,   -0.3827),
           Vec2f(0.7071,   -0.7071),
           Vec2f(0.5556,    0.8315),
           Vec2f(0.9808,   -0.1951),
           Vec2f(1.0000,    0.0000),
           Vec2f(0.1951,   -0.9808)};

  memset(map_out,0,w*h*sizeof(PixelSelectorStatus));



  float dw1 = setting_gradDownweightPerLevel;
  float dw2 = dw1*dw1;


  int n3=0, n2=0, n4=0;
  for(int y4=0;y4<h;y4+=(4*pot)) for(int x4=0;x4<w;x4+=(4*pot))
  {
    int my3 = std::min((4*pot), h-y4);
    int mx3 = std::min((4*pot), w-x4);
    int bestIdx4=-1; float bestVal4=0;
    Vec2f dir4 = directions[randomPattern[n2] & 0xF];
    for(int y3=0;y3<my3;y3+=(2*pot)) for(int x3=0;x3<mx3;x3+=(2*pot))
    {
      int x34 = x3+x4;
      int y34 = y3+y4;
      int my2 = std::min((2*pot), h-y34);
      int mx2 = std::min((2*pot), w-x34);
      int bestIdx3=-1; float bestVal3=0;
      Vec2f dir3 = directions[randomPattern[n2] & 0xF];
      for(int y2=0;y2<my2;y2+=pot) for(int x2=0;x2<mx2;x2+=pot)
      {
        int x234 = x2+x34;
        int y234 = y2+y34;
        int my1 = std::min(pot, h-y234);
        int mx1 = std::min(pot, w-x234);
        int bestIdx2=-1; float bestVal2=0;
        Vec2f dir2 = directions[randomPattern[n2] & 0xF];
        for(int y1=0;y1<my1;y1+=1) for(int x1=0;x1<mx1;x1+=1)
        {
          assert(x1+x234 < w);
          assert(y1+y234 < h);
          int idx = x1+x234 + w*(y1+y234);
          int xf = x1+x234;
          int yf = y1+y234;

          if(xf<4 || xf>=w-5 || yf<4 || yf>h-4) continue;


          float pixelTH0 =
            thsSmoothed[xf / patch_size_ + (yf / patch_size_) * thsStep];
          float pixelTH1 = pixelTH0*dw1;
          float pixelTH2 = pixelTH1*dw2;


          float ag0 = mapmax0[idx];
          if(ag0 > pixelTH0*thFactor)
          {
            Vec2f ag0d = map0[idx].tail<2>();
            float dirNorm = fabsf((float)(ag0d.dot(dir2)));
            if(!setting_selectDirectionDistribution) dirNorm = ag0;

            if(dirNorm > bestVal2)
            { bestVal2 = dirNorm; bestIdx2 = idx; bestIdx3 = -2; bestIdx4 = -2;}
          }
          if(bestIdx3==-2) continue;

          float ag1 = mapmax1[(int)(xf*0.5f+0.25f) + (int)(yf*0.5f+0.25f)*w1];
          if(ag1 > pixelTH1*thFactor)
          {
            Vec2f ag0d = map0[idx].tail<2>();
            float dirNorm = fabsf((float)(ag0d.dot(dir3)));
            if(!setting_selectDirectionDistribution) dirNorm = ag1;

            if(dirNorm > bestVal3)
            { bestVal3 = dirNorm; bestIdx3 = idx; bestIdx4 = -2;}
          }
          if(bestIdx4==-2) continue;

          float ag2 = mapmax2[(int)(xf*0.25f+0.125) + (int)(yf*0.25f+0.125)*w2];
          if(ag2 > pixelTH2*thFactor)
          {
            Vec2f ag0d = map0[idx].tail<2>();
            float dirNorm = fabsf((float)(ag0d.dot(dir4)));
            if(!setting_selectDirectionDistribution) dirNorm = ag2;

            if(dirNorm > bestVal4)
            { bestVal4 = dirNorm; bestIdx4 = idx; }
          }
        }

        if(bestIdx2>0)
        {
          map_out[bestIdx2] = 1;
          bestVal3 = 1e10;
          n2++;
        }
      }

      if(bestIdx3>0)
      {
        map_out[bestIdx3] = 2;
        bestVal4 = 1e10;
        n3++;
      }
    }

    if(bestIdx4>0)
    {
      map_out[bestIdx4] = 4;
      n4++;
    }
  }


  return Eigen::Vector3i(n2,n3,n4);
}

// Saliency

int SaliencyComputeHistQuantil(int * hist, float below) {
  int all_count = hist[256] * below + 0.5f;

  for (int i = 0; i < 256; i++) {
    all_count -= hist[i];
    if (all_count < 0) {
      return i;
    }
  }

  return 255;
}

void PixelSelector::makeSaliencyHists(const FrameHessian* const fh) {
  float* saliency_map = fh->saliency_;

  int w = wG[0];
  int h = hG[0];

  int p_w = w / patch_size_;
  int p_h = h / patch_size_;

  if (setting_blob_smoothing) {
    saliency_smoother_->SmoothByBlob(saliency_map, w, h);
  } else if (setting_segmentation_smoothing && fh->segmentation_ != NULL) {
    saliency_smoother_->SmoothBySegmentation(saliency_map, fh->segmentation_,
      w, h, setting_seg_weights_map);
  }

  float saliency_sum = 0.0f;

  for (int y = 0; y < p_h; y++) {
    for (int x = 0; x < p_w; x++) {
      if (saliency_map == NULL) {
        saliency_ths_[x + y * p_w] = 1.0f;
        continue;
      }
      
      float* curr_map = saliency_map + patch_size_ * x + patch_size_ * y * w;
      int* curr_hist = saliency_hist_;
      memset(curr_hist, 0, sizeof(int) * 257);

      for (int j = 0; j < patch_size_; j++) {
        for (int i = 0; i < patch_size_; i++) {
          int it = i + patch_size_ * x;
          int jt = j + patch_size_ * y;
          if (it > (w - 2) || jt > (h - 2) || it < 1 || jt < 1) {
            continue;
          }
          // int curr_s = static_cast<int>(curr_map[i + j * w]);
          // printf("%d, %f\n", curr_s, curr_map[i + j * w]);
          curr_hist[static_cast<int>(curr_map[i + j * w])]++;
          curr_hist[256]++;
        }
      }

      saliency_ths_[x + y * p_w] =
        SaliencyComputeHistQuantil(curr_hist, setting_minGradHistCut);// +
        // setting_minSaliencyHistAdd;

      saliency_sum += saliency_ths_[x + y * p_w];
    }
  }

  float saliency_mean = saliency_sum / (p_w * p_h);

  for (int y = 0; y < p_h; y++) {
    for (int x = 0; x < p_w; x++) {
      float sum = 0.0;
      float num = 0.0;
      if (x > 0) {
        if (y > 0) {
          num++;
          sum += saliency_ths_[x - 1 + (y - 1) * p_w];
        }
        if (y < (p_h - 1)) {
          num++;
          sum += saliency_ths_[x - 1 + (y + 1) * p_w];
        }
        num++;
        sum += saliency_ths_[x - 1 + y * p_w];
      }

      if (x < (p_w - 1)) {
        if (y > 0) {
          num++;
          sum += saliency_ths_[x + 1 + (y - 1) * p_w];
        }
        if (y < (p_h - 1)) {
          num++;
          sum += saliency_ths_[x + 1 + (y + 1) * p_w];
        }
        num++;
        sum += saliency_ths_[x + 1 + y * p_w];
      }

      if (y > 0) {
        num++;
        sum += saliency_ths_[x + (y - 1) * p_w];
      }
      if (y < (p_h - 1)) {
        num++;
        sum += saliency_ths_[x + (y + 1) * p_w];
      }
      num++;
      sum += saliency_ths_[x + y * p_w];

      if (setting_average_smoothing) {
        saliency_ths_smoothed_[x + y * p_w] = sum / num +
          (saliency_mean / setting_minSaliencyMeanWeight);
      } else if (setting_constant_smoothing) {
        saliency_ths_smoothed_[x + y * p_w] = sum / num +
          setting_minSaliencyHistAdd;
      } else {
        saliency_ths_smoothed_[x + y * p_w] = sum / num +
          (saliency_mean / setting_minSaliencyMeanWeight) +
          setting_minSaliencyHistAdd;
      }
    }
  }
}

void PixelSelector::setUpSaliencyRandomSampler() {
  int p_w = wG[0] / patch_size_;
  int p_h = hG[0] / patch_size_;
  std::vector<int> weights(p_w * p_h);
  for (int i = 0; i < (p_w * p_h); i++) {
    weights[i] = max(static_cast<int>(saliency_ths_smoothed_[i]), 1);
  }

  if (saliency_random_ != NULL) {
    delete saliency_random_;
  }

  saliency_random_ =
    new std::discrete_distribution<>(weights.begin(), weights.end());
}

int PixelSelector::getSaliencyWeightedRandomNumber() {
  return (*saliency_random_)(*gen_);
}

// Randomly choose a 32x32 block and do the same potential select inside
// this block. For each (4 * pot)x(4 * pot), ideally, we will select
// 16 points.
Eigen::Vector3i PixelSelector::saliencySelect(
    const FrameHessian* const fh,
    float* map_out,
    int pot,
    float thFactor) {
  assert(PYR_LEVELS > 2);

  Eigen::Vector3f const* const intensity_grad_map = fh->dI;
  float* grad_map_lv0 = fh->absSquaredGrad[0];
  float* grad_map_lv1 = fh->absSquaredGrad[1];
  float* grad_map_lv2 = fh->absSquaredGrad[2];
  int w = wG[0];
  int w1 = wG[1];
  int w2 = wG[2];
  int h = hG[0];

  const Vec2f directions[16] = {Vec2f(0,    1.0000),
                                Vec2f(0.3827,    0.9239),
                                Vec2f(0.1951,    0.9808),
                                Vec2f(0.9239,    0.3827),
                                Vec2f(0.7071,    0.7071),
                                Vec2f(0.3827,   -0.9239),
                                Vec2f(0.8315,    0.5556),
                                Vec2f(0.8315,   -0.5556),
                                Vec2f(0.5556,   -0.8315),
                                Vec2f(0.9808,    0.1951),
                                Vec2f(0.9239,   -0.3827),
                                Vec2f(0.7071,   -0.7071),
                                Vec2f(0.5556,    0.8315),
                                Vec2f(0.9808,   -0.1951),
                                Vec2f(1.0000,    0.0000),
                                Vec2f(0.1951,   -0.9808)};

  memset(map_out, 0, w * h * sizeof(PixelSelectorStatus));



  float dw1 = setting_gradDownweightPerLevel;
  float dw2 = dw1 * dw1;
  int p_w = w / patch_size_;
  int p_h = h / patch_size_;

  // Every 4pot x 4pot will sample 1 ~ 16 points.
  int setting_desiredImmatureDensity_int = setting_desiredImmatureDensity;
  // int num_iterations = std::min(p_w * p_h, setting_desiredImmatureDensity_int);
  int num_iterations = (w * h) / (pot * 2 * pot * 2);
  // printf("Number of iterations: %d\n", num_iterations);

  int n2 = 0;
  int n3 = 0;
  int n4 = 0;
  for (int i = 0; i < num_iterations; i++) {
    int random_block = getSaliencyWeightedRandomNumber();
    int w_start = (random_block % p_w) * patch_size_;
    int h_start = (random_block / p_w) * patch_size_;
    int w_end = std::min(w_start + patch_size_, w);
    int h_end = std::min(h_start + patch_size_, h);
    for (int y4 = h_start; y4 < h_end; y4 += (4 * pot)) {
      for (int x4 = w_start; x4 < w_end; x4 += (4 * pot)) {
        int my3 = std::min((4 * pot), h_end - y4);
        int mx3 = std::min((4 * pot), w_end - x4);
        int bestIdx4 = -1;
        float bestVal4 = 0;
        Vec2f dir4 = directions[randomPattern[n2] & 0xF];
        for (int y3 = 0; y3 < my3; y3 += (2 * pot)) {
          for (int x3 = 0; x3 < mx3; x3 += (2 * pot)) {
            int x34 = x3 + x4;
            int y34 = y3 + y4;
            int my2 = std::min((2 * pot), h_end - y34);
            int mx2 = std::min((2 * pot), w_end - x34);
            int bestIdx3 = -1;
            float bestVal3 = 0;
            Vec2f dir3 = directions[randomPattern[n2] & 0xF];
            for (int y2 = 0; y2 < my2; y2 += pot) {
              for (int x2 = 0; x2 < mx2; x2 += pot) {
                int x234 = x2 + x34;
                int y234 = y2 + y34;
                int my1 = std::min(pot, h_end - y234);
                int mx1 = std::min(pot, w_end - x234);
                int bestIdx2 = -1;
                float bestVal2 = 0;
                Vec2f dir2 = directions[randomPattern[n2] & 0xF];
                for (int y1 = 0; y1 < my1; y1 += 1) {
                  for (int x1 = 0; x1 < mx1; x1 += 1) {
                    assert(x1 + x234 < w);
                    assert(y1 + y234 < h);
                    int idx = x1 + x234 + w * (y1 + y234);
                    int xf = x1 + x234;
                    int yf = y1 + y234;

                    if (xf < 4 || xf >= (w - 5) || yf < 4 || yf > (h - 4)) {
                      continue;
                    }

                    if (map_out[idx] != 0) {
                      // already selected
                      continue;
                    }

                    float pixelTH0 =
                      thsSmoothed[xf / patch_size_ + (yf / patch_size_) * thsStep];
                    float pixelTH1 = pixelTH0 * dw1;
                    float pixelTH2 = pixelTH1 * dw2;


                    float ag0 = grad_map_lv0[idx];
                    if (ag0 > pixelTH0 * thFactor) {
                      Vec2f ag0d = intensity_grad_map[idx].tail<2>();
                      float dirNorm =
                        fabsf(static_cast<float>(ag0d.dot(dir2)));
                      if (!setting_selectDirectionDistribution) {
                        dirNorm = ag0;
                      }

                      if (dirNorm > bestVal2) {
                        bestVal2 = dirNorm;
                        bestIdx2 = idx;
                        bestIdx3 = -2;
                        bestIdx4 = -2;
                      }
                    }
                    if (bestIdx3 == -2) {
                      continue;
                    }

                    float ag1 =
                      grad_map_lv1[static_cast<int>(xf * 0.5f + 0.25f) + static_cast<int>(yf * 0.5f + 0.25f) * w1];
                    if (ag1 > pixelTH1 * thFactor) {
                      Vec2f ag0d = intensity_grad_map[idx].tail<2>();
                      float dirNorm = fabsf(static_cast<float>(ag0d.dot(dir3)));
                      if (!setting_selectDirectionDistribution) {
                        dirNorm = ag1;
                      }

                      if (dirNorm > bestVal3) {
                        bestVal3 = dirNorm;
                        bestIdx3 = idx;
                        bestIdx4 = -2;
                      }
                    }
                    if (bestIdx4 == -2) {
                      continue;
                    }

                    float ag2 =
                      grad_map_lv2[static_cast<int>(xf * 0.25f + 0.125) + static_cast<int>(yf * 0.25f + 0.125) * w2];
                    if (ag2 > pixelTH2*thFactor) {
                      Vec2f ag0d = intensity_grad_map[idx].tail<2>();
                      float dirNorm = fabsf(static_cast<float>(ag0d.dot(dir4)));
                      if (!setting_selectDirectionDistribution) {
                        dirNorm = ag2;
                      }

                      if (dirNorm > bestVal4) {
                        bestVal4 = dirNorm;
                        bestIdx4 = idx;
                      }
                    }
                  }
                }

                if (bestIdx2 > 0) {
                  map_out[bestIdx2] = 1;
                  bestVal3 = 1e10;
                  n2++;
                }
              }
            }

            if (bestIdx3 > 0) {
              map_out[bestIdx3] = 2;
              bestVal4 = 1e10;
              n3++;
            }
          }
        }

        if (bestIdx4 > 0) {
          map_out[bestIdx4] = 4;
          n4++;
        }
      }
    }
  }

  return Eigen::Vector3i(n2, n3, n4);
}


}

