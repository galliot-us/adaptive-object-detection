/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <cstring>
#include <iostream>
#include "nvdsinfer_custom_impl.h"
#include <cassert>
#include <cmath>
#include <tuple>
#include <memory>

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define CLIP(a,min,max) (MAX(MIN(a, max), min))
#define DIVIDE_AND_ROUND_UP(a, b) ((a + b - 1) / b)

struct FrcnnParams {
    int inputHeight;
    int inputWidth;
    int outputClassSize;
    float visualizeThreshold;
    int postNmsTopN;
    int outputBboxSize;
    std::vector<float> classifierRegressorStd;

};

struct MrcnnRawDetection {
    float y1, x1, y2, x2, class_id, score;
};

static void frcnn_batch_inverse_transform_classifier(const float* roi_after_nms, int roi_num_per_img,
                                        const float* classifier_cls, const float* classifier_regr, std::vector<float>& pred_boxes,
                                        std::vector<int>& pred_cls_ids, std::vector<float>& pred_probs, std::vector<int>& box_num_per_img,
                                        int N, const FrcnnParams &gFrcnnParams);

static void frcnn_parse_boxes(int img_num, int class_num, std::vector<float>& pred_boxes,
                 std::vector<float>& pred_probs, std::vector<int>& pred_cls_ids, std::vector<int>& box_num_per_img,
                 const FrcnnParams &gFrcnnParams,
                 std::vector<NvDsInferObjectDetectionInfo> &objectList);

static bool frcnn_parse_output(const int batchSize,
                 const float* out_class,
                 const float* out_reg,
                 const float* out_proposal,
                 const FrcnnParams &gFrcnnParams,
                 std::vector<NvDsInferObjectDetectionInfo> &objectList);
/* This is a sample bounding box parsing function for the sample Resnet10
 * detector model provided with the SDK. */

/* C-linkage to prevent name-mangling */
extern "C"
bool NvDsInferParseCustomResnet (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
        NvDsInferNetworkInfo  const &networkInfo,
        NvDsInferParseDetectionParams const &detectionParams,
        std::vector<NvDsInferObjectDetectionInfo> &objectList);

/* This is a sample bounding box parsing function for the tensorflow SSD models
 * detector model provided with the SDK. */

/* C-linkage to prevent name-mangling */
extern "C"
bool NvDsInferParseCustomTfSSD (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
        NvDsInferNetworkInfo  const &networkInfo,
        NvDsInferParseDetectionParams const &detectionParams,
        std::vector<NvDsInferObjectDetectionInfo> &objectList);

extern "C"
bool NvDsInferParseCustomSSDTLT (
         std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
         NvDsInferNetworkInfo  const &networkInfo,
         NvDsInferParseDetectionParams const &detectionParams,
         std::vector<NvDsInferObjectDetectionInfo> &objectList);

extern "C"
bool NvDsInferParseCustomSSDNeuralet (
         std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
         NvDsInferNetworkInfo  const &networkInfo,
         NvDsInferParseDetectionParams const &detectionParams,
         std::vector<NvDsInferObjectDetectionInfo> &objectList);

extern "C"
bool NvDsInferParseCustomYOLOV3TLT (
         std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
         NvDsInferNetworkInfo  const &networkInfo,
         NvDsInferParseDetectionParams const &detectionParams,
         std::vector<NvDsInferObjectDetectionInfo> &objectList);


/* This is a sample bounding box parsing function for the sample faster RCNN
 *
 * detector model provided with the SDK. */

/* C-linkage to prevent name-mangling */
extern "C"
bool NvDsInferParseCustomFrcnnTLT (
         std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
         NvDsInferNetworkInfo  const &networkInfo,
         NvDsInferParseDetectionParams const &detectionParams,
         std::vector<NvDsInferObjectDetectionInfo> &objectList);

extern "C"
bool NvDsInferParseCustomMrcnnTLT (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                   NvDsInferNetworkInfo  const &networkInfo,
                                   NvDsInferParseDetectionParams const &detectionParams,
                                   std::vector<NvDsInferInstanceMaskInfo> &objectList);

extern "C"
bool NvDsInferParseCustomResnet (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
        NvDsInferNetworkInfo  const &networkInfo,
        NvDsInferParseDetectionParams const &detectionParams,
        std::vector<NvDsInferObjectDetectionInfo> &objectList)
{
  static NvDsInferDimsCHW covLayerDims;
  static NvDsInferDimsCHW bboxLayerDims;
  static int bboxLayerIndex = -1;
  static int covLayerIndex = -1;
  static bool classMismatchWarn = false;
  int numClassesToParse;

  /* Find the bbox layer */
  if (bboxLayerIndex == -1) {
    for (unsigned int i = 0; i < outputLayersInfo.size(); i++) {
      if (strcmp(outputLayersInfo[i].layerName, "conv2d_bbox") == 0) {
        bboxLayerIndex = i;
        getDimsCHWFromDims(bboxLayerDims, outputLayersInfo[i].inferDims);
        break;
      }
    }
    if (bboxLayerIndex == -1) {
    std::cerr << "Could not find bbox layer buffer while parsing" << std::endl;
    return false;
    }
  }

  /* Find the cov layer */
  if (covLayerIndex == -1) {
    for (unsigned int i = 0; i < outputLayersInfo.size(); i++) {
      if (strcmp(outputLayersInfo[i].layerName, "conv2d_cov/Sigmoid") == 0) {
        covLayerIndex = i;
        getDimsCHWFromDims(covLayerDims, outputLayersInfo[i].inferDims);
        break;
      }
    }
    if (covLayerIndex == -1) {
    std::cerr << "Could not find bbox layer buffer while parsing" << std::endl;
    return false;
    }
  }

  /* Warn in case of mismatch in number of classes */
  if (!classMismatchWarn) {
    if (covLayerDims.c != detectionParams.numClassesConfigured) {
      std::cerr << "WARNING: Num classes mismatch. Configured:" <<
        detectionParams.numClassesConfigured << ", detected by network: " <<
        covLayerDims.c << std::endl;
    }
    classMismatchWarn = true;
  }

  /* Calculate the number of classes to parse */
  numClassesToParse = MIN (covLayerDims.c, detectionParams.numClassesConfigured);

  int gridW = covLayerDims.w;
  int gridH = covLayerDims.h;
  int gridSize = gridW * gridH;
  float gcCentersX[gridW];
  float gcCentersY[gridH];
  float bboxNormX = 35.0;
  float bboxNormY = 35.0;
  float *outputCovBuf = (float *) outputLayersInfo[covLayerIndex].buffer;
  float *outputBboxBuf = (float *) outputLayersInfo[bboxLayerIndex].buffer;
  int strideX = DIVIDE_AND_ROUND_UP(networkInfo.width, bboxLayerDims.w);
  int strideY = DIVIDE_AND_ROUND_UP(networkInfo.height, bboxLayerDims.h);

  for (int i = 0; i < gridW; i++)
  {
    gcCentersX[i] = (float)(i * strideX + 0.5);
    gcCentersX[i] /= (float)bboxNormX;

  }
  for (int i = 0; i < gridH; i++)
  {
    gcCentersY[i] = (float)(i * strideY + 0.5);
    gcCentersY[i] /= (float)bboxNormY;

  }

  for (int c = 0; c < numClassesToParse; c++)
  {
    float *outputX1 = outputBboxBuf + (c * 4 * bboxLayerDims.h * bboxLayerDims.w);

    float *outputY1 = outputX1 + gridSize;
    float *outputX2 = outputY1 + gridSize;
    float *outputY2 = outputX2 + gridSize;

    float threshold = detectionParams.perClassPreclusterThreshold[c];
    for (int h = 0; h < gridH; h++)
    {
      for (int w = 0; w < gridW; w++)
      {
        int i = w + h * gridW;
        if (outputCovBuf[c * gridSize + i] >= threshold)
        {
          NvDsInferObjectDetectionInfo object;
          float rectX1f, rectY1f, rectX2f, rectY2f;

          rectX1f = (outputX1[w + h * gridW] - gcCentersX[w]) * -bboxNormX;
          rectY1f = (outputY1[w + h * gridW] - gcCentersY[h]) * -bboxNormY;
          rectX2f = (outputX2[w + h * gridW] + gcCentersX[w]) * bboxNormX;
          rectY2f = (outputY2[w + h * gridW] + gcCentersY[h]) * bboxNormY;

          object.classId = c;
          object.detectionConfidence = outputCovBuf[c * gridSize + i];

          /* Clip object box co-ordinates to network resolution */
          object.left = CLIP(rectX1f, 0, networkInfo.width - 1);
          object.top = CLIP(rectY1f, 0, networkInfo.height - 1);
          object.width = CLIP(rectX2f, 0, networkInfo.width - 1) -
                             object.left + 1;
          object.height = CLIP(rectY2f, 0, networkInfo.height - 1) -
                             object.top + 1;

          objectList.push_back(object);
        }
      }
    }
  }
  return true;
}

extern "C"
bool NvDsInferParseCustomTfSSD (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo  const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferObjectDetectionInfo> &objectList)
{
    auto layerFinder = [&outputLayersInfo](const std::string &name)
        -> const NvDsInferLayerInfo *{
        for (auto &layer : outputLayersInfo) {
            if (layer.dataType == FLOAT &&
              (layer.layerName && name == layer.layerName)) {
                return &layer;
            }
        }
        return nullptr;
    };

    const NvDsInferLayerInfo *numDetectionLayer = layerFinder("num_detections");
    const NvDsInferLayerInfo *scoreLayer = layerFinder("detection_scores");
    const NvDsInferLayerInfo *classLayer = layerFinder("detection_classes");
    const NvDsInferLayerInfo *boxLayer = layerFinder("detection_boxes");
    if (!scoreLayer || !classLayer || !boxLayer) {
        std::cerr << "ERROR: some layers missing or unsupported data types "
                  << "in output tensors" << std::endl;
        return false;
    }

    unsigned int numDetections = classLayer->inferDims.d[0];
    if (numDetectionLayer && numDetectionLayer->buffer) {
        numDetections = (int)((float*)numDetectionLayer->buffer)[0];
    }
    if (numDetections > classLayer->inferDims.d[0]) {
        numDetections = classLayer->inferDims.d[0];
    }
    numDetections = std::max<int>(0, numDetections);
    for (unsigned int i = 0; i < numDetections; ++i) {
        NvDsInferObjectDetectionInfo res;
        res.detectionConfidence = ((float*)scoreLayer->buffer)[i];
        res.classId = ((float*)classLayer->buffer)[i];
        if (res.classId >= detectionParams.perClassPreclusterThreshold.size() ||
            res.detectionConfidence <
            detectionParams.perClassPreclusterThreshold[res.classId]) {
            continue;
        }
        enum {y1, x1, y2, x2};
        float rectX1f, rectY1f, rectX2f, rectY2f;
        rectX1f = ((float*)boxLayer->buffer)[i *4 + x1] * networkInfo.width;
        rectY1f = ((float*)boxLayer->buffer)[i *4 + y1] * networkInfo.height;
        rectX2f = ((float*)boxLayer->buffer)[i *4 + x2] * networkInfo.width;;
        rectY2f = ((float*)boxLayer->buffer)[i *4 + y2] * networkInfo.height;
        rectX1f = CLIP(rectX1f, 0.0f, networkInfo.width - 1);
        rectX2f = CLIP(rectX2f, 0.0f, networkInfo.width - 1);
        rectY1f = CLIP(rectY1f, 0.0f, networkInfo.height - 1);
        rectY2f = CLIP(rectY2f, 0.0f, networkInfo.height - 1);
        if (rectX2f <= rectX1f || rectY2f <= rectY1f) {
            continue;
        }
        res.left = rectX1f;
        res.top = rectY1f;
        res.width = rectX2f - rectX1f;
        res.height = rectY2f - rectY1f;
        if (res.width && res.height) {
            objectList.emplace_back(res);
        }
    }

    return true;
}

extern "C"
bool NvDsInferParseCustomSSDTLT (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                   NvDsInferNetworkInfo  const &networkInfo,
                                   NvDsInferParseDetectionParams const &detectionParams,
                                   std::vector<NvDsInferObjectDetectionInfo> &objectList) {
    static int nmsIndex = -1;

    /* Find the nms layer */
    if (nmsIndex == -1) {
        for (unsigned int i = 0; i < outputLayersInfo.size(); i++) {
            if (strcmp(outputLayersInfo[i].layerName, "NMS") == 0) {
                nmsIndex = i;
                break;
            }
        }
        if (nmsIndex == -1) {
            std::cerr << "Could not find nms layer buffer while parsing" << std::endl;
            return false;
        }
    }

    // Host memory for "nms"
    float* out_nms = (float *) outputLayersInfo[nmsIndex].buffer;

    const int batch_id = 0;
    const float threshold = detectionParams.perClassThreshold[0];

    // Set your keep_count / keep_top here
    const int keep_count = 200;
    const int keep_top_k = 200;

    float* det;

    for (int i = 0; i < keep_count; i++) {
        det = out_nms + batch_id * keep_top_k * 7 + i * 7;

        // Output format for each detection is stored in the below order
        // [image_id, label, confidence, xmin, ymin, xmax, ymax]
        if ( det[2] < threshold) continue;
        assert((unsigned int) det[1] <  detectionParams.numClassesConfigured);

        NvDsInferObjectDetectionInfo object;
        object.classId = (int) det[1];
        object.detectionConfidence = det[2];

        /* Clip object box co-ordinates to network resolution */
        object.left = CLIP(det[3] * networkInfo.width, 0, networkInfo.width - 1);
        object.top = CLIP(det[4] * networkInfo.height, 0, networkInfo.height - 1);
        object.width = CLIP((det[5] - det[3]) * networkInfo.width, 0, networkInfo.width - 1);
        object.height = CLIP((det[6] - det[4]) * networkInfo.height, 0, networkInfo.height - 1);

        objectList.push_back(object);
    }

    return true;
}

extern "C"
bool NvDsInferParseCustomSSDNeuralet (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                   NvDsInferNetworkInfo  const &networkInfo,
                                   NvDsInferParseDetectionParams const &detectionParams,
                                   std::vector<NvDsInferObjectDetectionInfo> &objectList) {
  static int nmsLayerIndex = -1;
  static int nms1LayerIndex = -1;
  static bool classMismatchWarn = false;
  int numClassesToParse;
  static const int NUM_CLASSES_SSD = 91;

  if (nmsLayerIndex == -1) {
    for (unsigned int i = 0; i < outputLayersInfo.size(); i++) {
      if (strcmp(outputLayersInfo[i].layerName, "NMS") == 0) {
        nmsLayerIndex = i;
        break;
      }
    }
    if (nmsLayerIndex == -1) {
    std::cerr << "Could not find NMS layer buffer while parsing" << std::endl;
    return false;
    }
  }

  if (nms1LayerIndex == -1) {
    for (unsigned int i = 0; i < outputLayersInfo.size(); i++) {
      if (strcmp(outputLayersInfo[i].layerName, "NMS_1") == 0) {
        nms1LayerIndex = i;
        break;
      }
    }
    if (nms1LayerIndex == -1) {
    std::cerr << "Could not find NMS_1 layer buffer while parsing" << std::endl;
    return false;
    }
  }

  if (!classMismatchWarn) {
    if (NUM_CLASSES_SSD !=
        detectionParams.numClassesConfigured) {
      std::cerr << "WARNING: Num classes mismatch. Configured:" <<
        detectionParams.numClassesConfigured << ", detected by network: " <<
        NUM_CLASSES_SSD << std::endl;
    }
    classMismatchWarn = true;
  }

  numClassesToParse = MIN (NUM_CLASSES_SSD,
      detectionParams.numClassesConfigured);

  int keepCount = *((int *) outputLayersInfo[nms1LayerIndex].buffer);
  float *detectionOut = (float *) outputLayersInfo[nmsLayerIndex].buffer;

  for (int i = 0; i < keepCount; ++i)
  {
    float* det = detectionOut + i * 7;
    int classId = det[1];

    if (classId >= numClassesToParse)
      continue;

    float threshold = detectionParams.perClassThreshold[classId];

    if (det[2] < threshold)
      continue;

    unsigned int rectx1, recty1, rectx2, recty2;
    NvDsInferObjectDetectionInfo object;

    rectx1 = det[3] * networkInfo.width;
    recty1 = det[4] * networkInfo.height;
    rectx2 = det[5] * networkInfo.width;
    recty2 = det[6] * networkInfo.height;

    object.classId = classId;
    object.detectionConfidence = det[2];

    /* Clip object box co-ordinates to network resolution */
    object.left = CLIP(rectx1, 0, networkInfo.width - 1);
    object.top = CLIP(recty1, 0, networkInfo.height - 1);
    object.width = CLIP(rectx2, 0, networkInfo.width - 1) -
      object.left + 1;
    object.height = CLIP(recty2, 0, networkInfo.height - 1) -
      object.top + 1;

    objectList.push_back(object);
  }

  return true;
}
extern "C"
bool NvDsInferParseCustomYOLOV3TLT (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                   NvDsInferNetworkInfo  const &networkInfo,
                                   NvDsInferParseDetectionParams const &detectionParams,
                                   std::vector<NvDsInferObjectDetectionInfo> &objectList) {
    if(outputLayersInfo.size() != 4)
    {
        std::cerr << "Mismatch in the number of output buffers."
                  << "Expected 4 output buffers, detected in the network :"
                  << outputLayersInfo.size() << std::endl;
        return false;
    }

    /* Host memory for "BatchedNMS"
       BatchedNMS has 4 output bindings, the order is:
       keepCount, bboxes, scores, classes
    */
    int* p_keep_count = (int *) outputLayersInfo[0].buffer;
    float* p_bboxes = (float *) outputLayersInfo[1].buffer;
    float* p_scores = (float *) outputLayersInfo[2].buffer;
    float* p_classes = (float *) outputLayersInfo[3].buffer;

    const float threshold = detectionParams.perClassThreshold[0];

    const int keep_top_k = 200;
    const char* log_enable = std::getenv("ENABLE_DEBUG");

    if(log_enable != NULL && std::stoi(log_enable)) {
        std::cout <<"keep cout"
              <<p_keep_count[0] << std::endl;
    }

    for (int i = 0; i < p_keep_count[0] && objectList.size() <= keep_top_k; i++) {

        if ( p_scores[i] < threshold) continue;
        assert((unsigned int) p_classes[i] < detectionParams.numClassesConfigured);

        if(log_enable != NULL && std::stoi(log_enable)) {
            std::cout << "label/conf/ x/y x/y -- "
                      << p_classes[i] << " " << p_scores[i] << " "
                      << p_bboxes[4*i] << " " << p_bboxes[4*i+1] << " " << p_bboxes[4*i+2] << " "<< p_bboxes[4*i+3] << " " << std::endl;
        }
        if(p_bboxes[4*i+2] < p_bboxes[4*i] || p_bboxes[4*i+3] < p_bboxes[4*i+1]) continue;

        NvDsInferObjectDetectionInfo object;
        object.classId = (int) p_classes[i];
        object.detectionConfidence = p_scores[i];

        /* Clip object box co-ordinates to network resolution */
        object.left = CLIP(p_bboxes[4*i], 0, networkInfo.width - 1);
        object.top = CLIP(p_bboxes[4*i+1], 0, networkInfo.height - 1);
        object.width = CLIP((p_bboxes[4*i+2] - p_bboxes[4*i]), 0, networkInfo.width - 1);
        object.height = CLIP((p_bboxes[4*i+3] - p_bboxes[4*i+1]), 0, networkInfo.height - 1);

        objectList.push_back(object);
    }
    return true;
}

static void frcnn_batch_inverse_transform_classifier(
         const float* roi_after_nms, int roi_num_per_img,
         const float* classifier_cls, const float* classifier_regr,
         std::vector<float>& pred_boxes, std::vector<int>& pred_cls_ids,
         std::vector<float>& pred_probs, std::vector<int>& box_num_per_img,
         int N, const FrcnnParams &gFrcnnParams) {
    auto max_index = [](const float* start, const float* end) -> int {
        float max_val = start[0];
        int max_pos = 0;

        for (int i = 1; start + i < end; ++i) {
            if (start[i] > max_val) {
                max_val = start[i];
                max_pos = i;
            }
        }

        return max_pos;
    };
    int box_num;

    for (int n = 0; n < N; ++n) {
        box_num = 0;

        for (int i = 0; i < roi_num_per_img; ++i) {
            auto max_idx = max_index(
                     classifier_cls + n * roi_num_per_img * gFrcnnParams.outputClassSize + i * gFrcnnParams.outputClassSize,
                     classifier_cls + n * roi_num_per_img * gFrcnnParams.outputClassSize + i * gFrcnnParams.outputClassSize +
                     gFrcnnParams.outputClassSize);

            if (max_idx == (gFrcnnParams.outputClassSize - 1) ||
                classifier_cls[n * roi_num_per_img * gFrcnnParams.outputClassSize + max_idx + i * gFrcnnParams.outputClassSize] <
                gFrcnnParams.visualizeThreshold) {
                continue;
            }

            // Inverse transform
            float tx, ty, tw, th;
            //(i, 20, 4)
            tx = classifier_regr[n * roi_num_per_img * gFrcnnParams.outputBboxSize + i * gFrcnnParams.outputBboxSize + max_idx * 4]
                     / gFrcnnParams.classifierRegressorStd[0];
            ty = classifier_regr[n * roi_num_per_img * gFrcnnParams.outputBboxSize + i * gFrcnnParams.outputBboxSize + max_idx * 4 + 1]
                     / gFrcnnParams.classifierRegressorStd[1];
            tw = classifier_regr[n * roi_num_per_img * gFrcnnParams.outputBboxSize + i * gFrcnnParams.outputBboxSize + max_idx * 4 + 2]
                     / gFrcnnParams.classifierRegressorStd[2];
            th = classifier_regr[n * roi_num_per_img * gFrcnnParams.outputBboxSize + i * gFrcnnParams.outputBboxSize + max_idx * 4 + 3]
                     / gFrcnnParams.classifierRegressorStd[3];
            float y = roi_after_nms[n * roi_num_per_img * 4 + 4 * i] * static_cast<float>(gFrcnnParams.inputHeight - 1.0f);
            float x = roi_after_nms[n * roi_num_per_img * 4 + 4 * i + 1] * static_cast<float>(gFrcnnParams.inputWidth - 1.0f);
            float ymax = roi_after_nms[n * roi_num_per_img * 4 + 4 * i + 2] * static_cast<float>(gFrcnnParams.inputHeight - 1.0f);
            float xmax = roi_after_nms[n * roi_num_per_img * 4 + 4 * i + 3] * static_cast<float>(gFrcnnParams.inputWidth - 1.0f);
            float w = xmax - x;
            float h = ymax - y;
            float cx = x + w / 2.0f;
            float cy = y + h / 2.0f;
            float cx1 = tx * w + cx;
            float cy1 = ty * h + cy;
            float w1 = std::exp(static_cast<double>(tw)) * w;
            float h1 = std::exp(static_cast<double>(th)) * h;
            float x1 = cx1 - w1 / 2.0f;
            float y1 = cy1 - h1 / 2.0f;
            auto clip
                = [](float in, float low, float high) -> float { return (in < low) ? low : (in > high ? high : in); };
            float x2 = x1 + w1;
            float y2 = y1 + h1;
            x1 = clip(x1, 0.0f, gFrcnnParams.inputWidth - 1.0f);
            y1 = clip(y1, 0.0f, gFrcnnParams.inputHeight - 1.0f);
            x2 = clip(x2, 0.0f, gFrcnnParams.inputWidth - 1.0f);
            y2 = clip(y2, 0.0f, gFrcnnParams.inputHeight - 1.0f);

            if (x2 > x1 && y2 > y1) {
                pred_boxes.push_back(x1);
                pred_boxes.push_back(y1);
                pred_boxes.push_back(x2);
                pred_boxes.push_back(y2);
                pred_probs.push_back(classifier_cls[n * roi_num_per_img * gFrcnnParams.outputClassSize +
                                                    max_idx + i * gFrcnnParams.outputClassSize]);
                pred_cls_ids.push_back(max_idx);
                ++box_num;
            }
        }

        box_num_per_img.push_back(box_num);
    }
}

static void frcnn_parse_boxes(int img_num, int class_num,
         std::vector<float>& pred_boxes, std::vector<float>& pred_probs,
         std::vector<int>& pred_cls_ids, std::vector<int>& box_num_per_img,
         const FrcnnParams& gFrcnnParams,
         std::vector<NvDsInferObjectDetectionInfo> &objectList) {
    int box_start_idx = 0;

    for (int i = 0; i < img_num; ++i) {
        for (int c = 0; c < (class_num - 1); ++c) {
            // skip the background
            for (int k = box_start_idx; k < box_start_idx + box_num_per_img[i]; ++k) {
                if (pred_cls_ids[k] == c) {
                    NvDsInferObjectDetectionInfo obj{static_cast<unsigned int>(c),
                     CLIP(pred_boxes[4*k], 0, gFrcnnParams.inputWidth - 1 ),
                     CLIP(pred_boxes[4*k+1], 0, gFrcnnParams.inputHeight - 1),
                     CLIP(pred_boxes[4*k+2] - pred_boxes[4*k], 0, gFrcnnParams.inputWidth - 1),
                     CLIP(pred_boxes[4*k+3] - pred_boxes[4*k+1], 0, gFrcnnParams.inputHeight - 1),
                     pred_probs[k]};
                   objectList.push_back(obj);
                }
            }
        }

        box_start_idx += box_num_per_img[i];
    }
}

static bool frcnn_parse_output(const int batchSize,
                 const float* out_class,
                 const float* out_reg,
                 const float* out_proposal,
                 const FrcnnParams& gFrcnnParams,
                 std::vector<NvDsInferObjectDetectionInfo> &objectList) {
    const int outputClassSize = gFrcnnParams.outputClassSize;
    std::vector<float> classifierRegressorStd;
    std::vector<float> pred_boxes;
    std::vector<int> pred_cls_ids;
    std::vector<float> pred_probs;
    std::vector<int> box_num_per_img;
    objectList.clear();

    int post_nms_top_n = gFrcnnParams.postNmsTopN;

    // Post processing for stage 2.
    frcnn_batch_inverse_transform_classifier(out_proposal, post_nms_top_n, out_class, out_reg, pred_boxes, pred_cls_ids,
                                       pred_probs, box_num_per_img, batchSize, gFrcnnParams);
    frcnn_parse_boxes(batchSize, outputClassSize, pred_boxes, pred_probs, pred_cls_ids, box_num_per_img, gFrcnnParams, objectList);
    return true;
}


extern "C"
bool NvDsInferParseCustomFrcnnTLT (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                   NvDsInferNetworkInfo  const &networkInfo,
                                   NvDsInferParseDetectionParams const &detectionParams,
                                   std::vector<NvDsInferObjectDetectionInfo> &objectList) {
    static NvDsInferDimsCHW covLayerDims;

    static int proposalIndex = -1;
    static int bboxLayerIndex = -1;
    static int covLayerIndex = -1;

    static bool classMismatchWarn = false;

    FrcnnParams gFrcnnParams;


    /* Find the proposal layer */
    if (proposalIndex == -1) {
        for (unsigned int i = 0; i < outputLayersInfo.size(); i++) {
            if (strcmp(outputLayersInfo[i].layerName, "proposal") == 0) {
                proposalIndex = i;
                break;
            }
        }
        if (proposalIndex == -1) {
            std::cerr << "Could not find proposal layer buffer while parsing" << std::endl;
            return false;
        }
    }

    /* Find the bbox layer */
    if (bboxLayerIndex == -1) {
        for (unsigned int i = 0; i < outputLayersInfo.size(); i++) {
            if (strcmp(outputLayersInfo[i].layerName, "dense_regress_td/BiasAdd") == 0) {
                bboxLayerIndex = i;
                break;
            }
        }
        if (bboxLayerIndex == -1) {
            std::cerr << "Could not find bbox layer buffer while parsing" << std::endl;
            return false;
        }
    }

    /* Find the cov layer */
    if (covLayerIndex == -1) {
        for (unsigned int i = 0; i < outputLayersInfo.size(); i++) {
            if (strcmp(outputLayersInfo[i].layerName, "dense_class_td/Softmax") == 0) {
                covLayerIndex = i;
                getDimsCHWFromDims(covLayerDims, outputLayersInfo[i].inferDims);
                break;
            }
        }
        if (covLayerIndex == -1) {
            std::cerr << "Could not find cov layer buffer while parsing" << std::endl;
            return false;
        }
    }

    /* Warn in case of mismatch in number of classes */
    if (!classMismatchWarn) {
        if (covLayerDims.h != detectionParams.numClassesConfigured) {
            std::cerr << "WARNING: Num classes mismatch. Configured:" <<
                      detectionParams.numClassesConfigured << ", detected by network: " <<
                      covLayerDims.c << " " << covLayerDims.h << " " << covLayerDims.w << std::endl;
        }
        classMismatchWarn = true;
    }

    gFrcnnParams.inputHeight = networkInfo.height;
    gFrcnnParams.inputWidth = networkInfo.width;
    gFrcnnParams.visualizeThreshold = detectionParams.perClassThreshold[0];
    gFrcnnParams.classifierRegressorStd.push_back(10.0f);
    gFrcnnParams.classifierRegressorStd.push_back(10.0f);
    gFrcnnParams.classifierRegressorStd.push_back(5.0f);
    gFrcnnParams.classifierRegressorStd.push_back(5.0f);
    gFrcnnParams.outputClassSize = detectionParams.numClassesConfigured;
    gFrcnnParams.outputBboxSize = (gFrcnnParams.outputClassSize - 1) * 4;
    gFrcnnParams.postNmsTopN = 300;

    // Host memory for "proposal"
    const float* out_proposal = (float *) outputLayersInfo[proposalIndex].buffer;

    // Host memory for "dense_class_4/Softmax"
    const float* out_class = (float *) outputLayersInfo[covLayerIndex].buffer;

    // Host memory for "dense_regress_4/BiasAdd"
    const float* out_reg = (float *) outputLayersInfo[bboxLayerIndex].buffer;

    const int batch_size = 1;

    frcnn_parse_output(batch_size, out_class, out_reg, out_proposal, gFrcnnParams, objectList);

    return true;
}

extern "C"
bool NvDsInferParseCustomMrcnnTLT (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                   NvDsInferNetworkInfo  const &networkInfo,
                                   NvDsInferParseDetectionParams const &detectionParams,
                                   std::vector<NvDsInferInstanceMaskInfo> &objectList) {
    auto layerFinder = [&outputLayersInfo](const std::string &name)
        -> const NvDsInferLayerInfo *{
        for (auto &layer : outputLayersInfo) {
            if (layer.dataType == FLOAT &&
              (layer.layerName && name == layer.layerName)) {
                return &layer;
            }
        }
        return nullptr;
    };

    const NvDsInferLayerInfo *detectionLayer = layerFinder("generate_detections");
    const NvDsInferLayerInfo *maskLayer = layerFinder("mask_head/mask_fcn_logits/BiasAdd");

    if (!detectionLayer || !maskLayer) {
        std::cerr << "ERROR: some layers missing or unsupported data types "
                  << "in output tensors" << std::endl;
        return false;
    }

    if(maskLayer->inferDims.numDims != 4U) {
        std::cerr << "Network output number of dims is : " <<
            maskLayer->inferDims.numDims << " expect is 4"<< std::endl;
        return false;
    }

    const unsigned int det_max_instances = maskLayer->inferDims.d[0];
    const unsigned int num_classes = maskLayer->inferDims.d[1];
    if(num_classes != detectionParams.numClassesConfigured) {
        std::cerr << "WARNING: Num classes mismatch. Configured:" <<
            detectionParams.numClassesConfigured << ", detected by network: " <<
            num_classes << std::endl;
    }
    const unsigned int mask_instance_height= maskLayer->inferDims.d[2];
    const unsigned int mask_instance_width = maskLayer->inferDims.d[3];

    auto out_det = reinterpret_cast<MrcnnRawDetection*>( detectionLayer->buffer);
    auto out_mask = reinterpret_cast<float(*)[mask_instance_width *
        mask_instance_height]>(maskLayer->buffer);

    for(auto i = 0U; i < det_max_instances; i++) {
        MrcnnRawDetection &rawDec = out_det[i];

        if(rawDec.score < detectionParams.perClassPreclusterThreshold[0])
            continue;

        NvDsInferInstanceMaskInfo obj;
        obj.left = CLIP(rawDec.x1, 0, networkInfo.width - 1);
        obj.top = CLIP(rawDec.y1, 0, networkInfo.height - 1);
        obj.width = CLIP(rawDec.x2, 0, networkInfo.width - 1) - rawDec.x1;
        obj.height = CLIP(rawDec.y2, 0, networkInfo.height - 1) - rawDec.y1;
        if(obj.width <= 0 || obj.height <= 0)
            continue;
        obj.classId = static_cast<int>(rawDec.class_id);
        obj.detectionConfidence = rawDec.score;

        obj.mask_size = sizeof(float)*mask_instance_width*mask_instance_height;
        obj.mask = new float[mask_instance_width*mask_instance_height];
        obj.mask_width = mask_instance_width;
        obj.mask_height = mask_instance_height;

        float *rawMask = reinterpret_cast<float *>(out_mask + i
                         * detectionParams.numClassesConfigured + obj.classId);
        memcpy (obj.mask, rawMask, sizeof(float)*mask_instance_width*mask_instance_height);

        objectList.push_back(obj);
    }

    return true;

}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomFrcnnTLT);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYOLOV3TLT);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomSSDTLT);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomSSDNeuralet);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomResnet);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomTfSSD);
CHECK_CUSTOM_INSTANCE_MASK_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomMrcnnTLT);
