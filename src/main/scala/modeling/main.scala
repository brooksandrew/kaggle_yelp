package modeling

import io.nn._
import io.kaggleSubmission._ 
import io.readCsvData._

import processing.alignedData
import processing.images._
import processing.kaggleSubmission._
import processing.makeDataSets._
import processing.scoring._

import training.cnn._

object runPipeline {
  def main(args: Array[String]): Unit = {
    
  // image processing on training data
  val labMap = readBizLabels("data/labels/train.csv")
  val bizMap = readBiz2ImgLabels("data/labels/train_photo_to_biz_ids.csv")
  val imgs = getImageIds("data/images/train", bizMap, bizMap.map(_._2).toSet.toList).slice(0,1500)
  val dataMap = processImages(imgs, resizeImgDim = 128) // nPixels = 64
  val alignedData = new alignedData(dataMap, bizMap, Option(labMap))()
  
  // training (one model/class at a time). Many microparameters hardcoded within
  val cnn0 = trainModel(alignedData, bizClass = 0, saveNN = "results/modelsV1/model0_iter200_img1500_6layer") 
  val cnn1 = trainModel(alignedData, bizClass = 1, saveNN = "results/modelsV0/model1_iter200_img1500_6layer") 
  val cnn2 = trainModel(alignedData, bizClass = 2, saveNN = "results/modelsV0/model2_iter200_img1500_6layer") 
  val cnn3 = trainModel(alignedData, bizClass = 3, saveNN = "results/modelsV0/model3_iter200_img1500_6layer") 
  val cnn4 = trainModel(alignedData, bizClass = 4, saveNN = "results/modelsV0/model4_iter200_img1500_6layer") 
  val cnn5 = trainModel(alignedData, bizClass = 5, saveNN = "results/modelsV0/model5_iter200_img1500_6layer") 
  val cnn6 = trainModel(alignedData, bizClass = 6, saveNN = "results/modelsV0/model6_iter200_img1500_6layer") 
  val cnn7 = trainModel(alignedData, bizClass = 7, saveNN = "results/modelsV0/model7_iter200_img1500_6layer") 
  val cnn8 = trainModel(alignedData, bizClass = 8, saveNN = "results/modelsV0/model8_iter200_img1500_6layer") 

//  // processing test data for scoring
  val bizMapTE = readBiz2ImgLabels("data/labels/test_photo_to_biz.csv")
  val imgsTE = getImageIds("data/images/test/", bizMapTE, bizMapTE.map(_._2).toSet.slice(0,12).toList).slice(0, 100)
  val dataMapTE = processImages(imgsTE, resizeImgDim = 128)
  val alignedDataTE = new alignedData(dataMapTE, bizMapTE, None)()
  
  // creating csv file to submit to kaggle (scores all models)
  val kaggleResults = createKaggleSubmitObj(alignedDataTE, "results/ModelsV0/")
  val kaggleSubmitResults = writeKaggleSubmissionFile("results/kaggleSubmission/kaggleSubmit_3_24_2016_trash.csv", kaggleResults, thresh = 0.5)
  
  // example of how to score just model
  val dsTE = makeDataSetTE(alignedDataTE)
  val model = loadNN("results/modelsV0/model1.json", "results/modelsV0/model1.bin")
  val predsTE = scoreModel(model, dsTE)
  val bizScoreAgg = aggImgScores2Biz(predsTE, alignedDataTE)
  println(bizScoreAgg)
  
  }
}