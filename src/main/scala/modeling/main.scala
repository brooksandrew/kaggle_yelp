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
    
  // image processing
  val labMap = readBizLabels("/Users/abrooks/Box Sync/Andrew Brooks/kaggle_yelp/data/labels/train.csv")
  val bizMap = readBiz2ImgLabels("/Users/abrooks/Documents/kaggle_yelp_photo/train_photo_to_biz_ids.csv")
  val imgs = getImageIds("/Users/abrooks/Documents/kaggle_yelp_photo/train_photos/", bizMap, bizMap.map(_._2).toSet.slice(0,1000).toList).slice(0,10)
  val dataMap = processImages(imgs, resizeImgDim = 128) // nPixels = 64
  val alignedData = new alignedData(dataMap, bizMap, labMap)()
  
  // training (one model/class)
  println("processing DONE")
  val cnn = trainModel(alignedData, bizClass = 2, saveNN = "results/modelsV1/model2_iter3_img1500_trash") // many microparameters hardcoded within 
  
//  // scoring (one model/class)
  val bizMapTE = readBiz2ImgLabels("/Users/abrooks/Documents/kaggle_yelp_photo/test_photo_to_biz.csv")
  val imgsTE = getImageIds("/Users/abrooks/Documents/kaggle_yelp_photo/test_photos/", bizMapTE, bizMapTE.map(_._2).toSet.slice(0,12).toList)
  val dataMapTE = processImages(imgsTE, resizeImgDim = 128, nPixels = 64)
  val alignedDataTE = new alignedData(dataMapTE, bizMapTE)(List(2,3))
  val dsTE = makeDataSetTE(alignedDataTE)
  val model = loadNN("/Users/abrooks/Box Sync/Andrew Brooks/kaggle_yelp/data/results/coefficients1.json", "/Users/abrooks/Box Sync/Andrew Brooks/kaggle_yelp/data/results/coefficients1.bin")
  val predsTE = scoreModel(model, dsTE)
  val bizScoreAgg = aggImgScores2Biz(predsTE, alignedDataTE)
  
  // creating csv file to submit to kaggle
  val kaggleResults = createKaggleSubmitObj(alignedDataTE, "/Users/abrooks/Box Sync/Andrew Brooks/kaggle_yelp/results/")
  val kaggleSubmitResults = writeKaggleSubmissionFile("results/kaggleSubmission/kaggleSubmit.csv", kaggleResults, thresh = 0.5)
  
   println(kaggleSubmitResults)
   println(kaggleResults)
  
  }
}