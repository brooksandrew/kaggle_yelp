package modeling.exploreModel

import modeling.io.nn._
import modeling.io.readCsvData._

import modeling.processing.alignedData
import modeling.processing.images._
import modeling.processing.makeDataSets._
import modeling.processing.scoring._


object p_explore_data extends App {
  
  val path4interpreter = "/Users/abrooks/Documents/github/kaggle_yelp/"
  
  val labMap = readBizLabels(path4interpreter + "data/labels/train.csv")
  val bizMap = readBiz2ImgLabels(path4interpreter + "data/labels/train_photo_to_biz_ids.csv")
  val imgs = getImageIds(path4interpreter + "data/images/train", bizMap, bizMap.map(_._2).toSet.toList.slice(1500,1515))
  val dataMap = processImages(imgs, resizeImgDim = 64) // nPixels = 64
  val alignedData = new alignedData(dataMap, bizMap, Option(labMap))()
  
  bizMap.keys
  
  
  val dsTE = makeDataSetTE(alignedData)
  val model = loadNN(path4interpreter + "results/modelsV0/model2_img16k_epoch15_batch128_pixels64_nout100_200.json", path4interpreter + "results/modelsV0/model2_img16k_epoch15_batch128_pixels64_nout100_200.bin")
  val predsTE = scoreModel(model, dsTE)
  
  val bizScoreAgg = aggImgScores2Biz(predsTE, alignedData)
  println(bizScoreAgg)
  println(alignedData.getImgCntsPerBiz)
  println(alignedData.getBizLabels)
  
}