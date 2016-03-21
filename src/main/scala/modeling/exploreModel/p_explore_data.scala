package modeling.exploreModel

import modeling.io.nn._


object p_explore_data {
  
  
  
  def getClassCounts(alignedData: List[(Int, String, Vector[Int], Set[Int])]): Vector[(Int, Int)] = {
    (0 to 8).map(x => (x, alignedData.map(_._4.contains(x)).count(_ == true))).toVector
  }
  
  
  
  
  val model = loadNN("/Users/abrooks/Box Sync/Andrew Brooks/kaggle_yelp/results/modelsV1/model0_.json", "/Users/abrooks/Box Sync/Andrew Brooks/kaggle_yelp/results/coefficients1.bin")
  //val predsTE = scoreModel(model, dsTE)
  
  
  
  
  
  
}