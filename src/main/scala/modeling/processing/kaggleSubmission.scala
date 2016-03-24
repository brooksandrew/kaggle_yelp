package modeling.processing

import modeling.io.nn._
import modeling.processing.scoring._
import modeling.processing.makeDataSets.makeDataSetTE

object kaggleSubmission {
  
   /** Create csv to submit to kaggle with predicted classes from all 9 categories for each business in the test image set */
  
  def createKaggleSubmitObj(alignedData: alignedData,
                            modelPath: String, 
                            model0: String = "model0",
                            model1: String = "model1",
                            model2: String = "model2",
                            model3: String = "model3",
                            model4: String = "model4",
                            model5: String = "model5",
                            model6: String = "model6",
                            model7: String = "model7",
                            model8: String = "model8"
                            ) : List[(String, Vector[Double])] = {
    
    // new code which works in REPL    
    // creates a List for each model (class) containing a map from the bizID to the probability of belonging in that class 
    val big = for(m <- List(model0, model1, model2, model3, model4, model5, model6, model7, model8))
      yield {
        val ds = makeDataSetTE(alignedData)
        val model = loadNN(modelPath + m + ".json", modelPath + m + ".bin")
        val scores = scoreModel(model, ds)
        val bizScores = aggImgScores2Biz(scores, alignedData)
        bizScores.toMap
    }
     
    // transforming the data structure above into a List for each bizID containing a Tuple (bizid, List[Double]) where the Vector[Double] is the 
    // the vector of probabilities 
     alignedData.data.map(_._2).distinct map ( x =>
       (x, big.map(x2 =>  x2(x)).toVector)
     )
    
  }
}