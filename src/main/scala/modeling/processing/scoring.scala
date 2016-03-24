package modeling.processing

import org.nd4j.linalg.api.ndarray.INDArray
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork

object scoring {
  
  def scoreModel(model: MultiLayerNetwork, ds: INDArray) = {
    model.output(ds)
  }
  
  /** Take model predictions from scoreModel and merge with alignedData*/
  
  def aggImgScores2Biz(scores: INDArray, alignedData: alignedData ) = {
    assert(scores.size(0) == alignedData.data.length, "alignedData and scores length are different.  They must be equal")
    def getRowIndices4Biz(mylist: List[String], mybiz: String): List[Int] = mylist.zipWithIndex.filter(x => x._1 == mybiz).map(_._2)
    def mean(xs: List[Double]) = xs.sum / xs.size

    alignedData.getBizIds.distinct.map(x => (x, {
      val irows = getRowIndices4Biz(alignedData.getBizIds, x)
      val ret = for(row <- irows) yield scores.getRow(row).getColumn(1).toString.toDouble
      mean(ret)
    }))
    
  }
  
}