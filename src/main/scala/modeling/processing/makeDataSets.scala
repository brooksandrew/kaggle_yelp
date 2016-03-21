package modeling.processing

import org.nd4j.linalg.dataset.{DataSet}
import org.nd4s.Implicits._ // should work with nd4s package
import org.nd4j.linalg.api.ndarray.INDArray

object makeDataSets {
  
  /** Creates DataSet object from the data structure of data structure from alignLables function in form List[(imgID, bizID, labels, pixelVector)]  */
  
  def makeDataSet(alignedData: List[(Int, String, Vector[Int], Set[Int])], bizClass: Int): DataSet = {
    val alignedXData = alignedData.map(x => x._3).toNDArray
    val alignedLabs = alignedData.map(x => if (x._4.contains(bizClass)) Vector(1, 0) else Vector(0, 1)).toNDArray
    new DataSet(alignedXData, alignedLabs) 
  }
  
  def makeDataSetTE(alignedData: List[(Int, String, Vector[Int])]): INDArray = {
    alignedData.map(x => x._3).toNDArray
  }
  
  
}