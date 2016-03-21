package modeling.processing

import org.nd4j.linalg.dataset.{DataSet}
import org.nd4s.Implicits._ // should work with nd4s package
import org.nd4j.linalg.api.ndarray.INDArray


object aligningData {
  
   /** 
   *  
   *  Creates List tuples where each tuple includes:
   *  1) the key referring to the image (pid) (obs identifier) 
   *  2) the bizid referring (of which there are several for images assigned to) 
   *  3) label for the obs
   *  4) feature vector
   *  */
  
   def alignBizImgIds(dataMap: Map[Int, Vector[Int]], bizMap: Map[Int, String])
    (rowindices: List[Int] = dataMap.keySet.toList): List[(Int, String, Vector[Int])] = {
      for { pid <- rowindices
          val imgHasBiz = bizMap.get(pid) // returns None if img does not have a bizID
          val bid = if(imgHasBiz != None) imgHasBiz.get else "-1"
          if (dataMap.keys.toSet.contains(pid) && imgHasBiz != None)
      } yield { 
          (pid, bid, dataMap(pid))
      }
  }
  
  
  def alignLabels(dataMap: Map[Int, Vector[Int]], bizMap: Map[Int, String], labMap: Map[String, Set[Int]])
    (rowindices: List[Int] = dataMap.keySet.toList): List[(Int, String, Vector[Int], Set[Int])] = {
      def flatten1[A, B, C, D](t: ((A, B, C), D)): (A, B, C, D) = (t._1._1, t._1._2, t._1._3, t._2)
      val al = alignBizImgIds(dataMap, bizMap)()
      for { p <- al
      } yield {
        val bid = p._2
        val labs = if (labMap.keySet.contains(bid)) labMap(bid) else Set[Int]() 
        flatten1(p, labs) 
      }
  }
  
  
  
}