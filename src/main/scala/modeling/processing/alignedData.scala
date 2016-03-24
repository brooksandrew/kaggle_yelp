package modeling.processing

//import modeling.processing.aligningData.alignBizImgIds

class alignedData(dataMap: Map[Int, Vector[Int]], bizMap: Map[Int, String], labMap: Map[String, Set[Int]])
                 (rowindices: List[Int] = dataMap.keySet.toList) {
  
  // initializing alignedData with empty labMap when it is not provided (we are working with training data)
  def this(dataMap: Map[Int, Vector[Int]], bizMap: Map[Int, String])(rowindices: List[Int]) = this(dataMap, bizMap, Map("" -> Set[Int]()))(rowindices)

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
  
  // pre-computing and saving data as a val so method does not need to re-compute each time it is called. 
  lazy val data = alignLabels(dataMap, bizMap, labMap)(rowindices)
  
  // getter functions
  def getImgIds = data.map(_._1)
  def getBizIds = data.map(_._2)
  def getImgVectors = data.map(_._3)
  def getBizLabels = data.map(_._4)
  
  
  
  
}