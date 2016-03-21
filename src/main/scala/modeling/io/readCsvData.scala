package modeling.io

import scala.io.Source

object readCsvData {
  
  /** Generic function to load in CSV */
  
  def readcsv(csv: String, rows: List[Int]=List(-1)):  List[List[String]] = {
    val src = Source.fromFile(csv) 
    def reading(csv: String): List[List[String]] = {  
      src.getLines.map(x => x.split(",").toList)
         .toList
    }
    //src.close
    try {
        if(rows==List(-1)) reading(csv)
        else rows.map(reading(csv))
    } finally {
        src.close
    }
        
  }
  
  
  /** Create map from bizid to labels of form bizid -> Set(labels)  */
  
  def readBizLabels(csv: String, rows: List[Int]=List(-1)): Map[String, Set[Int]]  = {
    val src = readcsv(csv)
    src.drop(1) // should make this conditional or handle in pattern-matching
       .map(x => x match {
          case x :: Nil => (x(0).toString, Set[Int]())
          case _ => (x(0).toString, x(1).split(" ").map(y => y.toInt).toSet)
          }).toMap
  }
  
  /** Create map from imgID to bizID of form imgID -> busID  */
   
  def readBiz2ImgLabels(csv: String, rows: List[Int] = List(-1)): Map[Int, String]  = {
    val src = readcsv(csv)
    src.drop(1) // should make this conditional or handle in pattern-matching
       .map(x => x match {
         case x :: Nil => (x(0).toInt, "-1")
          case _ => (x(0).toInt, x(1).split(" ").head)
       }).toMap
  }

  
  
  
}