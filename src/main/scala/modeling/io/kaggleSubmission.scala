package modeling.io

import java.io.File

object kaggleSubmission {
  
    def writeKaggleSubmissionFile(outcsv: String, kaggleObj: List[(String, Vector[Double])], thresh: Double): Unit = {
    
    // prints to a csv or other txt file
    def printToFile(f: java.io.File)(op: java.io.PrintWriter => Unit) {
      val p = new java.io.PrintWriter(f)
      try { op(p) } finally { p.close() }
    }
    
    // assigning cutoffs for each class
    def findIndicesAboveThresh(x: Vector[Double]): Vector[Int] = {
      x.zipWithIndex.filter(x => x._1 >= thresh).map(_._2)
    }
    
    // create vector of rows to write to csv
    val ret = (for(i <- 0 until kaggleObj.length) yield {
      (kaggleObj(i)._1 + "," + findIndicesAboveThresh(kaggleObj(i)._2).mkString(" "))
    }).toVector
    
    // actually write text file
    printToFile(new File(outcsv)) {
      p => (Vector("business_ids,labels") ++ ret).foreach(p.println)
    }
    
  }
  
  
}