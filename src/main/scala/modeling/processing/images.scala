package modeling.processing

import java.io.File
import javax.imageio.ImageIO
import scala.util.matching.Regex
import imgUtils._


object images {
  
  
  /**  Define RegEx to extract jpg name from the image class which is used to match against training labels */
  val patt_get_jpg_name = new Regex("[0-9]")
  
  /** Collects all images associated with a BizId. */
  def getImgIdsForBizId(bizMap: Map[Int, String], bizIds: List[String]): List[Int] = {
    bizMap.filter(x => bizIds.exists(y => y == x._2)).map(_._1).toList
  }
  
   /** Get a list of images to load and process
      *
      * @param photoDir directory where the raw images reside
      * @param ids optional parameter to subset the images loaded from photoDir.
      * 
      * @example println(getImageIds("/Users/abrooks/Documents/kaggle_yelp_photo/train_photos/", ids=List.range(0,10)))
      */
  
  def getImageIds(photoDir: String, bizMap: Map[Int, String] = Map(-1 -> "-1"), bizIds: List[String] = List("-1")): List[String] = {
    val d =  new File(photoDir) // new File("data/images/") // too many photos?
    val imgsPath = d.listFiles().map(x => x.toString).toList
    
    if (bizMap == Map(-1 -> "-1") || bizIds == List(-1)) {
      imgsPath 
    } else {
      val imgsMap = imgsPath.map(x => patt_get_jpg_name.findAllIn(x).mkString.toInt -> x).toMap
      val imgsPathSub = getImgIdsForBizId(bizMap, bizIds)
      imgsPathSub.map(x => imgsMap(x))
    }
  }
  
  
  
   /** Read and process images into a photoID -> vector map
      *
      * @param imgs list of images to read-in.  created from getImageIds function.
      * @param resizeImgDim dimension to rescale square images to
      * @param nPixels number of pixels to maintain.  mainly used to sample image to drastically reduce runtime while testing features. 
      * 
      * @example
        val imgs = getImageIds("/Users/abrooks/Documents/kaggle_yelp_photo/train_photos/", ids=List(0,1,2,3,4))
        println(processImages(imgs, resizeImgDim = 128, nPixels = 16))    */
  
  def processImages(imgs: List[String], resizeImgDim: Int = 128, nPixels: Int = -1): Map[Int, Vector[Int]] = {       
    imgs.map(x => 
      patt_get_jpg_name.findAllIn(x).mkString.toInt -> { 
        val img0 = ImageIO.read(new File(x))
         .makeSquare
         .resizeImg(resizeImgDim, resizeImgDim) // (200, 200)
         .image2gray
       if(nPixels != -1) img0.slice(0, nPixels)
       else img0
     }   
   ).filter( x => x._2 != ())
    .toMap
    
  }
      
 

  
  
  
  
  
  
}