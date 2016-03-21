name := "kaggle_yelp"

scalaVersion := "2.11.5"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "1.6.0" % "provided",
  "org.nd4j" % "nd4j-x86" % "0.4-rc3.8",
  "org.nd4j" % "nd4s_2.11" % "0.4-rc3.8",
  "org.deeplearning4j" % "deeplearning4j-core" % "0.4-rc3.8",
  "org.imgscalr" % "imgscalr-lib" % "4.2",
  "com.sksamuel.scrimage" %% "scrimage-core" % "2.1.0",
  "com.sksamuel.scrimage" %% "scrimage-io-extra" % "2.1.0",
  "com.sksamuel.scrimage" %% "scrimage-filters" % "2.1.0"
)

//"org.nd4j" % "nd4s_2.11" % "0.4-rc3.8"
//"org.nd4j" % "nd4j-x86" % "0.4-rc3.8"

// these are in the dl4j build.sbt
 // "commons-io" % "commons-io" % "2.4",
 // "com.google.guava" % "guava" % "19.0",
 // "org.deeplearning4j" % "deeplearning4j-core" % "0.4-rc3.8",
 // "org.deeplearning4j" % "deeplearning4j-nlp" % "0.4-rc3.8",
 // "org.deeplearning4j" % "deeplearning4j-ui" % "0.4-rc3.8",
 // "org.jblas" % "jblas" % "1.2.4",
 // "org.nd4j" % "canova-nd4j-codec" % "0.0.0.14",
 // "org.nd4j" % "canova-nd4j-image" % "0.0.0.14",
 // "org.nd4j" % "nd4j-x86" % "0.4-rc3.8"